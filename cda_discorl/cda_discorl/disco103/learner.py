"""Disco103 in place of PPO's loss on the selected modules.

The rule produces its targets from whole trajectories (its meta-network runs
an LSTM backward over time, and its value targets are TD(lambda)), while
RLlib's PPO Learner shuffles timesteps into minibatches. So the targets are
computed once per update batch, by a connector piece right after GAE, where
the batch still holds whole episodes in order; they are written to the batch
as columns, and each minibatch's loss reads them per row. With several epochs
the targets stay fixed across them, as PPO's advantages do.

Per discrete action component, the loss is DiscoRL's agent loss
(`disco_rl/update_rules/disco.py`):

  pi_cost         * KL(pi_hat || pi)
  y_cost          * KL(y_hat || y)
  z_cost          * KL(z_hat || z[a])
  aux_policy_cost * KL(pi_next || aux_pi[a])   (terminal steps masked)
  value_cost      * CE(q[a], two_hot(h(V(q[a]) + td)))

with the published costs. The continuous `size` component keeps PPO's clipped
surrogate and entropy bonus (`disco103_size_head: "ppo"`), and the scalar
critic keeps PPO's value loss, since GAE's advantages for `size` come from it.

Rewards are multiplied by `disco103_reward_scale` before the rule sees them:
the rule was discovered on Atari-scale rewards, and this market's are around
1e-3 per step.

The target network is an EMA copy of the module (coefficient 0.9, the rule's
`target_params_coeff`), updated after every optimiser step. It and each
stream's meta-network state are part of the Learner's state.
"""

from __future__ import annotations

import copy
from typing import Dict

import numpy as np
import torch
from ray.rllib.algorithms.ppo.ppo import LEARNER_RESULTS_VF_EXPLAINED_VAR_KEY
from ray.rllib.connectors.connector_v2 import ConnectorV2
from ray.rllib.connectors.learner import GeneralAdvantageEstimation
from ray.rllib.core.columns import Columns
from ray.rllib.evaluation.postprocessing import Postprocessing
from ray.rllib.utils.torch_utils import explained_variance

from gym_continuousDoubleAuction.train.extensions import settings_for
from gym_continuousDoubleAuction.train.model.action_mask import MASK_LOGIT

from cda_discorl.categorical_value import h, h_inv, support, two_hot
from cda_discorl.disco103.heads import out_key
from cda_discorl.policy_prediction import per_head_kl

NAME = "discorl"

#: Key in the Learner's state.
STATE_KEY = "discorl_disco103"

#: Batch column prefix for the targets: `discorl_d103_target/<component>/<name>`.
TARGET_PREFIX = "discorl_d103_target"


def target_key(component: str, name: str) -> str:
    return f"{TARGET_PREFIX}/{component}/{name}"


def _kl(target_logits, logits):
    log_t = torch.log_softmax(target_logits, -1)
    return (log_t.exp() * (log_t - torch.log_softmax(logits, -1))).sum(-1)


def _to_time_major(flat: np.ndarray, lens, pad_last=True) -> np.ndarray:
    """`[sum(lens), ...]` rows, episode after episode, to `[max(lens), n, ...]`.

    Shorter episodes are padded by repeating their last row, which the rule
    then sees as terminal (see `_terminal_time_major`)."""
    out = np.zeros((max(lens), len(lens)) + flat.shape[1:], dtype=flat.dtype)
    start = 0
    for i, n in enumerate(lens):
        out[:n, i] = flat[start : start + n]
        if pad_last and n < out.shape[0]:
            out[n:, i] = flat[start + n - 1]
        start += n
    return out


def _from_time_major(tm: np.ndarray, lens) -> np.ndarray:
    """`[T, n, ...]` targets back to rows; each episode's bootstrap row gets zeros."""
    rows = []
    for i, n in enumerate(lens):
        part = np.zeros((n,) + tm.shape[2:], dtype=tm.dtype)
        part[: n - 1] = tm[: n - 1, i]
        rows.append(part)
    return np.concatenate(rows, axis=0)


class Disco103Targets(ConnectorV2):
    """Writes Disco103's targets for the selected modules into the batch."""

    def __init__(self, learner, module_ids, reward_scale, **kwargs):
        super().__init__(**kwargs)
        self.learner = learner
        self.module_ids = list(module_ids)
        self.reward_scale = reward_scale

    def __call__(self, *, rl_module, batch, episodes, **kwargs):
        episodes = list(
            self.single_agent_episode_iterator(episodes, agents_that_stepped_only=False)
        )
        for module_id in self.module_ids:
            if module_id not in batch:
                continue
            lens = [len(e) for e in episodes if e.module_id in (None, module_id)]
            self.learner._d103_write_targets(
                rl_module[module_id].unwrapped(),
                module_id,
                batch[module_id],
                lens,
                self.reward_scale,
            )
        return batch


class Disco103Learner:
    """Learner mixin: Disco103's loss on the selected modules."""

    _d103 = None

    def build(self) -> None:
        super().build()
        s = settings_for(self.config.learner_config_dict, NAME)
        if not s.get("disco103"):
            return
        from cda_discorl.disco103.bridge import Bridge

        self._d103 = Bridge(int(s["value_bins"]), float(s["value_max_abs"]))
        self._d103_bins = int(s["value_bins"])
        self._d103_max_abs = float(s["value_max_abs"])
        self._d103_coeff = float(self._d103.hyper_params["target_params_coeff"])
        self._d103_targets: Dict[str, torch.nn.Module] = {}
        self._d103_streams: Dict[str, Dict[str, dict]] = {}
        for module_id in s["modules"]:
            module = self.module[module_id].unwrapped()
            target = copy.deepcopy(module)
            target.requires_grad_(False)
            self._d103_targets[module_id] = target
            self._d103_streams[module_id] = {
                name: self._d103.new_state() for name, _ in module._d103_slices
            }
        self._learner_connector.insert_after(
            GeneralAdvantageEstimation,
            Disco103Targets(self, s["modules"], float(s["disco103_reward_scale"])),
        )

    # -- targets, once per update batch --------------------------------------

    def _d103_write_targets(self, module, module_id, mbatch, lens, reward_scale) -> None:
        from gym_continuousDoubleAuction.train.model.moe_learner import _collect

        with torch.no_grad():
            online = module._forward_train(mbatch)
            target = self._d103_targets[module_id]._forward_train(mbatch)
        # An MoE encoder leaves routing stats behind on every pass; drop these.
        _collect(module.encoder)
        _collect(self._d103_targets[module_id].encoder)
        to_np = lambda t: t.detach().float().cpu().numpy()  # noqa: E731
        tm = lambda t: _to_time_major(to_np(t), lens)  # noqa: E731

        rewards = to_np(mbatch[Columns.REWARDS]) * reward_scale
        terminated = to_np(mbatch[Columns.TERMINATEDS]).astype(bool)
        # Each episode: n rows, the last the bootstrap step PPO appends. The
        # rule takes T + 1 observations and T transitions.
        r_tm = _to_time_major(rewards, lens, pad_last=False)[:-1]
        term_tm = _to_time_major(terminated, lens, pad_last=False)[:-1]
        for i, n in enumerate(lens):
            term_tm[n - 1 :, i] = True  # padding past an episode's end is terminal

        device = mbatch[Columns.REWARDS].device
        new_cols = {}
        for name, sl in module._d103_slices:
            actions = _to_time_major(to_np(mbatch[Columns.ACTIONS][name]).astype(np.int64), lens)
            agent_out = {
                "logits": tm(online[Columns.ACTION_DIST_INPUTS][..., sl]),
                "y": tm(online[out_key(name, "y")]),
                "z": tm(online[out_key(name, "z")]),
                "aux_pi": tm(online[out_key(name, "aux_pi")]),
                "q": tm(online[out_key(name, "q")]),
            }
            target_out = {
                "logits": tm(target[Columns.ACTION_DIST_INPUTS][..., sl]),
                "y": tm(target[out_key(name, "y")]),
                "z": tm(target[out_key(name, "z")]),
                "q": tm(target[out_key(name, "q")]),
            }
            out, self._d103_streams[module_id][name] = self._d103.targets(
                self._d103_streams[module_id][name],
                actions=actions,
                rewards=r_tm,
                is_terminal=term_tm,
                agent_out=agent_out,
                behaviour_logits=tm(mbatch[Columns.ACTION_DIST_INPUTS][..., sl]),
                target_out=target_out,
            )
            for key in ("pi", "y", "z", "q_td"):
                new_cols[target_key(name, key)] = _from_time_major(out[key], lens)
            # The next step's logits, for the policy-prediction term.
            nxt = np.concatenate([agent_out["logits"][1:], agent_out["logits"][-1:]], axis=0)
            new_cols[target_key(name, "next_logits")] = _from_time_major(nxt, lens)
        for key, value in new_cols.items():
            mbatch[key] = torch.as_tensor(value, device=device)

    # -- the loss, per minibatch ---------------------------------------------

    def _policy_value_loss(self, *, module_id, config, batch, fwd_out):
        module = self.module[module_id].unwrapped()
        if self._d103 is None or not getattr(module, "_d103_on", False):
            return super()._policy_value_loss(
                module_id=module_id, config=config, batch=batch, fwd_out=fwd_out
            )

        mask = batch.get(Columns.LOSS_MASK)

        def mean(x):
            return x[mask].sum() / mask.sum() if mask is not None else x.mean()

        hp = self._d103.hyper_params
        points = support(
            self._d103_bins, self._d103_max_abs, device=fwd_out[Columns.ACTION_DIST_INPUTS].device
        )
        not_terminal = 1.0 - batch[Columns.TERMINATEDS].float()
        logs = {}
        total = 0.0
        for name, sl in module._d103_slices:
            a = batch[Columns.ACTIONS][name].long()

            def pick(t):
                idx = a.view(*a.shape, 1, 1).expand(*a.shape, 1, t.shape[-1])
                return t.gather(-2, idx).squeeze(-2)

            logits = fwd_out[Columns.ACTION_DIST_INPUTS][..., sl]
            # The meta-net can put mass on an action the env masked out; against
            # a masked logit that KL is unbounded, so the target is masked too.
            pi_hat = torch.where(
                logits <= MASK_LOGIT / 2, MASK_LOGIT, batch[target_key(name, "pi")]
            )
            pi = mean(_kl(pi_hat, logits))
            y = mean(_kl(batch[target_key(name, "y")], fwd_out[out_key(name, "y")]))
            z = mean(_kl(batch[target_key(name, "z")], pick(fwd_out[out_key(name, "z")])))
            aux = mean(
                per_head_kl(
                    batch[target_key(name, "next_logits")],
                    pick(fwd_out[out_key(name, "aux_pi")]),
                    [logits.shape[-1]],
                )
                * not_terminal
            )
            q_a = pick(fwd_out[out_key(name, "q")])
            with torch.no_grad():
                v = h_inv((torch.softmax(q_a, -1) * points).sum(-1))
                target = two_hot(h(v + batch[target_key(name, "q_td")].to(v.dtype)), points)
            value = mean(-(target * torch.log_softmax(q_a, -1)).sum(-1))

            total = total + (
                hp["pi_cost"] * pi
                + hp["y_cost"] * y
                + hp["z_cost"] * z
                + hp["aux_policy_cost"] * aux
                + hp["value_cost"] * value
            )
            for key, val in (("pi", pi), ("y", y), ("z", z), ("aux", aux), ("q", value)):
                logs[f"discorl_d103_{name}_{key}_loss"] = val

        total = total + self._d103_size_and_critic_loss(
            module, module_id, config, batch, fwd_out, mean, logs
        )
        self.metrics.log_dict(logs, key=module_id, window=1)
        return total

    def _d103_size_and_critic_loss(self, module, module_id, config, batch, fwd_out, mean, logs):
        """PPO's clipped surrogate and entropy on the continuous components, plus its critic."""
        curr = module.get_train_action_dist_cls().from_logits(fwd_out[Columns.ACTION_DIST_INPUTS])
        prev = module.get_exploration_action_dist_cls().from_logits(
            batch[Columns.ACTION_DIST_INPUTS]
        )
        discrete = {name for name, _ in module._d103_slices}
        loss = 0.0
        adv = batch[Postprocessing.ADVANTAGES]
        entropy_coeff = self.entropy_coeff_schedulers_per_module[module_id].get_current_value()
        for name, child in curr._original_struct.items():
            if name in discrete:
                continue
            act = batch[Columns.ACTIONS][name]
            ratio = torch.exp(child.logp(act) - prev._original_struct[name].logp(act))
            surrogate = torch.min(
                adv * ratio, adv * torch.clamp(ratio, 1 - config.clip_param, 1 + config.clip_param)
            )
            entropy = child.entropy()
            loss = loss + mean(-surrogate - entropy_coeff * entropy)
            logs[f"discorl_d103_{name}_policy_loss"] = mean(-surrogate)

        values = module.compute_values(batch, embeddings=fwd_out.get(Columns.EMBEDDINGS))
        vf = torch.clamp(
            (values - batch[Postprocessing.VALUE_TARGETS]) ** 2, 0, config.vf_clip_param
        )
        vf_loss = mean(vf)
        logs["vf_loss"] = vf_loss
        logs[LEARNER_RESULTS_VF_EXPLAINED_VAR_KEY] = explained_variance(
            batch[Postprocessing.VALUE_TARGETS], values
        )
        return loss + config.vf_loss_coeff * vf_loss

    # -- the target network and the state ------------------------------------

    def apply_gradients(self, gradients_dict) -> None:
        super().apply_gradients(gradients_dict)
        if self._d103 is None:
            return
        c = self._d103_coeff
        with torch.no_grad():
            for module_id, target in self._d103_targets.items():
                online = self.module[module_id].unwrapped()
                for t, p in zip(target.parameters(), online.parameters()):
                    t.mul_(c).add_(p.detach(), alpha=1.0 - c)

    def get_state(self, components=None, *, not_components=None, **kwargs):
        state = super().get_state(components, not_components=not_components, **kwargs)
        if self._d103 is not None and components is None:
            from cda_discorl.disco103.bridge import state_to_numpy

            state[STATE_KEY] = {
                "targets": {
                    mid: {k: v.detach().cpu() for k, v in t.state_dict().items()}
                    for mid, t in self._d103_targets.items()
                },
                "streams": state_to_numpy(self._d103_streams),
            }
        return state

    def set_state(self, state) -> None:
        state = dict(state)
        saved = state.pop(STATE_KEY, None)
        super().set_state(state)
        if self._d103 is not None and saved is not None:
            from cda_discorl.disco103.bridge import state_from_numpy

            for mid, weights in saved["targets"].items():
                if mid in self._d103_targets:
                    self._d103_targets[mid].load_state_dict(weights)
            self._d103_streams = state_from_numpy(saved["streams"])
