"""Target policy: anchor the policy to a slow-moving copy of itself.

DiscoRL's agent keeps target parameters, an exponential moving average of its
own (`target_params = c * target_params + (1 - c) * params`, with `c` 0.9 in
the paper's agent), and its update rule reads the target network's policy.
Here that copy is used directly as a trust region: the loss gains

    target_kl_cost * KL(pi_target || pi)

on the selected modules, and the copy moves `1 - target_coeff` of the way to
the online weights after every optimiser step. PPO's clipping bounds how far
one update moves the policy from the one that sampled the data; this bounds
how far a run of updates drifts from where the policy has recently been,
which matters when the league keeps swapping the opponents under it.

The copy lives on the Learner, not the module, so env runners never see it;
its weights are part of the Learner's state and survive a checkpoint.
"""

from __future__ import annotations

import copy
from typing import Dict

import torch
from ray.rllib.core.columns import Columns

from gym_continuousDoubleAuction.train.extensions import settings_for

NAME = "discorl"

#: Key in the Learner's state holding each target module's weights.
STATE_KEY = "discorl_target_policy"

#: Metric logged per module.
TARGET_KL_KEY = "discorl_target_kl"


class TargetPolicyLearner:
    """Learner mixin: an EMA copy per selected module and a KL to it."""

    _targets: Dict[str, torch.nn.Module] = None

    def build(self) -> None:
        super().build()
        s = settings_for(self.config.learner_config_dict, NAME)
        if not s.get("target_policy"):
            return
        self._target_coeff = float(s["target_coeff"])
        self._target_kl_cost = float(s["target_kl_cost"])
        self._targets = {}
        for module_id in s["modules"]:
            target = copy.deepcopy(self.module[module_id].unwrapped())
            target.requires_grad_(False)
            self._targets[module_id] = target

    def _policy_value_loss(self, *, module_id, config, batch, fwd_out):
        loss = super()._policy_value_loss(
            module_id=module_id, config=config, batch=batch, fwd_out=fwd_out
        )
        if not self._targets or module_id not in self._targets:
            return loss

        module = self.module[module_id].unwrapped()
        dist_cls = module.get_train_action_dist_cls()
        with torch.no_grad():
            target_inputs = self._targets[module_id]._forward(batch)[Columns.ACTION_DIST_INPUTS]
        target_dist = dist_cls.from_logits(target_inputs)
        online_dist = dist_cls.from_logits(fwd_out[Columns.ACTION_DIST_INPUTS])
        kl = target_dist.kl(online_dist)
        if Columns.LOSS_MASK in batch:
            mask = batch[Columns.LOSS_MASK]
            mean_kl = kl[mask].sum() / mask.sum()
        else:
            mean_kl = kl.mean()

        self.metrics.log_value((module_id, TARGET_KL_KEY), mean_kl, window=1)
        return loss + self._target_kl_cost * mean_kl

    def apply_gradients(self, gradients_dict) -> None:
        super().apply_gradients(gradients_dict)
        if not self._targets:
            return
        c = self._target_coeff
        with torch.no_grad():
            for module_id, target in self._targets.items():
                online = self.module[module_id].unwrapped()
                for t, p in zip(target.parameters(), online.parameters()):
                    t.mul_(c).add_(p.detach(), alpha=1.0 - c)

    def get_state(self, components=None, *, not_components=None, **kwargs):
        state = super().get_state(components, not_components=not_components, **kwargs)
        if self._targets and components is None:
            state[STATE_KEY] = {
                mid: {k: v.detach().cpu() for k, v in t.state_dict().items()}
                for mid, t in self._targets.items()
            }
        return state

    def set_state(self, state) -> None:
        state = dict(state)
        saved = state.pop(STATE_KEY, None)
        super().set_state(state)
        if self._targets and saved is not None:
            for mid, weights in saved.items():
                if mid in self._targets:
                    self._targets[mid].load_state_dict(weights)
