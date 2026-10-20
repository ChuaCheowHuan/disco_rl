"""Policy prediction: an auxiliary head that predicts the next step's policy.

DiscoRL's agent has an `aux_pi` output: for every action it could take now, a
prediction of the policy it will have at the next step. It is trained toward
the actual next-step logits (stop-gradient), with terminal steps masked out
(`disco_rl/update_rules/disco.py`). The head is never acted on; it shapes the
shared representation toward what the agent's own action does to its next
decision.

This market's action is a Dict, so the head is conditioned on the
`category` action (the main decision) and predicts the logits of every
discrete component at the next step. Its loss is the sum, over those
components, of KL(next-step policy || prediction), weighted by
`policy_prediction_cost`.

Pieces:

  PolicyPredictionModule  - adds `aux_pi` from the actor embedding, written to
                            `fwd_out` on the train pass. Not built on
                            inference-only (env-runner) modules.
  PolicyPredictionLearner - adds RLlib's next-observation piece to the
                            Learner's connector when the run does not already
                            have it, computes the next-step logits under
                            `no_grad`, and adds the loss.

Needs a stateless encoder: the next step's policy of a recurrent module
depends on a state the train batch does not carry.
"""

from __future__ import annotations

from typing import List, Tuple

import gymnasium as gym
import torch
from torch import nn
from ray.rllib.connectors.learner import (
    AddNextObservationsFromEpisodesToTrainBatch,
    AddOneTsToEpisodesAndTruncate,
)
from ray.rllib.core.columns import Columns

from gym_continuousDoubleAuction.train.extensions import settings_for

NAME = "discorl"

#: `fwd_out` key holding the head's output, `[..., n_category, n_discrete_logits]`.
AUX_PI = "discorl_aux_pi"

#: Metric logged per module.
POLICY_PREDICTION_LOSS_KEY = "discorl_policy_prediction_loss"


def discrete_slices(action_space: gym.spaces.Dict) -> List[Tuple[str, slice]]:
    """Each Discrete component's slice in the flat `ACTION_DIST_INPUTS`.

    Components appear in the Dict's order; a Box contributes `2 * prod(shape)`
    inputs (RLlib's diagonal Gaussian), as in gym-CDA's `category_logit_slice`.
    """
    out, offset = [], 0
    for name, space in action_space.spaces.items():
        if isinstance(space, gym.spaces.Discrete):
            width = int(space.n)
            out.append((name, slice(offset, offset + width)))
        elif isinstance(space, gym.spaces.Box):
            width = 2 * int(torch.tensor(space.shape).prod())
        else:
            raise TypeError(f"policy_prediction: unsupported action component {name!r}")
        offset += width
    return out


def gather_discrete(logits: torch.Tensor, slices) -> torch.Tensor:
    return torch.cat([logits[..., s] for _, s in slices], dim=-1)


def per_head_kl(target: torch.Tensor, pred: torch.Tensor, widths: List[int]) -> torch.Tensor:
    """Sum over components of KL(softmax(target) || softmax(pred)), per row."""
    total = 0.0
    for t, p in zip(torch.split(target, widths, -1), torch.split(pred, widths, -1)):
        log_t = torch.log_softmax(t, -1)
        total = total + (log_t.exp() * (log_t - torch.log_softmax(p, -1))).sum(-1)
    return total


class PolicyPredictionModule:
    """Module mixin: the `aux_pi` head, on the train pass only."""

    _pp_on = False

    def setup(self):
        super().setup()
        s = type(self).extension_settings[NAME]
        if not s["policy_prediction"] or self.inference_only:
            return
        # `is_stateful()` is not usable until setup returns; this is what it checks.
        if self.get_initial_state():
            raise ValueError(
                "policy_prediction needs a stateless encoder: a recurrent module's "
                "next-step policy depends on a state the train batch does not carry."
            )
        self._pp_on = True
        self._pp_slices = discrete_slices(self.action_space)
        self._pp_widths = [s.stop - s.start for _, s in self._pp_slices]
        self._pp_n_category = int(self.action_space["category"].n)
        width = sum(self._pp_widths)
        embed = next(m for m in self.pi.modules() if isinstance(m, nn.Linear)).in_features
        self.aux_pi = nn.Linear(embed, self._pp_n_category * width)
        self._pp_actor_emb = None
        self.pi.register_forward_pre_hook(self._pp_capture)

    def _pp_capture(self, _module, args):
        self._pp_actor_emb = args[0]

    def get_non_inference_attributes(self):
        attrs = list(super().get_non_inference_attributes())
        return attrs + (["aux_pi"] if self._pp_on else [])

    def _forward_train(self, batch, **kwargs):
        out = super()._forward_train(batch, **kwargs)
        if self._pp_on:
            pred = self.aux_pi(self._pp_actor_emb)
            out[AUX_PI] = pred.unflatten(-1, (self._pp_n_category, sum(self._pp_widths)))
            self._pp_actor_emb = None
        return out


class PolicyPredictionLearner:
    """Learner mixin: next observations in the batch, and the prediction loss."""

    _pp_cost = None

    def build(self) -> None:
        super().build()
        s = settings_for(self.config.learner_config_dict, NAME)
        if not s.get("policy_prediction"):
            return
        self._pp_cost = float(s["policy_prediction_cost"])
        names = [type(c).__name__ for c in self._learner_connector.connectors]
        if AddNextObservationsFromEpisodesToTrainBatch.__name__ not in names:
            # After the extra timestep PPO adds to each episode, as gym-CDA's
            # own next-observation piece is.
            self._learner_connector.insert_after(
                AddOneTsToEpisodesAndTruncate, AddNextObservationsFromEpisodesToTrainBatch()
            )

    def _policy_value_loss(self, *, module_id, config, batch, fwd_out):
        loss = super()._policy_value_loss(
            module_id=module_id, config=config, batch=batch, fwd_out=fwd_out
        )
        module = self.module[module_id].unwrapped()
        if self._pp_cost is None or not getattr(module, "_pp_on", False):
            return loss

        with torch.no_grad():
            next_logits = module._forward({Columns.OBS: batch[Columns.NEXT_OBS]})[
                Columns.ACTION_DIST_INPUTS
            ]
            # An MoE encoder leaves routing stats behind on every pass; drop
            # this one's so they are not counted with the next train pass.
            from gym_continuousDoubleAuction.train.model.moe_learner import _collect

            _collect(module.encoder)
        target = gather_discrete(next_logits, module._pp_slices)

        category = batch[Columns.ACTIONS]["category"].long()
        aux = fwd_out[AUX_PI]
        index = category.view(*category.shape, 1, 1).expand(*category.shape, 1, aux.shape[-1])
        pred = aux.gather(-2, index).squeeze(-2)

        per_step = per_head_kl(target, pred, module._pp_widths)
        per_step = per_step * (1.0 - batch[Columns.TERMINATEDS].float())
        if Columns.LOSS_MASK in batch:
            mask = batch[Columns.LOSS_MASK]
            mean = per_step[mask].sum() / mask.sum()
        else:
            mean = per_step.mean()

        self.metrics.log_value((module_id, POLICY_PREDICTION_LOSS_KEY), mean, window=1)
        return loss + self._pp_cost * mean
