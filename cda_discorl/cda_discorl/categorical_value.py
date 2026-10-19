"""Categorical value: a two-hot critic over an h-transformed support.

DiscoRL's agent predicts its value as a distribution over `value_bins` points
rather than as one number, trained with a cross-entropy against a two-hot
encoding of the return. The support lives in h-space (Pohlen et al., 2018),
which compresses large returns so one set of bins covers small and large
scales alike.

Two pieces, because RLlib keeps the value head on the module and the loss on
the Learner:

  CategoricalValueModule  - swaps the value head's last layer for `value_bins`
                            logits. `compute_values` still returns one scalar
                            per step (the expectation, mapped back from
                            h-space), so GAE, the value targets and every
                            metric RLlib computes from it are unchanged.
  CategoricalValueLearner - on a module that has the head, runs PPO's loss with
                            `vf_loss_coeff` set to 0 and adds the cross-entropy
                            in its place, weighted by the real `vf_loss_coeff`.
                            PPO's squared-error value loss is still logged as a
                            diagnostic, so `vf_explained_var` stays comparable
                            with a plain PPO module.

`value_max_abs` bounds the support in h-space. Bins are `2 * value_max_abs /
(value_bins - 1)` apart; the paper's 300 with 601 bins is one unit, and returns
whose h-value is well under one unit all land between the two middle bins. Set
it to the scale of this run's returns.
"""

from __future__ import annotations

import torch
from torch import nn
from ray.rllib.core.columns import Columns
from ray.rllib.evaluation.postprocessing import Postprocessing

from gym_continuousDoubleAuction.train.extensions import settings_for

NAME = "discorl"

#: The h-transform's linear term, as in Pohlen et al. and rlax.
H_EPS = 1e-3

#: Metric logged per module, beside PPO's own `vf_loss`.
CATEGORICAL_VF_LOSS_KEY = "discorl_categorical_vf_loss"


def h(x: torch.Tensor) -> torch.Tensor:
    """sign(x) (sqrt(|x| + 1) - 1) + eps x."""
    return torch.sign(x) * (torch.sqrt(torch.abs(x) + 1.0) - 1.0) + H_EPS * x


def h_inv(y: torch.Tensor) -> torch.Tensor:
    """The exact inverse of `h`."""
    root = torch.sqrt(1.0 + 4.0 * H_EPS * (torch.abs(y) + 1.0 + H_EPS))
    return torch.sign(y) * (((root - 1.0) / (2.0 * H_EPS)) ** 2 - 1.0)


def support(bins: int, max_abs: float, device=None) -> torch.Tensor:
    return torch.linspace(-max_abs, max_abs, bins, device=device)


def two_hot(y: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """Probabilities on `points` whose mean is `y`, clipped to the support."""
    y = y.clamp(points[0], points[-1])
    step = points[1] - points[0]
    pos = (y - points[0]) / step
    lower = pos.floor().clamp(0, len(points) - 2)
    upper_w = (pos - lower).clamp(0.0, 1.0)
    out = torch.zeros(*y.shape, len(points), device=y.device, dtype=y.dtype)
    out.scatter_(-1, lower.long().unsqueeze(-1), (1.0 - upper_w).unsqueeze(-1))
    out.scatter_add_(-1, (lower.long() + 1).unsqueeze(-1), upper_w.unsqueeze(-1))
    return out


def _last_scalar_linear(head: nn.Module):
    """The value head's final `Linear(n, 1)`, and the module holding it."""
    found = None
    for parent in head.modules():
        for name, child in parent.named_children():
            if isinstance(child, nn.Linear) and child.out_features == 1:
                found = (parent, name, child)
    if found is None:
        raise ValueError("categorical_value: the value head has no Linear(n, 1) to replace.")
    return found


class CategoricalValueModule:
    """Module mixin: `value_bins` logits from the value head, one scalar out.

    Inert unless `categorical_value` is on: the extension composes one module
    mixin for all its components, and each switches itself.
    """

    _cv_on = False

    def setup(self):
        super().setup()
        s = type(self).extension_settings[NAME]
        if not s["categorical_value"]:
            return
        self._cv_on = True
        self._cv_bins = int(s["value_bins"])
        self._cv_max_abs = float(s["value_max_abs"])
        # An inference-only module (on an env runner) has no value head.
        vf = getattr(self, "vf", None)
        if vf is None:
            return
        parent, name, old = _last_scalar_linear(vf)
        setattr(parent, name, nn.Linear(old.in_features, self._cv_bins, bias=old.bias is not None))

    def compute_value_logits(self, batch, embeddings=None) -> torch.Tensor:
        """The value head's logits over the h-space support, `[..., value_bins]`."""
        if embeddings is None:
            # Let the base class run the encoder; capture the head's input.
            captured = {}
            handle = self.vf.register_forward_hook(lambda mod, inp, out: captured.update(out=out))
            try:
                super().compute_values(batch)
            finally:
                handle.remove()
            return captured["out"]
        return self.vf(embeddings)

    def compute_values(self, batch, embeddings=None):
        if not self._cv_on:
            return super().compute_values(batch, embeddings=embeddings)
        logits = self.compute_value_logits(batch, embeddings=embeddings)
        points = support(self._cv_bins, self._cv_max_abs, device=logits.device)
        return h_inv((torch.softmax(logits, dim=-1) * points).sum(-1))


class CategoricalValueLearner:
    """Learner mixin: cross-entropy on the two-hot return in place of PPO's MSE."""

    _cv_configs = None

    def _policy_value_loss(self, *, module_id, config, batch, fwd_out):
        module = self.module[module_id].unwrapped()
        if not getattr(module, "_cv_on", False):
            return super()._policy_value_loss(
                module_id=module_id, config=config, batch=batch, fwd_out=fwd_out
            )

        # PPO's loss without its value term. One copy per module, made once.
        if self._cv_configs is None:
            self._cv_configs = {}
        if module_id not in self._cv_configs:
            silent = config.copy(copy_frozen=False)
            silent.vf_loss_coeff = 0.0
            self._cv_configs[module_id] = silent
        loss = super()._policy_value_loss(
            module_id=module_id, config=self._cv_configs[module_id], batch=batch, fwd_out=fwd_out
        )

        s = settings_for(self.config.learner_config_dict, NAME)
        logits = module.compute_value_logits(batch, embeddings=fwd_out.get(Columns.EMBEDDINGS))
        points = support(int(s["value_bins"]), float(s["value_max_abs"]), device=logits.device)
        target = two_hot(h(batch[Postprocessing.VALUE_TARGETS].to(logits.dtype)), points)
        ce = -(target * torch.log_softmax(logits, dim=-1)).sum(-1)
        if Columns.LOSS_MASK in batch:
            mask = batch[Columns.LOSS_MASK]
            ce_mean = ce[mask].sum() / mask.sum()
        else:
            ce_mean = ce.mean()

        self.metrics.log_value((module_id, CATEGORICAL_VF_LOSS_KEY), ce_mean, window=1)
        return loss + config.vf_loss_coeff * ce_mean
