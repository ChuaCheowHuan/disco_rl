"""Advantage EMA: normalise advantages by lifetime moments, not each batch's.

RLlib's GAE connector standardises each batch's advantages by that batch's own
mean and standard deviation. DiscoRL instead keeps an exponential moving
average of the advantage's first and second moments over the agent's whole
lifetime, debiased as Adam debiases its moments, and normalises with those
(`disco_rl/utils.py`, `MovingAverage`). A batch of unusually large or small
advantages then keeps its scale relative to the past, instead of being
stretched to unit variance.

The piece sits right after GAE in the Learner's connector pipeline. GAE drops
the raw advantages, so they are recomputed as `value_targets - V(s)` with the
same weights GAE just used, then normalised with the updated EMA. The moments
are part of the Learner's state, so they survive a checkpoint.
"""

from __future__ import annotations

from typing import Dict, Iterable

import torch
from ray.rllib.connectors.connector_v2 import ConnectorV2
from ray.rllib.connectors.learner import GeneralAdvantageEstimation
from ray.rllib.core.columns import Columns
from ray.rllib.evaluation.postprocessing import Postprocessing

from gym_continuousDoubleAuction.train.extensions import settings_for

NAME = "discorl"

#: Key in the Learner's state holding the per-module moments.
STATE_KEY = "discorl_adv_ema"

#: DiscoRL's normalisation epsilons (`MovingAverage.normalize`).
EPS = 1e-6
ROOT_EPS = 1e-12


class EmaMoments:
    """One stream's debiased EMA of the first and second moment."""

    def __init__(self, decay: float):
        self.decay = decay
        self.moment1 = 0.0
        self.moment2 = 0.0
        self.decay_product = 1.0

    def update(self, values: torch.Tensor) -> None:
        d = self.decay
        self.moment1 = d * self.moment1 + (1.0 - d) * float(values.mean())
        self.moment2 = d * self.moment2 + (1.0 - d) * float((values**2).mean())
        self.decay_product *= d

    def normalize(self, values: torch.Tensor) -> torch.Tensor:
        debias = 1.0 / (1.0 - self.decay_product)
        mean = self.moment1 * debias
        var = max(0.0, self.moment2 * debias - mean**2)
        return (values - mean) / ((var + ROOT_EPS) ** 0.5 + EPS)

    def get_state(self) -> Dict[str, float]:
        return {
            "moment1": self.moment1,
            "moment2": self.moment2,
            "decay_product": self.decay_product,
        }

    def set_state(self, state: Dict[str, float]) -> None:
        self.moment1 = state["moment1"]
        self.moment2 = state["moment2"]
        self.decay_product = state["decay_product"]


class EmaAdvantages(ConnectorV2):
    """Replaces GAE's per-batch standardisation on the given modules."""

    def __init__(self, module_ids: Iterable[str], decay: float, **kwargs):
        super().__init__(**kwargs)
        self.moments = {mid: EmaMoments(decay) for mid in module_ids}

    def __call__(self, *, rl_module, batch, episodes=None, **kwargs):
        for module_id, moments in self.moments.items():
            if module_id not in batch:
                continue
            mbatch = batch[module_id]
            with torch.no_grad():
                values = rl_module[module_id].unwrapped().compute_values(mbatch)
            raw = mbatch[Postprocessing.VALUE_TARGETS] - values
            valid = mbatch[Columns.LOSS_MASK] if Columns.LOSS_MASK in mbatch else None
            moments.update(raw[valid] if valid is not None else raw)
            mbatch[Postprocessing.ADVANTAGES] = moments.normalize(raw).to(raw.dtype)
        return batch

    def get_state(self, components=None, *, not_components=None, **kwargs):
        return {mid: m.get_state() for mid, m in self.moments.items()}

    def set_state(self, state) -> None:
        for mid, s in state.items():
            if mid in self.moments:
                self.moments[mid].set_state(s)


class AdvEmaLearner:
    """Learner mixin: puts `EmaAdvantages` after GAE and keeps its moments in the state."""

    _adv_ema = None

    def build(self) -> None:
        super().build()
        s = settings_for(self.config.learner_config_dict, NAME)
        if not s.get("adv_ema"):
            return
        # `modules` was checked against the trainable modules when the config was built.
        self._adv_ema = EmaAdvantages(s["modules"], decay=float(s["adv_ema_decay"]))
        self._learner_connector.insert_after(GeneralAdvantageEstimation, self._adv_ema)

    def get_state(self, components=None, *, not_components=None, **kwargs):
        state = super().get_state(components, not_components=not_components, **kwargs)
        if self._adv_ema is not None and components is None:
            state[STATE_KEY] = self._adv_ema.get_state()
        return state

    def set_state(self, state) -> None:
        state = dict(state)
        saved = state.pop(STATE_KEY, None)
        super().set_state(state)
        if self._adv_ema is not None and saved is not None:
            self._adv_ema.set_state(saved)
