"""Env variety: redraw some market rules at the start of every episode.

DiscoRL's rule was discovered across many environments, and generalised
because of it. A league here trains on one fixed market. With `env_variety`
on, every episode draws each setting in `env_variety_keys` from its list in
`env_variety_choices`, so the agents learn under a spread of matching rules,
clearing modes and fees instead of one.

Only settings the env can change between episodes without a rebuild are
supported, and none of them changes the observation or action space:

  matching_rule  - read when `reset` builds the episode's new book
  step_clearing  - read on every step
  maker_fee_bps,
  taker_fee_bps  - held by each trader's account, which `reset` keeps

The draw happens before the env's own reset, from a generator re-seeded
whenever `reset` is given a seed, so a seeded episode repeats its market. The
settings in force are on `env.discorl_env_settings`. Applies to the whole env,
so to every agent, whichever modules `modules` selects.
"""

from __future__ import annotations

from decimal import Decimal
from itertools import product
from typing import Any, Dict

import numpy as np

from gym_continuousDoubleAuction.envs.account.account import validate_fee_rates
from gym_continuousDoubleAuction.envs.exchg.exchg_helper import Exchg_Helper
from gym_continuousDoubleAuction.envs.orderbook.orderbook import OrderBook

#: The keys this component can vary, and the values each may take.
SUPPORTED = {
    "matching_rule": OrderBook.MATCHING_RULES,
    "step_clearing": Exchg_Helper.STEP_CLEARINGS,
    "maker_fee_bps": None,
    "taker_fee_bps": None,
}

FEE_KEYS = ("maker_fee_bps", "taker_fee_bps")


def validate_choices(keys, choices) -> None:
    """Raise unless every key is supported and every combination it can draw is valid."""
    unsupported = sorted(set(keys) - set(SUPPORTED))
    if unsupported:
        raise ValueError(
            f"env_variety_keys may only name {sorted(SUPPORTED)}, which the env can change "
            f"between episodes without changing its spaces; got {unsupported}."
        )
    for key in keys:
        values = choices.get(key)
        if not isinstance(values, list) or not values:
            raise ValueError(f"env_variety_choices[{key!r}] must be a non-empty list.")
        allowed = SUPPORTED[key]
        if allowed is not None and not set(values) <= set(allowed):
            raise ValueError(f"env_variety_choices[{key!r}] may only hold {list(allowed)}.")
    if any(k in keys for k in FEE_KEYS):
        makers = choices["maker_fee_bps"] if "maker_fee_bps" in keys else [None]
        takers = choices["taker_fee_bps"] if "taker_fee_bps" in keys else [None]
        for maker, taker in product(makers, takers):
            if maker is not None and taker is not None:
                validate_fee_rates(maker, taker)


def _apply(env, drawn: Dict[str, Any]) -> None:
    if "matching_rule" in drawn:
        env.matching_rule = drawn["matching_rule"]
    if "step_clearing" in drawn:
        env.step_clearing = drawn["step_clearing"]
    fees = {k: drawn[k] for k in FEE_KEYS if k in drawn}
    if fees:
        maker = fees.get("maker_fee_bps", env.maker_fee_bps)
        taker = fees.get("taker_fee_bps", env.taker_fee_bps)
        validate_fee_rates(maker, taker)
        env.maker_fee_bps, env.taker_fee_bps = maker, taker
        for trader in env.traders:
            trader.acc.maker_fee_bps = Decimal(str(maker))
            trader.acc.taker_fee_bps = Decimal(str(taker))


def wrap(env, settings):
    """The extension's env wrapper: a per-episode draw ahead of `reset`."""
    if not settings["env_variety"]:
        return env
    keys = list(settings["env_variety_keys"])
    choices = settings["env_variety_choices"]
    rng = np.random.default_rng()
    original_reset = env.reset

    def reset(*, seed=None, options=None):
        nonlocal rng
        if seed is not None:
            rng = np.random.default_rng(seed)
        drawn = {k: choices[k][int(rng.integers(len(choices[k])))] for k in keys}
        _apply(env, drawn)
        env.discorl_env_settings = drawn
        return original_reset(seed=seed, options=options)

    env.reset = reset
    env.discorl_env_settings = {}
    return env
