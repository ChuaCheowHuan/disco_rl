"""The `discorl` settings block: every key, its default, and the rules between them.

gym-CDA fills in these defaults and rejects any key not listed here before
`validate` sees the block, so `validate` can index every key directly.

A run sets the block in its train config:

    "extensions_enabled": ["discorl"],
    "extension_settings": {"discorl": {"modules": ["policy_0"], "target_policy": true}}

Every component is off by default, so enabling the extension with no settings
changes nothing; `modules` limits the components to some trainable modules,
leaving the rest as plain PPO for an in-run comparison.

The components are the DiscoRL mechanisms that do not depend on the reward's
scale, plus `value_norm`, which removes that dependence from the critic. The
categorical critic and the frozen Disco103 rule are not here: at this market's
scale and budget neither learned (see the README).
"""

from __future__ import annotations

from typing import Any, Dict

from gym_continuousDoubleAuction.train.policy.policy_handler import POLICY_PREFIX

from cda_discorl import env_variety

DEFAULTS: Dict[str, Any] = {
    # Trainable modules the components apply to; the others stay plain PPO.
    "modules": ["policy_0"],
    # KL from the policy to an EMA copy of itself.
    "target_policy": False,
    "target_coeff": 0.9,
    "target_kl_cost": 0.1,
    # Advantages normalised by a lifetime EMA of their scale, not per batch.
    "adv_ema": False,
    "adv_ema_decay": 0.99,
    # Critic learns returns normalised by a debiased EMA of the value targets.
    "value_norm": False,
    "value_norm_decay": 0.99,
    # Per-episode draw of these env settings, each from its list of choices.
    "env_variety": False,
    "env_variety_keys": ["matching_rule", "step_clearing", "maker_fee_bps", "taker_fee_bps"],
    "env_variety_choices": {
        "matching_rule": ["fifo", "pro_rata"],
        "step_clearing": ["sequential", "batch"],
        "maker_fee_bps": [0, 1],
        "taker_fee_bps": [0, 2],
    },
    # Auxiliary head predicting the next step's policy. Experimental.
    "policy_prediction": False,
    "policy_prediction_cost": 1.0,
}

#: The on/off switches, one per component.
COMPONENTS = (
    "target_policy",
    "adv_ema",
    "value_norm",
    "env_variety",
    "policy_prediction",
)

#: Components whose code has landed. A switch for any other one is refused
#: rather than accepted and ignored, which would read as "it made no difference".
BUILT: frozenset = frozenset(COMPONENTS)

#: Components whose state lives on one Learner. With several Learners each would
#: keep its own copy, fitted to its own shard of the batch.
PER_LEARNER_STATE = ("target_policy", "adv_ema", "value_norm")

#: Settings that add a tensor to the module; changing one on a resume fails as structural.
STRUCTURAL_KEYS = ("value_norm", "policy_prediction")


def _fail(message: str) -> None:
    raise ValueError(f"extension_settings['discorl']: {message}")


def _in_unit_interval(settings, key: str) -> None:
    if not 0.0 <= settings[key] < 1.0:
        _fail(f"{key} must be in [0, 1), got {settings[key]!r}.")


def _non_negative(settings, key: str) -> None:
    if not settings[key] >= 0:
        _fail(f"{key} must be non-negative, got {settings[key]!r}.")


def validate(settings: Dict[str, Any], cfg) -> None:
    """Raise on a bad value or combination. Called with defaults filled in."""
    trainable = [f"{POLICY_PREFIX}{i}" for i in range(cfg.num_trained_agents)]
    modules = settings["modules"]
    if not isinstance(modules, list) or not modules:
        _fail("modules must be a non-empty list of trainable module ids.")
    unknown = sorted(set(modules) - set(trainable))
    if unknown:
        _fail(f"modules names {unknown}, which are not trainable. Trainable: {trainable}.")

    for name in COMPONENTS:
        if not isinstance(settings[name], bool):
            _fail(f"{name} must be true or false, got {settings[name]!r}.")
    unbuilt = sorted(n for n in COMPONENTS if settings[n] and n not in BUILT)
    if unbuilt:
        _fail(f"{unbuilt} are not built yet in this version of cda-discorl.")

    per_learner = [n for n in PER_LEARNER_STATE if settings[n]]
    if per_learner and cfg.num_learners > 1:
        _fail(f"{per_learner} keep per-Learner state and need num_learners <= 1.")

    _in_unit_interval(settings, "target_coeff")
    _non_negative(settings, "target_kl_cost")
    _in_unit_interval(settings, "adv_ema_decay")
    _in_unit_interval(settings, "value_norm_decay")
    _non_negative(settings, "policy_prediction_cost")

    if settings["policy_prediction"] and cfg.encoder_type == "lstm":
        _fail("policy_prediction needs a stateless encoder; encoder_type is lstm.")

    env_keys = settings["env_variety_keys"]
    missing = sorted(set(env_keys) - set(cfg.env_config))
    if missing:
        _fail(f"env_variety_keys names {missing}, which the env config does not have.")
    try:
        env_variety.validate_choices(env_keys, settings["env_variety_choices"])
    except ValueError as err:
        _fail(str(err))
