"""The `discorl` settings block: every key, its default, and the rules between them.

gym-CDA fills in these defaults and rejects any key not listed here before
`validate` sees the block, so `validate` can index every key directly.

A run sets the block in its train config:

    "extensions_enabled": ["discorl"],
    "extension_settings": {"discorl": {"modules": ["policy_0"], "adv_ema": true}}

Every component is off by default, so enabling the extension with no settings
changes nothing; `modules` limits the components to some trainable modules,
leaving the rest as plain PPO for an in-run comparison.
"""

from __future__ import annotations

from typing import Any, Dict

from gym_continuousDoubleAuction.train.checkpoint import STRUCTURAL_CONFIG_KEYS
from gym_continuousDoubleAuction.train.policy.policy_handler import POLICY_PREFIX

DEFAULTS: Dict[str, Any] = {
    # Trainable modules the components apply to; the others stay plain PPO.
    "modules": ["policy_0"],
    # Two-hot categorical critic over a symmetric, h-transformed support.
    "categorical_value": False,
    "value_bins": 601,
    "value_max_abs": 300.0,
    # KL from the policy to an EMA copy of itself.
    "target_policy": False,
    "target_coeff": 0.9,
    "target_kl_cost": 0.1,
    # Auxiliary head predicting the next step's policy.
    "policy_prediction": False,
    "policy_prediction_cost": 1.0,
    # Advantages normalised by a lifetime EMA of their scale, not per batch.
    "adv_ema": False,
    "adv_ema_decay": 0.99,
    # Per-episode randomisation of these env settings.
    "env_variety": False,
    "env_variety_keys": ["matching_rule", "step_clearing", "maker_fee_bps", "taker_fee_bps"],
    # The frozen Disco103 update rule in place of PPO's loss.
    "disco103": False,
    "disco103_reward_scale": 100.0,
    "disco103_size_head": "ppo",
}

#: The on/off switches, one per component.
COMPONENTS = (
    "categorical_value",
    "target_policy",
    "policy_prediction",
    "adv_ema",
    "env_variety",
    "disco103",
)

#: Components whose code has landed. A switch for any other one is refused
#: rather than accepted and ignored, which would read as "it made no difference".
BUILT: frozenset = frozenset({"categorical_value", "adv_ema", "target_policy", "policy_prediction"})

#: Components Disco103 supplies itself: its targets already include a
#: categorical value, a policy target and a policy-prediction output.
DISCO103_REPLACES = ("categorical_value", "target_policy", "policy_prediction")

#: Components whose state lives on one Learner. With several Learners each would
#: keep its own copy, fitted to its own shard of the batch.
PER_LEARNER_STATE = ("target_policy", "adv_ema", "disco103")

#: Settings that add or resize a tensor; changing one on a resume fails as structural.
STRUCTURAL_KEYS = ("categorical_value", "value_bins", "policy_prediction", "disco103")

#: Values `disco103_size_head` accepts. Disco103 handles one discrete action, so
#: the continuous size head keeps PPO's loss.
SIZE_HEAD_MODES = ("ppo",)


def _fail(message: str) -> None:
    raise ValueError(f"extension_settings['discorl']: {message}")


def _in_unit_interval(settings, key: str) -> None:
    if not 0.0 <= settings[key] < 1.0:
        _fail(f"{key} must be in [0, 1), got {settings[key]!r}.")


def _positive(settings, key: str) -> None:
    if not settings[key] > 0:
        _fail(f"{key} must be positive, got {settings[key]!r}.")


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

    if settings["disco103"]:
        overlap = [n for n in DISCO103_REPLACES if settings[n]]
        if overlap:
            _fail(f"disco103 supplies {overlap} itself; turn them off.")
    per_learner = [n for n in PER_LEARNER_STATE if settings[n]]
    if per_learner and cfg.num_learners > 1:
        _fail(f"{per_learner} keep per-Learner state and need num_learners <= 1.")

    if settings["value_bins"] < 3 or settings["value_bins"] % 2 == 0:
        _fail(f"value_bins must be odd and at least 3, got {settings['value_bins']!r}.")
    _positive(settings, "value_max_abs")
    _in_unit_interval(settings, "target_coeff")
    _non_negative(settings, "target_kl_cost")
    _non_negative(settings, "policy_prediction_cost")
    _in_unit_interval(settings, "adv_ema_decay")
    _positive(settings, "disco103_reward_scale")
    if settings["disco103_size_head"] not in SIZE_HEAD_MODES:
        _fail(f"disco103_size_head must be one of {list(SIZE_HEAD_MODES)}.")

    if settings["policy_prediction"] and cfg.encoder_type == "lstm":
        _fail("policy_prediction needs a stateless encoder; encoder_type is lstm.")

    env_keys = settings["env_variety_keys"]
    missing = sorted(set(env_keys) - set(cfg.env_config))
    if missing:
        _fail(f"env_variety_keys names {missing}, which the env config does not have.")
    structural = sorted(
        k for k in env_keys if f"env_config.{k}" in STRUCTURAL_CONFIG_KEYS or k == "num_of_agents"
    )
    if structural:
        _fail(f"env_variety_keys may not vary {structural}: they change the spaces.")
