"""The package as gym-CDA sees it: installed, discovered, validated, and inert.

These run against an installed gym-CDA, through its public extension point, so
a change there that breaks this package fails here.
"""

from types import SimpleNamespace

import pytest

from gym_continuousDoubleAuction.train import extensions
from gym_continuousDoubleAuction.train.model.moe_learner import CDAPPOTorchLearner
from gym_continuousDoubleAuction.train.train import TrainConfig, _learner_class

import cda_discorl
from cda_discorl import config

BASE_CFG = dict(
    num_agents=4,
    num_trained_agents=2,
    max_step=64,
    num_episodes_per_iter=2,
    num_epochs=1,
    minibatch_size=64,
    num_env_runners=0,
    num_learners=0,
    num_gpus_per_learner=0,
    episode_data_dir=None,
    log_level="ERROR",
)


def _cfg(settings=None, **kwargs):
    return TrainConfig(
        **{**BASE_CFG, **kwargs},
        extensions_enabled=["discorl"],
        extension_settings={"discorl": settings or {}},
    )


class TestDiscovery:
    def test_the_entry_point_resolves_to_this_package(self):
        extensions.discover.cache_clear()
        assert extensions.discover()["discorl"] is cda_discorl.EXTENSION

    def test_the_recorded_version_is_the_package_version(self):
        from importlib.metadata import version

        assert cda_discorl.EXTENSION.version == version("cda-discorl")


class TestEnabledWithEverythingOffChangesNothing:
    def test_every_component_defaults_off(self):
        assert not any(config.DEFAULTS[name] for name in config.COMPONENTS)

    def test_the_learner_only_gains_the_switched_mixin(self):
        composed = _learner_class(_cfg())
        assert composed.__mro__[1] is cda_discorl.EXTENSION.learner_mixin
        assert composed.__mro__[-len(CDAPPOTorchLearner.__mro__) :] == CDAPPOTorchLearner.__mro__

    def test_no_module_gets_the_module_mixin(self):
        assert cda_discorl.EXTENSION.module_ids(config.DEFAULTS, ["policy_0", "policy_1"]) == []

    def test_the_env_is_unchanged(self):
        env = SimpleNamespace(reset=lambda **kw: None)
        reset = env.reset
        assert extensions.wrap_env(env, ["discorl"], {}) is env
        assert env.reset is reset

    def test_the_settings_reach_the_learner_config(self):
        group = {extensions.EXTENSIONS_CONFIG_KEY: extensions.learner_config(_cfg())}
        assert extensions.settings_for(group, "discorl") == config.DEFAULTS
        assert extensions.versions(group) == {"discorl": cda_discorl.__version__}


class TestValidation:
    def test_an_unknown_key_is_refused_by_gym_cda(self):
        with pytest.raises(ValueError, match="unknown keys"):
            _cfg({"target_kl": 0.1})

    @pytest.mark.parametrize("name", config.COMPONENTS)
    def test_a_component_not_built_yet_is_refused(self, monkeypatch, name):
        monkeypatch.setattr(config, "BUILT", frozenset(config.COMPONENTS) - {name})
        with pytest.raises(ValueError, match="not built yet"):
            _cfg({name: True})

    def test_modules_must_be_trainable(self):
        with pytest.raises(ValueError, match="not trainable"):
            _cfg({"modules": ["policy_3"]})

    def test_modules_may_not_be_empty(self):
        with pytest.raises(ValueError, match="non-empty"):
            _cfg({"modules": []})

    def test_both_trainable_modules_may_be_selected(self):
        _cfg({"modules": ["policy_0", "policy_1"]})

    def test_a_switch_must_be_a_bool(self):
        with pytest.raises(ValueError, match="true or false"):
            _cfg({"adv_ema": 1})

    @pytest.mark.parametrize(
        "key, value",
        [
            ("target_coeff", 1.0),
            ("target_kl_cost", -0.1),
            ("policy_prediction_cost", -1.0),
            ("adv_ema_decay", -0.1),
            ("value_norm_decay", 1.0),
        ],
    )
    def test_out_of_range_values_are_refused(self, key, value):
        with pytest.raises(ValueError, match=key):
            _cfg({key: value})

    def test_env_variety_keys_must_be_env_settings(self):
        with pytest.raises(ValueError, match="does not have"):
            _cfg({"env_variety_keys": ["no_such_key"]})

    @pytest.mark.parametrize("key", ["n_hist", "book_mode", "num_of_agents"])
    def test_env_variety_may_not_vary_a_structural_key(self, key):
        with pytest.raises(ValueError, match="may only name"):
            _cfg({"env_variety_keys": [key]})


class TestCombinationRules:
    """Checked directly, with every switch treated as built."""

    @staticmethod
    def _check(monkeypatch, settings, **cfg_kwargs):
        monkeypatch.setattr(config, "BUILT", frozenset(config.COMPONENTS))
        cfg = TrainConfig(**{**BASE_CFG, **cfg_kwargs})
        config.validate({**config.DEFAULTS, **settings}, cfg)

    @pytest.mark.parametrize("name", config.PER_LEARNER_STATE)
    def test_per_learner_state_refuses_several_learners(self, monkeypatch, name):
        with pytest.raises(ValueError, match="num_learners"):
            self._check(monkeypatch, {name: True}, num_learners=2)

    def test_every_component_together_is_allowed(self, monkeypatch):
        self._check(monkeypatch, {name: True for name in config.COMPONENTS})


class TestResume:
    def test_structural_settings_are_registered(self):
        group = {extensions.EXTENSIONS_CONFIG_KEY: extensions.learner_config(_cfg())}
        assert set(extensions.structural_keys(group)) == {
            f"ext.discorl.{k}" for k in config.STRUCTURAL_KEYS
        }
