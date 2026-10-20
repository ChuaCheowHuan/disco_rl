"""The env variety component: validation, the draw on a real env, and a real run."""

import shutil
import tempfile
from decimal import Decimal

import numpy as np
import pytest
import ray

from gym_continuousDoubleAuction.envs.continuousDoubleAuction_env import (
    continuousDoubleAuctionEnv,
)
from gym_continuousDoubleAuction.train import extensions
from gym_continuousDoubleAuction.train.train import TrainConfig, build_config

from cda_discorl import config
from cda_discorl import env_variety as ev

CHOICES = config.DEFAULTS["env_variety_choices"]
KEYS = config.DEFAULTS["env_variety_keys"]


def _settings(**overrides):
    return {**config.DEFAULTS, "env_variety": True, **overrides}


class TestValidation:
    def test_the_defaults_are_valid(self):
        ev.validate_choices(KEYS, CHOICES)

    def test_an_unsupported_key_is_refused(self):
        with pytest.raises(ValueError, match="may only name"):
            ev.validate_choices(["n_hist"], {"n_hist": [1, 2]})

    def test_a_key_needs_choices(self):
        with pytest.raises(ValueError, match="non-empty list"):
            ev.validate_choices(["matching_rule"], {"matching_rule": []})

    def test_a_choice_outside_the_envs_rules_is_refused(self):
        with pytest.raises(ValueError, match="may only hold"):
            ev.validate_choices(["step_clearing"], {"step_clearing": ["sequential", "auction"]})

    def test_every_fee_pair_it_could_draw_must_be_valid(self):
        with pytest.raises(ValueError, match="maker_fee_bps"):
            ev.validate_choices(
                ["maker_fee_bps", "taker_fee_bps"],
                {"maker_fee_bps": [0, -3], "taker_fee_bps": [1, 5]},
            )

    def test_the_config_rejects_bad_choices(self, monkeypatch):
        cfg = TrainConfig(num_agents=4, num_trained_agents=2, episode_data_dir=None)
        bad = {**CHOICES, "matching_rule": ["lifo"]}
        with pytest.raises(ValueError, match="may only hold"):
            config.validate(_settings(env_variety_choices=bad), cfg)


def _env():
    return continuousDoubleAuctionEnv({"num_of_agents": 4, "max_step": 20})


class TestTheDraw:
    def test_off_leaves_the_env_alone(self):
        env = _env()
        reset = env.reset
        assert ev.wrap(env, {**config.DEFAULTS}) is env
        assert env.reset == reset

    def test_each_reset_applies_its_draw_before_the_book_is_built(self):
        env = ev.wrap(_env(), _settings())
        for seed in range(6):
            env.reset(seed=seed)
            drawn = env.discorl_env_settings
            assert set(drawn) == set(KEYS)
            assert env.LOB.matching_rule == drawn["matching_rule"]
            assert env.step_clearing == drawn["step_clearing"]
            for trader in env.traders:
                assert trader.acc.maker_fee_bps == Decimal(str(drawn["maker_fee_bps"]))
                assert trader.acc.taker_fee_bps == Decimal(str(drawn["taker_fee_bps"]))

    def test_a_seed_repeats_its_market(self):
        a, b = ev.wrap(_env(), _settings()), ev.wrap(_env(), _settings())
        a.reset(seed=11)
        b.reset(seed=11)
        assert a.discorl_env_settings == b.discorl_env_settings

    def test_every_choice_is_drawn(self):
        env = ev.wrap(_env(), _settings())
        env.reset(seed=0)
        seen = {k: set() for k in KEYS}
        for _ in range(60):
            env.reset()
            for k, v in env.discorl_env_settings.items():
                seen[k].add(v)
        assert all(seen[k] == set(CHOICES[k]) for k in KEYS)

    def test_only_the_named_keys_vary(self):
        env = ev.wrap(_env(), _settings(env_variety_keys=["step_clearing"]))
        env.reset(seed=3)
        assert list(env.discorl_env_settings) == ["step_clearing"]
        assert env.LOB.matching_rule == "fifo"

    def test_money_is_conserved_under_drawn_fees(self):
        settings = _settings(
            env_variety_keys=["maker_fee_bps", "taker_fee_bps"],
            env_variety_choices={**CHOICES, "maker_fee_bps": [1], "taker_fee_bps": [5]},
        )
        env = ev.wrap(_env(), settings)
        env.reset(seed=4)
        for _ in range(20):
            env.step({a: env.action_spaces[a].sample() for a in env.agents})
        start = float(env.init_cash) * len(env.traders)
        total = sum(float(t.acc.nav) for t in env.traders) + float(env.fees_collected)
        assert total == pytest.approx(start)
        assert float(env.fees_collected) > 0


TEST_CFG = dict(
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
    seed=7,
    extensions_enabled=["discorl"],
    extension_settings={"discorl": {"env_variety": True}},
)


class TestInARealRun:
    @classmethod
    def setup_class(cls):
        extensions.discover.cache_clear()
        ray.init(ignore_reinit_error=True, include_dashboard=False, log_to_driver=False, num_cpus=2)
        cls.log_dir = tempfile.mkdtemp()
        cfg = TrainConfig(**TEST_CFG, log_base_dir=cls.log_dir)
        ppo_config, _cb = build_config(cfg)
        cls.algo = ppo_config.build_algo()
        cls.result = cls.algo.train()

    @classmethod
    def teardown_class(cls):
        cls.algo.stop()
        ray.shutdown()
        shutil.rmtree(cls.log_dir, ignore_errors=True)

    def test_the_sampled_env_drew_a_market(self):
        (env,) = self.algo.env_runner.env.envs
        assert set(env.unwrapped.discorl_env_settings) == set(KEYS)

    def test_training_ran(self):
        assert np.isfinite(float(self.result["learners"]["policy_0"]["total_loss"]))
