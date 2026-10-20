"""The value normalisation component, piece by piece and inside a real run."""

import shutil
import tempfile

import numpy as np
import pytest
import ray
import torch
from torch import nn
from ray.rllib.core.columns import Columns
from ray.rllib.evaluation.postprocessing import Postprocessing

from gym_continuousDoubleAuction.train import extensions
from gym_continuousDoubleAuction.train.train import TrainConfig, build_config

from cda_discorl import adv_ema, config
from cda_discorl import value_norm as vn


class _Base(nn.Module):
    """Stands in for the composed PPO module: an encoder-free value head."""

    def setup(self):
        self.vf = nn.Sequential(nn.Linear(4, 1))

    def compute_values(self, batch, embeddings=None):
        if embeddings is None:
            embeddings = batch["emb"]
        return self.vf(embeddings).squeeze(-1)


def _module(on, decay=0.9):
    settings = {**config.DEFAULTS, "value_norm": on, "value_norm_decay": decay}
    cls = type(
        "M",
        (vn.ValueNormModule, _Base),
        {"extension_settings": {"discorl": settings}},
    )
    m = cls()
    m.setup()
    return m


class TestModule:
    def test_off_leaves_values_alone(self):
        m = _module(on=False)
        emb = torch.randn(3, 4)
        assert torch.equal(m.compute_values({}, embeddings=emb), m.vf(emb).squeeze(-1))

    def test_before_any_update_the_scale_is_the_identity(self):
        m = _module(on=True)
        assert m.value_norm_stats() == (0.0, 1.0)

    def test_the_moments_match_the_advantage_emas(self):
        torch.manual_seed(0)
        m = _module(on=True, decay=0.9)
        ref = adv_ema.EmaMoments(0.9)
        for scale in (0.004, 0.01, 0.002):
            t = torch.randn(64, dtype=torch.float64) * scale + 0.001
            m.update_value_norm(t)
            ref.update(t)
        mean, std = m.value_norm_stats()
        x = torch.tensor([0.003], dtype=torch.float64)
        assert float((x - mean) / std) == pytest.approx(float(ref.normalize(x)))

    def test_values_are_mapped_back_unless_raw(self):
        m = _module(on=True)
        m.update_value_norm(torch.tensor([0.01, 0.03]))
        mean, std = m.value_norm_stats()
        emb = torch.randn(5, 4)
        raw = m.vf(emb).squeeze(-1)
        assert torch.allclose(m.compute_values({}, embeddings=emb), mean + std * raw)
        m._vn_raw = True
        assert torch.allclose(m.compute_values({}, embeddings=emb), raw)

    def test_the_moments_are_in_the_state_dict(self):
        m = _module(on=True)
        m.update_value_norm(torch.tensor([0.5]))
        n = _module(on=True)
        n.load_state_dict(m.state_dict())
        assert n.value_norm_stats() == m.value_norm_stats()


class _Metrics:
    def __init__(self):
        self.logged = {}

    def log_value(self, key, value, window=None):
        self.logged[key] = float(value)


class _PPO:
    """Records what PPO's loss saw: the targets and the module's values."""

    def _policy_value_loss(self, *, module_id, config, batch, fwd_out):
        module = self.module[module_id].unwrapped()
        self.seen_targets = batch[Postprocessing.VALUE_TARGETS]
        self.seen_values = module.compute_values(batch, embeddings=fwd_out[Columns.EMBEDDINGS])
        return torch.tensor(1.0)


class _Learner(vn.ValueNormLearner, _PPO):
    def __init__(self, module):
        self.module = {"p": type("W", (), {"unwrapped": lambda _self: module})()}
        self.metrics = _Metrics()


class TestLearner:
    def test_the_loss_sees_normalised_targets_and_predictions(self):
        m = _module(on=True)
        m.update_value_norm(torch.tensor([0.002, 0.006]))
        mean, std = m.value_norm_stats()
        learner = _Learner(m)
        targets = torch.tensor([0.001, 0.004, 0.007])
        emb = torch.randn(3, 4)
        learner._policy_value_loss(
            module_id="p",
            config=None,
            batch={Postprocessing.VALUE_TARGETS: targets},
            fwd_out={Columns.EMBEDDINGS: emb},
        )
        assert torch.allclose(learner.seen_targets, (targets - mean) / std)
        assert torch.allclose(learner.seen_values, m.vf(emb).squeeze(-1))
        assert m._vn_raw is False
        assert learner.metrics.logged[("p", vn.STD_KEY)] == pytest.approx(std)

    def test_a_module_without_it_sees_raw_targets(self):
        m = _module(on=False)
        learner = _Learner(m)
        targets = torch.tensor([0.001, 0.004])
        learner._policy_value_loss(
            module_id="p",
            config=None,
            batch={Postprocessing.VALUE_TARGETS: targets},
            fwd_out={Columns.EMBEDDINGS: torch.randn(2, 4)},
        )
        assert learner.seen_targets is targets


class TestConnector:
    def test_padding_is_left_out_and_other_modules_untouched(self):
        m0, m1 = _module(on=True, decay=0.9), _module(on=True, decay=0.9)
        wrap = lambda m: type("W", (), {"unwrapped": lambda _self: m})()  # noqa: E731
        batch = {
            "policy_0": {
                Postprocessing.VALUE_TARGETS: torch.tensor([1.0, 3.0, 100.0]),
                Columns.LOSS_MASK: torch.tensor([True, True, False]),
            },
            "policy_1": {Postprocessing.VALUE_TARGETS: torch.tensor([5.0])},
        }
        vn.ValueNormMoments(["policy_0"])(
            rl_module={"policy_0": wrap(m0), "policy_1": wrap(m1)}, batch=batch
        )
        assert float(m0._vn_moment1) == pytest.approx(0.1 * 2.0)
        assert m1.value_norm_stats() == (0.0, 1.0)


class TestValidation:
    def test_the_decay_must_be_below_one(self):
        cfg = TrainConfig(num_agents=4, num_trained_agents=2, episode_data_dir=None)
        with pytest.raises(ValueError, match="value_norm_decay"):
            config.validate({**config.DEFAULTS, "value_norm_decay": 1.0}, cfg)


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
    extension_settings={"discorl": {"value_norm": True, "value_norm_decay": 0.9}},
)


class TestInARealRun:
    """policy_0's critic learns normalised returns; policy_1 stays plain PPO."""

    @classmethod
    def setup_class(cls):
        extensions.discover.cache_clear()
        ray.init(ignore_reinit_error=True, include_dashboard=False, log_to_driver=False, num_cpus=2)
        cls.log_dir = tempfile.mkdtemp()
        cls.cfg = TrainConfig(**TEST_CFG, log_base_dir=cls.log_dir)
        ppo_config, _cb = build_config(cls.cfg)
        cls.algo = ppo_config.build_algo()
        cls.result = cls.algo.train()
        cls.learner = cls.algo.learner_group._learner

    @classmethod
    def teardown_class(cls):
        cls.algo.stop()
        ray.shutdown()
        shutil.rmtree(cls.log_dir, ignore_errors=True)

    def test_the_moments_moved_once(self):
        module = self.learner.module["policy_0"].unwrapped()
        assert float(module._vn_decay_product) == pytest.approx(0.9)
        assert np.isfinite(float(self.result["learners"]["policy_0"][vn.STD_KEY]))

    def test_the_other_module_has_none(self):
        assert not getattr(self.learner.module["policy_1"].unwrapped(), "_vn_on", False)
        assert vn.STD_KEY not in self.result["learners"]["policy_1"]

    def test_the_checkpoint_restores_the_moments(self):
        from ray.rllib.algorithms.algorithm import Algorithm

        from gym_continuousDoubleAuction.train.checkpoint import save_checkpoint

        path = save_checkpoint(self.algo, self.cfg, 1)
        restored = Algorithm.from_checkpoint(path)
        try:
            got = restored.learner_group._learner.module["policy_0"].unwrapped()
            want = self.learner.module["policy_0"].unwrapped()
            # The module state travels as float32, so allow its rounding.
            assert got.value_norm_stats() == pytest.approx(want.value_norm_stats(), rel=1e-5)
        finally:
            restored.stop()
