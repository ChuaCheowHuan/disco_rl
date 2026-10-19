"""The categorical value component, piece by piece and inside a real run."""

import shutil
import tempfile

import pytest
import ray
import torch
from torch import nn
from ray.rllib.core.columns import Columns
from ray.rllib.evaluation.postprocessing import Postprocessing

from gym_continuousDoubleAuction.train import extensions
from gym_continuousDoubleAuction.train.train import TrainConfig, build_config

from cda_discorl import categorical_value as cv
from cda_discorl import config


class TestTransforms:
    def test_h_inv_inverts_h(self):
        x = torch.tensor([-1e4, -3.0, -1e-3, 0.0, 1e-3, 0.5, 250.0, 1e4], dtype=torch.float64)
        assert torch.allclose(cv.h_inv(cv.h(x)), x, rtol=1e-9, atol=1e-9)

    def test_two_hot_sums_to_one_and_has_the_target_mean(self):
        points = cv.support(601, 300.0, device="cpu").double()
        y = torch.tensor([-299.5, -0.25, 0.0, 0.3, 17.75], dtype=torch.float64)
        probs = cv.two_hot(y, points)
        assert torch.allclose(probs.sum(-1), torch.ones(5, dtype=torch.float64))
        assert torch.allclose((probs * points).sum(-1), y)
        assert int((probs > 0).sum(-1).max()) <= 2

    def test_two_hot_clips_to_the_support(self):
        points = cv.support(5, 2.0, device="cpu")
        probs = cv.two_hot(torch.tensor([-9.0, 9.0]), points)
        assert probs[0, 0] == 1.0 and probs[1, -1] == 1.0


class _Base:
    """Stands in for the composed PPO module: an encoder-free value head."""

    def setup(self):
        self.vf = nn.Sequential(nn.Linear(8, 1))

    def compute_values(self, batch, embeddings=None):
        if embeddings is None:
            embeddings = batch["emb"]
        return self.vf(embeddings).squeeze(-1)


def _module(on, bins=11, max_abs=5.0):
    settings = {
        **config.DEFAULTS,
        "categorical_value": on,
        "value_bins": bins,
        "value_max_abs": max_abs,
    }
    cls = type(
        "M", (cv.CategoricalValueModule, _Base), {"extension_settings": {"discorl": settings}}
    )
    m = cls()
    m.setup()
    return m


class TestModule:
    def test_off_leaves_the_head_alone(self):
        m = _module(on=False)
        assert m.vf[0].out_features == 1
        assert m.compute_values({}, embeddings=torch.zeros(3, 8)).shape == (3,)

    def test_on_gives_one_logit_per_bin_and_a_scalar_value(self):
        m = _module(on=True)
        assert m.vf[0].out_features == 11
        emb = torch.randn(4, 8)
        logits = m.compute_value_logits({}, embeddings=emb)
        assert logits.shape == (4, 11)
        expected = cv.h_inv((torch.softmax(logits, -1) * cv.support(11, 5.0)).sum(-1))
        assert torch.allclose(m.compute_values({}, embeddings=emb), expected)

    def test_logits_without_embeddings_run_the_base_path(self):
        m = _module(on=True)
        emb = torch.randn(2, 8)
        assert torch.equal(m.compute_value_logits({"emb": emb}), m.vf(emb))


class _Metrics:
    def __init__(self):
        self.logged = {}

    def log_value(self, key, value, window=None):
        self.logged[key] = float(value)


class _Config:
    def __init__(self, coeff):
        self.vf_loss_coeff = coeff

    def copy(self, copy_frozen=False):
        return _Config(self.vf_loss_coeff)


class _PPO:
    """Records the vf_loss_coeff PPO's loss was computed with."""

    def _policy_value_loss(self, *, module_id, config, batch, fwd_out):
        self.seen_coeff = config.vf_loss_coeff
        return torch.tensor(10.0)


class _Learner(cv.CategoricalValueLearner, _PPO):
    def __init__(self, module, settings):
        self.module = {"p": type("W", (), {"unwrapped": lambda _self: module})()}
        self.metrics = _Metrics()
        self.config = type(
            "C",
            (),
            {"learner_config_dict": {"extensions": {"extension_settings": {"discorl": settings}}}},
        )()


class TestLearner:
    def _batch(self, n=6):
        return {Postprocessing.VALUE_TARGETS: torch.linspace(-2, 2, n)}

    def test_replaces_the_squared_error_with_cross_entropy(self):
        m = _module(on=True)
        settings = type(m).extension_settings["discorl"]
        learner = _Learner(m, settings)
        batch = self._batch()
        emb = torch.randn(6, 8)
        loss = learner._policy_value_loss(
            module_id="p", config=_Config(0.5), batch=batch, fwd_out={Columns.EMBEDDINGS: emb}
        )
        assert learner.seen_coeff == 0.0
        logits = m.vf(emb)
        target = cv.two_hot(cv.h(batch[Postprocessing.VALUE_TARGETS]), cv.support(11, 5.0))
        ce = -(target * torch.log_softmax(logits, -1)).sum(-1).mean()
        assert torch.allclose(loss, 10.0 + 0.5 * ce)
        assert learner.metrics.logged[("p", cv.CATEGORICAL_VF_LOSS_KEY)] == pytest.approx(float(ce))

    def test_respects_the_loss_mask(self):
        m = _module(on=True)
        learner = _Learner(m, type(m).extension_settings["discorl"])
        batch = {**self._batch(), Columns.LOSS_MASK: torch.tensor([True] * 3 + [False] * 3)}
        emb = torch.randn(6, 8)
        loss = learner._policy_value_loss(
            module_id="p", config=_Config(1.0), batch=batch, fwd_out={Columns.EMBEDDINGS: emb}
        )
        target = cv.two_hot(cv.h(batch[Postprocessing.VALUE_TARGETS]), cv.support(11, 5.0))
        ce = -(target * torch.log_softmax(m.vf(emb), -1)).sum(-1)[:3].mean()
        assert torch.allclose(loss, 10.0 + ce)

    def test_a_module_without_it_gets_plain_ppo(self):
        m = _module(on=False)
        learner = _Learner(m, type(m).extension_settings["discorl"])
        loss = learner._policy_value_loss(
            module_id="p", config=_Config(0.5), batch=self._batch(), fwd_out={}
        )
        assert learner.seen_coeff == 0.5 and float(loss) == 10.0


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
    extension_settings={
        "discorl": {"categorical_value": True, "value_bins": 51, "value_max_abs": 5.0}
    },
)


class TestInARealRun:
    """policy_0 gets the categorical critic; policy_1 stays plain PPO."""

    @classmethod
    def setup_class(cls):
        extensions.discover.cache_clear()
        ray.init(ignore_reinit_error=True, include_dashboard=False, log_to_driver=False, num_cpus=2)
        cls.log_dir = tempfile.mkdtemp()
        cfg = TrainConfig(**TEST_CFG, log_base_dir=cls.log_dir)
        ppo_config, _cb = build_config(cfg)
        cls.algo = ppo_config.build_algo()
        cls.result = cls.algo.train()
        cls.modules = cls.algo.learner_group._learner.module

    @classmethod
    def teardown_class(cls):
        cls.algo.stop()
        ray.shutdown()
        shutil.rmtree(cls.log_dir, ignore_errors=True)

    def test_only_the_selected_module_has_the_categorical_head(self):
        assert self.modules["policy_0"].unwrapped()._cv_on
        assert not getattr(self.modules["policy_1"].unwrapped(), "_cv_on", False)

    def test_the_cross_entropy_trained_only_the_selected_module(self):
        assert self.result["learners"]["policy_0"][cv.CATEGORICAL_VF_LOSS_KEY] > 0
        assert cv.CATEGORICAL_VF_LOSS_KEY not in self.result["learners"]["policy_1"]

    def test_values_are_finite_scalars(self):
        for module_id in ("policy_0", "policy_1"):
            metrics = self.result["learners"][module_id]
            assert torch.isfinite(torch.tensor(float(metrics["vf_explained_var"])))

    def test_the_checkpoint_restores_the_categorical_head(self):
        from ray.rllib.algorithms.algorithm import Algorithm

        from gym_continuousDoubleAuction.train.checkpoint import save_checkpoint

        cfg = TrainConfig(**TEST_CFG, log_base_dir=self.log_dir)
        path = save_checkpoint(self.algo, cfg, 1)
        restored = Algorithm.from_checkpoint(path)
        try:
            module = restored.learner_group._learner.module["policy_0"].unwrapped()
            original = self.modules["policy_0"].unwrapped()
            assert module._cv_on
            for a, b in zip(module.vf.parameters(), original.vf.parameters()):
                assert torch.equal(a.detach().cpu(), b.detach().cpu())
        finally:
            restored.stop()
