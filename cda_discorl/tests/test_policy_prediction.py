"""The policy prediction component, piece by piece and inside a real run."""

import shutil
import tempfile

import gymnasium as gym
import numpy as np
import pytest
import ray
import torch

from gym_continuousDoubleAuction.train import extensions
from gym_continuousDoubleAuction.train.train import TrainConfig, build_config

from cda_discorl import config
from cda_discorl import policy_prediction as pp

SPACE = gym.spaces.Dict(
    {
        "category": gym.spaces.Discrete(3),
        "size": gym.spaces.Box(0.0, 1.0, (1,), np.float32),
        "price": gym.spaces.Discrete(4),
    }
)


class TestPieces:
    def test_discrete_slices_skip_the_box(self):
        # gymnasium sorts a Dict's keys: category, price, size.
        assert pp.discrete_slices(SPACE) == [("category", slice(0, 3)), ("price", slice(3, 7))]

    def test_gather_takes_the_discrete_logits_in_order(self):
        space = gym.spaces.Dict(
            {
                "a": gym.spaces.Discrete(2),
                "b": gym.spaces.Box(0, 1, (1,)),
                "c": gym.spaces.Discrete(3),
            }
        )
        logits = torch.arange(7.0).expand(2, 7)
        got = pp.gather_discrete(logits, pp.discrete_slices(space))
        assert got[0].tolist() == [0, 1, 4, 5, 6]

    def test_kl_is_zero_for_equal_logits_and_sums_over_components(self):
        t = torch.randn(4, 7)
        assert torch.allclose(pp.per_head_kl(t, t, [3, 4]), torch.zeros(4), atol=1e-6)
        p = torch.randn(4, 7)
        a = torch.distributions.Categorical(logits=t[:, :3])
        b = torch.distributions.Categorical(logits=p[:, :3])
        c = torch.distributions.Categorical(logits=t[:, 3:])
        d = torch.distributions.Categorical(logits=p[:, 3:])
        expected = torch.distributions.kl_divergence(a, b) + torch.distributions.kl_divergence(c, d)
        assert torch.allclose(pp.per_head_kl(t, p, [3, 4]), expected, atol=1e-6)

    def test_a_masked_action_in_the_target_contributes_nothing(self):
        t = torch.tensor([[0.0, -1.0e9, 0.0]])
        p = torch.tensor([[0.0, 5.0, 0.0]])
        assert torch.isfinite(pp.per_head_kl(t, p, [3])).all()


def _cfg(**kwargs):
    base = dict(num_agents=4, num_trained_agents=2, episode_data_dir=None, log_level="ERROR")
    return TrainConfig(**{**base, **kwargs})


def test_an_lstm_encoder_is_refused(monkeypatch):
    monkeypatch.setattr(config, "BUILT", frozenset(config.COMPONENTS))
    with pytest.raises(ValueError, match="stateless"):
        config.validate({**config.DEFAULTS, "policy_prediction": True}, _cfg(encoder_type="lstm"))


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
    extension_settings={"discorl": {"policy_prediction": True}},
)


class TestInARealRun:
    """policy_0 gets the aux head and its loss; policy_1 stays plain PPO."""

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

    def test_the_next_observation_piece_is_added_once(self):
        names = [type(c).__name__ for c in self.learner._learner_connector.connectors]
        assert names.count("AddNextObservationsFromEpisodesToTrainBatch") == 1
        assert names.index("AddNextObservationsFromEpisodesToTrainBatch") == (
            names.index("AddOneTsToEpisodesAndTruncate") + 1
        )

    def test_only_the_selected_learner_module_has_the_head(self):
        assert hasattr(self.learner.module["policy_0"].unwrapped(), "aux_pi")
        assert not hasattr(self.learner.module["policy_1"].unwrapped(), "aux_pi")

    def test_the_loss_is_logged_finite_for_the_selected_module_only(self):
        value = float(self.result["learners"]["policy_0"][pp.POLICY_PREDICTION_LOSS_KEY])
        assert np.isfinite(value) and value >= 0
        assert pp.POLICY_PREDICTION_LOSS_KEY not in self.result["learners"]["policy_1"]

    def test_the_checkpoint_restores_the_head(self):
        from ray.rllib.algorithms.algorithm import Algorithm

        from gym_continuousDoubleAuction.train.checkpoint import save_checkpoint

        path = save_checkpoint(self.algo, self.cfg, 1)
        restored = Algorithm.from_checkpoint(path)
        try:
            got = restored.learner_group._learner.module["policy_0"].unwrapped().aux_pi
            want = self.learner.module["policy_0"].unwrapped().aux_pi
            assert torch.equal(got.weight.detach().cpu(), want.weight.detach().cpu())
        finally:
            restored.stop()
