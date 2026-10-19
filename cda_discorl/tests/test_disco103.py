"""Disco103 in place of PPO's loss: the bridge, the plumbing, and a real run.

Needs the `[disco103]` extra; skipped without it.
"""

import shutil
import tempfile

import numpy as np
import pytest
import ray
import torch

pytest.importorskip("jax")
pytest.importorskip("disco_rl.update_rules.disco")

from gym_continuousDoubleAuction.train import extensions  # noqa: E402
from gym_continuousDoubleAuction.train.train import TrainConfig, build_config  # noqa: E402

from cda_discorl import config  # noqa: E402
from cda_discorl.disco103 import bridge, learner  # noqa: E402


class TestBridge:
    def test_the_published_weights_load_whole(self):
        params = bridge.load_weights()
        assert len(params) == 21
        assert sum(len(v) for v in params.values()) == 42

    def test_targets_have_the_rules_shapes_and_carry_state(self):
        b = bridge.Bridge(601, 300.0)
        a, t, n = 4, 6, 2
        rng = np.random.default_rng(0)
        out = {
            "logits": rng.normal(size=(t + 1, n, a)),
            "y": rng.normal(size=(t + 1, n, 600)),
            "z": rng.normal(size=(t + 1, n, a, 600)),
            "aux_pi": rng.normal(size=(t + 1, n, a, a)),
            "q": rng.normal(size=(t + 1, n, a, 601)),
        }
        state = b.new_state()
        targets, new_state = b.targets(
            state,
            actions=rng.integers(a, size=(t + 1, n)),
            rewards=rng.normal(size=(t, n)),
            is_terminal=np.zeros((t, n), bool),
            agent_out=out,
            behaviour_logits=out["logits"],
            target_out=out,
        )
        assert targets["pi"].shape == (t, n, a)
        assert targets["y"].shape == targets["z"].shape == (t, n, 600)
        assert targets["q_td"].shape == (t, n)
        assert all(np.isfinite(v).all() for v in targets.values())
        assert float(new_state["adv_ema_state"].decay_product) < 1.0


class TestTimeMajor:
    def test_round_trip_pads_with_the_last_row_and_zeroes_bootstrap_rows(self):
        flat = np.arange(7.0)  # episodes of 3 and 4 rows
        tm = learner._to_time_major(flat, [3, 4])
        assert tm[:, 0].tolist() == [0, 1, 2, 2]
        assert tm[:, 1].tolist() == [3, 4, 5, 6]
        back = learner._from_time_major(tm, [3, 4])
        assert back.tolist() == [0, 1, 0, 3, 4, 5, 0]


def _cfg(**kwargs):
    base = dict(num_agents=4, num_trained_agents=2, episode_data_dir=None, log_level="ERROR")
    return TrainConfig(**{**base, **kwargs})


def test_an_lstm_encoder_is_refused():
    with pytest.raises(ValueError, match="stateless"):
        config.validate({**config.DEFAULTS, "disco103": True}, _cfg(encoder_type="lstm"))


TEST_CFG = dict(
    num_agents=4,
    num_trained_agents=2,
    max_step=64,
    num_episodes_per_iter=2,
    num_epochs=2,
    minibatch_size=64,
    num_env_runners=0,
    num_learners=0,
    num_gpus_per_learner=0,
    episode_data_dir=None,
    log_level="ERROR",
    seed=7,
    extensions_enabled=["discorl"],
    extension_settings={"discorl": {"disco103": True}},
)


class TestInARealRun:
    """policy_0 learns by Disco103; policy_1 stays plain PPO."""

    @classmethod
    def setup_class(cls):
        extensions.discover.cache_clear()
        ray.init(ignore_reinit_error=True, include_dashboard=False, log_to_driver=False, num_cpus=2)
        cls.log_dir = tempfile.mkdtemp()
        cls.cfg = TrainConfig(**TEST_CFG, log_base_dir=cls.log_dir)
        ppo_config, _cb = build_config(cls.cfg)
        cls.algo = ppo_config.build_algo()
        cls.learner = cls.algo.learner_group._learner
        cls.result = cls.algo.train()

    @classmethod
    def teardown_class(cls):
        cls.algo.stop()
        ray.shutdown()
        shutil.rmtree(cls.log_dir, ignore_errors=True)

    def test_every_discrete_component_has_finite_disco_losses(self):
        metrics = self.result["learners"]["policy_0"]
        module = self.learner.module["policy_0"].unwrapped()
        for name, _ in module._d103_slices:
            for term in ("pi", "y", "z", "aux", "q"):
                assert np.isfinite(float(metrics[f"discorl_d103_{name}_{term}_loss"])), (name, term)
        assert np.isfinite(float(metrics["total_loss"]))

    def test_the_critic_reports_its_explained_variance(self):
        # train.compare reads it; without it a Disco103 row has no critic column.
        assert np.isfinite(float(self.result["learners"]["policy_0"]["vf_explained_var"]))

    def test_the_continuous_component_keeps_ppo(self):
        assert "discorl_d103_size_policy_loss" in self.result["learners"]["policy_0"]

    def test_the_other_module_is_plain_ppo(self):
        assert not any("d103" in k for k in self.result["learners"]["policy_1"])
        assert not hasattr(self.learner.module["policy_1"].unwrapped(), "disco_heads")

    def test_the_meta_state_advanced_once_per_update(self):
        streams = self.learner._d103_streams["policy_0"]
        for state in streams.values():
            assert float(state["adv_ema_state"].decay_product) == pytest.approx(0.99)

    def test_the_checkpoint_restores_targets_and_meta_state(self):
        from ray.rllib.algorithms.algorithm import Algorithm

        from gym_continuousDoubleAuction.train.checkpoint import save_checkpoint

        path = save_checkpoint(self.algo, self.cfg, 1)
        restored = Algorithm.from_checkpoint(path)
        try:
            got = restored.learner_group._learner
            for k, v in self.learner._d103_targets["policy_0"].state_dict().items():
                assert torch.equal(got._d103_targets["policy_0"].state_dict()[k].cpu(), v.cpu()), k
            want = self.learner._d103_streams["policy_0"]["category"]["rnn_state"]
            have = got._d103_streams["policy_0"]["category"]["rnn_state"]
            assert np.allclose(
                np.asarray(want["lstm"]["meta_rnn_state"].hidden),
                np.asarray(have["lstm"]["meta_rnn_state"].hidden),
            )
        finally:
            restored.stop()
