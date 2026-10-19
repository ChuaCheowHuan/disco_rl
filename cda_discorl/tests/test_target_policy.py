"""The target policy component, piece by piece and inside a real run."""

import shutil
import tempfile

import pytest
import ray
import torch
from torch import nn

from gym_continuousDoubleAuction.train import extensions
from gym_continuousDoubleAuction.train.train import TrainConfig, build_config

from cda_discorl import target_policy as tp


class _Wrapped:
    def __init__(self, m):
        self.m = m

    def unwrapped(self):
        return self.m


class _Base:
    def apply_gradients(self, gradients_dict):
        self.applied = True


class _Learner(tp.TargetPolicyLearner, _Base):
    def __init__(self, online, target, coeff):
        self.module = {"p": _Wrapped(online)}
        self._targets = {"p": target}
        self._target_coeff = coeff


class TestEmaUpdate:
    def test_each_optimiser_step_moves_the_target_by_one_minus_coeff(self):
        online, target = nn.Linear(3, 2), nn.Linear(3, 2)
        before = [p.detach().clone() for p in target.parameters()]
        learner = _Learner(online, target, coeff=0.9)
        learner.apply_gradients({})
        assert learner.applied
        for t, b, o in zip(target.parameters(), before, online.parameters()):
            assert torch.allclose(t, 0.9 * b + 0.1 * o)

    def test_the_target_takes_no_gradient(self):
        online, target = nn.Linear(3, 2), nn.Linear(3, 2)
        target.requires_grad_(False)
        _Learner(online, target, coeff=0.5).apply_gradients({})
        assert not any(p.requires_grad for p in target.parameters())

    def test_nothing_happens_when_off(self):
        learner = _Learner(nn.Linear(3, 2), nn.Linear(3, 2), coeff=0.9)
        learner._targets = None
        learner.apply_gradients({})
        assert learner.applied


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
    extension_settings={"discorl": {"target_policy": True, "target_coeff": 0.5}},
)


class TestInARealRun:
    """policy_0 is anchored to its EMA copy; policy_1 stays plain PPO."""

    @classmethod
    def setup_class(cls):
        extensions.discover.cache_clear()
        ray.init(ignore_reinit_error=True, include_dashboard=False, log_to_driver=False, num_cpus=2)
        cls.log_dir = tempfile.mkdtemp()
        cls.cfg = TrainConfig(**TEST_CFG, log_base_dir=cls.log_dir)
        ppo_config, _cb = build_config(cls.cfg)
        cls.algo = ppo_config.build_algo()
        cls.learner = cls.algo.learner_group._learner
        cls.initial = {
            k: v.detach().clone() for k, v in cls.learner._targets["policy_0"].state_dict().items()
        }
        cls.result = cls.algo.train()

    @classmethod
    def teardown_class(cls):
        cls.algo.stop()
        ray.shutdown()
        shutil.rmtree(cls.log_dir, ignore_errors=True)

    def test_only_the_selected_module_has_a_target(self):
        assert list(self.learner._targets) == ["policy_0"]

    def test_the_kl_is_logged_finite_for_the_selected_module_only(self):
        kl = self.result["learners"]["policy_0"][tp.TARGET_KL_KEY]
        assert torch.isfinite(torch.tensor(float(kl))) and float(kl) >= 0
        assert tp.TARGET_KL_KEY not in self.result["learners"]["policy_1"]

    def test_the_target_moved_and_lags_the_online_weights(self):
        target = self.learner._targets["policy_0"].state_dict()
        online = self.learner.module["policy_0"].unwrapped().state_dict()
        moved = [k for k in self.initial if not torch.equal(target[k], self.initial[k])]
        assert moved
        assert any(not torch.equal(target[k], online[k]) for k in moved)

    def test_the_checkpoint_restores_the_target(self):
        from ray.rllib.algorithms.algorithm import Algorithm

        from gym_continuousDoubleAuction.train.checkpoint import save_checkpoint

        path = save_checkpoint(self.algo, self.cfg, 1)
        restored = Algorithm.from_checkpoint(path)
        try:
            got = restored.learner_group._learner._targets["policy_0"].state_dict()
            for k, v in self.learner._targets["policy_0"].state_dict().items():
                assert torch.equal(got[k].cpu(), v.cpu()), k
        finally:
            restored.stop()


@pytest.mark.parametrize("coeff", [0.0, 0.9])
def test_coeff_zero_tracks_the_online_weights_exactly(coeff):
    online, target = nn.Linear(2, 2), nn.Linear(2, 2)
    _Learner(online, target, coeff=coeff).apply_gradients({})
    same = all(torch.equal(t, o) for t, o in zip(target.parameters(), online.parameters()))
    assert same == (coeff == 0.0)
