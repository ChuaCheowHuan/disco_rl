"""A real gym-CDA run with discorl enabled and every component off.

Proves the skeleton is inert where it matters: the Algorithm builds, one
iteration trains, and the checkpoint records the extension so a restore
without the package is refused by name.
"""

import shutil
import tempfile

import numpy as np
import ray

from gym_continuousDoubleAuction.train import extensions
from gym_continuousDoubleAuction.train.checkpoint import _read_league_state, save_checkpoint
from gym_continuousDoubleAuction.train.model.moe_learner import (
    CDAPPOTorchLearner,
    CDAPPOTorchRLModule,
)
from gym_continuousDoubleAuction.train.train import TrainConfig, build_config

import cda_discorl
from cda_discorl import config

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
)


class TestEnabledButInert:
    @classmethod
    def setup_class(cls):
        extensions.discover.cache_clear()
        ray.init(ignore_reinit_error=True, include_dashboard=False, log_to_driver=False, num_cpus=2)
        cls.log_dir = tempfile.mkdtemp()
        cls.cfg = TrainConfig(**TEST_CFG, log_base_dir=cls.log_dir)
        ppo_config, _cb = build_config(cls.cfg)
        cls.algo = ppo_config.build_algo()
        cls.result = cls.algo.train()

    @classmethod
    def teardown_class(cls):
        cls.algo.stop()
        ray.shutdown()
        shutil.rmtree(cls.log_dir, ignore_errors=True)

    def test_the_learner_runs_plain_ppo(self):
        assert isinstance(self.algo.learner_group._learner, CDAPPOTorchLearner)
        for module_id in ("policy_0", "policy_1"):
            assert not any(k.startswith("discorl") for k in self.result["learners"][module_id])

    def test_the_modules_are_unchanged(self):
        assert type(self.algo.env_runner.module["policy_0"]) is CDAPPOTorchRLModule

    def test_one_iteration_trained_both_modules(self):
        for module_id in ("policy_0", "policy_1"):
            assert "total_loss" in self.result["learners"][module_id]

    def test_the_checkpoint_records_the_extension(self):
        path = save_checkpoint(self.algo, self.cfg, 1)
        recorded = _read_league_state(path)[extensions.EXTENSIONS_STATE_KEY]
        assert recorded == {"discorl": cda_discorl.__version__}


class TestEveryBuiltComponentTogether:
    """All built components on policy_0, with a remote env runner.

    A remote runner builds inference-only modules and receives weights from the
    Learner, so this is where an extra head missing from the inference-only
    list would fail.
    """

    @classmethod
    def setup_class(cls):
        extensions.discover.cache_clear()
        ray.init(ignore_reinit_error=True, include_dashboard=False, log_to_driver=False, num_cpus=3)
        cls.log_dir = tempfile.mkdtemp()
        settings = {n: True for n in config.BUILT}
        cfg = TrainConfig(
            **{**TEST_CFG, "num_env_runners": 1},
            extension_settings={"discorl": settings},
            log_base_dir=cls.log_dir,
        )
        ppo_config, _cb = build_config(cfg)
        cls.algo = ppo_config.build_algo()
        cls.results = [cls.algo.train() for _ in range(2)]

    @classmethod
    def teardown_class(cls):
        cls.algo.stop()
        ray.shutdown()
        shutil.rmtree(cls.log_dir, ignore_errors=True)

    def test_two_iterations_train_with_finite_losses(self):
        for result in self.results:
            metrics = result["learners"]["policy_0"]
            assert np.isfinite(float(metrics["total_loss"]))
            assert any(k.startswith("discorl") for k in metrics)
