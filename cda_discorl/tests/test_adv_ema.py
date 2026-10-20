"""The advantage EMA component, piece by piece and inside a real run."""

import shutil
import tempfile

import pytest
import ray
import torch
from ray.rllib.core.columns import Columns
from ray.rllib.evaluation.postprocessing import Postprocessing

from gym_continuousDoubleAuction.train import extensions
from gym_continuousDoubleAuction.train.train import TrainConfig, build_config

from cda_discorl import adv_ema


def _reference(batches, decay):
    """DiscoRL's MovingAverage, step by step, on plain floats."""
    m1 = m2 = 0.0
    prod = 1.0
    out = []
    for b in batches:
        m1 = decay * m1 + (1 - decay) * b.mean().item()
        m2 = decay * m2 + (1 - decay) * (b**2).mean().item()
        prod *= decay
        mean = m1 / (1 - prod)
        var = max(0.0, m2 / (1 - prod) - mean**2)
        out.append((b - mean) / ((var + 1e-12) ** 0.5 + 1e-6))
    return out


class TestMoments:
    def test_matches_discorls_moving_average(self):
        torch.manual_seed(0)
        batches = [torch.randn(32, dtype=torch.float64) * s + s for s in (1.0, 3.0, 0.1, 10.0)]
        m = adv_ema.EmaMoments(decay=0.9)
        got = []
        for b in batches:
            m.update(b)
            got.append(m.normalize(b))
        for a, b in zip(got, _reference(batches, 0.9)):
            assert torch.allclose(a, b)

    def test_the_first_batch_is_standardised_as_before(self):
        """Debiasing makes step one's moments the batch's own."""
        b = torch.tensor([1.0, 2.0, 3.0, 6.0], dtype=torch.float64)
        m = adv_ema.EmaMoments(decay=0.99)
        m.update(b)
        expected = (b - b.mean()) / b.std(unbiased=False)
        assert torch.allclose(m.normalize(b), expected, atol=1e-5)

    def test_a_small_batch_keeps_its_scale_against_the_past(self):
        m = adv_ema.EmaMoments(decay=0.99)
        m.update(torch.randn(256) * 10.0)
        small = torch.randn(256) * 0.1
        m.update(small)
        assert m.normalize(small).std() < 0.1

    def test_state_round_trips(self):
        m = adv_ema.EmaMoments(decay=0.9)
        m.update(torch.randn(8))
        n = adv_ema.EmaMoments(decay=0.9)
        n.set_state(m.get_state())
        assert n.get_state() == m.get_state()


class _Module:
    def __init__(self, values):
        self.values = values

    def unwrapped(self):
        return self

    def compute_values(self, batch):
        return self.values


class TestConnector:
    def test_normalises_only_the_selected_modules(self):
        targets = torch.tensor([1.0, 2.0, 4.0, 8.0])
        values = torch.tensor([0.5, 0.5, 0.5, 0.5])
        untouched = torch.tensor([9.0, 9.0, 9.0, 9.0])
        batch = {
            "policy_0": {
                Postprocessing.VALUE_TARGETS: targets,
                Postprocessing.ADVANTAGES: untouched,
            },
            "policy_1": {
                Postprocessing.VALUE_TARGETS: targets,
                Postprocessing.ADVANTAGES: untouched,
            },
        }
        rl_module = {"policy_0": _Module(values), "policy_1": _Module(values)}
        piece = adv_ema.EmaAdvantages(["policy_0"], decay=0.9)
        piece(rl_module=rl_module, batch=batch)

        expected = adv_ema.EmaMoments(0.9)
        expected.update(targets - values)
        assert torch.allclose(
            batch["policy_0"][Postprocessing.ADVANTAGES], expected.normalize(targets - values)
        )
        assert batch["policy_1"][Postprocessing.ADVANTAGES] is untouched

    def test_padding_is_left_out_of_the_moments(self):
        targets = torch.tensor([1.0, 3.0, 100.0])
        batch = {
            "policy_0": {
                Postprocessing.VALUE_TARGETS: targets,
                Columns.LOSS_MASK: torch.tensor([True, True, False]),
            }
        }
        piece = adv_ema.EmaAdvantages(["policy_0"], decay=0.9)
        piece(rl_module={"policy_0": _Module(torch.zeros(3))}, batch=batch)
        assert piece.moments["policy_0"].moment1 == pytest.approx(0.1 * 2.0)


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
    extension_settings={"discorl": {"adv_ema": True, "adv_ema_decay": 0.9}},
)


class TestInARealRun:
    """policy_0's advantages use the EMA; policy_1 keeps GAE's standardisation."""

    @classmethod
    def setup_class(cls):
        extensions.discover.cache_clear()
        ray.init(ignore_reinit_error=True, include_dashboard=False, log_to_driver=False, num_cpus=2)
        cls.log_dir = tempfile.mkdtemp()
        cls.cfg = TrainConfig(**TEST_CFG, log_base_dir=cls.log_dir)
        ppo_config, _cb = build_config(cls.cfg)
        cls.algo = ppo_config.build_algo()
        cls.algo.train()
        cls.learner = cls.algo.learner_group._learner

    @classmethod
    def teardown_class(cls):
        cls.algo.stop()
        ray.shutdown()
        shutil.rmtree(cls.log_dir, ignore_errors=True)

    def test_the_piece_runs_right_after_gae(self):
        names = [type(p).__name__ for p in self.learner._learner_connector.connectors]
        assert names[names.index("GeneralAdvantageEstimation") + 1] == "EmaAdvantages"

    def test_only_the_selected_module_has_moments_and_they_moved(self):
        moments = self.learner._adv_ema.moments
        assert list(moments) == ["policy_0"]
        assert moments["policy_0"].decay_product == pytest.approx(0.9)

    def test_the_checkpoint_restores_the_moments(self):
        from ray.rllib.algorithms.algorithm import Algorithm

        from gym_continuousDoubleAuction.train.checkpoint import save_checkpoint

        path = save_checkpoint(self.algo, self.cfg, 1)
        restored = Algorithm.from_checkpoint(path)
        try:
            got = restored.learner_group._learner._adv_ema.moments["policy_0"].get_state()
            assert got == self.learner._adv_ema.moments["policy_0"].get_state()
        finally:
            restored.stop()
