"""The JAX side of Disco103: the frozen meta-network and its per-stream state.

Imported only when `disco103` is on, because it needs the `[disco103]` extra
(JAX, Haiku, rlax and the upstream `disco_rl` package).

`Bridge.targets` takes one action component's time-major rollout as numpy
arrays, runs `DiscoUpdateRule.unroll_meta_net` with the published Disco103
weights, and returns the targets the torch loss is trained toward. Each
(module, action component) pair is one stream with its own meta-RNN state and
advantage/TD moving averages, carried across updates as the rule expects.

The target network's outputs are computed in torch and handed in; the rule's
own target-parameter update is fed an empty tree and its result discarded,
since the torch Learner keeps that copy.
"""

from __future__ import annotations

from importlib import resources
from typing import Any, Dict

import jax
import jax.numpy as jnp
import numpy as np

from disco_rl import agent as disco_agent
from disco_rl import types
from disco_rl.update_rules import disco

#: Size of the y and z predictions; fixed by the published weights.
PREDICTION_SIZE = 600


def load_weights() -> Dict[str, Dict[str, np.ndarray]]:
    """The Disco103 meta-network parameters, as the upstream colab unpacks them."""
    path = resources.files("disco_rl.update_rules") / "weights" / "disco_103.npz"
    with path.open("rb") as f:
        flat = dict(np.load(f))
    params: Dict[str, Dict[str, np.ndarray]] = {}
    for key in flat:
        module = "/".join(key.split("/")[:-1])
        params[module] = {"b": flat[f"{module}/b"], "w": flat[f"{module}/w"]}
    return params


class Bridge:
    """Disco103 with this run's value support, ready to produce targets."""

    def __init__(self, value_bins: int, value_max_abs: float):
        settings = disco_agent.get_settings_disco()
        rule_kwargs = dict(settings.update_rule)
        rule_kwargs.update(num_bins=value_bins, max_abs_value=value_max_abs)
        self.rule = disco.DiscoUpdateRule(**rule_kwargs)
        self.hyper_params = dict(settings.hyper_params)
        self.params = jax.tree.map(jnp.asarray, load_weights())
        # Not jitted: under `jit` a tracer fails the upstream meta-net's
        # `isinstance(logits, chex.Array)` check with current JAX.
        self._unroll = self._unroll_impl

    def new_state(self) -> Dict[str, Any]:
        _, rnn_state = self.rule.init_params(jax.random.PRNGKey(0))
        return dict(
            rnn_state=rnn_state,
            adv_ema_state=self.rule._adv_ema.init_state(),
            td_ema_state=self.rule._td_ema.init_state(),
            target_params={},
        )

    def _unroll_impl(self, meta_state, actions, rewards, is_terminal, agent_out, behaviour, target):
        rollout = types.UpdateRuleInputs(
            observations=jnp.zeros(actions.shape),
            actions=actions,
            rewards=rewards,
            is_terminal=is_terminal,
            agent_out=agent_out,
            behaviour_agent_out=behaviour,
        )
        out, new_state = self.rule.unroll_meta_net(
            self.params,
            {},
            None,
            meta_state,
            rollout,
            self.hyper_params,
            lambda *_: (target, None),
            jax.random.PRNGKey(0),
        )
        keep = {k: out[k] for k in ("pi", "y", "z", "q_td")}
        return keep, new_state

    def targets(
        self, state, *, actions, rewards, is_terminal, agent_out, behaviour_logits, target_out
    ):
        """Targets for one stream. Arrays are time-major, `[T + 1, B, ...]` except
        `rewards` and `is_terminal`, `[T, B]`. Returns `(targets, new_state)`."""
        as_j = lambda x: jnp.asarray(x, jnp.float32)  # noqa: E731
        out, new_state = self._unroll(
            state,
            jnp.asarray(actions, jnp.int32),
            as_j(rewards),
            jnp.asarray(is_terminal, bool),
            jax.tree.map(as_j, agent_out),
            {"logits": as_j(behaviour_logits)},
            jax.tree.map(as_j, target_out),
        )
        new_state = dict(new_state, target_params={})
        return {k: np.asarray(v) for k, v in out.items()}, new_state


def state_to_numpy(state):
    return jax.tree.map(np.asarray, state)


def state_from_numpy(state):
    return jax.tree.map(jnp.asarray, state)
