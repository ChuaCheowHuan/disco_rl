# cda-discorl

DiscoRL components for
[gym-continuousDoubleAuction](https://github.com/ChuaCheowHuan/gym-continuousDoubleAuction)
(gym-CDA), packaged as a training extension. It lives beside the upstream `disco_rl` package,
which it does not modify.

Installing it changes no run. A run uses it only when its train config lists it:

```json
"extensions_enabled": ["discorl"],
"extension_settings": {"discorl": {"modules": ["policy_0"]}}
```

To remove it, drop the name from `extensions_enabled` and uninstall the package. gym-CDA needs no
other edit. Its extension contract is gym-CDA's `docs/30_extensions.md`.

## Install

gym-CDA is not on PyPI, so install it first, then this package:

```bash
pip install "gym_continuousDoubleAuction @ git+https://github.com/ChuaCheowHuan/gym-continuousDoubleAuction"
pip install "git+https://github.com/ChuaCheowHuan/disco_rl#subdirectory=cda_discorl"
```

Add the `[disco103]` extra for the frozen Disco103 update rule. It installs the upstream `disco_rl`
package from this repository, which brings JAX, Haiku, rlax and the published weights.

## Settings

Every component is off by default. `cda_discorl/config.py` lists every key with its default and
the rules between them.

| Component | Switch | What it does | Status |
|---|---|---|---|
| Categorical value | `categorical_value` | Two-hot critic over `value_bins` with an h-transform | Built |
| Target policy | `target_policy` | KL from the policy to an EMA copy of itself | Built |
| Policy prediction | `policy_prediction` | Auxiliary head predicting the next step's policy | Built |
| Advantage EMA | `adv_ema` | Normalises advantages by a lifetime EMA of their scale | Built |
| Env variety | `env_variety` | Redraws some market rules at the start of every episode | Built |
| Disco103 | `disco103` | The frozen discovered rule in place of PPO's loss | Built |

A switch for a component that is not built yet is refused, rather than accepted and ignored. All
six are built in this version.
`modules` picks which trainable modules get the components; the others stay plain PPO, so one
league run compares the two.

Rules: `disco103` supplies its own categorical value, policy target and policy prediction, so it
refuses those three switches. `target_policy`, `adv_ema` and `disco103` keep per-Learner state
and refuse `num_learners > 1`. `env_variety_keys` may only name the four settings the env can change
between episodes.

A resume must enable the same extension with the same settings as the checkpoint; gym-CDA
refuses anything else, and refuses to restore a checkpoint that used `discorl` where this package
is not installed.

## Disco103

With `disco103` on, the selected modules learn by the frozen update rule DeepMind discovered
(Oh et al., 2025), using the published `disco_103.npz` weights, in place of PPO's loss.

The rule's agent has, per discrete action, a state prediction `y`, an action-conditional
prediction `z`, a next-step policy prediction `aux_pi` and a categorical action value `q`. Here
each discrete component of the action (category, price, price offset, order slot) gets its own
set from the actor embedding, and is its own stream to the meta-network, with its own meta-RNN
state and moving averages. The loss per component is the rule's own: KL to the meta-network's
`pi_hat`, `y_hat` and `z_hat`, the next-step policy prediction, and the categorical TD value loss,
with the published costs (1, 1, 1, 1, 0.2). Each term is logged as
`discorl_d103_<component>_<term>_loss`.

How it fits RLlib's PPO Learner:

- **Targets once per update batch.** The rule needs whole trajectories, and PPO shuffles timesteps
  into minibatches. So right after GAE, where the batch still holds whole episodes, the targets
  are computed for every step and stored in the batch; each minibatch reads its rows. With several
  epochs they stay fixed across them, as PPO's advantages do. Episodes of different lengths are
  padded with terminal steps, which also enter the rule's moving averages.
- **The continuous size keeps PPO** (`disco103_size_head: "ppo"`): its clipped surrogate and
  entropy bonus, with GAE's advantages from the scalar critic, which keeps PPO's value loss.
- **Rewards are scaled** by `disco103_reward_scale` (100) before the rule sees them. It was
  discovered on Atari-scale rewards; this market's are around 1e-3 per step.
- **Masked actions stay masked.** The meta-network can put weight on an action the env masked
  out; the policy target is masked the same way before the KL.
- **The target network** is an EMA copy (coefficient 0.9, the rule's own), updated after every
  optimiser step. It and the meta-network state are saved with the Learner's state.
- The meta-network runs in JAX on the CPU, un-jitted (current JAX fails an upstream type check
  under `jit`), about 0.3 s per component per update at 64 steps.

It needs a stateless encoder, so it refuses `encoder_type: lstm`, and it refuses
`categorical_value`, `target_policy` and `policy_prediction`, which it supplies itself.

## Categorical value

The value head predicts a distribution over `value_bins` points spread evenly over
`[-value_max_abs, value_max_abs]` in h-space, `h(x) = sign(x)(sqrt(|x| + 1) - 1) + 0.001x`, trained
with a cross-entropy against a two-hot encoding of the value target. It replaces PPO's squared
error, weighted by the same `vf_loss_coeff`. The scalar value RLlib uses for GAE is the
expectation, mapped back out of h-space, so nothing downstream changes. PPO's `vf_loss` is still
logged beside the new `discorl_categorical_vf_loss`.

The paper's support (601 bins over +/-300) has bins one h-unit apart. This market's per-step rewards
are around 1e-3, so its returns sit inside the two middle bins. Set `value_max_abs` to the size of
this run's returns in h-space; for a return of `R` that is about `R / 2` when `R` is small.

## Target policy

DiscoRL's agent keeps target weights, an exponential moving average of its own. With
`target_policy` on, each selected module gets such a copy on the Learner, and the loss gains
`target_kl_cost * KL(pi_target || pi)`. After every optimiser step the copy moves
`1 - target_coeff` of the way to the online weights. PPO's clipping limits how far one update
moves the policy; this limits how far a run of updates drifts from where the policy has recently
been, which matters when the league keeps changing the opponents. The KL is logged as
`discorl_target_kl`, and the copy is saved with the Learner's state.

## Policy prediction

DiscoRL's agent has an auxiliary output that predicts, for each action it could take now, the
policy it will have at the next step, trained toward the actual next-step policy. Nothing acts on
it; it shapes the shared representation. This market's action is a Dict, so with
`policy_prediction` on the head is conditioned on the `category` action and predicts the logits of
every discrete component (category, price, price offset, order slot) at the next step. The loss is
the sum over those components of KL(next-step policy || prediction), with terminal steps masked
out, weighted by `policy_prediction_cost` and logged as `discorl_policy_prediction_loss`.

It adds the next observation to the Learner's train batch when the run does not already carry it
(JEPA runs do). It needs a stateless encoder, so it refuses `encoder_type: lstm`. The head lives
only on the Learner's copy of a module; env runners do not build it.

## Env variety

DiscoRL's rule generalised because it was discovered across many environments; a league here
trains on one fixed market. With `env_variety` on, every episode draws each setting in
`env_variety_keys` from its list in `env_variety_choices`:

| Key | Default choices | When the env reads it |
|---|---|---|
| `matching_rule` | `fifo`, `pro_rata` | When `reset` builds the episode's book |
| `step_clearing` | `sequential`, `batch` | Every step |
| `maker_fee_bps` | `0`, `1` | Each trader's account, which `reset` keeps |
| `taker_fee_bps` | `0`, `2` | The same |

Only these four are supported: none changes the observation or action space, and none needs the
env rebuilt. Every fee pair the lists could produce is checked against the env's own fee rules
when the config is built. A seeded `reset` repeats its draw. The settings in force are on
`env.discorl_env_settings`. The draw applies to the whole market, so to every agent, whichever
modules `modules` selects.

## Advantage EMA

RLlib standardises each batch's advantages by that batch's own mean and standard deviation.
With `adv_ema` on, the selected modules use DiscoRL's normaliser instead: an exponential moving
average (decay `adv_ema_decay`) of the advantage's first and second moments over the whole run,
debiased as Adam debiases its moments. The first batch is standardised exactly as before; later
batches keep their scale relative to the past rather than being stretched to unit variance.

It runs right after RLlib's GAE step on the Learner. GAE does not keep the raw advantages, so
they are recomputed as `value_targets - V(s)` with the same weights. The moments are saved with
the Learner's state, so a resume continues them.

## Tests

```bash
cd cda_discorl && python -m pytest -q
```

They run against the installed gym-CDA and include one real training iteration.
