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

Add the `[disco103]` extra for the frozen Disco103 update rule (JAX, Haiku and the upstream
`disco_rl` package).

## Settings

Every component is off by default. `cda_discorl/config.py` lists every key with its default and
the rules between them.

| Component | Switch | What it does | Status |
|---|---|---|---|
| Categorical value | `categorical_value` | Two-hot critic over `value_bins` with an h-transform | Built |
| Target policy | `target_policy` | KL from the policy to an EMA copy of itself | Not built yet |
| Policy prediction | `policy_prediction` | Auxiliary head predicting the next step's policy | Not built yet |
| Advantage EMA | `adv_ema` | Normalises advantages by a lifetime EMA of their scale | Built |
| Env variety | `env_variety` | Randomises non-structural env settings per episode | Not built yet |
| Disco103 | `disco103` | The frozen discovered rule in place of PPO's loss | Not built yet |

A switch for a component that is not built yet is refused, rather than accepted and ignored.
`modules` picks which trainable modules get the components; the others stay plain PPO, so one
league run compares the two.

Rules: `disco103` supplies its own categorical value, policy target and policy prediction, so it
refuses those three switches. `target_policy`, `adv_ema` and `disco103` keep per-Learner state
and refuse `num_learners > 1`. `env_variety_keys` may not name a setting that changes the observation or
action space.

A resume must enable the same extension with the same settings as the checkpoint; gym-CDA
refuses anything else, and refuses to restore a checkpoint that used `discorl` where this package
is not installed.

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
