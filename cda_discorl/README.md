# cda-discorl

Learning mechanisms from *Discovering state-of-the-art reinforcement learning algorithms*
(Oh et al., 2025) for the PPO learner in
[gym-continuousDoubleAuction](https://github.com/ChuaCheowHuan/gym-continuousDoubleAuction)
(gym-CDA), packaged as a training extension. It lives beside the upstream `disco_rl` package,
which it does not modify.

Installing it changes no run. A run uses it only when its train config lists it:

```json
"extensions_enabled": ["discorl"],
"extension_settings": {"discorl": {"modules": ["policy_0"], "target_policy": true}}
```

To remove it, drop the name from `extensions_enabled` and uninstall the package. gym-CDA needs no
other edit. Its extension contract is gym-CDA's `docs/30_extensions.md`.

## Install

gym-CDA is not on PyPI, so install it first, then this package:

```bash
pip install "gym_continuousDoubleAuction @ git+https://github.com/ChuaCheowHuan/gym-continuousDoubleAuction"
pip install "git+https://github.com/ChuaCheowHuan/disco_rl#subdirectory=cda_discorl"
```

## What is in it, and why

An earlier version also carried DiscoRL's categorical critic and the frozen Disco103 update rule.
Measured against plain PPO in this market, neither learned. Their fixed numbers assume
Atari-scale returns, and this market's value targets have a standard deviation of about 0.004.
Disco103 also needs agent lifetimes in the millions of steps.

This version keeps the mechanisms that do not depend on the reward's scale. It adds `value_norm`,
which removes that dependence from the critic.

| Component | Switch | What it does | Status |
|---|---|---|---|
| Target policy | `target_policy` | KL from the policy to an EMA copy of itself | Recommended, pending a longer run |
| Advantage EMA | `adv_ema` | Normalises advantages by a lifetime EMA of their scale | Recommended |
| Value normalisation | `value_norm` | Critic learns returns on a unit scale | New, being measured |
| Env variety | `env_variety` | Redraws some market rules at the start of every episode | Recommended for generalisation |
| Policy prediction | `policy_prediction` | Auxiliary head predicting the next step's policy | Experimental |

In a 3-seed, 30-iteration comparison, none separated from PPO on return. Target policy had the
best mean, and every recommended component kept the critic at least as good as PPO's.

Every component is off by default, and `cda_discorl/config.py` lists every key with its default.
`modules` picks which trainable modules get the components; the others stay plain PPO, so one
league run compares the two. `target_policy`, `adv_ema` and `value_norm` keep per-Learner state
and refuse `num_learners > 1`. A switch for a component that is not built is refused rather than
ignored.

A resume must enable the same extension with the same settings as the checkpoint. gym-CDA refuses
anything else, and refuses to restore a checkpoint that used `discorl` where this package is not
installed.

## Target policy

DiscoRL's agent keeps target weights, an exponential moving average of its own. With
`target_policy` on, each selected module gets such a copy on the Learner, and the loss gains
`target_kl_cost * KL(pi_target || pi)`. After every optimiser step the copy moves
`1 - target_coeff` of the way to the online weights.

PPO's clipping limits how far one update moves the policy. This limits how far a run of updates
drifts from where the policy has recently been, which matters when the league keeps changing the
opponents. The KL is logged as `discorl_target_kl`, and the copy is saved with the Learner's state.

## Advantage EMA

RLlib standardises each batch's advantages by that batch's own mean and standard deviation.
With `adv_ema` on, the selected modules use DiscoRL's normaliser instead: an exponential moving
average (decay `adv_ema_decay`) of the advantage's first and second moments over the whole run,
debiased as Adam debiases its moments. The first batch is standardised exactly as before. Later
batches keep their scale relative to the past rather than being stretched to unit variance.

It runs right after RLlib's GAE step on the Learner. GAE does not keep the raw advantages, so
they are recomputed as `value_targets - V(s)` with the same weights. The moments are saved with
the Learner's state, so a resume continues them.

## Value normalisation

With `value_norm` on, the value head predicts the return normalised by a debiased EMA (decay
`value_norm_decay`) of the value targets' mean and standard deviation. PPO's critic loss then runs
on normalised targets against that prediction. `compute_values` maps the prediction back to
reward units, so GAE, the advantages and every metric RLlib reports are unchanged in meaning.

The moments are module buffers, saved and restored with the weights, and logged as
`discorl_value_norm_mean` and `discorl_value_norm_std`. They are not PopArt: when the moments move,
the value moves with them for one update.

## Env variety

DiscoRL's rule generalised because it was discovered across many environments, and a league here
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

Because fees change returns, compare an env-variety run with others on held-out markets, not on
training return.

## Policy prediction

DiscoRL's agent has an auxiliary output that predicts, for each action it could take now, the
policy it will have at the next step, trained toward the actual next-step policy. Nothing acts on
it; it shapes the shared representation.

This market's action is a Dict. So with `policy_prediction` on, the head is conditioned on the
`category` action and predicts the logits of every discrete component (category, price, price
offset, order slot) at the next step. The loss is the sum over those components of
KL(next-step policy || prediction), with terminal steps masked out. It is weighted by
`policy_prediction_cost` and logged as `discorl_policy_prediction_loss`.

It adds the next observation to the Learner's train batch when the run does not already carry it
(JEPA runs do). It needs a stateless encoder, so it refuses `encoder_type: lstm`. The head lives
only on the Learner's copy of a module; env runners do not build it.

## Tests

```bash
cd cda_discorl && python -m pytest -q
```

They run against the installed gym-CDA and include real training iterations. CI is in
`ci/cda_discorl.yml`; move it to `.github/workflows/` to enable it.
