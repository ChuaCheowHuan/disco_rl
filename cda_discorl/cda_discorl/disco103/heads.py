"""The agent outputs Disco103 reads and trains, per discrete action component.

DiscoRL's agent emits, besides its policy logits:

  y       [Y]       a state prediction
  z       [A, Y]    an action-conditional prediction
  aux_pi  [A, A]    the next step's policy, per action
  q       [A, bins] a categorical action value

for one discrete action of `A` choices, with `Y` = 600. This market's action is
a Dict, so every Discrete component gets its own set, from the same actor
embedding the policy head reads. The heads exist only on the Learner's copy
of a module; env runners never build them.
"""

from __future__ import annotations

from torch import nn

from cda_discorl.policy_prediction import discrete_slices

NAME = "discorl"

#: Disco103's y/z width; fixed by the published weights.
PREDICTION_SIZE = 600

#: DiscoRL's agent initialises its heads with this weight std.
HEAD_INIT_STD = 1e-2

#: `fwd_out` key prefix: `discorl_d103/<component>/<y|z|aux_pi|q>`.
OUT_PREFIX = "discorl_d103"


def out_key(component: str, name: str) -> str:
    return f"{OUT_PREFIX}/{component}/{name}"


def _head(in_features: int, out_features: int) -> nn.Linear:
    layer = nn.Linear(in_features, out_features)
    nn.init.normal_(layer.weight, std=HEAD_INIT_STD)
    nn.init.zeros_(layer.bias)
    return layer


class Disco103Module:
    """Module mixin: Disco103's heads per discrete component, on the train pass."""

    _d103_on = False

    def setup(self):
        super().setup()
        s = type(self).extension_settings[NAME]
        if not s["disco103"] or self.inference_only:
            return
        # `is_stateful()` is not usable until setup returns; this is what it checks.
        if self.get_initial_state():
            raise ValueError("disco103 needs a stateless encoder.")
        self._d103_on = True
        self._d103_bins = int(s["value_bins"])
        self._d103_slices = discrete_slices(self.action_space)
        embed = next(m for m in self.pi.modules() if isinstance(m, nn.Linear)).in_features
        heads = {}
        for name, sl in self._d103_slices:
            a = sl.stop - sl.start
            heads[name] = nn.ModuleDict(
                {
                    "y": _head(embed, PREDICTION_SIZE),
                    "z": _head(embed, a * PREDICTION_SIZE),
                    "aux_pi": _head(embed, a * a),
                    "q": _head(embed, a * self._d103_bins),
                }
            )
        self.disco_heads = nn.ModuleDict(heads)
        self._d103_actor_emb = None
        self.pi.register_forward_pre_hook(self._d103_capture)

    def _d103_capture(self, _module, args):
        self._d103_actor_emb = args[0]

    def get_non_inference_attributes(self):
        attrs = list(super().get_non_inference_attributes())
        return attrs + (["disco_heads"] if self._d103_on else [])

    def _forward_train(self, batch, **kwargs):
        out = super()._forward_train(batch, **kwargs)
        if self._d103_on:
            emb = self._d103_actor_emb
            for name, sl in self._d103_slices:
                a = sl.stop - sl.start
                heads = self.disco_heads[name]
                out[out_key(name, "y")] = heads["y"](emb)
                out[out_key(name, "z")] = heads["z"](emb).unflatten(-1, (a, PREDICTION_SIZE))
                out[out_key(name, "aux_pi")] = heads["aux_pi"](emb).unflatten(-1, (a, a))
                out[out_key(name, "q")] = heads["q"](emb).unflatten(-1, (a, self._d103_bins))
            self._d103_actor_emb = None
        return out
