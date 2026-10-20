"""The extension's one Learner mixin and one module mixin, built from the components.

gym-CDA composes a single mixin of each kind per extension, so every
component's mixin is listed here and switches itself on its own setting.
Order matters only between components that override the same method; each
calls `super()`, so all of them run.
"""

from cda_discorl.adv_ema import AdvEmaLearner
from cda_discorl.policy_prediction import PolicyPredictionLearner, PolicyPredictionModule
from cda_discorl.target_policy import TargetPolicyLearner
from cda_discorl.value_norm import ValueNormLearner, ValueNormModule

#: Components that need the module mixin on the modules they apply to.
MODULE_COMPONENTS = ("value_norm", "policy_prediction")


class DiscoLearnerMixin(
    # First, so every critic loss after it sees the normalised targets.
    ValueNormLearner,
    TargetPolicyLearner,
    PolicyPredictionLearner,
    AdvEmaLearner,
):
    pass


class DiscoModuleMixin(ValueNormModule, PolicyPredictionModule):
    pass


def module_ids(settings, trainable_ids):
    """The selected modules, when a component that changes the module is on."""
    if not any(settings[name] for name in MODULE_COMPONENTS):
        return []
    return [m for m in trainable_ids if m in settings["modules"]]
