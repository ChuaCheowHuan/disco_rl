"""DiscoRL components for gym-continuousDoubleAuction, as an installable extension.

Installing this package registers it with gym-CDA through the
`gym_cda.extensions` entry point; a run uses it only when `"discorl"` is in
`extensions_enabled`. See gym-CDA's docs/30_extensions.md for the contract and
`config.py` for the settings.

A switch for a component that has not landed is refused (`config.BUILT`).
"""

from gym_continuousDoubleAuction.train.extensions import Extension

from cda_discorl import components, config, env_variety

__version__ = "0.1.0"

EXTENSION = Extension(
    name="discorl",
    version=__version__,
    defaults=config.DEFAULTS,
    validate=config.validate,
    learner_mixin=components.DiscoLearnerMixin,
    module_mixin=components.DiscoModuleMixin,
    module_ids=components.module_ids,
    env_wrapper=env_variety.wrap,
    structural_keys=config.STRUCTURAL_KEYS,
)
