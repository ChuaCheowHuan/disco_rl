"""DiscoRL components for gym-continuousDoubleAuction, as an installable extension.

Installing this package registers it with gym-CDA through the
`gym_cda.extensions` entry point; a run uses it only when `"discorl"` is in
`extensions_enabled`. See gym-CDA's docs/30_extensions.md for the contract and
`config.py` for the settings.

Each component lands in its own release. Until one has, its switch is refused.
"""

from gym_continuousDoubleAuction.train.extensions import Extension

from cda_discorl import config

__version__ = "0.1.0"

EXTENSION = Extension(
    name="discorl",
    version=__version__,
    defaults=config.DEFAULTS,
    validate=config.validate,
    structural_keys=config.STRUCTURAL_KEYS,
)
