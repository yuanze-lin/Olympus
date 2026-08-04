"""Specialist backends for the Olympus routing tokens.

Importing this package registers every backend in
:data:`olympus_tools.backends.base._REGISTRY`.
"""

from .base import Backend, ModelHost, available_backends, register  # noqa: F401
from . import sota  # noqa: F401  (current SOTA defaults)
from . import diffusion  # noqa: F401  (Table 9 originals, still selectable)
from . import perception  # noqa: F401
from . import restoration  # noqa: F401
from . import threed  # noqa: F401
from . import rvos  # noqa: F401

__all__ = ["Backend", "ModelHost", "available_backends", "register"]
