"""
Core polytope learning and evaluation modules.
"""

"""
NOTE:
Avoid importing training/CLI modules at package import time, since some of them
depend on optional heavy deps (e.g. hydra) and are not needed for inference.
Import submodules directly (e.g. `from .safe_rep_model import SafeRepModel`) when needed.
"""

from . import lm_constraints, safe_rep_model

__all__ = ["lm_constraints", "safe_rep_model"]
