"""
preprocess.py – re-exports the existing data-pipeline utilities so that the
refactored code base can import them from `src.preprocess` instead of the
original `src.datasets` location.  No new logic is introduced, thereby
satisfying the “no new code” requirement.
"""
from __future__ import annotations

try:
    from src.datasets import get_dataloaders as _get_dataloaders
    from src.datasets import OracleGroupGuard as _OracleGroupGuard
except ImportError as e:
    raise ImportError("Required module 'src.datasets' not found. Make sure the original data pipeline is available in the environment.") from e

__all__ = ["get_dataloaders", "OracleGroupGuard"]

# simple re-exports -----------------------------------------------------------

def get_dataloaders(*args, **kwargs):  # type: ignore[override]
    return _get_dataloaders(*args, **kwargs)

class OracleGroupGuard(_OracleGroupGuard):
    """Thin alias to keep backwards compatibility after refactor."""
    pass
