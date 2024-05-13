"""
Infrastructure for testing Numba.

Numba releases often lag for a few months after Python releases, so we don't
want Numba to be a blocker for Python 3.X support. So this minimally emulates
the Numba module, while allowing for the fact that Numba may not be installed.
"""

import pytest

try:
    from numba import float64, guvectorize  # type: ignore[import-untyped]
except ImportError:
    float64 = []

    def guvectorize(_a, _b):  # type: ignore[no-untyped-def]
        """When Numba is unavailable, skip tests using the decorated function."""

        def decorator(_):  # type: ignore[no-untyped-def]
            def skip(*_args, **_kwargs):  # type: ignore[no-untyped-def]
                pytest.skip("Numba not available")

            return skip

        return decorator


__all__ = ["guvectorize", "float64"]
