from __future__ import annotations

from typing import Any

from polars.utils._polars_version import get_polars_version

try:
    from polars.polars import __build__
except ImportError:
    __build__ = {}

__build__["version"] = get_polars_version() or "<missing>"


def build_info() -> dict[str, Any]:
    """
    Return detailed Polars build information.

    The dictionary with build information contains the following keys:

    - `"build"`
    - `"info-time"`
    - `"dependencies"`
    - `"features"`
    - `"host"`
    - `"target"`
    - `"git"`
    - `"version"`

    If Polars was compiled without the `build_info` feature flag, only the `"version"`
    key is included.

    Notes
    -----
    `pyo3-built`_ is used to generate the build information.

    .. _pyo3-built: https://github.com/PyO3/pyo3-built
    """
    return __build__
