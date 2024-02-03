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
    Return a dict with Polars build information.

    If Polars was compiled with "build_info" feature gate return the full build info,
    otherwise only version is included. The full build information dict contains
    the following keys ['build', 'info-time', 'dependencies', 'features', 'host',
    'target', 'git', 'version'].
    """
    return __build__
