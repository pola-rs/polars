from __future__ import annotations

from typing import Any

from polars.utils.polars_version import get_polars_version

try:
    from polars.polars import _build_info_
except ImportError:
    _build_info_ = {}

_build_info_["version"] = get_polars_version() or "<missing>"


def build_info() -> dict[str, Any]:
    """
    Return a dict with Polars build information.

    If Polars was compiled with "build_info" feature gate return the full build info,
    otherwise only version is included. The full build information dict contains
    the following keys ['build', 'info-time', 'dependencies', 'features', 'host',
    'target', 'git', 'version'].
    """
    return _build_info_
