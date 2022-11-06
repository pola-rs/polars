from __future__ import annotations

from typing import Any

try:
    from polars.polars import version

    _version_ = version()
except ImportError:
    _version_ = "<missing>"

try:
    from polars.polars import _build_info_
except ImportError:
    _build_info_ = {}

_build_info_["version"] = _version_


def build_info() -> dict[str, Any]:
    """
    Return a dict with polars build information.

    If polars was compiled with "build_info" feature gate return the full build info,
    otherwise only version is included. The full build information dict contains
    the following keys ['build', 'info-time', 'dependencies', 'features', 'host',
    'target', 'git', 'version'].
    """
    return _build_info_
