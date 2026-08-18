"""Shared utilities for query monitoring."""

from __future__ import annotations

import os
from typing import Final

MONITORING_ENV_VAR: Final[str] = "POLARS_QUERY_MONITORING"


def monitoring_enabled_globally() -> bool:
    return os.environ.get(MONITORING_ENV_VAR) == "1"


def activate_monitoring() -> None:
    """Ensure `polars_cloud` is installed and this session is authenticated."""
    from polars._dependencies import import_optional

    import_optional(
        "polars_cloud",
        err_prefix="query monitoring requires the",
        install_message=(
            "Please install using the command `pip install 'polars-cloud>=0.11.0'`."
        ),
    ).authenticate()
