from typing import Any

from polars._utils.deprecation import issue_deprecation_warning
from polars.api import get_shared_lib_location

__all__ = ["_get_shared_lib_location"]


def _get_shared_lib_location(main_file: Any) -> str:
    issue_deprecation_warning(
        "`polars.utils` is deprecated and will be made private. Use `from polars.api import get_shared_lib_location` instead.",
        version="0.20.14",
    )
    return get_shared_lib_location(main_file)
