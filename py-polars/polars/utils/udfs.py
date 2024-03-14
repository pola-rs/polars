"""Deprecated module. Do not use."""

import os
from typing import Any

from polars._utils.deprecation import issue_deprecation_warning

__all__ = ["_get_shared_lib_location"]


def _get_shared_lib_location(main_file: Any) -> str:
    """
    Get the location of the dynamic library file.

    .. deprecated:: 0.20.16
        Use :func:`polars.plugins.register_plugin` instead.
    """
    issue_deprecation_warning(
        "`_get_shared_lib_location` is deprecated and will be removed in the next breaking release."
        " The new `register_plugin` function has this functionality built in."
        " Use `from polars.plugins import register_plugin` to import that function."
        " Check the user guide for the currently-recommended way to register a plugin:"
        " https://docs.pola.rs/user-guide/expressions/plugins",
        version="0.20.16",
    )
    directory = os.path.dirname(main_file)  # noqa: PTH120
    return os.path.join(  # noqa: PTH118
        directory, next(filter(_is_shared_lib, os.listdir(directory)))
    )


def _is_shared_lib(file: str) -> bool:
    return file.endswith((".so", ".dll", ".pyd"))
