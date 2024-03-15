"""Deprecated module. Do not use."""

import os
from typing import Any

from polars._utils.deprecation import deprecate_function

__all__ = ["_get_shared_lib_location"]


@deprecate_function(
    "It will be removed in the next breaking release."
    " The new `register_plugin_function` function has this functionality built in."
    " Use `from polars.plugins import register_plugin_function` to import that function."
    " Check the user guide for the currently-recommended way to register a plugin:"
    " https://docs.pola.rs/user-guide/expressions/plugins",
    version="0.20.16",
)
def _get_shared_lib_location(main_file: Any) -> str:
    """
    Get the location of the dynamic library file.

    .. deprecated:: 0.20.16
        Use :func:`polars.plugins.register_plugin_function` instead.

    Warnings
    --------
    This function is deprecated and will be removed in the next breaking release.
    The new `polars.plugins.register_plugin_function` function has this
    functionality built in. Use `from polars.plugins import register_plugin_function`
    to import that function.

    Check the user guide for the recommended way to register a plugin:
    https://docs.pola.rs/user-guide/expressions/plugins
    """
    directory = os.path.dirname(main_file)  # noqa: PTH120
    return os.path.join(  # noqa: PTH118
        directory, next(filter(_is_shared_lib, os.listdir(directory)))
    )


def _is_shared_lib(file: str) -> bool:
    return file.endswith((".so", ".dll", ".pyd"))
