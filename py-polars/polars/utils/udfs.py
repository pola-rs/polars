"""Deprecated module. Do not use."""

import os
from typing import Any

__all__ = ["_get_shared_lib_location"]


def _get_shared_lib_location(main_file: Any) -> str:
    directory = os.path.dirname(main_file)  # noqa: PTH120
    return os.path.join(  # noqa: PTH118
        directory, next(filter(_is_shared_lib, os.listdir(directory)))
    )


def _is_shared_lib(file: str) -> bool:
    return file.endswith((".so", ".dll", ".pyd"))
