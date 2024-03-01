from __future__ import annotations

import os
from typing import Any

__all__ = ["get_shared_lib_location"]


def get_shared_lib_location(main_file: Any) -> str:
    """Get location of Shared Object file."""
    directory = os.path.dirname(main_file)  # noqa: PTH120
    return os.path.join(  # noqa: PTH118
        directory, next(filter(_is_shared_lib, os.listdir(directory)))
    )


def _is_shared_lib(file: str) -> bool:
    return file.endswith((".so", ".dll", ".pyd"))
