from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


class HostilePathLike(os.PathLike[str]):
    """An ``os.PathLike`` whose ``str()`` is deliberately *not* the path.

    Its ``__str__`` returns a non-path value, so consuming it correctly requires
    ``os.fspath()`` rather than ``str()`` — a regression guard for gh #17828.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = os.fspath(path)

    def __fspath__(self) -> str:
        return self._path

    def __repr__(self) -> str:
        return "<HostilePathLike: path hidden>"

    __str__ = __repr__
