from __future__ import annotations

import inspect
import warnings
from pathlib import Path
from typing import Any

import polars as pl


def find_stacklevel() -> int:
    """
    Find the first place in the stack that is not inside Polars.

    Taken from:
    https://github.com/pandas-dev/pandas/blob/ab89c53f48df67709a533b6a95ce3d911871a0a8/pandas/util/_exceptions.py#L30-L51
    """
    pkg_dir = str(Path(pl.__file__).parent)

    # https://stackoverflow.com/questions/17407119/python-inspect-stack-is-slow
    frame = inspect.currentframe()
    n = 0
    try:
        while frame:
            fname = inspect.getfile(frame)
            if fname.startswith(pkg_dir) or (
                (qualname := getattr(frame.f_code, "co_qualname", None))
                # ignore @singledispatch wrappers
                and qualname.startswith("singledispatch.")
            ):
                frame = frame.f_back
                n += 1
            else:
                break
    finally:
        # https://docs.python.org/3/library/inspect.html
        # > Though the cycle detector will catch these, destruction of the frames
        # > (and local variables) can be made deterministic by removing the cycle
        # > in a 'finally' clause.
        del frame
    return n


def issue_warning(message: str, category: type[Warning], **kwargs: Any) -> None:
    """
    Issue a warning.

    Parameters
    ----------
    message
        The message associated with the warning.
    category
        The warning category.
    **kwargs
        Additional arguments for `warnings.warn`. Note that the `stacklevel` is
        determined automatically.
    """
    warnings.warn(
        message=message, category=category, stacklevel=find_stacklevel(), **kwargs
    )


# this is called from rust
def _polars_warn(msg: str, category: type[Warning] = UserWarning) -> None:
    warnings.warn(
        msg,
        category=category,
        stacklevel=find_stacklevel(),
    )
