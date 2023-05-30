from __future__ import annotations

from typing import TYPE_CHECKING, Any

from polars.dependencies import pickle

if TYPE_CHECKING:
    from polars import DataFrame


def _deserialize_and_execute(
    buf: bytes, with_columns: list[str] | None, *args: Any
) -> DataFrame:
    """
    Deserialize and execute the given function for the projected columns.

    Called from polars-lazy. Polars-lazy provides the bytes of the pickled function and
    the projected columns.

    Parameters
    ----------
    buf
        Pickled function
    with_columns
        Columns that are projected
    *args
        Additional function arguments.

    """
    func = pickle.loads(buf)
    return func(with_columns, *args)
