from io import StringIO
from typing import TYPE_CHECKING

from polars.io.csv.functions import read_csv

if TYPE_CHECKING:
    from polars import DataFrame


def read_clipboard() -> "DataFrame":
    """
    Reads contents from clipboard into dataframe.

    Returns
    -------
    DataFrame

    Examples
    --------
    >>> df = pl.read_clipboard()  # doctest: +SKIP

    """
    df_str = _read_clipboard()
    return read_csv(StringIO(df_str), separator="\t")


def _read_clipboard() -> str:
    from tkinter import Tk

    r = Tk()
    r.withdraw()

    s = r.clipboard_get()

    r.update()
    r.destroy()

    return s


def _write_clipboard(s: str) -> None:
    from tkinter import Tk

    r = Tk()
    r.withdraw()

    r.clipboard_clear()
    r.clipboard_append(s)

    # 100ms delay before destroy, otherwise tkinter exits too early
    # see https://bugs.python.org/issue23760
    r.after(100, r.destroy)
