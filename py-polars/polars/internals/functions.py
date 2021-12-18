from datetime import datetime, timedelta
from typing import Optional, Sequence, Union, overload

from polars import internals as pli
from polars.datatypes import py_type_to_dtype
from polars.utils import _datetime_to_pl_timestamp, _timedelta_to_pl_duration

try:
    from polars.polars import concat_df as _concat_df
    from polars.polars import concat_lf as _concat_lf
    from polars.polars import concat_series as _concat_series
    from polars.polars import py_date_range as _py_date_range
    from polars.polars import py_diag_concat_df as _diag_concat_df

    _DOCUMENTING = False
except ImportError:  # pragma: no cover
    _DOCUMENTING = True


def get_dummies(df: "pli.DataFrame") -> "pli.DataFrame":
    """
    Convert categorical variables into dummy/indicator variables.

    Parameters
    ----------
    df
        DataFrame to convert.
    """
    return df.to_dummies()


@overload
def concat(
    items: Sequence["pli.DataFrame"],
    rechunk: bool = True,
    how: str = "vertical",
) -> "pli.DataFrame":
    ...


@overload
def concat(
    items: Sequence["pli.Series"],
    rechunk: bool = True,
    how: str = "vertical",
) -> "pli.Series":
    ...


def concat(
    items: Union[
        Sequence["pli.DataFrame"], Sequence["pli.Series"], Sequence["pli.LazyFrame"]
    ],
    rechunk: bool = True,
    how: str = "vertical",
) -> Union["pli.DataFrame", "pli.Series", "pli.LazyFrame"]:
    """
    Aggregate all the Dataframes/Series in a List of DataFrames/Series to a single DataFrame/Series.

    Parameters
    ----------
    items
        DataFrames/Series/LazyFrames to concatenate.
    rechunk
        rechunk the final DataFrame/Series.
    how
        Only used if the items are DataFrames.

        On of {"vertical", "diagonal"}.
        Vertical: Applies multiple `vstack` operations.
        Diagonal: Finds a union between the column schemas and fills missing column values with null.
    """
    if not len(items) > 0:
        raise ValueError("cannot concat empty list")

    out: Union["pli.Series", "pli.DataFrame", "pli.LazyFrame"]
    if isinstance(items[0], pli.DataFrame):
        if how == "vertical":
            out = pli.wrap_df(_concat_df(items))
        elif how == "diagonal":
            out = pli.wrap_df(_diag_concat_df(items))
        else:
            raise ValueError(
                f"how should be one of {'vertical', 'diagonal'}, got {how}"
            )
    elif isinstance(items[0], pli.LazyFrame):
        return pli.wrap_ldf(_concat_lf(items, rechunk))
    else:
        out = pli.wrap_s(_concat_series(items))

    if rechunk:
        return out.rechunk()
    return out


def repeat(
    val: Union[int, float, str, bool], n: int, name: Optional[str] = None
) -> "pli.Series":
    """
    Repeat a single value n times and collect into a Series.

    Parameters
    ----------
    val
        Value to repeat.
    n
        Number of repeats.
    name
        Optional name of the Series.
    """
    if name is None:
        name = ""

    dtype = py_type_to_dtype(type(val))
    s = pli.Series._repeat(name, val, n, dtype)
    return s


def arg_where(mask: "pli.Series") -> "pli.Series":
    """
    Get index values where Boolean mask evaluate True.

    Parameters
    ----------
    mask
        Boolean Series.

    Returns
    -------
    UInt32 Series
    """
    return mask.arg_true()


def date_range(
    low: datetime,
    high: datetime,
    interval: Union[str, timedelta],
    closed: Optional[str] = "both",
    name: Optional[str] = None,
) -> "pli.Series":
    """
    Create a date range of type `Datetime`.

    Parameters
    ----------
    low
        Lower bound of the date range
    high
        Upper bound of the date range
    interval
        Interval periods
        A python timedelta object or a polars duration `str`
        e.g.: "3d12h4m25s" # 3 days, 12 hours, 4 minutes, and 25 seconds
    closed {None, 'left', 'right', 'both', 'none'}
        Make the interval closed to the 'left', 'right', 'none' or 'both' sides.
    name
        Name of the output Series

    Returns
    -------
    A Series of type `Datetime`

    Examples
    --------
    >>> from datetime import datetime
    >>> pl.date_range(datetime(1985, 1, 1), datetime(2015, 7, 1), "1d12h")
    shape: (7426,)
    Series: '' [datetime]
    [
        1985-01-01 00:00:00
        1985-01-02 12:00:00
        1985-01-04 00:00:00
        1985-01-05 12:00:00
        1985-01-07 00:00:00
        1985-01-08 12:00:00
        1985-01-10 00:00:00
        1985-01-11 12:00:00
        1985-01-13 00:00:00
        1985-01-14 12:00:00
        1985-01-16 00:00:00
        1985-01-17 12:00:00
        ...
        2015-06-14 00:00:00
        2015-06-15 12:00:00
        2015-06-17 00:00:00
        2015-06-18 12:00:00
        2015-06-20 00:00:00
        2015-06-21 12:00:00
        2015-06-23 00:00:00
        2015-06-24 12:00:00
        2015-06-26 00:00:00
        2015-06-27 12:00:00
        2015-06-29 00:00:00
        2015-06-30 12:00:00
    ]

    """
    if isinstance(interval, timedelta):
        interval = _timedelta_to_pl_duration(interval)
    start = _datetime_to_pl_timestamp(low)
    stop = _datetime_to_pl_timestamp(high)
    if name is None:
        name = ""

    return pli.wrap_s(_py_date_range(start, stop, interval, closed, name))
