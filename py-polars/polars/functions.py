from datetime import datetime, timedelta
from typing import Optional, Sequence, Union

import numpy as np

import polars as pl

try:
    from polars.datatypes import py_type_to_polars_type
    from polars.polars import concat_df as _concat_df
    from polars.polars import concat_series as _concat_series

    _DOCUMENTING = False
except ImportError:
    _DOCUMENTING = True

__all__ = ["get_dummies", "concat", "repeat", "arg_where", "date_range"]


def get_dummies(df: "pl.DataFrame") -> "pl.DataFrame":
    """
    Convert categorical variables into dummy/indicator variables.

    Parameters
    ----------
    df
        DataFrame to convert.
    """
    return df.to_dummies()


def concat(
    items: Union[Sequence["pl.DataFrame"], Sequence["pl.Series"]], rechunk: bool = True
) -> Union["pl.DataFrame", "pl.Series"]:
    """
    Aggregate all the Dataframes/Series in a List of DataFrames/Series to a single DataFrame/Series.

    Parameters
    ----------
    items
        DataFrames/Series to concatenate.
    rechunk
        rechunk the final DataFrame/Series.
    """
    if not len(items) > 0:
        raise ValueError("cannot concat empty list")

    out: Union["pl.Series", "pl.DataFrame"]
    if isinstance(items[0], pl.DataFrame):
        out = pl.wrap_df(_concat_df(items))
    else:
        out = pl.wrap_s(_concat_series(items))

    if rechunk:
        return out.rechunk()  # type: ignore
    return out


def repeat(
    val: Union[int, float, str, bool], n: int, name: Optional[str] = None
) -> "pl.Series":
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

    dtype = py_type_to_polars_type(type(val))
    s = pl.Series._repeat(name, val, n, dtype)
    return s


def arg_where(mask: "pl.Series") -> "pl.Series":
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
    low: datetime, high: datetime, interval: timedelta, name: Optional[str] = None
) -> pl.Series:
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
    name
        Name of the output Series
    Returns
    -------
    A Series of type `Datetime`

    Examples
    --------
    >>> import polars as pl
    >>> from datetime import datetime, timedelta
    >>> pl.date_range(datetime(1985, 1, 1), datetime(2015, 7, 1), timedelta(days=1, hours=12))
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
    return pl.Series(
        name=name,
        values=np.arange(low, high, interval, dtype="datetime64[ms]").astype(int),
    ).cast(pl.Datetime)
