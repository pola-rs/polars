from datetime import date, datetime, timedelta
from typing import Optional, Sequence, Tuple, Union, overload

from polars import internals as pli
from polars.datatypes import Date
from polars.utils import (
    _datetime_to_pl_timestamp,
    _timedelta_to_pl_duration,
    in_nanoseconds_window,
)

try:
    from polars.polars import concat_df as _concat_df
    from polars.polars import concat_lf as _concat_lf
    from polars.polars import concat_series as _concat_series
    from polars.polars import py_date_range as _py_date_range
    from polars.polars import py_diag_concat_df as _diag_concat_df
    from polars.polars import py_hor_concat_df as _hor_concat_df

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


@overload
def concat(
    items: Sequence["pli.LazyFrame"],
    rechunk: bool = True,
    how: str = "vertical",
) -> "pli.LazyFrame":
    ...


@overload
def concat(
    items: Sequence["pli.Expr"],
    rechunk: bool = True,
    how: str = "vertical",
) -> "pli.Expr":
    ...


def concat(
    items: Union[
        Sequence["pli.DataFrame"],
        Sequence["pli.Series"],
        Sequence["pli.LazyFrame"],
        Sequence["pli.Expr"],
    ],
    rechunk: bool = True,
    how: str = "vertical",
) -> Union["pli.DataFrame", "pli.Series", "pli.LazyFrame", "pli.Expr"]:
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

        One of {"vertical", "diagonal", "horizontal"}.

        - Vertical: Applies multiple `vstack` operations.
        - Diagonal: Finds a union between the column schemas and fills missing column values with null.
        - Horizontal: Stacks Series horizontally and fills with nulls if the lengths don't match.

    Examples
    --------
    >>> df1 = pl.DataFrame({"a": [1], "b": [3]})
    >>> df2 = pl.DataFrame({"a": [2], "b": [4]})
    >>> pl.concat([df1, df2])
    shape: (2, 2)
    ┌─────┬─────┐
    │ a   ┆ b   │
    │ --- ┆ --- │
    │ i64 ┆ i64 │
    ╞═════╪═════╡
    │ 1   ┆ 3   │
    ├╌╌╌╌╌┼╌╌╌╌╌┤
    │ 2   ┆ 4   │
    └─────┴─────┘

    """
    if not len(items) > 0:
        raise ValueError("cannot concat empty list")

    out: Union["pli.Series", "pli.DataFrame", "pli.LazyFrame", "pli.Expr"]
    first = items[0]
    if isinstance(first, pli.DataFrame):
        if how == "vertical":
            out = pli.wrap_df(_concat_df(items))
        elif how == "diagonal":
            out = pli.wrap_df(_diag_concat_df(items))
        elif how == "horizontal":
            out = pli.wrap_df(_hor_concat_df(items))
        else:
            raise ValueError(
                f"how should be one of {'vertical', 'diagonal'}, got {how}"
            )
    elif isinstance(first, pli.LazyFrame):
        return pli.wrap_ldf(_concat_lf(items, rechunk))
    elif isinstance(first, pli.Series):
        out = pli.wrap_s(_concat_series(items))
    elif isinstance(first, pli.Expr):
        out = first
        for e in items[1:]:
            out = out.append(e)  # type: ignore
    else:
        raise ValueError(f"did not expect type: {type(first)} in 'pl.concat'.")

    if rechunk:
        return out.rechunk()
    return out


def _ensure_datetime(value: Union[date, datetime]) -> Tuple[datetime, bool]:
    is_date_type = False
    if isinstance(value, date) and not isinstance(value, datetime):
        value = datetime(value.year, value.month, value.day)
        is_date_type = True
    return value, is_date_type


def _interval_granularity(interval: str) -> str:
    return interval[-2:].lstrip("0123456789")


def date_range(
    low: Union[date, datetime],
    high: Union[date, datetime],
    interval: Union[str, timedelta],
    closed: Optional[str] = "both",
    name: Optional[str] = None,
    time_unit: Optional[str] = None,
) -> "pli.Series":
    """
    Create a range of type `Datetime` (or `Date`).

    Parameters
    ----------
    low
        Lower bound of the date range.
    high
        Upper bound of the date range.
    interval
        Interval periods
        A python timedelta object or a polars duration `str`
        e.g.: "3d12h4m25s" # 3 days, 12 hours, 4 minutes, and 25 seconds
    closed {None, 'left', 'right', 'both', 'none'}
        Make the interval closed to the 'left', 'right', 'none' or 'both' sides.
    name
        Name of the output Series.
    time_unit
        Set the time unit; one of {'ns', 'us', 'ms'}.

    Notes
    -----
    If both `low` and `high` are passed as date types (not datetime), and the
    interval granularity is no finer than 1d, the returned range is also of
    type date. All other permutations return a datetime Series.

    Returns
    -------
    A Series of type `Datetime` or `Date`.

    Examples
    --------
    >>> from datetime import datetime, date
    >>> pl.date_range(datetime(1985, 1, 1), datetime(2015, 7, 1), "1d12h")
    shape: (7426,)
    Series: '' [datetime[ns]]
    [
        1985-01-01 00:00:00
        1985-01-02 12:00:00
        1985-01-04 00:00:00
        1985-01-05 12:00:00
        1985-01-07 00:00:00
        ...
        2015-06-24 12:00:00
        2015-06-26 00:00:00
        2015-06-27 12:00:00
        2015-06-29 00:00:00
        2015-06-30 12:00:00
    ]

    >>> pl.date_range(date(2022, 1, 1), date(2022, 3, 1), "1mo", name="drange")
    shape: (3,)
    Series: 'drange' [date]
    [
        2022-01-01
        2022-02-01
        2022-03-01
    ]
    """
    if isinstance(interval, timedelta):
        interval = _timedelta_to_pl_duration(interval)

    low, low_is_date = _ensure_datetime(low)
    high, high_is_date = _ensure_datetime(high)

    if in_nanoseconds_window(low) and in_nanoseconds_window(high) and time_unit is None:
        tu = "ns"
    elif time_unit is not None:
        tu = time_unit
    else:
        tu = "ms"

    start = _datetime_to_pl_timestamp(low, tu)
    stop = _datetime_to_pl_timestamp(high, tu)
    if name is None:
        name = ""

    dt_range = pli.wrap_s(_py_date_range(start, stop, interval, closed, name, tu))
    if (
        low_is_date
        and high_is_date
        and not _interval_granularity(interval).endswith(("h", "m", "s"))
    ):
        dt_range = dt_range.cast(Date)

    return dt_range
