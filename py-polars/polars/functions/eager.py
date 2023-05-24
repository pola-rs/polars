from __future__ import annotations

import contextlib
import warnings
from datetime import datetime, time, timedelta
from functools import reduce
from itertools import chain
from typing import TYPE_CHECKING, Iterable, List, Sequence, cast, overload

import polars._reexport as pl
from polars import functions as F
from polars.datatypes import Date
from polars.expr.datetime import TIME_ZONE_DEPRECATION_MESSAGE
from polars.type_aliases import FrameType
from polars.utils._parse_expr_input import expr_to_lit_or_expr
from polars.utils._wrap import wrap_df, wrap_expr, wrap_ldf, wrap_s
from polars.utils.convert import (
    _datetime_to_pl_timestamp,
    _time_to_pl_time,
    _timedelta_to_pl_duration,
    _tzinfo_to_str,
)
from polars.utils.decorators import deprecated_alias
from polars.utils.various import find_stacklevel, ordered_unique

with contextlib.suppress(ImportError):  # Module not available when building docs
    import polars.polars as plr

if TYPE_CHECKING:
    import sys
    from datetime import date

    from polars import DataFrame, Expr, LazyFrame, Series
    from polars.type_aliases import (
        ClosedInterval,
        ConcatMethod,
        JoinStrategy,
        PolarsType,
        TimeUnit,
    )

    if sys.version_info >= (3, 8):
        from typing import Literal
    else:
        from typing_extensions import Literal


def concat(
    items: Iterable[PolarsType],
    *,
    how: ConcatMethod = "vertical",
    rechunk: bool = True,
    parallel: bool = True,
) -> PolarsType:
    """
    Combine multiple DataFrames, LazyFrames, or Series into a single object.

    Parameters
    ----------
    items
        DataFrames, LazyFrames, or Series to concatenate.
    how : {'vertical', 'diagonal', 'horizontal', 'align'}
        Series only support the `vertical` strategy.
        LazyFrames do not support the `horizontal` strategy.

        * vertical: Applies multiple `vstack` operations.
        * diagonal: Finds a union between the column schemas and fills missing column
          values with ``null``.
        * horizontal: Stacks Series from DataFrames horizontally and fills with ``null``
          if the lengths don't match.
        * align: Combines frames horizontally, auto-determining the common key columns
          and aligning rows using the same logic as ``align_frames``; this behaviour is
          patterned after a full outer join, but does not handle column-name collision.
          (If you need more control, you should use a suitable join method instead).
    rechunk
        Make sure that the result data is in contiguous memory.
    parallel
        Only relevant for LazyFrames. This determines if the concatenated
        lazy computations may be executed in parallel.

    Examples
    --------
    >>> df1 = pl.DataFrame({"a": [1], "b": [3]})
    >>> df2 = pl.DataFrame({"a": [2], "b": [4]})
    >>> pl.concat([df1, df2])  # default is 'vertical' strategy
    shape: (2, 2)
    ┌─────┬─────┐
    │ a   ┆ b   │
    │ --- ┆ --- │
    │ i64 ┆ i64 │
    ╞═════╪═════╡
    │ 1   ┆ 3   │
    │ 2   ┆ 4   │
    └─────┴─────┘

    >>> df_h1 = pl.DataFrame({"l1": [1, 2], "l2": [3, 4]})
    >>> df_h2 = pl.DataFrame({"r1": [5, 6], "r2": [7, 8], "r3": [9, 10]})
    >>> pl.concat([df_h1, df_h2], how="horizontal")
    shape: (2, 5)
    ┌─────┬─────┬─────┬─────┬─────┐
    │ l1  ┆ l2  ┆ r1  ┆ r2  ┆ r3  │
    │ --- ┆ --- ┆ --- ┆ --- ┆ --- │
    │ i64 ┆ i64 ┆ i64 ┆ i64 ┆ i64 │
    ╞═════╪═════╪═════╪═════╪═════╡
    │ 1   ┆ 3   ┆ 5   ┆ 7   ┆ 9   │
    │ 2   ┆ 4   ┆ 6   ┆ 8   ┆ 10  │
    └─────┴─────┴─────┴─────┴─────┘

    >>> df_d1 = pl.DataFrame({"a": [1], "b": [3]})
    >>> df_d2 = pl.DataFrame({"a": [2], "c": [4]})
    >>> pl.concat([df_d1, df_d2], how="diagonal")
    shape: (2, 3)
    ┌─────┬──────┬──────┐
    │ a   ┆ b    ┆ c    │
    │ --- ┆ ---  ┆ ---  │
    │ i64 ┆ i64  ┆ i64  │
    ╞═════╪══════╪══════╡
    │ 1   ┆ 3    ┆ null │
    │ 2   ┆ null ┆ 4    │
    └─────┴──────┴──────┘

    >>> df_a1 = pl.DataFrame({"id": [1, 2], "x": [3, 4]})
    >>> df_a2 = pl.DataFrame({"id": [2, 3], "y": [5, 6]})
    >>> df_a3 = pl.DataFrame({"id": [1, 3], "z": [7, 8]})
    >>> pl.concat([df_a1, df_a2, df_a3], how="align")
    shape: (3, 4)
    ┌─────┬──────┬──────┬──────┐
    │ id  ┆ x    ┆ y    ┆ z    │
    │ --- ┆ ---  ┆ ---  ┆ ---  │
    │ i64 ┆ i64  ┆ i64  ┆ i64  │
    ╞═════╪══════╪══════╪══════╡
    │ 1   ┆ 3    ┆ null ┆ 7    │
    │ 2   ┆ 4    ┆ 5    ┆ null │
    │ 3   ┆ null ┆ 6    ┆ 8    │
    └─────┴──────┴──────┴──────┘

    """
    # unpack/standardise (handles generator input)
    elems = list(items)

    if not len(elems) > 0:
        raise ValueError("cannot concat empty list")
    elif len(elems) == 1 and isinstance(
        elems[0], (pl.DataFrame, pl.Series, pl.LazyFrame)
    ):
        return elems[0]

    if how == "align":
        if not isinstance(elems[0], (pl.DataFrame, pl.LazyFrame)):
            raise RuntimeError(
                f"'align' strategy is not supported for {type(elems[0]).__name__}"
            )

        # establish common columns, maintaining the order in which they appear
        all_columns = list(chain.from_iterable(e.columns for e in elems))
        key = {v: k for k, v in enumerate(ordered_unique(all_columns))}
        common_cols = sorted(
            reduce(
                lambda x, y: set(x) & set(y),  # type: ignore[arg-type, return-value]
                chain(e.columns for e in elems),
            ),
            key=lambda k: key.get(k, 0),
        )
        # align the frame data using an outer join with no suffix-resolution
        # (so we raise an error in case of column collision, like "horizontal")
        lf: LazyFrame = reduce(
            lambda x, y: x.join(y, how="outer", on=common_cols, suffix=""),
            [df.lazy() for df in elems],
        ).sort(by=common_cols)

        eager = isinstance(elems[0], pl.DataFrame)
        return lf.collect() if eager else lf  # type: ignore[return-value]

    out: Series | DataFrame | LazyFrame | Expr
    first = elems[0]

    if isinstance(first, pl.DataFrame):
        if how == "vertical":
            out = wrap_df(plr.concat_df(elems))
        elif how == "diagonal":
            out = wrap_df(plr.diag_concat_df(elems))
        elif how == "horizontal":
            out = wrap_df(plr.hor_concat_df(elems))
        else:
            raise ValueError(
                f"`how` must be one of {{'vertical','diagonal','horizontal','align'}}, "
                f"got {how!r}"
            )
    elif isinstance(first, pl.LazyFrame):
        if how == "vertical":
            return wrap_ldf(plr.concat_lf(elems, rechunk, parallel))
        if how == "diagonal":
            return wrap_ldf(plr.diag_concat_lf(elems, rechunk, parallel))
        else:
            raise ValueError(
                "'LazyFrame' only allows {'vertical','diagonal','align'} concat strategies."
            )
    elif isinstance(first, pl.Series):
        if how == "vertical":
            out = wrap_s(plr.concat_series(elems))
        else:
            raise ValueError("'Series' only allows {'vertical'} concat strategy.")

    elif isinstance(first, pl.Expr):
        out = first
        for e in elems[1:]:
            out = out.append(e)
    else:
        raise ValueError(f"did not expect type: {type(first)} in 'pl.concat'.")

    if rechunk:
        return out.rechunk()
    return out


def _ensure_datetime(value: date | datetime) -> tuple[datetime, bool]:
    is_date_type = False
    if not isinstance(value, datetime):
        value = datetime(value.year, value.month, value.day)
        is_date_type = True
    return value, is_date_type


def _interval_granularity(interval: str) -> str:
    return interval[-2:].lstrip("0123456789")


@overload
def date_range(
    start: date | datetime | Expr | str,
    end: date | datetime | Expr | str,
    interval: str | timedelta = ...,
    *,
    eager: Literal[False] = ...,
    closed: ClosedInterval = ...,
    name: str | None = None,
    time_unit: TimeUnit | None = ...,
    time_zone: str | None = ...,
) -> Expr:
    ...


@overload
def date_range(
    start: date | datetime | Expr | str,
    end: date | datetime | Expr | str,
    interval: str | timedelta = ...,
    *,
    eager: Literal[True],
    closed: ClosedInterval = ...,
    name: str | None = ...,
    time_unit: TimeUnit | None = ...,
    time_zone: str | None = ...,
) -> Series:
    ...


@overload
def date_range(
    start: date | datetime | Expr | str,
    end: date | datetime | Expr | str,
    interval: str | timedelta = ...,
    *,
    eager: bool,
    closed: ClosedInterval = ...,
    name: str | None = ...,
    time_unit: TimeUnit | None = ...,
    time_zone: str | None = ...,
) -> Series | Expr:
    ...


@deprecated_alias(low="start", high="end")
def date_range(
    start: date | datetime | Expr | str,
    end: date | datetime | Expr | str,
    interval: str | timedelta = "1d",
    *,
    eager: bool = False,
    closed: ClosedInterval = "both",
    name: str | None = None,
    time_unit: TimeUnit | None = None,
    time_zone: str | None = None,
) -> Series | Expr:
    """
    Create a range of type `Datetime` (or `Date`).

    Parameters
    ----------
    start
        Lower bound of the date range, given as a date, datetime, Expr, or column name.
    end
        Upper bound of the date range, given as a date, datetime, Expr, or column name.
    interval
        Interval of the range periods; can be a python timedelta object like
        ``timedelta(days=10)`` or a polars duration string, such as ``3d12h4m25s``
        (representing 3 days, 12 hours, 4 minutes, and 25 seconds).
    eager
        Evaluate immediately and return a ``Series``. If set to ``False`` (default),
        return an expression instead.
    closed : {'both', 'left', 'right', 'none'}
        Define whether the temporal window interval is closed or not.
    name
        Name of the output Series.
    time_unit : {None, 'ns', 'us', 'ms'}
        Set the time unit.
    time_zone:
        Optional timezone


    Notes
    -----
    If both ``start`` and ``end`` are passed as date types (not datetime), and the
    interval granularity is no finer than 1d, the returned range is also of
    type date. All other permutations return a datetime Series.

    Returns
    -------
    A Series of type `Datetime` or `Date`.

    Examples
    --------
    Using polars duration string to specify the interval:

    >>> from datetime import date
    >>> pl.date_range(
    ...     date(2022, 1, 1), date(2022, 3, 1), "1mo", name="dtrange", eager=True
    ... )
    shape: (3,)
    Series: 'dtrange' [date]
    [
        2022-01-01
        2022-02-01
        2022-03-01
    ]

    Using `timedelta` object to specify the interval:

    >>> from datetime import datetime, timedelta
    >>> pl.date_range(
    ...     datetime(1985, 1, 1),
    ...     datetime(1985, 1, 10),
    ...     timedelta(days=1, hours=12),
    ...     time_unit="ms",
    ...     eager=True,
    ... )
    shape: (7,)
    Series: '' [datetime[ms]]
    [
        1985-01-01 00:00:00
        1985-01-02 12:00:00
        1985-01-04 00:00:00
        1985-01-05 12:00:00
        1985-01-07 00:00:00
        1985-01-08 12:00:00
        1985-01-10 00:00:00
    ]

    Specify a time zone

    >>> pl.date_range(
    ...     datetime(2022, 1, 1),
    ...     datetime(2022, 3, 1),
    ...     "1mo",
    ...     time_zone="America/New_York",
    ...     eager=True,
    ... )
    shape: (3,)
    Series: '' [datetime[μs, America/New_York]]
    [
        2022-01-01 00:00:00 EST
        2022-02-01 00:00:00 EST
        2022-03-01 00:00:00 EST
    ]

    Combine with ``month_end`` to get the last day of the month:

    >>> (
    ...     pl.date_range(
    ...         datetime(2022, 1, 1), datetime(2022, 3, 1), "1mo", eager=True
    ...     ).dt.month_end()
    ... )
    shape: (3,)
    Series: '' [datetime[μs]]
    [
        2022-01-31 00:00:00
        2022-02-28 00:00:00
        2022-03-31 00:00:00
    ]

    """
    from polars.dependencies import zoneinfo

    if time_zone is not None and time_zone not in zoneinfo.available_timezones():
        warnings.warn(
            TIME_ZONE_DEPRECATION_MESSAGE,
            DeprecationWarning,
            stacklevel=find_stacklevel(),
        )

    if name is None:
        name = ""
    if isinstance(interval, timedelta):
        interval = _timedelta_to_pl_duration(interval)
    elif " " in interval:
        interval = interval.replace(" ", "")

    if (
        not eager
        or isinstance(start, (str, pl.Expr))
        or isinstance(end, (str, pl.Expr))
    ):
        start = expr_to_lit_or_expr(start, str_to_lit=False)._pyexpr
        end = expr_to_lit_or_expr(end, str_to_lit=False)._pyexpr
        return wrap_expr(
            plr.date_range_lazy(start, end, interval, closed, name, time_zone)
        )

    start, start_is_date = _ensure_datetime(start)
    end, end_is_date = _ensure_datetime(end)

    if start.tzinfo is not None or time_zone is not None:
        if start.tzinfo != end.tzinfo:
            raise ValueError(
                "Cannot mix different timezone aware datetimes."
                f" Got: '{start.tzinfo}' and '{end.tzinfo}'."
            )

        if time_zone is not None and start.tzinfo is not None:
            if _tzinfo_to_str(start.tzinfo) != time_zone:
                raise ValueError(
                    "Given time_zone is different from that of timezone aware datetimes."
                    f" Given: '{time_zone}', got: '{start.tzinfo}'."
                )
        if time_zone is None and start.tzinfo is not None:
            time_zone = _tzinfo_to_str(start.tzinfo)

    time_unit_: TimeUnit
    if time_unit is not None:
        time_unit_ = time_unit
    elif "ns" in interval:
        time_unit_ = "ns"
    else:
        time_unit_ = "us"

    start_pl = _datetime_to_pl_timestamp(start, time_unit_)
    end_pl = _datetime_to_pl_timestamp(end, time_unit_)
    dt_range = wrap_s(
        plr.date_range(start_pl, end_pl, interval, closed, name, time_unit_, time_zone)
    )
    if (
        start_is_date
        and end_is_date
        and not _interval_granularity(interval).endswith(("h", "m", "s"))
    ):
        dt_range = dt_range.cast(Date)

    return dt_range


@overload
def time_range(
    start: time | Expr | str | None = ...,
    end: time | Expr | str | None = ...,
    interval: str | timedelta = ...,
    *,
    closed: ClosedInterval = ...,
    eager: Literal[False] = ...,
    name: str | None = ...,
) -> Expr:
    ...


@overload
def time_range(
    start: time | Expr | str | None = ...,
    end: time | Expr | str | None = ...,
    interval: str | timedelta = ...,
    *,
    closed: ClosedInterval = ...,
    eager: Literal[True],
    name: str | None = ...,
) -> Series:
    ...


@overload
def time_range(
    start: time | Expr | str | None = ...,
    end: time | Expr | str | None = ...,
    interval: str | timedelta = ...,
    *,
    closed: ClosedInterval = ...,
    eager: bool,
    name: str | None = ...,
) -> Series | Expr:
    ...


def time_range(
    start: time | Expr | str | None = None,
    end: time | Expr | str | None = None,
    interval: str | timedelta = "1h",
    *,
    closed: ClosedInterval = "both",
    eager: bool = False,
    name: str | None = None,
) -> Series | Expr:
    """
    Create a range of type `Time`.

    Parameters
    ----------
    start
        Lower bound of the time range, given as a time, Expr, or column name.
        If omitted, will default to ``time(0,0,0,0)``.
    end
        Upper bound of the time range, given as a time, Expr, or column name.
        If omitted, will default to ``time(23,59,59,999999)``.
    interval
        Interval of the range periods; can be a python timedelta object like
        ``timedelta(minutes=10)`` or a polars duration string, such as ``1h30m25s``
        (representing 1 hour, 30 minutes, and 25 seconds).
    closed : {'both', 'left', 'right', 'none'}
        Define whether the temporal window interval is closed or not.
    eager:
        Evaluate immediately and return a ``Series``. If set to ``False`` (default),
        return an expression instead.
    name
        Name of the output Series.

        .. deprecated:: 0.18.0
            This argument is deprecated. Use the ``alias`` method instead.

    Returns
    -------
    A Series of type `Time`.

    Examples
    --------
    Create a Series that starts at 14:00, with intervals of 3 hours and 15 mins:

    >>> from datetime import time
    >>> pl.time_range(
    ...     start=time(14, 0),
    ...     interval=timedelta(hours=3, minutes=15),
    ...     eager=True,
    ... )
    shape: (4,)
    Series: 'time' [time]
    [
        14:00:00
        17:15:00
        20:30:00
        23:45:00
    ]

    Generate a DataFrame with two columns made of eager ``time_range`` Series,
    and create a third column using ``time_range`` in expression context:

    >>> lf = pl.LazyFrame(
    ...     {
    ...         "start": pl.time_range(interval="6h", eager=True),
    ...         "stop": pl.time_range(start=time(2, 59), interval="5h59m", eager=True),
    ...     }
    ... ).with_columns(
    ...     intervals=pl.time_range("start", "stop", interval="1h29m", eager=False)
    ... )
    >>> lf.collect()
    shape: (4, 3)
    ┌──────────┬──────────┬────────────────────────────────┐
    │ start    ┆ stop     ┆ intervals                      │
    │ ---      ┆ ---      ┆ ---                            │
    │ time     ┆ time     ┆ list[time]                     │
    ╞══════════╪══════════╪════════════════════════════════╡
    │ 00:00:00 ┆ 02:59:00 ┆ [00:00:00, 01:29:00, 02:58:00] │
    │ 06:00:00 ┆ 08:58:00 ┆ [06:00:00, 07:29:00, 08:58:00] │
    │ 12:00:00 ┆ 14:57:00 ┆ [12:00:00, 13:29:00]           │
    │ 18:00:00 ┆ 20:56:00 ┆ [18:00:00, 19:29:00]           │
    └──────────┴──────────┴────────────────────────────────┘

    """
    if name is not None:
        warnings.warn(
            "the `name` argument is deprecated. Use the `alias` method instead.",
            DeprecationWarning,
            stacklevel=find_stacklevel(),
        )

    if isinstance(interval, timedelta):
        interval = _timedelta_to_pl_duration(interval)
    elif " " in interval:
        interval = interval.replace(" ", "").lower()

    for unit in ("y", "mo", "w", "d"):
        if unit in interval:
            raise ValueError(f"invalid interval unit for time_range: found {unit!r}")

    default_start = time(0, 0, 0)
    default_end = time(23, 59, 59, 999999)
    if (
        not eager
        or isinstance(start, (str, pl.Expr))
        or isinstance(end, (str, pl.Expr))
    ):
        start_expr = (
            F.lit(default_start)
            if start is None
            else expr_to_lit_or_expr(start, str_to_lit=False)
        )._pyexpr

        end_expr = (
            F.lit(default_end)
            if end is None
            else expr_to_lit_or_expr(end, str_to_lit=False)
        )._pyexpr

        tm_expr = wrap_expr(plr.time_range_lazy(start_expr, end_expr, interval, closed))
        if name is not None:
            tm_expr = tm_expr.alias(name)
        return tm_expr
    else:
        tm_srs = wrap_s(
            plr.time_range(
                _time_to_pl_time(default_start if start is None else start),
                _time_to_pl_time(default_end if end is None else end),
                interval,
                closed,
            )
        )
        if name is not None:
            tm_srs = tm_srs.alias(name)
        return tm_srs


def align_frames(
    *frames: FrameType,
    on: str | Expr | Sequence[str] | Sequence[Expr] | Sequence[str | Expr],
    how: JoinStrategy = "outer",
    select: str | Expr | Sequence[str | Expr] | None = None,
    descending: bool | Sequence[bool] = False,
) -> list[FrameType]:
    r"""
    Align a sequence of frames using common values from one or more columns as a key.

    Frames that do not contain the given key values have rows injected (with nulls
    filling the non-key columns), and each resulting frame is sorted by the key.

    The original column order of input frames is not changed unless ``select`` is
    specified (in which case the final column order is determined from that). In the
    case where duplicate key values exist, the alignment behaviour is determined by
    the given alignment strategy specified in the ``how`` parameter (by default this
    is a full outer join, but if your data is suitable you can get a large speedup
    by setting ``how="left"`` instead).

    Note that this function does not result in a joined frame - you receive the same
    number of frames back that you passed in, but each is now aligned by key and has
    the same number of rows.

    Parameters
    ----------
    frames
        Sequence of DataFrames or LazyFrames.
    on
        One or more columns whose unique values will be used to align the frames.
    select
        Optional post-alignment column select to constrain and/or order
        the columns returned from the newly aligned frames.
    descending
        Sort the alignment column values in descending order; can be a single
        boolean or a list of booleans associated with each column in ``on``.
    how
        By default the row alignment values are determined using a full outer join
        strategy across all frames; if you know that the first frame contains all
        required keys, you can set ``how="left"`` for a large performance increase.

    Examples
    --------
    >>> from datetime import date
    >>> df1 = pl.DataFrame(
    ...     {
    ...         "dt": [date(2022, 9, 1), date(2022, 9, 2), date(2022, 9, 3)],
    ...         "x": [3.5, 4.0, 1.0],
    ...         "y": [10.0, 2.5, 1.5],
    ...     }
    ... )
    >>> df2 = pl.DataFrame(
    ...     {
    ...         "dt": [date(2022, 9, 2), date(2022, 9, 3), date(2022, 9, 1)],
    ...         "x": [8.0, 1.0, 3.5],
    ...         "y": [1.5, 12.0, 5.0],
    ...     }
    ... )
    >>> df3 = pl.DataFrame(
    ...     {
    ...         "dt": [date(2022, 9, 3), date(2022, 9, 2)],
    ...         "x": [2.0, 5.0],
    ...         "y": [2.5, 2.0],
    ...     }
    ... )  # doctest: +IGNORE_RESULT
    >>> pl.Config.set_tbl_formatting("UTF8_FULL")  # doctest: +IGNORE_RESULT
    #
    # df1                              df2                              df3
    # shape: (3, 3)                    shape: (3, 3)                    shape: (2, 3)
    # ┌────────────┬─────┬──────┐      ┌────────────┬─────┬──────┐      ┌────────────┬─────┬─────┐
    # │ dt         ┆ x   ┆ y    │      │ dt         ┆ x   ┆ y    │      │ dt         ┆ x   ┆ y   │
    # │ ---        ┆ --- ┆ ---  │      │ ---        ┆ --- ┆ ---  │      │ ---        ┆ --- ┆ --- │
    # │ date       ┆ f64 ┆ f64  │      │ date       ┆ f64 ┆ f64  │      │ date       ┆ f64 ┆ f64 │
    # ╞════════════╪═════╪══════╡      ╞════════════╪═════╪══════╡      ╞════════════╪═════╪═════╡
    # │ 2022-09-01 ┆ 3.5 ┆ 10.0 │\  ,->│ 2022-09-02 ┆ 8.0 ┆ 1.5  │\  ,->│ 2022-09-03 ┆ 2.0 ┆ 2.5 │
    # ├╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌┤ \/   ├╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌┤ \/   ├╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌┤
    # │ 2022-09-02 ┆ 4.0 ┆ 2.5  │_/\,->│ 2022-09-03 ┆ 1.0 ┆ 12.0 │_/`-->│ 2022-09-02 ┆ 5.0 ┆ 2.0 │
    # ├╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌┤  /\  ├╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌┤      └────────────┴─────┴─────┘
    # │ 2022-09-03 ┆ 1.0 ┆ 1.5  │_/  `>│ 2022-09-01 ┆ 3.5 ┆ 5.0  │-//-
    # └────────────┴─────┴──────┘      └────────────┴─────┴──────┘
    ...

    Align frames by the "dt" column:

    >>> af1, af2, af3 = pl.align_frames(
    ...     df1, df2, df3, on="dt"
    ... )  # doctest: +IGNORE_RESULT
    #
    # df1                              df2                              df3
    # shape: (3, 3)                    shape: (3, 3)                    shape: (3, 3)
    # ┌────────────┬─────┬──────┐      ┌────────────┬─────┬──────┐      ┌────────────┬──────┬──────┐
    # │ dt         ┆ x   ┆ y    │      │ dt         ┆ x   ┆ y    │      │ dt         ┆ x    ┆ y    │
    # │ ---        ┆ --- ┆ ---  │      │ ---        ┆ --- ┆ ---  │      │ ---        ┆ ---  ┆ ---  │
    # │ date       ┆ f64 ┆ f64  │      │ date       ┆ f64 ┆ f64  │      │ date       ┆ f64  ┆ f64  │
    # ╞════════════╪═════╪══════╡      ╞════════════╪═════╪══════╡      ╞════════════╪══════╪══════╡
    # │ 2022-09-01 ┆ 3.5 ┆ 10.0 │----->│ 2022-09-01 ┆ 3.5 ┆ 5.0  │----->│ 2022-09-01 ┆ null ┆ null │
    # ├╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌┤      ├╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌┤      ├╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌┼╌╌╌╌╌╌┤
    # │ 2022-09-02 ┆ 4.0 ┆ 2.5  │----->│ 2022-09-02 ┆ 8.0 ┆ 1.5  │----->│ 2022-09-02 ┆ 5.0  ┆ 2.0  │
    # ├╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌┤      ├╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌┤      ├╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌┼╌╌╌╌╌╌┤
    # │ 2022-09-03 ┆ 1.0 ┆ 1.5  │----->│ 2022-09-03 ┆ 1.0 ┆ 12.0 │----->│ 2022-09-03 ┆ 2.0  ┆ 2.5  │
    # └────────────┴─────┴──────┘      └────────────┴─────┴──────┘      └────────────┴──────┴──────┘
    ...

    Align frames by "dt" using "left" alignment, but keep only cols "x" and "y":

    >>> af1, af2, af3 = pl.align_frames(
    ...     df1, df2, df3, on="dt", select=["x", "y"], how="left"
    ... )  # doctest: +IGNORE_RESULT
    #
    # af1                 af2                 af3
    # shape: (3, 3)       shape: (3, 3)       shape: (3, 3)
    # ┌─────┬──────┐      ┌─────┬──────┐      ┌──────┬──────┐
    # │ x   ┆ y    │      │ x   ┆ y    │      │ x    ┆ y    │
    # │ --- ┆ ---  │      │ --- ┆ ---  │      │ ---  ┆ ---  │
    # │ f64 ┆ f64  │      │ f64 ┆ f64  │      │ f64  ┆ f64  │
    # ╞═════╪══════╡      ╞═════╪══════╡      ╞══════╪══════╡
    # │ 3.5 ┆ 10.0 │      │ 3.5 ┆ 5.0  │      │ null ┆ null │
    # ├╌╌╌╌╌┼╌╌╌╌╌╌┤      ├╌╌╌╌╌┼╌╌╌╌╌╌┤      ├╌╌╌╌╌╌┼╌╌╌╌╌╌┤
    # │ 4.0 ┆ 2.5  │      │ 8.0 ┆ 1.5  │      │ 5.0  ┆ 2.0  │
    # ├╌╌╌╌╌┼╌╌╌╌╌╌┤      ├╌╌╌╌╌┼╌╌╌╌╌╌┤      ├╌╌╌╌╌╌┼╌╌╌╌╌╌┤
    # │ 1.0 ┆ 1.5  │      │ 1.0 ┆ 12.0 │      │ 2.0  ┆ 2.5  │
    # └─────┴──────┘      └─────┴──────┘      └──────┴──────┘
    ...

    Now data is aligned, and you can easily calculate the row-wise dot product:

    >>> (af1 * af2 * af3).fill_null(0).select(pl.sum(pl.col("*")).alias("dot"))
    shape: (3, 1)
    ┌───────┐
    │ dot   │
    │ ---   │
    │ f64   │
    ╞═══════╡
    │ 0.0   │
    ├╌╌╌╌╌╌╌┤
    │ 167.5 │
    ├╌╌╌╌╌╌╌┤
    │ 47.0  │
    └───────┘

    """  # noqa: W505
    if not frames:
        return []
    elif len({type(f) for f in frames}) != 1:
        raise TypeError(
            "Input frames must be of a consistent type (all LazyFrame or all DataFrame)"
        )

    on = [on] if (isinstance(on, str) or not isinstance(on, Sequence)) else on
    align_on = [(c.meta.output_name() if isinstance(c, pl.Expr) else c) for c in on]

    # create aligned master frame (this is the most expensive part; afterwards
    # we just subselect out the columns representing the component frames)
    eager = isinstance(frames[0], pl.DataFrame)
    alignment_frame: LazyFrame = (
        reduce(  # type: ignore[attr-defined]
            lambda x, y: x.lazy().join(  # type: ignore[arg-type, return-value]
                y.lazy(), how=how, on=align_on, suffix=str(id(y))
            ),
            frames,
        )
        .sort(by=align_on, descending=descending)
        .collect()
        .lazy()
    )

    # select-out aligned components from the master frame
    aligned_cols = set(alignment_frame.columns)
    aligned_frames = []
    for df in frames:
        sfx = str(id(df))
        df_cols = [
            F.col(f"{c}{sfx}").alias(c) if f"{c}{sfx}" in aligned_cols else F.col(c)
            for c in df.columns
        ]
        f = alignment_frame.select(*df_cols)
        if select is not None:
            f = f.select(select)
        aligned_frames.append(f)

    return cast(
        List[FrameType], F.collect_all(aligned_frames) if eager else aligned_frames
    )
