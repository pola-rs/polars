from __future__ import annotations

import contextlib
import warnings
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Iterable, Sequence, overload

from polars import internals as pli
from polars.datatypes import Date
from polars.utils._parse_expr_input import expr_to_lit_or_expr
from polars.utils._wrap import wrap_df, wrap_expr, wrap_ldf, wrap_s
from polars.utils.convert import (
    _datetime_to_pl_timestamp,
    _timedelta_to_pl_duration,
    _tzinfo_to_str,
)
from polars.utils.decorators import deprecate_nonkeyword_arguments, deprecated_alias

with contextlib.suppress(ImportError):  # Module not available when building docs
    from polars.polars import concat_df as _concat_df
    from polars.polars import concat_lf as _concat_lf
    from polars.polars import concat_series as _concat_series
    from polars.polars import py_date_range as _py_date_range
    from polars.polars import py_date_range_lazy as _py_date_range_lazy
    from polars.polars import py_diag_concat_df as _diag_concat_df
    from polars.polars import py_diag_concat_lf as _diag_concat_lf
    from polars.polars import py_hor_concat_df as _hor_concat_df


if TYPE_CHECKING:
    import sys
    from datetime import date

    from polars.dataframe import DataFrame
    from polars.expr.expr import Expr
    from polars.lazyframe import LazyFrame
    from polars.series import Series
    from polars.type_aliases import (
        ClosedInterval,
        ConcatMethod,
        PolarsDataType,
        TimeUnit,
    )

    if sys.version_info >= (3, 8):
        from typing import Literal
    else:
        from typing_extensions import Literal


def get_dummies(
    df: DataFrame,
    *,
    columns: str | Sequence[str] | None = None,
    separator: str = "_",
) -> DataFrame:
    """
    Convert categorical variables into dummy/indicator variables.

    .. deprecated:: 0.16.8
        `pl.get_dummies(df)` has been deprecated; use `df.to_dummies()`

    Parameters
    ----------
    df
        DataFrame to convert.
    columns
        Name of the column(s) that should be converted to dummy variables.
        If set to ``None`` (default), convert all columns.
    separator
        Separator/delimiter used when generating column names.

    Examples
    --------
    >>> df = pl.DataFrame(
    ...     {
    ...         "foo": [1, 2],
    ...         "bar": [3, 4],
    ...         "ham": ["a", "b"],
    ...     }
    ... )
    >>> pl.get_dummies(df.to_dummies(), columns=["foo", "bar"])
    shape: (2, 6)
    ┌───────┬───────┬───────┬───────┬───────┬───────┐
    │ foo_1 ┆ foo_2 ┆ bar_3 ┆ bar_4 ┆ ham_a ┆ ham_b │
    │ ---   ┆ ---   ┆ ---   ┆ ---   ┆ ---   ┆ ---   │
    │ u8    ┆ u8    ┆ u8    ┆ u8    ┆ u8    ┆ u8    │
    ╞═══════╪═══════╪═══════╪═══════╪═══════╪═══════╡
    │ 1     ┆ 0     ┆ 1     ┆ 0     ┆ 1     ┆ 0     │
    │ 0     ┆ 1     ┆ 0     ┆ 1     ┆ 0     ┆ 1     │
    └───────┴───────┴───────┴───────┴───────┴───────┘

    """
    warnings.warn(
        "`pl.get_dummies(df)` has been deprecated; use `df.to_dummies()`",
        category=DeprecationWarning,
        stacklevel=2,
    )
    return df.to_dummies(columns=columns, separator=separator)


@overload
def concat(
    items: Iterable[DataFrame],
    rechunk: bool = True,
    how: ConcatMethod = "vertical",
    parallel: bool = True,
) -> DataFrame:
    ...


@overload
def concat(
    items: Iterable[Series],
    rechunk: bool = True,
    how: ConcatMethod = "vertical",
    parallel: bool = True,
) -> Series:
    ...


@overload
def concat(
    items: Iterable[LazyFrame],
    rechunk: bool = True,
    how: ConcatMethod = "vertical",
    parallel: bool = True,
) -> LazyFrame:
    ...


@overload
def concat(
    items: Iterable[Expr],
    rechunk: bool = True,
    how: ConcatMethod = "vertical",
    parallel: bool = True,
) -> Expr:
    ...


@deprecate_nonkeyword_arguments()
def concat(
    items: (
        Iterable[DataFrame] | Iterable[Series] | Iterable[LazyFrame] | Iterable[Expr]
    ),
    rechunk: bool = True,
    how: ConcatMethod = "vertical",
    parallel: bool = True,
) -> DataFrame | Series | LazyFrame | Expr:
    """
    Aggregate multiple Dataframes/Series to a single DataFrame/Series.

    Parameters
    ----------
    items
        DataFrames/Series/LazyFrames to concatenate.
    rechunk
        Make sure that all data is in contiguous memory.
    how : {'vertical', 'diagonal', 'horizontal'}
        Lazy only supports the 'vertical' strategy.

        - Vertical: applies multiple `vstack` operations.
        - Diagonal: finds a union between the column schemas and fills missing column
            values with null.
        - Horizontal: stacks Series horizontally and fills with nulls if the lengths
            don't match.
    parallel
        Only relevant for LazyFrames. This determines if the concatenated
        lazy computations may be executed in parallel.

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
    │ 2   ┆ 4   │
    └─────┴─────┘

    >>> df_h1 = pl.DataFrame(
    ...     {
    ...         "l1": [1, 2],
    ...         "l2": [3, 4],
    ...     }
    ... )
    >>> df_h2 = pl.DataFrame(
    ...     {
    ...         "r1": [5, 6],
    ...         "r2": [7, 8],
    ...         "r3": [9, 10],
    ...     }
    ... )
    >>> pl.concat(
    ...     [
    ...         df_h1,
    ...         df_h2,
    ...     ],
    ...     how="horizontal",
    ... )
    shape: (2, 5)
    ┌─────┬─────┬─────┬─────┬─────┐
    │ l1  ┆ l2  ┆ r1  ┆ r2  ┆ r3  │
    │ --- ┆ --- ┆ --- ┆ --- ┆ --- │
    │ i64 ┆ i64 ┆ i64 ┆ i64 ┆ i64 │
    ╞═════╪═════╪═════╪═════╪═════╡
    │ 1   ┆ 3   ┆ 5   ┆ 7   ┆ 9   │
    │ 2   ┆ 4   ┆ 6   ┆ 8   ┆ 10  │
    └─────┴─────┴─────┴─────┴─────┘

    >>> df_d1 = pl.DataFrame(
    ...     {
    ...         "a": [1],
    ...         "b": [3],
    ...     }
    ... )
    >>> df_d2 = pl.DataFrame(
    ...     {
    ...         "a": [2],
    ...         "d": [4],
    ...     }
    ... )
    >>> pl.concat(
    ...     [
    ...         df_d1,
    ...         df_d2,
    ...     ],
    ...     how="diagonal",
    ... )
    shape: (2, 3)
    ┌─────┬──────┬──────┐
    │ a   ┆ b    ┆ d    │
    │ --- ┆ ---  ┆ ---  │
    │ i64 ┆ i64  ┆ i64  │
    ╞═════╪══════╪══════╡
    │ 1   ┆ 3    ┆ null │
    │ 2   ┆ null ┆ 4    │
    └─────┴──────┴──────┘

    """
    # unpack/standardise (offers simple support for generator input)
    elems = list(items)

    if not len(elems) > 0:
        raise ValueError("cannot concat empty list")

    out: Series | DataFrame | LazyFrame | Expr
    first = elems[0]
    if isinstance(first, pli.DataFrame):
        if how == "vertical":
            out = wrap_df(_concat_df(elems))
        elif how == "diagonal":
            out = wrap_df(_diag_concat_df(elems))
        elif how == "horizontal":
            out = wrap_df(_hor_concat_df(elems))
        else:
            raise ValueError(
                f"how must be one of {{'vertical', 'diagonal', 'horizontal'}}, "
                f"got {how}"
            )
    elif isinstance(first, pli.LazyFrame):
        if how == "vertical":
            return wrap_ldf(_concat_lf(elems, rechunk, parallel))
        if how == "diagonal":
            return wrap_ldf(_diag_concat_lf(elems, rechunk, parallel))
        else:
            raise ValueError(
                "Lazy only allows {{'vertical', 'diagonal'}} concat strategy."
            )
    elif isinstance(first, pli.Series):
        out = wrap_s(_concat_series(elems))
    elif isinstance(first, pli.Expr):
        out = first
        for e in elems[1:]:
            out = out.append(e)  # type: ignore[arg-type]
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
    low: Expr,
    high: date | datetime | Expr | str,
    interval: str | timedelta = ...,
    *,
    lazy: Literal[False] = ...,
    closed: ClosedInterval = ...,
    name: str | None = ...,
    time_unit: TimeUnit | None = ...,
    time_zone: str | None = ...,
) -> Expr:
    ...


@overload
def date_range(
    low: date | datetime | Expr | str,
    high: Expr,
    interval: str | timedelta = ...,
    *,
    lazy: Literal[False] = ...,
    closed: ClosedInterval = ...,
    name: str | None = ...,
    time_unit: TimeUnit | None = ...,
    time_zone: str | None = ...,
) -> Expr:
    ...


@overload
def date_range(
    low: date | datetime | str,
    high: date | datetime | str,
    interval: str | timedelta = ...,
    *,
    lazy: Literal[False] = ...,
    closed: ClosedInterval = ...,
    name: str | None = ...,
    time_unit: TimeUnit | None = ...,
    time_zone: str | None = ...,
) -> Series:
    ...


@overload
def date_range(
    low: date | datetime | Expr | str,
    high: date | datetime | Expr | str,
    interval: str | timedelta = ...,
    *,
    lazy: Literal[True],
    closed: ClosedInterval = ...,
    name: str | None = None,
    time_unit: TimeUnit | None = ...,
    time_zone: str | None = ...,
) -> Expr:
    ...


def date_range(
    low: date | datetime | Expr | str,
    high: date | datetime | Expr | str,
    interval: str | timedelta = "1d",
    *,
    lazy: bool = False,
    closed: ClosedInterval = "both",
    name: str | None = None,
    time_unit: TimeUnit | None = None,
    time_zone: str | None = None,
) -> Series | Expr:
    """
    Create a range of type `Datetime` (or `Date`).

    Parameters
    ----------
    low
        Lower bound of the date range, given as a date, datetime, Expr, or column name.
    high
        Upper bound of the date range, given as a date, datetime, Expr, or column name.
    interval
        Interval periods. It can be a python timedelta object, like
        ``timedelta(days=10)``, or a polars duration string, such as ``3d12h4m25s``
        representing 3 days, 12 hours, 4 minutes, and 25 seconds.
    lazy:
        Return an expression.
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
    If both ``low`` and ``high`` are passed as date types (not datetime), and the
    interval granularity is no finer than 1d, the returned range is also of
    type date. All other permutations return a datetime Series.

    Returns
    -------
    A Series of type `Datetime` or `Date`.

    Examples
    --------
    Using polars duration string to specify the interval:

    >>> from datetime import date
    >>> pl.date_range(date(2022, 1, 1), date(2022, 3, 1), "1mo", name="dtrange")
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
    ... )
    shape: (3,)
    Series: '' [datetime[μs, America/New_York]]
    [
        2022-01-01 00:00:00 EST
        2022-02-01 00:00:00 EST
        2022-03-01 00:00:00 EST
    ]

    """
    if name is None:
        name = ""
    if isinstance(interval, timedelta):
        interval = _timedelta_to_pl_duration(interval)
    elif " " in interval:
        interval = interval.replace(" ", "")

    if isinstance(low, (str, pli.Expr)) or isinstance(high, (str, pli.Expr)) or lazy:
        low = expr_to_lit_or_expr(low, str_to_lit=False)._pyexpr
        high = expr_to_lit_or_expr(high, str_to_lit=False)._pyexpr
        return wrap_expr(
            _py_date_range_lazy(low, high, interval, closed, name, time_zone)
        )

    low, low_is_date = _ensure_datetime(low)
    high, high_is_date = _ensure_datetime(high)

    if low.tzinfo is not None or time_zone is not None:
        if low.tzinfo != high.tzinfo:
            raise ValueError(
                "Cannot mix different timezone aware datetimes."
                f" Got: '{low.tzinfo}' and '{high.tzinfo}'."
            )

        if time_zone is not None and low.tzinfo is not None:
            if _tzinfo_to_str(low.tzinfo) != time_zone:
                raise ValueError(
                    "Given time_zone is different from that of timezone aware datetimes."
                    f" Given: '{time_zone}', got: '{low.tzinfo}'."
                )
        if time_zone is None and low.tzinfo is not None:
            time_zone = _tzinfo_to_str(low.tzinfo)

    tu: TimeUnit
    if time_unit is not None:
        tu = time_unit
    elif "ns" in interval:
        tu = "ns"
    else:
        tu = "us"

    start = _datetime_to_pl_timestamp(low, tu)
    stop = _datetime_to_pl_timestamp(high, tu)
    dt_range = wrap_s(
        _py_date_range(start, stop, interval, closed, name, tu, time_zone)
    )
    if (
        low_is_date
        and high_is_date
        and not _interval_granularity(interval).endswith(("h", "m", "s"))
    ):
        dt_range = dt_range.cast(Date)

    return dt_range


def cut(
    s: Series,
    bins: list[float],
    labels: list[str] | None = None,
    break_point_label: str = "break_point",
    category_label: str = "category",
) -> DataFrame:
    """
    Bin values into discrete values.

    .. deprecated:: 0.16.8
        `pl.cut(series, ...)` has been deprecated; use `series.cut(...)`

    Parameters
    ----------
    s
        Series to bin.
    bins
        Bins to create.
    labels
        Labels to assign to the bins. If given the length of labels must be
        len(bins) + 1.
    break_point_label
        Name given to the breakpoint column.
    category_label
        Name given to the category column.

    Returns
    -------
    DataFrame

    Warnings
    --------
    This functionality is experimental and may change without it being considered a
    breaking change.

    Examples
    --------
    >>> a = pl.Series("a", [v / 10 for v in range(-30, 30, 5)])
    >>> pl.cut(a, bins=[-1, 1])
    shape: (12, 3)
    ┌──────┬─────────────┬──────────────┐
    │ a    ┆ break_point ┆ category     │
    │ ---  ┆ ---         ┆ ---          │
    │ f64  ┆ f64         ┆ cat          │
    ╞══════╪═════════════╪══════════════╡
    │ -3.0 ┆ -1.0        ┆ (-inf, -1.0] │
    │ -2.5 ┆ -1.0        ┆ (-inf, -1.0] │
    │ -2.0 ┆ -1.0        ┆ (-inf, -1.0] │
    │ -1.5 ┆ -1.0        ┆ (-inf, -1.0] │
    │ …    ┆ …           ┆ …            │
    │ 1.0  ┆ 1.0         ┆ (-1.0, 1.0]  │
    │ 1.5  ┆ inf         ┆ (1.0, inf]   │
    │ 2.0  ┆ inf         ┆ (1.0, inf]   │
    │ 2.5  ┆ inf         ┆ (1.0, inf]   │
    └──────┴─────────────┴──────────────┘

    """
    warnings.warn(
        "`pl.cut(series)` has been deprecated; use `series.cut()`",
        category=DeprecationWarning,
        stacklevel=2,
    )
    return s.cut(bins, labels, break_point_label, category_label)


@overload
def align_frames(
    *frames: DataFrame,
    on: str | Expr | Sequence[str] | Sequence[Expr] | Sequence[str | Expr],
    select: str | Expr | Sequence[str | Expr] | None = None,
    descending: bool | Sequence[bool] = False,
) -> list[DataFrame]:
    ...


@overload
def align_frames(
    *frames: LazyFrame,
    on: str | Expr | Sequence[str] | Sequence[Expr] | Sequence[str | Expr],
    select: str | Expr | Sequence[str | Expr] | None = None,
    descending: bool | Sequence[bool] = False,
) -> list[LazyFrame]:
    ...


@deprecated_alias(reverse="descending")
def align_frames(
    *frames: DataFrame | LazyFrame,
    on: str | Expr | Sequence[str] | Sequence[Expr] | Sequence[str | Expr],
    select: str | Expr | Sequence[str | Expr] | None = None,
    descending: bool | Sequence[bool] = False,
) -> list[DataFrame] | list[LazyFrame]:
    r"""
    Align a sequence of frames using the unique values from one or more columns as a key.

    Frames that do not contain the given key values have rows injected (with nulls
    filling the non-key columns), and each resulting frame is sorted by the key.

    The original column order of input frames is not changed unless ``select`` is
    specified (in which case the final column order is determined from that).

    Note that this does not result in a joined frame - you receive the same number
    of frames back that you passed in, but each is now aligned by key and has
    the same number of rows.

    Parameters
    ----------
    frames
        sequence of DataFrames or LazyFrames.
    on
        one or more columns whose unique values will be used to align the frames.
    select
        optional post-alignment column select to constrain and/or order
        the columns returned from the newly aligned frames.
    descending
        sort the alignment column values in descending order; can be a single
        boolean or a list of booleans associated with each column in ``on``.

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

    Align frames by "dt", but keep only cols "x" and "y":

    >>> af1, af2, af3 = pl.align_frames(
    ...     df1, df2, df3, on="dt", select=["x", "y"]
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
        return []  # type: ignore[return-value]
    elif len({type(f) for f in frames}) != 1:
        raise TypeError(
            "Input frames must be of a consistent type (all LazyFrame or all DataFrame)"
        )

    # establish the superset of all "on" column values, sort, and cache
    eager = isinstance(frames[0], pli.DataFrame)
    alignment_frame = (
        concat([df.lazy().select(on) for df in frames])
        .unique(maintain_order=False)
        .sort(by=on, descending=descending)
    )
    alignment_frame = (
        alignment_frame.collect().lazy() if eager else alignment_frame.cache()
    )
    # finally, align all frames
    aligned_frames = [
        alignment_frame.join(
            other=df.lazy(),
            on=alignment_frame.columns,
            how="left",
        ).select(df.columns)
        for df in frames
    ]
    if select is not None:
        aligned_frames = [df.select(select) for df in aligned_frames]

    return [df.collect() for df in aligned_frames] if eager else aligned_frames


def ones(n: int, dtype: PolarsDataType | None = None) -> Series:
    """
    Return a new Series of given length and type, filled with ones.

    Parameters
    ----------
    n
        Number of elements in the ``Series``
    dtype
        DataType of the elements, defaults to ``polars.Float64``

    Notes
    -----
    In the lazy API you should probably not use this, but use ``lit(1)``
    instead.

    Examples
    --------
    >>> pl.ones(5, pl.Int64)
    shape: (5,)
    Series: '' [i64]
    [
        1
        1
        1
        1
        1
    ]

    """
    s = pli.Series([1.0])
    if dtype:
        s = s.cast(dtype)
    return s.new_from_index(0, n)


def zeros(n: int, dtype: PolarsDataType | None = None) -> Series:
    """
    Return a new Series of given length and type, filled with zeros.

    Parameters
    ----------
    n
        Number of elements in the ``Series``
    dtype
        DataType of the elements, defaults to ``polars.Float64``

    Notes
    -----
    In the lazy API you should probably not use this, but use ``lit(0)``
    instead.

    Examples
    --------
    >>> pl.zeros(5, pl.Int64)
    shape: (5,)
    Series: '' [i64]
    [
        0
        0
        0
        0
        0
    ]

    """
    s = pli.Series([0.0])
    if dtype:
        s = s.cast(dtype)
    return s.new_from_index(0, n)
