from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import TYPE_CHECKING, Sequence, overload

from polars import internals as pli
from polars.datatypes import Categorical, Date, Float64
from polars.utils import _datetime_to_pl_timestamp, _timedelta_to_pl_duration

try:
    from polars.polars import concat_df as _concat_df
    from polars.polars import concat_lf as _concat_lf
    from polars.polars import concat_series as _concat_series
    from polars.polars import py_date_range as _py_date_range
    from polars.polars import py_diag_concat_df as _diag_concat_df
    from polars.polars import py_hor_concat_df as _hor_concat_df

    _DOCUMENTING = False
except ImportError:
    _DOCUMENTING = True

if TYPE_CHECKING:
    from polars.internals.type_aliases import ClosedWindow, ConcatMethod, TimeUnit


def get_dummies(
    df: pli.DataFrame, *, columns: list[str] | None = None
) -> pli.DataFrame:
    """
    Convert categorical variables into dummy/indicator variables.

    Parameters
    ----------
    df
        DataFrame to convert.
    columns
        A subset of columns to convert to dummy variables. ``None`` means
        "all columns".

    """
    return df.to_dummies(columns=columns)


@overload
def concat(
    items: Sequence[pli.DataFrame],
    rechunk: bool = True,
    how: ConcatMethod = "vertical",
) -> pli.DataFrame:
    ...


@overload
def concat(
    items: Sequence[pli.Series],
    rechunk: bool = True,
    how: ConcatMethod = "vertical",
) -> pli.Series:
    ...


@overload
def concat(
    items: Sequence[pli.LazyFrame],
    rechunk: bool = True,
    how: ConcatMethod = "vertical",
) -> pli.LazyFrame:
    ...


@overload
def concat(
    items: Sequence[pli.Expr],
    rechunk: bool = True,
    how: ConcatMethod = "vertical",
) -> pli.Expr:
    ...


def concat(
    items: (
        Sequence[pli.DataFrame]
        | Sequence[pli.Series]
        | Sequence[pli.LazyFrame]
        | Sequence[pli.Expr]
    ),
    rechunk: bool = True,
    how: ConcatMethod = "vertical",
) -> pli.DataFrame | pli.Series | pli.LazyFrame | pli.Expr:
    """
    Aggregate multiple Dataframes/Series to a single DataFrame/Series.

    Parameters
    ----------
    items
        DataFrames/Series/LazyFrames to concatenate.
    rechunk
        Make sure that all data is in contiguous memory.
    how : {'vertical', 'diagonal', 'horizontal'}
        Only used if the items are DataFrames.

        - Vertical: applies multiple `vstack` operations.
        - Diagonal: finds a union between the column schemas and fills missing column
            values with null.
        - Horizontal: stacks Series horizontally and fills with nulls if the lengths
            don't match.

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

    out: pli.Series | pli.DataFrame | pli.LazyFrame | pli.Expr
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
                f"how must be one of {{'vertical', 'diagonal'}}, got {how}"
            )
    elif isinstance(first, pli.LazyFrame):
        return pli.wrap_ldf(_concat_lf(items, rechunk))
    elif isinstance(first, pli.Series):
        out = pli.wrap_s(_concat_series(items))
    elif isinstance(first, pli.Expr):
        out = first
        for e in items[1:]:
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


def date_range(
    low: date | datetime,
    high: date | datetime,
    interval: str | timedelta,
    closed: ClosedWindow = "both",
    name: str | None = None,
    time_unit: TimeUnit | None = None,
) -> pli.Series:
    """
    Create a range of type `Datetime` (or `Date`).

    Parameters
    ----------
    low
        Lower bound of the date range.
    high
        Upper bound of the date range.
    interval
        Interval periods. It can be a python timedelta object, like
        ``timedelta(days=10)``, or a polars duration string, such as ``3d12h4m25s``
        representing 3 days, 12 hours, 4 minutes, and 25 seconds.
    closed : {'both', 'left', 'right', 'none'}
        Define whether the temporal window interval is closed or not.
    name
        Name of the output Series.
    time_unit : {None, 'ns', 'us', 'ms'}
        Set the time unit.

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
    >>> pl.date_range(date(2022, 1, 1), date(2022, 3, 1), "1mo", name="drange")
    shape: (3,)
    Series: 'drange' [date]
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

    """
    if isinstance(interval, timedelta):
        interval = _timedelta_to_pl_duration(interval)
    elif " " in interval:
        interval = interval.replace(" ", "")

    low, low_is_date = _ensure_datetime(low)
    high, high_is_date = _ensure_datetime(high)

    tu: TimeUnit
    if time_unit is not None:
        tu = time_unit
    elif "ns" in interval:
        tu = "ns"
    else:
        tu = "us"

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


def cut(
    s: pli.Series,
    bins: list[float],
    labels: list[str] | None = None,
    break_point_label: str = "break_point",
    category_label: str = "category",
) -> pli.DataFrame:
    """
    Bin values into discrete values.

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
    ├╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ -2.5 ┆ -1.0        ┆ (-inf, -1.0] │
    ├╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ -2.0 ┆ -1.0        ┆ (-inf, -1.0] │
    ├╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ -1.5 ┆ -1.0        ┆ (-inf, -1.0] │
    ├╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ ...  ┆ ...         ┆ ...          │
    ├╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ 1.0  ┆ 1.0         ┆ (-1.0, 1.0]  │
    ├╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ 1.5  ┆ inf         ┆ (1.0, inf]   │
    ├╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ 2.0  ┆ inf         ┆ (1.0, inf]   │
    ├╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ 2.5  ┆ inf         ┆ (1.0, inf]   │
    └──────┴─────────────┴──────────────┘

    """
    var_nm = s.name

    cuts_df = pli.DataFrame(
        [
            pli.Series(
                name=break_point_label, values=bins, dtype=Float64
            ).extend_constant(float("inf"), 1)
        ]
    )

    if labels:
        if len(labels) != len(bins) + 1:
            raise ValueError("expected more labels")
        cuts_df = cuts_df.with_column(pli.Series(name=category_label, values=labels))
    else:
        cuts_df = cuts_df.with_column(
            pli.format(
                "({}, {}]",
                pli.col(break_point_label).shift_and_fill(1, float("-inf")),
                pli.col(break_point_label),
            ).alias(category_label)
        )

    cuts_df = cuts_df.with_column(pli.col(category_label).cast(Categorical))

    result = (
        s.sort()
        .to_frame()
        .join_asof(
            cuts_df,
            left_on=var_nm,
            right_on=break_point_label,
            strategy="forward",
        )
    )
    return result


@overload
def align_frames(
    *frames: pli.DataFrame,
    on: str | pli.Expr | Sequence[str] | Sequence[pli.Expr] | Sequence[str | pli.Expr],
    select: str | pli.Expr | Sequence[str | pli.Expr] | None = None,
) -> list[pli.DataFrame]:
    ...


@overload
def align_frames(
    *frames: pli.LazyFrame,
    on: str | pli.Expr | Sequence[str] | Sequence[pli.Expr] | Sequence[str | pli.Expr],
    select: str | pli.Expr | Sequence[str | pli.Expr] | None = None,
) -> list[pli.LazyFrame]:
    ...


def align_frames(
    *frames: pli.DataFrame | pli.LazyFrame,
    on: str | pli.Expr | Sequence[str] | Sequence[pli.Expr] | Sequence[str | pli.Expr],
    select: str | pli.Expr | Sequence[str | pli.Expr] | None = None,
) -> list[pli.DataFrame] | list[pli.LazyFrame]:
    """
    Align a sequence of frames using the values from one or more columns as a key.

    Frames that do not contain the given key values have rows injected (with nulls
    filling the non-key columns), and each resulting frame is sorted by the key.

    The original column order of input frames is not changed unless `select` is
    specified (in which case the final column order is determined from that).

    Note that this does not result in a joined frame - you receive the same number
    of frames back that you passed in, but each is now aligned by key and has
    the same number of rows.

    Parameters
    ----------
    frames
        sequence of DataFrames or LazyFrames.
    on
        one or more columns whose values will be used to align the frames.
    select
        optional post-alignment column select to constrain and/or order
        the columns returned from the newly aligned frames.

    Examples
    --------
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
    ... )

    # df1                              df2                              df3
    # shape: (3, 3)                    shape: (3, 3)                    shape: (2, 3)
    # ┌────────────┬─────┬──────┐      ┌────────────┬─────┬──────┐      ┌────────────┬─────┬─────┐
    # │ dt         ┆ x   ┆ y    │      │ dt         ┆ x   ┆ y    │      │ dt         ┆ x   ┆ y   │
    # │ ---        ┆ --- ┆ ---  │      │ ---        ┆ --- ┆ ---  │      │ ---        ┆ --- ┆ --- │
    # │ date       ┆ f64 ┆ f64  │      │ date       ┆ f64 ┆ f64  │      │ date       ┆ f64 ┆ f64 │
    # ╞════════════╪═════╪══════╡      ╞════════════╪═════╪══════╡      ╞════════════╪═════╪═════╡
    # │ 2022-09-01 ┆ 3.5 ┆ 10.0 │      │ 2022-09-02 ┆ 8.0 ┆ 1.5  │      │ 2022-09-03 ┆ 2.0 ┆ 2.5 │
    # ├╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌┤      ├╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌┤      ├╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌┤
    # │ 2022-09-02 ┆ 4.0 ┆ 2.5  │      │ 2022-09-03 ┆ 1.0 ┆ 12.0 │      │ 2022-09-02 ┆ 5.0 ┆ 2.0 │
    # ├╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌┤      ├╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌┤      └────────────┴─────┴─────┘
    # │ 2022-09-03 ┆ 1.0 ┆ 1.5  │      │ 2022-09-01 ┆ 3.5 ┆ 5.0  │
    # └────────────┴─────┴──────┘      └────────────┴─────┴──────┘

    >>> # align frames on the values in "dt", but keep only cols "x" and "y":
    >>> af1, af2, af3 = pl.align_frames(df1, df2, df3, on="dt", select=["x", "y"])

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

    >>> # now frames are aligned, can easily calculate the row-wise dot product:
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

    >>> # as above, but keeping the alignment column ("dt") in the final frame:
    >>> af1, af2, af3 = pl.align_frames(df1, df2, df3, on="dt")

    # af1                              af2                              af3
    # shape: (3, 3)                    shape: (3, 3)                    shape: (3, 3)
    # ┌────────────┬─────┬──────┐      ┌────────────┬─────┬──────┐      ┌────────────┬──────┬──────┐
    # │ dt         ┆ x   ┆ y    │      │ dt         ┆ x   ┆ y    │      │ dt         ┆ x    ┆ y    │
    # │ ---        ┆ --- ┆ ---  │      │ ---        ┆ --- ┆ ---  │      │ ---        ┆ ---  ┆ ---  │
    # │ date       ┆ f64 ┆ f64  │      │ date       ┆ f64 ┆ f64  │      │ date       ┆ f64  ┆ f64  │
    # ╞════════════╪═════╪══════╡      ╞════════════╪═════╪══════╡      ╞════════════╪══════╪══════╡
    # │ 2022-09-01 ┆ 3.5 ┆ 10.0 │      │ 2022-09-01 ┆ 3.5 ┆ 5.0  │      │ 2022-09-01 ┆ null ┆ null │
    # ├╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌┤      ├╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌┤      ├╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌┼╌╌╌╌╌╌┤
    # │ 2022-09-02 ┆ 4.0 ┆ 2.5  │      │ 2022-09-02 ┆ 8.0 ┆ 1.5  │      │ 2022-09-02 ┆ 5.0  ┆ 2.0  │
    # ├╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌┤      ├╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌┤      ├╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌┼╌╌╌╌╌╌┤
    # │ 2022-09-03 ┆ 1.0 ┆ 1.5  │      │ 2022-09-03 ┆ 1.0 ┆ 12.0 │      │ 2022-09-03 ┆ 2.0  ┆ 2.5  │
    # └────────────┴─────┴──────┘      └────────────┴─────┴──────┘      └────────────┴──────┴──────┘

    >>> (af1[["x", "y"]] * af2[["x", "y"]] * af3[["x", "y"]]).fill_null(0).select(
    ...     pl.sum(pl.col("*")).alias("dot")
    ... ).insert_at_idx(0, af1["dt"])
    shape: (3, 2)
    ┌────────────┬───────┐
    │ dt         ┆ dot   │
    │ ---        ┆ ---   │
    │ date       ┆ f64   │
    ╞════════════╪═══════╡
    │ 2022-09-01 ┆ 0.0   │
    ├╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┤
    │ 2022-09-02 ┆ 167.5 │
    ├╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┤
    │ 2022-09-03 ┆ 47.0  │
    └────────────┴───────┘

    """  # noqa: E501
    if not frames:
        return []  # type: ignore[return-value]
    elif len({type(f) for f in frames}) != 1:
        raise TypeError(
            "Input frames must be of a consistent type (all LazyFrame or all DataFrame)"
        )

    eager = isinstance(frames[0], pli.DataFrame)
    alignment_frame = concat([df.lazy().select(on) for df in frames]).unique(
        maintain_order=False
    )
    if eager:  # collect once, outside the alignment joins
        alignment_frame = alignment_frame.collect().lazy()

    aligned_frames = [
        alignment_frame.join(
            other=df.lazy(),
            on=alignment_frame.columns,
            how="left",
        )
        .select(df.columns)
        .sort(by=on)
        for df in frames
    ]
    if select is not None:
        aligned_frames = [df.select(select) for df in aligned_frames]

    return [df.collect() for df in aligned_frames] if eager else aligned_frames
