from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Optional, Sequence, overload

from polars import internals as pli
from polars.datatypes import Categorical, Date, Float64
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
except ImportError:
    _DOCUMENTING = True


def get_dummies(df: pli.DataFrame) -> pli.DataFrame:
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
    items: Sequence[pli.DataFrame],
    rechunk: bool = True,
    how: str = "vertical",
) -> pli.DataFrame:
    ...


@overload
def concat(
    items: Sequence[pli.Series],
    rechunk: bool = True,
    how: str = "vertical",
) -> pli.Series:
    ...


@overload
def concat(
    items: Sequence[pli.LazyFrame],
    rechunk: bool = True,
    how: str = "vertical",
) -> pli.LazyFrame:
    ...


@overload
def concat(
    items: Sequence[pli.Expr],
    rechunk: bool = True,
    how: str = "vertical",
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
    how: str = "vertical",
) -> pli.DataFrame | pli.Series | pli.LazyFrame | pli.Expr:
    """
    Aggregate multiple Dataframes/Series to a single DataFrame/Series.

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
        - Diagonal: Finds a union between the column schemas and fills missing column
            values with null.
        - Horizontal: Stacks Series horizontally and fills with nulls if the lengths
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
                f"how should be one of {'vertical', 'diagonal'}, got {how}"
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
    closed: str | None = "both",
    name: str | None = None,
    time_unit: str | None = None,
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
    closed : {None, 'left', 'right', 'both', 'none'}
        Make the interval closed to the 'left', 'right', 'none' or 'both' sides.
    name
        Name of the output Series.
    time_unit : {'ns', 'us', 'ms'}
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


def cut(
    s: pli.Series,
    bins: list[float],
    labels: Optional[list[str]] = None,
    break_point_label: str = "break_point",
    category_label: str = "category",
) -> pli.DataFrame:
    """
    Bin values into discrete values.

    .. warning::
        This function is experimental and might change without it being considered a
        breaking change.

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
