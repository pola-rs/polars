from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Iterable, overload

from polars import functions as F
from polars.datatypes import Date, Struct, Time
from polars.utils._parse_expr_input import (
    parse_as_expression,
    parse_as_list_of_expressions,
)
from polars.utils._wrap import wrap_expr

with contextlib.suppress(ImportError):  # Module not available when building docs
    import polars.polars as plr


if TYPE_CHECKING:
    import sys

    from polars import Expr, Series
    from polars.type_aliases import IntoExpr, SchemaDict

    if sys.version_info >= (3, 8):
        from typing import Literal
    else:
        from typing_extensions import Literal


def datetime_(
    year: Expr | str | int,
    month: Expr | str | int,
    day: Expr | str | int,
    hour: Expr | str | int | None = None,
    minute: Expr | str | int | None = None,
    second: Expr | str | int | None = None,
    microsecond: Expr | str | int | None = None,
) -> Expr:
    """
    Create a Polars literal expression of type Datetime.

    Parameters
    ----------
    year
        column or literal.
    month
        column or literal, ranging from 1-12.
    day
        column or literal, ranging from 1-31.
    hour
        column or literal, ranging from 0-23.
    minute
        column or literal, ranging from 0-59.
    second
        column or literal, ranging from 0-59.
    microsecond
        column or literal, ranging from 0-999999.

    Returns
    -------
    Expr of type `pl.Datetime`

    """
    year_expr = parse_as_expression(year)._pyexpr
    month_expr = parse_as_expression(month)._pyexpr
    day_expr = parse_as_expression(day)._pyexpr

    if hour is not None:
        hour = parse_as_expression(hour)._pyexpr
    if minute is not None:
        minute = parse_as_expression(minute)._pyexpr
    if second is not None:
        second = parse_as_expression(second)._pyexpr
    if microsecond is not None:
        microsecond = parse_as_expression(microsecond)._pyexpr

    return wrap_expr(
        plr.datetime(
            year_expr,
            month_expr,
            day_expr,
            hour,
            minute,
            second,
            microsecond,
        )
    )


def date_(
    year: Expr | str | int,
    month: Expr | str | int,
    day: Expr | str | int,
) -> Expr:
    """
    Create a Polars literal expression of type Date.

    Parameters
    ----------
    year
        column or literal.
    month
        column or literal, ranging from 1-12.
    day
        column or literal, ranging from 1-31.

    Returns
    -------
    Expr of type pl.Date

    """
    return datetime_(year, month, day).cast(Date).alias("date")


def time_(
    hour: Expr | str | int | None = None,
    minute: Expr | str | int | None = None,
    second: Expr | str | int | None = None,
    microsecond: Expr | str | int | None = None,
) -> Expr:
    """
    Create a Polars literal expression of type Time.

    Parameters
    ----------
    hour
        column or literal, ranging from 0-23.
    minute
        column or literal, ranging from 0-59.
    second
        column or literal, ranging from 0-59.
    microsecond
        column or literal, ranging from 0-999999.

    Returns
    -------
    Expr of type pl.Date

    """
    epoch_start = (1970, 1, 1)
    return (
        datetime_(*epoch_start, hour, minute, second, microsecond)
        .cast(Time)
        .alias("time")
    )


def duration(
    *,
    days: Expr | str | int | None = None,
    seconds: Expr | str | int | None = None,
    nanoseconds: Expr | str | int | None = None,
    microseconds: Expr | str | int | None = None,
    milliseconds: Expr | str | int | None = None,
    minutes: Expr | str | int | None = None,
    hours: Expr | str | int | None = None,
    weeks: Expr | str | int | None = None,
) -> Expr:
    """
    Create polars `Duration` from distinct time components.

    Returns
    -------
    Expr of type `pl.Duration`

    Examples
    --------
    >>> from datetime import datetime
    >>> df = pl.DataFrame(
    ...     {
    ...         "dt": [datetime(2022, 1, 1), datetime(2022, 1, 2)],
    ...         "add": [1, 2],
    ...     }
    ... )
    >>> print(df)
    shape: (2, 2)
    ┌─────────────────────┬─────┐
    │ dt                  ┆ add │
    │ ---                 ┆ --- │
    │ datetime[μs]        ┆ i64 │
    ╞═════════════════════╪═════╡
    │ 2022-01-01 00:00:00 ┆ 1   │
    │ 2022-01-02 00:00:00 ┆ 2   │
    └─────────────────────┴─────┘
    >>> with pl.Config(tbl_width_chars=120):
    ...     df.select(
    ...         (pl.col("dt") + pl.duration(weeks="add")).alias("add_weeks"),
    ...         (pl.col("dt") + pl.duration(days="add")).alias("add_days"),
    ...         (pl.col("dt") + pl.duration(seconds="add")).alias("add_seconds"),
    ...         (pl.col("dt") + pl.duration(milliseconds="add")).alias("add_millis"),
    ...         (pl.col("dt") + pl.duration(hours="add")).alias("add_hours"),
    ...     )
    ...
    shape: (2, 5)
    ┌─────────────────────┬─────────────────────┬─────────────────────┬─────────────────────────┬─────────────────────┐
    │ add_weeks           ┆ add_days            ┆ add_seconds         ┆ add_millis              ┆ add_hours           │
    │ ---                 ┆ ---                 ┆ ---                 ┆ ---                     ┆ ---                 │
    │ datetime[μs]        ┆ datetime[μs]        ┆ datetime[μs]        ┆ datetime[μs]            ┆ datetime[μs]        │
    ╞═════════════════════╪═════════════════════╪═════════════════════╪═════════════════════════╪═════════════════════╡
    │ 2022-01-08 00:00:00 ┆ 2022-01-02 00:00:00 ┆ 2022-01-01 00:00:01 ┆ 2022-01-01 00:00:00.001 ┆ 2022-01-01 01:00:00 │
    │ 2022-01-16 00:00:00 ┆ 2022-01-04 00:00:00 ┆ 2022-01-02 00:00:02 ┆ 2022-01-02 00:00:00.002 ┆ 2022-01-02 02:00:00 │
    └─────────────────────┴─────────────────────┴─────────────────────┴─────────────────────────┴─────────────────────┘

    """  # noqa: W505
    if hours is not None:
        hours = parse_as_expression(hours)._pyexpr
    if minutes is not None:
        minutes = parse_as_expression(minutes)._pyexpr
    if seconds is not None:
        seconds = parse_as_expression(seconds)._pyexpr
    if milliseconds is not None:
        milliseconds = parse_as_expression(milliseconds)._pyexpr
    if microseconds is not None:
        microseconds = parse_as_expression(microseconds)._pyexpr
    if nanoseconds is not None:
        nanoseconds = parse_as_expression(nanoseconds)._pyexpr
    if days is not None:
        days = parse_as_expression(days)._pyexpr
    if weeks is not None:
        weeks = parse_as_expression(weeks)._pyexpr

    return wrap_expr(
        plr.duration(
            days,
            seconds,
            nanoseconds,
            microseconds,
            milliseconds,
            minutes,
            hours,
            weeks,
        )
    )


def concat_list(exprs: IntoExpr | Iterable[IntoExpr], *more_exprs: IntoExpr) -> Expr:
    """
    Horizontally concatenate columns into a single list column.

    Operates in linear time.

    Parameters
    ----------
    exprs
        Columns to concatenate into a single list column. Accepts expression input.
        Strings are parsed as column names, other non-expression inputs are parsed as
        literals.
    *more_exprs
        Additional columns to concatenate into a single list column, specified as
        positional arguments.

    Examples
    --------
    Create lagged columns and collect them into a list. This mimics a rolling window.

    >>> df = pl.DataFrame({"A": [1.0, 2.0, 9.0, 2.0, 13.0]})
    >>> df = df.select([pl.col("A").shift(i).alias(f"A_lag_{i}") for i in range(3)])
    >>> df.select(
    ...     pl.concat_list([f"A_lag_{i}" for i in range(3)][::-1]).alias("A_rolling")
    ... )
    shape: (5, 1)
    ┌───────────────────┐
    │ A_rolling         │
    │ ---               │
    │ list[f64]         │
    ╞═══════════════════╡
    │ [null, null, 1.0] │
    │ [null, 1.0, 2.0]  │
    │ [1.0, 2.0, 9.0]   │
    │ [2.0, 9.0, 2.0]   │
    │ [9.0, 2.0, 13.0]  │
    └───────────────────┘

    """
    exprs = parse_as_list_of_expressions(exprs, *more_exprs)
    return wrap_expr(plr.concat_list(exprs))


@overload
def struct(
    exprs: IntoExpr | Iterable[IntoExpr] = ...,
    *more_exprs: IntoExpr,
    eager: Literal[False] = ...,
    schema: SchemaDict | None = ...,
    **named_exprs: IntoExpr,
) -> Expr:
    ...


@overload
def struct(
    exprs: IntoExpr | Iterable[IntoExpr] = ...,
    *more_exprs: IntoExpr,
    eager: Literal[True],
    schema: SchemaDict | None = ...,
    **named_exprs: IntoExpr,
) -> Series:
    ...


@overload
def struct(
    exprs: IntoExpr | Iterable[IntoExpr] = ...,
    *more_exprs: IntoExpr,
    eager: bool,
    schema: SchemaDict | None = ...,
    **named_exprs: IntoExpr,
) -> Expr | Series:
    ...


def struct(
    exprs: IntoExpr | Iterable[IntoExpr] = None,
    *more_exprs: IntoExpr,
    eager: bool = False,
    schema: SchemaDict | None = None,
    **named_exprs: IntoExpr,
) -> Expr | Series:
    """
    Collect columns into a struct column.

    Parameters
    ----------
    exprs
        Column(s) to collect into a struct column. Accepts expression input. Strings are
        parsed as column names, other non-expression inputs are parsed as literals.
    *more_exprs
        Additional columns to collect into the struct column, specified as positional
        arguments.
    eager
        Evaluate immediately and return a ``Series``. If set to ``False`` (default),
        return an expression instead.
    schema
        Optional schema that explicitly defines the struct field dtypes. If no columns
        or expressions are provided, schema keys are used to define columns.
    **named_exprs
        Additional columns to collect into the struct column, specified as keyword
        arguments. The columns will be renamed to the keyword used.

    Examples
    --------
    Collect all columns of a dataframe into a struct by passing ``pl.all()``.

    >>> df = pl.DataFrame(
    ...     {
    ...         "int": [1, 2],
    ...         "str": ["a", "b"],
    ...         "bool": [True, None],
    ...         "list": [[1, 2], [3]],
    ...     }
    ... )
    >>> df.select(pl.struct(pl.all()).alias("my_struct"))
    shape: (2, 1)
    ┌─────────────────────┐
    │ my_struct           │
    │ ---                 │
    │ struct[4]           │
    ╞═════════════════════╡
    │ {1,"a",true,[1, 2]} │
    │ {2,"b",null,[3]}    │
    └─────────────────────┘

    Collect selected columns into a struct by either passing a list of columns, or by
    specifying each column as a positional argument.

    >>> df.select(pl.struct("int", False).alias("my_struct"))
    shape: (2, 1)
    ┌───────────┐
    │ my_struct │
    │ ---       │
    │ struct[2] │
    ╞═══════════╡
    │ {1,false} │
    │ {2,false} │
    └───────────┘

    Use keyword arguments to easily name each struct field.

    >>> df.select(pl.struct(p="int", q="bool").alias("my_struct")).schema
    {'my_struct': Struct([Field('p', Int64), Field('q', Boolean)])}

    """
    exprs = parse_as_list_of_expressions(exprs, *more_exprs, **named_exprs)  # type: ignore[arg-type]
    expr = wrap_expr(plr.as_struct(exprs))

    if schema:
        if not exprs:
            # no columns or expressions provided; create one from schema keys
            expr = wrap_expr(
                plr.as_struct(parse_as_list_of_expressions(list(schema.keys())))
            )
        expr = expr.cast(Struct(schema), strict=False)

    if eager:
        return F.select(expr).to_series()
    else:
        return expr


def concat_str(
    exprs: IntoExpr | Iterable[IntoExpr],
    *more_exprs: IntoExpr,
    separator: str = "",
) -> Expr:
    """
    Horizontally concatenate columns into a single string column.

    Operates in linear time.

    Parameters
    ----------
    exprs
        Columns to concatenate into a single string column. Accepts expression input.
        Strings are parsed as column names, other non-expression inputs are parsed as
        literals. Non-``Utf8`` columns are cast to ``Utf8``.
    *more_exprs
        Additional columns to concatenate into a single string column, specified as
        positional arguments.
    separator
        String that will be used to separate the values of each column.

    Examples
    --------
    >>> df = pl.DataFrame(
    ...     {
    ...         "a": [1, 2, 3],
    ...         "b": ["dogs", "cats", None],
    ...         "c": ["play", "swim", "walk"],
    ...     }
    ... )
    >>> df.with_columns(
    ...     pl.concat_str(
    ...         [
    ...             pl.col("a") * 2,
    ...             pl.col("b"),
    ...             pl.col("c"),
    ...         ],
    ...         separator=" ",
    ...     ).alias("full_sentence"),
    ... )
    shape: (3, 4)
    ┌─────┬──────┬──────┬───────────────┐
    │ a   ┆ b    ┆ c    ┆ full_sentence │
    │ --- ┆ ---  ┆ ---  ┆ ---           │
    │ i64 ┆ str  ┆ str  ┆ str           │
    ╞═════╪══════╪══════╪═══════════════╡
    │ 1   ┆ dogs ┆ play ┆ 2 dogs play   │
    │ 2   ┆ cats ┆ swim ┆ 4 cats swim   │
    │ 3   ┆ null ┆ walk ┆ null          │
    └─────┴──────┴──────┴───────────────┘

    """
    exprs = parse_as_list_of_expressions(exprs, *more_exprs)
    return wrap_expr(plr.concat_str(exprs, separator))


def format(f_string: str, *args: Expr | str) -> Expr:
    """
    Format expressions as a string.

    Parameters
    ----------
    f_string
        A string that with placeholders.
        For example: "hello_{}" or "{}_world
    args
        Expression(s) that fill the placeholders

    Examples
    --------
    >>> df = pl.DataFrame(
    ...     {
    ...         "a": ["a", "b", "c"],
    ...         "b": [1, 2, 3],
    ...     }
    ... )
    >>> df.select(
    ...     [
    ...         pl.format("foo_{}_bar_{}", pl.col("a"), "b").alias("fmt"),
    ...     ]
    ... )
    shape: (3, 1)
    ┌─────────────┐
    │ fmt         │
    │ ---         │
    │ str         │
    ╞═════════════╡
    │ foo_a_bar_1 │
    │ foo_b_bar_2 │
    │ foo_c_bar_3 │
    └─────────────┘

    """
    if f_string.count("{}") != len(args):
        raise ValueError("number of placeholders should equal the number of arguments")

    exprs = []

    arguments = iter(args)
    for i, s in enumerate(f_string.split("{}")):
        if i > 0:
            e = parse_as_expression(next(arguments))
            exprs.append(e)

        if len(s) > 0:
            exprs.append(F.lit(s))

    return concat_str(exprs, separator="")
