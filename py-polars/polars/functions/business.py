from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Iterable

from polars._utils.parse_expr_input import parse_as_expression
from polars._utils.wrap import wrap_expr

with contextlib.suppress(ImportError):  # Module not available when building docs
    import polars.polars as plr

if TYPE_CHECKING:
    from datetime import date

    from polars import Expr
    from polars.type_aliases import DayOfWeek, IntoExprColumn

DAY_NAMES = (
    "Mon",
    "Tue",
    "Wed",
    "Thu",
    "Fri",
    "Sat",
    "Sun",
)


def _make_week_mask(
    weekend: Iterable[str] | None,
) -> tuple[bool, ...]:
    if weekend is None:
        return tuple([True] * 7)
    if isinstance(weekend, str):
        weekend_set = {weekend}
    else:
        weekend_set = set(weekend)
    for day in weekend_set:
        if day not in DAY_NAMES:
            msg = f"Expected one of {DAY_NAMES}, got: {day}"
            raise ValueError(msg)
    return tuple(
        [
            False if v in weekend else True  # noqa: SIM211
            for v in DAY_NAMES
        ]
    )


def business_day_count(
    start: date | IntoExprColumn,
    end: date | IntoExprColumn,
    weekend: DayOfWeek | Iterable[DayOfWeek] | None = ("Sat", "Sun"),
) -> Expr:
    """
    Count the number of business days between `start` and `end` (not including `end`).

    Parameters
    ----------
    start
        Start dates.
    end
        End dates.
    weekend
        Which days of the week to exclude. The default is `('Sat', 'Sun')`, but you
        can also pass, for example, `weekend=('Fri', 'Sat')`, `weekend='Sun'`,
        or `weekend=None`.

        Allowed values in the tuple are 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat',
        and 'Sun'.

    Returns
    -------
    Expr

    Examples
    --------
    >>> from datetime import date
    >>> df = pl.DataFrame(
    ...     {
    ...         "start": [date(2020, 1, 1), date(2020, 1, 2)],
    ...         "end": [date(2020, 1, 2), date(2020, 1, 10)],
    ...     }
    ... )
    >>> df.with_columns(
    ...     total_day_count=(pl.col("end") - pl.col("start")).dt.total_days(),
    ...     business_day_count=pl.business_day_count("start", "end"),
    ... )
    shape: (2, 4)
    ┌────────────┬────────────┬─────────────────┬────────────────────┐
    │ start      ┆ end        ┆ total_day_count ┆ business_day_count │
    │ ---        ┆ ---        ┆ ---             ┆ ---                │
    │ date       ┆ date       ┆ i64             ┆ i32                │
    ╞════════════╪════════════╪═════════════════╪════════════════════╡
    │ 2020-01-01 ┆ 2020-01-02 ┆ 1               ┆ 1                  │
    │ 2020-01-02 ┆ 2020-01-10 ┆ 8               ┆ 6                  │
    └────────────┴────────────┴─────────────────┴────────────────────┘

    Note how the two "count" columns differ due to the weekend (2020-01-04 - 2020-01-05)
    not being counted by `business_day_count`.

    You can pass a custom weekend - for example, if you only take Sunday off:

    >>> df.with_columns(
    ...     total_day_count=(pl.col("end") - pl.col("start")).dt.total_days(),
    ...     business_day_count=pl.business_day_count("start", "end", weekend="Sun"),
    ... )
    shape: (2, 4)
    ┌────────────┬────────────┬─────────────────┬────────────────────┐
    │ start      ┆ end        ┆ total_day_count ┆ business_day_count │
    │ ---        ┆ ---        ┆ ---             ┆ ---                │
    │ date       ┆ date       ┆ i64             ┆ i32                │
    ╞════════════╪════════════╪═════════════════╪════════════════════╡
    │ 2020-01-01 ┆ 2020-01-02 ┆ 1               ┆ 1                  │
    │ 2020-01-02 ┆ 2020-01-10 ┆ 8               ┆ 7                  │
    └────────────┴────────────┴─────────────────┴────────────────────┘
    """
    start_pyexpr = parse_as_expression(start)
    end_pyexpr = parse_as_expression(end)
    week_mask = _make_week_mask(weekend)
    return wrap_expr(plr.business_day_count(start_pyexpr, end_pyexpr, week_mask))
