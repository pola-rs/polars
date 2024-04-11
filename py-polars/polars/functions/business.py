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
    from polars.type_aliases import IntoExprColumn


def business_day_count(
    start: date | IntoExprColumn,
    end: date | IntoExprColumn,
    week_mask: Iterable[bool] = (True, True, True, True, True, False, False),
) -> Expr:
    """
    Count the number of business days between `start` and `end` (not including `end`).

    Parameters
    ----------
    start
        Start dates.
    end
        End dates.
    week_mask
        Which days of the week to count. The default is Monday to Friday.
        If you wanted to count only Monday to Thursday, you would pass
        `(True, True, True, True, False, False, False)`.

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

    >>> week_mask = (True, True, True, True, True, True, False)
    >>> df.with_columns(
    ...     total_day_count=(pl.col("end") - pl.col("start")).dt.total_days(),
    ...     business_day_count=pl.business_day_count("start", "end", week_mask),
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
    return wrap_expr(plr.business_day_count(start_pyexpr, end_pyexpr, week_mask))
