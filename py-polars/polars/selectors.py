from __future__ import annotations

import re
from datetime import timezone
from typing import TYPE_CHECKING, Any, Collection, Mapping, TypeVar

from polars import functions as F
from polars.datatypes import (
    FLOAT_DTYPES,
    INTEGER_DTYPES,
    NUMERIC_DTYPES,
    SIGNED_INTEGER_DTYPES,
    TEMPORAL_DTYPES,
    UNSIGNED_INTEGER_DTYPES,
    Categorical,
    Datetime,
    Duration,
    Utf8,
    is_polars_dtype,
)
from polars.expr import Expr
from polars.utils.deprecation import deprecate_function

if TYPE_CHECKING:
    import sys

    from polars import DataFrame, LazyFrame
    from polars.datatypes import PolarsDataType
    from polars.type_aliases import TimeUnit

    if sys.version_info >= (3, 11):
        from typing import Self
    else:
        from typing_extensions import Self


def is_selector(obj: Any) -> bool:
    """
    Indicate whether the given object/expression is a selector.

    Examples
    --------
    >>> from polars.selectors import is_selector
    >>> import polars.selectors as cs
    >>> is_selector(pl.col("colx"))
    False
    >>> is_selector(cs.first() | cs.last())
    True
    """
    # note: don't want to expose the "_selector_proxy_" object
    return isinstance(obj, _selector_proxy_)


def expand_selector(
    target: DataFrame | LazyFrame | Mapping[str, PolarsDataType], selector: SelectorType
) -> tuple[str, ...]:
    """
    Expand a selector to column names with respect to a specific frame or schema target.

    Parameters
    ----------
    target
        A polars DataFrame, LazyFrame or schema.
    selector
        An arbitrary polars selector (or compound selector).

    Examples
    --------
    >>> import polars.selectors as cs
    >>> df = pl.DataFrame(
    ...     {
    ...         "colx": ["a", "b", "c"],
    ...         "coly": [123, 456, 789],
    ...         "colz": [2.0, 5.5, 8.0],
    ...     }
    ... )

    Expand selector with respect to an existing `DataFrame`:

    >>> cs.expand_selector(df, cs.numeric())
    ('coly', 'colz')
    >>> cs.expand_selector(df, cs.first() | cs.last())
    ('colx', 'colz')

    This also works with `LazyFrame`:

    >>> cs.expand_selector(df.lazy(), ~(cs.first() | cs.last()))
    ('coly',)

    Expand selector with respect to a standalone schema:

    >>> schema = {
    ...     "colx": pl.Float32,
    ...     "coly": pl.Float64,
    ...     "colz": pl.Date,
    ... }
    >>> cs.expand_selector(schema, cs.float())
    ('colx', 'coly')

    """
    if isinstance(target, Mapping):
        from polars.dataframe import DataFrame

        target = DataFrame(schema=target)

    return tuple(target.select(selector).columns)


@deprecate_function(
    message="This function has been superseded by `expand_selector`; please update accordingly",
    version="0.18.14",
)
def selector_column_names(
    frame: DataFrame | LazyFrame, selector: SelectorType
) -> tuple[str, ...]:
    """
    Return the column names that would be selected from the given frame.

    .. deprecated:: 0.18.14
       Use :func:`expand_selector` instead.

    Parameters
    ----------
    frame
        A polars DataFrame or LazyFrame.
    selector
        An arbitrary polars selector (or compound selector).

    """
    return expand_selector(target=frame, selector=selector)


def _expand_selectors(
    frame: DataFrame | LazyFrame, items: Any, *more_items: Any
) -> list[Any]:
    """
    Internal function that expands any selectors to column names in the given input.

    Non-selector values are left as-is.

    Examples
    --------
    >>> from polars.selectors import _expand_selectors
    >>> import polars.selectors as cs
    >>> df = pl.DataFrame(
    ...     {
    ...         "colw": ["a", "b"],
    ...         "colx": ["x", "y"],
    ...         "coly": [123, 456],
    ...         "colz": [2.0, 5.5],
    ...     }
    ... )
    >>> _expand_selectors(df, ["colx", cs.numeric()])
    ['colx', 'coly', 'colz']
    >>> _expand_selectors(df, cs.string(), cs.float())
    ['colw', 'colx', 'colz']

    """
    expanded: list[Any] = []
    for item in (
        *(
            items
            if isinstance(items, Collection) and not isinstance(items, str)
            else [items]
        ),
        *more_items,
    ):
        if is_selector(item):
            selector_cols = expand_selector(frame, item)
            expanded.extend(selector_cols)
        else:
            expanded.append(item)
    return expanded


class _selector_proxy_(Expr):
    """Base column selector expression/proxy."""

    _attrs: dict[str, Any]
    _repr_override: str

    def __init__(
        self,
        expr: Expr,
        name: str,
        parameters: dict[str, Any] | None = None,
    ):
        self._pyexpr = expr._pyexpr
        self._attrs = {
            "params": parameters,
            "name": name,
        }

    def __invert__(self) -> Self:
        """Invert the selector."""
        if hasattr(self, "_attrs"):
            inverted = all() - self
            inverted._repr_override = f"~{self!r}"  # type: ignore[attr-defined]
        else:
            inverted = ~self.as_expr()
        return inverted  # type: ignore[return-value]

    def __repr__(self) -> str:
        if not hasattr(self, "_attrs"):
            return re.sub(
                r"<[\w.]+_selector_proxy_[\w ]+>", "<selector>", super().__repr__()
            )
        elif hasattr(self, "_repr_override"):
            return self._repr_override
        else:
            selector_name, params = self._attrs["name"], self._attrs["params"]
            set_ops = {"and": "&", "or": "|", "sub": "-"}
            if selector_name in set_ops:
                op = set_ops[selector_name]
                return "(%s)" % f" {op} ".join(repr(p) for p in params.values())
            else:
                str_params = ",".join(
                    (repr(v)[1:-1] if k.startswith("*") else f"{k}={v!r}")
                    for k, v in (params or {}).items()
                )
                return f"cs.{selector_name}({str_params})"

    def __sub__(self, other: Any) -> SelectorType | Expr:  # type: ignore[override]
        if isinstance(other, _selector_proxy_) and hasattr(other, "_attrs"):
            return _selector_proxy_(
                self.meta._as_selector().meta._selector_sub(other),
                parameters={"self": self, "other": other},
                name="sub",
            )
        else:
            return self.as_expr().__sub__(other)

    def __and__(self, other: Any) -> SelectorType | Expr:  # type: ignore[override]
        if isinstance(other, _selector_proxy_) and hasattr(other, "_attrs"):
            return _selector_proxy_(
                self.meta._as_selector().meta._selector_and(other),
                parameters={"self": self, "other": other},
                name="and",
            )
        else:
            return self.as_expr().__and__(other)

    def __or__(self, other: Any) -> SelectorType | Expr:  # type: ignore[override]
        if isinstance(other, _selector_proxy_) and hasattr(other, "_attrs"):
            return _selector_proxy_(
                self.meta._as_selector().meta._selector_add(other),
                parameters={"self": self, "other": other},
                name="or",
            )
        else:
            return self.as_expr().__or__(other)

    def __rand__(self, other: Any) -> SelectorType | Expr:  # type: ignore[override]
        # order of operation doesn't matter
        if isinstance(other, _selector_proxy_) and hasattr(other, "_attrs"):
            return self.__and__(other)
        else:
            return self.as_expr().__rand__(other)

    def __ror__(self, other: Any) -> SelectorType | Expr:  # type: ignore[override]
        # order of operation doesn't matter
        if isinstance(other, _selector_proxy_) and hasattr(other, "_attrs"):
            return self.__or__(other)
        else:
            return self.as_expr().__ror__(other)

    def as_expr(self) -> Expr:
        """
        Materialize the ``selector`` into a normal expression.

        This ensures that the operators ``|``, ``&``, ``~`` and ``-``
        are applied on the data and not on the selector sets.
        """
        return Expr._from_pyexpr(self._pyexpr)


def _re_string(string: str | Collection[str]) -> str:
    """Return escaped regex, potentially representing multiple string fragments."""
    if isinstance(string, str):
        rx = f"{re.escape(string)}"
    else:
        strings: list[str] = []
        for st in string:
            if isinstance(st, Collection) and not isinstance(st, str):  # type: ignore[redundant-expr]
                strings.extend(st)
            else:
                strings.append(st)
        rx = "|".join(re.escape(x) for x in strings)
    return f"({rx})"


SelectorType = TypeVar("SelectorType", Expr, _selector_proxy_)


def all() -> SelectorType:
    """
    Select all columns.

    Examples
    --------
    >>> from datetime import date
    >>> import polars.selectors as cs
    >>> df = pl.DataFrame(
    ...     {
    ...         "dt": [date(1999, 12, 31), date(2024, 1, 1)],
    ...         "value": [1_234_500, 5_000_555],
    ...     },
    ...     schema_overrides={"value": pl.Int32},
    ... )

    Select all columns, casting them to string:

    >>> df.select(cs.all().cast(pl.Utf8))
    shape: (2, 2)
    ┌────────────┬─────────┐
    │ dt         ┆ value   │
    │ ---        ┆ ---     │
    │ str        ┆ str     │
    ╞════════════╪═════════╡
    │ 1999-12-31 ┆ 1234500 │
    │ 2024-01-01 ┆ 5000555 │
    └────────────┴─────────┘

    Select all columns *except* for those matching the given dtypes:

    >>> df.select(cs.all() - cs.numeric())
    shape: (2, 1)
    ┌────────────┐
    │ dt         │
    │ ---        │
    │ date       │
    ╞════════════╡
    │ 1999-12-31 │
    │ 2024-01-01 │
    └────────────┘

    See Also
    --------
    first : Select the first column in the current scope.
    last : Select the last column in the current scope.

    """
    return _selector_proxy_(F.all(), name="all")


def by_dtype(
    *dtypes: PolarsDataType | Collection[PolarsDataType],
) -> SelectorType:
    """
    Select all columns matching the given dtypes.

    Examples
    --------
    >>> from datetime import date
    >>> import polars.selectors as cs
    >>> df = pl.DataFrame(
    ...     {
    ...         "dt": [date(1999, 12, 31), date(2024, 1, 1), date(2010, 7, 5)],
    ...         "value": [1_234_500, 5_000_555, -4_500_000],
    ...         "other": ["foo", "bar", "foo"],
    ...     }
    ... )

    Select all columns with date or integer dtypes:

    >>> df.select(cs.by_dtype(pl.Date, pl.INTEGER_DTYPES))
    shape: (3, 2)
    ┌────────────┬──────────┐
    │ dt         ┆ value    │
    │ ---        ┆ ---      │
    │ date       ┆ i64      │
    ╞════════════╪══════════╡
    │ 1999-12-31 ┆ 1234500  │
    │ 2024-01-01 ┆ 5000555  │
    │ 2010-07-05 ┆ -4500000 │
    └────────────┴──────────┘

    Select all columns that are not of date or integer dtype:

    >>> df.select(~cs.by_dtype(pl.Date, pl.INTEGER_DTYPES))
    shape: (3, 1)
    ┌───────┐
    │ other │
    │ ---   │
    │ str   │
    ╞═══════╡
    │ foo   │
    │ bar   │
    │ foo   │
    └───────┘

    Group by string columns and sum the numeric columns:

    >>> df.groupby(cs.string()).agg(cs.numeric().sum()).sort(by="other")
    shape: (2, 2)
    ┌───────┬──────────┐
    │ other ┆ value    │
    │ ---   ┆ ---      │
    │ str   ┆ i64      │
    ╞═══════╪══════════╡
    │ bar   ┆ 5000555  │
    │ foo   ┆ -3265500 │
    └───────┴──────────┘

    See Also
    --------
    integer : Select all integer columns.
    float : Select all float columns.
    numeric : Select all numeric columns.
    temporal : Select all temporal columns.

    """
    all_dtypes: list[PolarsDataType] = []
    for tp in dtypes:
        if is_polars_dtype(tp):
            all_dtypes.append(tp)  # type: ignore[arg-type]
        elif isinstance(tp, Collection):
            for t in tp:
                if not is_polars_dtype(t):
                    raise TypeError(f"Invalid dtype: {t!r}")
                all_dtypes.append(t)
        else:
            raise TypeError(f"Invalid dtype: {tp!r}")

    return _selector_proxy_(
        F.col(*all_dtypes), name="by_dtype", parameters={"dtypes": all_dtypes}
    )


def by_name(*names: str | Collection[str]) -> SelectorType:
    """
    Select all columns matching the given names.

    Parameters
    ----------
    *names
        One or more names of columns to select.

    Examples
    --------
    >>> import polars.selectors as cs
    >>> df = pl.DataFrame(
    ...     {
    ...         "foo": ["x", "y"],
    ...         "bar": [123, 456],
    ...         "baz": [2.0, 5.5],
    ...         "zap": [False, True],
    ...     }
    ... )

    Select columns by name:

    >>> df.select(cs.by_name("foo", "bar"))
    shape: (2, 2)
    ┌─────┬─────┐
    │ foo ┆ bar │
    │ --- ┆ --- │
    │ str ┆ i64 │
    ╞═════╪═════╡
    │ x   ┆ 123 │
    │ y   ┆ 456 │
    └─────┴─────┘

    Match all columns *except* for those given:

    >>> df.select(~cs.by_name("foo", "bar"))
    shape: (2, 2)
    ┌─────┬───────┐
    │ baz ┆ zap   │
    │ --- ┆ ---   │
    │ f64 ┆ bool  │
    ╞═════╪═══════╡
    │ 2.0 ┆ false │
    │ 5.5 ┆ true  │
    └─────┴───────┘

    See Also
    --------
    by_dtype : Select all columns matching the given dtypes.

    """
    all_names = []
    for nm in names:
        if isinstance(nm, str):
            all_names.append(nm)
        elif isinstance(nm, Collection):
            for n in nm:
                if not isinstance(n, str):
                    raise TypeError(f"Invalid name: {n!r}")
                all_names.append(n)
        else:
            TypeError(f"Invalid name: {nm!r}")

    return _selector_proxy_(
        F.col(*all_names), name="by_name", parameters={"*names": all_names}
    )


def contains(substring: str | Collection[str]) -> SelectorType:
    """
    Select columns that contain the given literal substring(s).

    Parameters
    ----------
    substring
        Substring(s) that matching column names should contain.

    Examples
    --------
    >>> import polars.selectors as cs
    >>> df = pl.DataFrame(
    ...     {
    ...         "foo": ["x", "y"],
    ...         "bar": [123, 456],
    ...         "baz": [2.0, 5.5],
    ...         "zap": [False, True],
    ...     }
    ... )

    Select columns that contain the substring 'ba':

    >>> df.select(cs.contains("ba"))
    shape: (2, 2)
    ┌─────┬─────┐
    │ bar ┆ baz │
    │ --- ┆ --- │
    │ i64 ┆ f64 │
    ╞═════╪═════╡
    │ 123 ┆ 2.0 │
    │ 456 ┆ 5.5 │
    └─────┴─────┘

    Select columns that contain the substring 'ba' or the letter 'z':

    >>> df.select(cs.contains(("ba", "z")))
    shape: (2, 3)
    ┌─────┬─────┬───────┐
    │ bar ┆ baz ┆ zap   │
    │ --- ┆ --- ┆ ---   │
    │ i64 ┆ f64 ┆ bool  │
    ╞═════╪═════╪═══════╡
    │ 123 ┆ 2.0 ┆ false │
    │ 456 ┆ 5.5 ┆ true  │
    └─────┴─────┴───────┘

    Select all columns *except* for those that contain the substring 'ba':

    >>> df.select(~cs.contains("ba"))
    shape: (2, 2)
    ┌─────┬───────┐
    │ foo ┆ zap   │
    │ --- ┆ ---   │
    │ str ┆ bool  │
    ╞═════╪═══════╡
    │ x   ┆ false │
    │ y   ┆ true  │
    └─────┴───────┘

    See Also
    --------
    matches : Select all columns that match the given regex pattern.
    ends_with : Select columns that end with the given substring(s).
    starts_with : Select columns that start with the given substring(s).

    """
    escaped_substring = _re_string(substring)
    raw_params = f"^.*{escaped_substring}.*$"

    return _selector_proxy_(
        F.col(raw_params),
        name="contains",
        parameters={"substring": escaped_substring},
    )


def datetime(
    time_unit: TimeUnit | Collection[TimeUnit] | None = None,
    time_zone: (str | timezone | Collection[str | timezone | None] | None) = (
        "*",
        None,
    ),
) -> SelectorType:
    """
    Select all datetime columns, optionally filtering by time unit/zone.

    Parameters
    ----------
    time_unit
        One (or more) of the allowed timeunit precision strings, "ms", "us", and "ns".
        Omit to select columns with any valid timeunit.
    time_zone
        * One or more timezone strings, as defined in zoneinfo (to see valid options
          run ``import zoneinfo; zoneinfo.available_timezones()`` for a full list).
        * Set ``None`` to select Datetime columns that do not have a timezone.
        * Set "*" to select Datetime columns that have *any* timezone.

    Examples
    --------
    >>> from datetime import datetime, date
    >>> import polars.selectors as cs
    >>> df = pl.DataFrame(
    ...     {
    ...         "tstamp_tokyo": [
    ...             datetime(1999, 7, 20, 20, 20, 16, 987654),
    ...             datetime(2000, 5, 15, 21, 21, 21, 123465),
    ...         ],
    ...         "tstamp_utc": [
    ...             datetime(2023, 4, 10, 12, 14, 16, 999000),
    ...             datetime(2025, 8, 25, 14, 18, 22, 666000),
    ...         ],
    ...         "tstamp": [
    ...             datetime(2000, 11, 20, 18, 12, 16, 600000),
    ...             datetime(2020, 10, 30, 10, 20, 25, 123000),
    ...         ],
    ...         "dt": [date(1999, 12, 31), date(2010, 7, 5)],
    ...     },
    ...     schema_overrides={
    ...         "tstamp_tokyo": pl.Datetime("ns", "Asia/Tokyo"),
    ...         "tstamp_utc": pl.Datetime("us", "UTC"),
    ...     },
    ... )

    Select all datetime columns:

    >>> df.select(cs.datetime())
    shape: (2, 3)
    ┌────────────────────────────────┬─────────────────────────────┬─────────────────────────┐
    │ tstamp_tokyo                   ┆ tstamp_utc                  ┆ tstamp                  │
    │ ---                            ┆ ---                         ┆ ---                     │
    │ datetime[ns, Asia/Tokyo]       ┆ datetime[μs, UTC]           ┆ datetime[μs]            │
    ╞════════════════════════════════╪═════════════════════════════╪═════════════════════════╡
    │ 1999-07-21 05:20:16.987654 JST ┆ 2023-04-10 12:14:16.999 UTC ┆ 2000-11-20 18:12:16.600 │
    │ 2000-05-16 06:21:21.123465 JST ┆ 2025-08-25 14:18:22.666 UTC ┆ 2020-10-30 10:20:25.123 │
    └────────────────────────────────┴─────────────────────────────┴─────────────────────────┘

    Select all datetime columns that have 'us' precision:

    >>> df.select(cs.datetime("us"))
    shape: (2, 2)
    ┌─────────────────────────────┬─────────────────────────┐
    │ tstamp_utc                  ┆ tstamp                  │
    │ ---                         ┆ ---                     │
    │ datetime[μs, UTC]           ┆ datetime[μs]            │
    ╞═════════════════════════════╪═════════════════════════╡
    │ 2023-04-10 12:14:16.999 UTC ┆ 2000-11-20 18:12:16.600 │
    │ 2025-08-25 14:18:22.666 UTC ┆ 2020-10-30 10:20:25.123 │
    └─────────────────────────────┴─────────────────────────┘

    Select all datetime columns that have *any* timezone:

    >>> df.select(cs.datetime(time_zone="*"))
    shape: (2, 2)
    ┌────────────────────────────────┬─────────────────────────────┐
    │ tstamp_tokyo                   ┆ tstamp_utc                  │
    │ ---                            ┆ ---                         │
    │ datetime[ns, Asia/Tokyo]       ┆ datetime[μs, UTC]           │
    ╞════════════════════════════════╪═════════════════════════════╡
    │ 1999-07-21 05:20:16.987654 JST ┆ 2023-04-10 12:14:16.999 UTC │
    │ 2000-05-16 06:21:21.123465 JST ┆ 2025-08-25 14:18:22.666 UTC │
    └────────────────────────────────┴─────────────────────────────┘

    Select all datetime columns that have a *specific* timezone:

    >>> df.select(cs.datetime(time_zone="UTC"))
    shape: (2, 1)
    ┌─────────────────────────────┐
    │ tstamp_utc                  │
    │ ---                         │
    │ datetime[μs, UTC]           │
    ╞═════════════════════════════╡
    │ 2023-04-10 12:14:16.999 UTC │
    │ 2025-08-25 14:18:22.666 UTC │
    └─────────────────────────────┘

    Select all datetime columns that have NO timezone:

    >>> df.select(cs.datetime(time_zone=None))
    shape: (2, 1)
    ┌─────────────────────────┐
    │ tstamp                  │
    │ ---                     │
    │ datetime[μs]            │
    ╞═════════════════════════╡
    │ 2000-11-20 18:12:16.600 │
    │ 2020-10-30 10:20:25.123 │
    └─────────────────────────┘

    Select all columns *except* for datetime columns:

    >>> df.select(~cs.datetime())
    shape: (2, 1)
    ┌────────────┐
    │ dt         │
    │ ---        │
    │ date       │
    ╞════════════╡
    │ 1999-12-31 │
    │ 2010-07-05 │
    └────────────┘

    """  # noqa: W505
    if time_unit is None:
        time_unit = ["ms", "us", "ns"]
    else:
        time_unit = [time_unit] if isinstance(time_unit, str) else list(time_unit)

    if time_zone is None:
        time_zone = [None]
    elif time_zone:
        time_zone = (
            [time_zone] if isinstance(time_zone, (str, timezone)) else list(time_zone)
        )

    datetime_dtypes = []
    for tu in time_unit:
        for tz in time_zone:  # type: ignore[union-attr]
            datetime_dtypes.append(Datetime(tu, tz))

    return _selector_proxy_(
        F.col(datetime_dtypes),
        name="datetime",
        parameters={"time_unit": time_unit, "time_zone": time_zone},
    )


def duration(
    time_unit: TimeUnit | Collection[TimeUnit] | None = None,
) -> SelectorType:
    """
    Select all duration columns, optionally filtering by time unit.

    Parameters
    ----------
    time_unit
        One (or more) of the allowed timeunit precision strings, "ms", "us", and "ns".
        Omit to select columns with any valid timeunit.

    Examples
    --------
    >>> from datetime import date, timedelta
    >>> import polars.selectors as cs
    >>> df = pl.DataFrame(
    ...     {
    ...         "dt": [date(2022, 1, 31), date(2025, 7, 5)],
    ...         "td1": [
    ...             timedelta(days=1, milliseconds=123456),
    ...             timedelta(days=1, hours=23, microseconds=987000),
    ...         ],
    ...         "td2": [
    ...             timedelta(days=7, microseconds=456789),
    ...             timedelta(days=14, minutes=999, seconds=59),
    ...         ],
    ...         "td3": [
    ...             timedelta(weeks=4, days=-10, microseconds=999999),
    ...             timedelta(weeks=3, milliseconds=123456, microseconds=1),
    ...         ],
    ...     },
    ...     schema_overrides={
    ...         "td1": pl.Duration("ms"),
    ...         "td2": pl.Duration("us"),
    ...         "td3": pl.Duration("ns"),
    ...     },
    ... )

    Select all duration columns:

    >>> df.select(cs.duration())
    shape: (2, 3)
    ┌────────────────┬─────────────────┬────────────────────┐
    │ td1            ┆ td2             ┆ td3                │
    │ ---            ┆ ---             ┆ ---                │
    │ duration[ms]   ┆ duration[μs]    ┆ duration[ns]       │
    ╞════════════════╪═════════════════╪════════════════════╡
    │ 1d 2m 3s 456ms ┆ 7d 456789µs     ┆ 18d 999999µs       │
    │ 1d 23h 987ms   ┆ 14d 16h 39m 59s ┆ 21d 2m 3s 456001µs │
    └────────────────┴─────────────────┴────────────────────┘

    Select all duration columns that have 'ms' precision:

    >>> df.select(cs.duration("ms"))
    shape: (2, 1)
    ┌────────────────┐
    │ td1            │
    │ ---            │
    │ duration[ms]   │
    ╞════════════════╡
    │ 1d 2m 3s 456ms │
    │ 1d 23h 987ms   │
    └────────────────┘

    Select all duration columns that have 'ms' OR 'ns' precision:

    >>> df.select(cs.duration(["ms", "ns"]))
    shape: (2, 2)
    ┌────────────────┬────────────────────┐
    │ td1            ┆ td3                │
    │ ---            ┆ ---                │
    │ duration[ms]   ┆ duration[ns]       │
    ╞════════════════╪════════════════════╡
    │ 1d 2m 3s 456ms ┆ 18d 999999µs       │
    │ 1d 23h 987ms   ┆ 21d 2m 3s 456001µs │
    └────────────────┴────────────────────┘

    Select all columns *except* for duration columns:

    >>> df.select(~cs.duration())
    shape: (2, 1)
    ┌────────────┐
    │ dt         │
    │ ---        │
    │ date       │
    ╞════════════╡
    │ 2022-01-31 │
    │ 2025-07-05 │
    └────────────┘

    """
    if time_unit is None:
        time_unit = ["ms", "us", "ns"]
    else:
        time_unit = [time_unit] if isinstance(time_unit, str) else list(time_unit)

    duration_dtypes = [Duration(tu) for tu in time_unit]
    return _selector_proxy_(
        F.col(duration_dtypes),
        name="duration",
        parameters={"time_unit": time_unit},
    )


def ends_with(*suffix: str) -> SelectorType:
    """
    Select columns that end with the given substring(s).

    Parameters
    ----------
    suffix
        Substring(s) that matching column names should end with.

    Examples
    --------
    >>> import polars.selectors as cs
    >>> df = pl.DataFrame(
    ...     {
    ...         "foo": ["x", "y"],
    ...         "bar": [123, 456],
    ...         "baz": [2.0, 5.5],
    ...         "zap": [False, True],
    ...     }
    ... )

    Select columns that end with the substring 'z':

    >>> df.select(cs.ends_with("z"))
    shape: (2, 1)
    ┌─────┐
    │ baz │
    │ --- │
    │ f64 │
    ╞═════╡
    │ 2.0 │
    │ 5.5 │
    └─────┘

    Select columns that end with *either* the letter 'z' or 'r':

    >>> df.select(cs.ends_with("z", "r"))
    shape: (2, 2)
    ┌─────┬─────┐
    │ bar ┆ baz │
    │ --- ┆ --- │
    │ i64 ┆ f64 │
    ╞═════╪═════╡
    │ 123 ┆ 2.0 │
    │ 456 ┆ 5.5 │
    └─────┴─────┘

    Select all columns *except* for those that end with the substring 'z':

    >>> df.select(~cs.ends_with("z"))
    shape: (2, 3)
    ┌─────┬─────┬───────┐
    │ foo ┆ bar ┆ zap   │
    │ --- ┆ --- ┆ ---   │
    │ str ┆ i64 ┆ bool  │
    ╞═════╪═════╪═══════╡
    │ x   ┆ 123 ┆ false │
    │ y   ┆ 456 ┆ true  │
    └─────┴─────┴───────┘

    See Also
    --------
    contains : Select columns that contain the given literal substring(s).
    matches : Select all columns that match the given regex pattern.
    starts_with : Select columns that start with the given substring(s).

    """
    escaped_suffix = _re_string(suffix)
    raw_params = f"^.*{escaped_suffix}$"

    return _selector_proxy_(
        F.col(raw_params),
        name="ends_with",
        parameters={"*suffix": escaped_suffix},
    )


def first() -> SelectorType:
    """
    Select the first column in the current scope.

    Examples
    --------
    >>> import polars.selectors as cs
    >>> df = pl.DataFrame(
    ...     {
    ...         "foo": ["x", "y"],
    ...         "bar": [123, 456],
    ...         "baz": [2.0, 5.5],
    ...         "zap": [0, 1],
    ...     }
    ... )

    Select the first column:

    >>> df.select(cs.first())
    shape: (2, 1)
    ┌─────┐
    │ foo │
    │ --- │
    │ str │
    ╞═════╡
    │ x   │
    │ y   │
    └─────┘

    Select everything  *except* for the first column:

    >>> df.select(~cs.first())
    shape: (2, 3)
    ┌─────┬─────┬─────┐
    │ bar ┆ baz ┆ zap │
    │ --- ┆ --- ┆ --- │
    │ i64 ┆ f64 ┆ i64 │
    ╞═════╪═════╪═════╡
    │ 123 ┆ 2.0 ┆ 0   │
    │ 456 ┆ 5.5 ┆ 1   │
    └─────┴─────┴─────┘

    See Also
    --------
    all : Select all columns.
    last : Select the last column in the current scope.

    """
    return _selector_proxy_(F.first(), name="first")


def float() -> SelectorType:
    """
    Select all float columns.

    Examples
    --------
    >>> import polars.selectors as cs
    >>> df = pl.DataFrame(
    ...     {
    ...         "foo": ["x", "y"],
    ...         "bar": [123, 456],
    ...         "baz": [2.0, 5.5],
    ...         "zap": [0.0, 1.0],
    ...     },
    ...     schema_overrides={"baz": pl.Float32, "zap": pl.Float64},
    ... )

    Select all float columns:

    >>> df.select(cs.float())
    shape: (2, 2)
    ┌─────┬─────┐
    │ baz ┆ zap │
    │ --- ┆ --- │
    │ f32 ┆ f64 │
    ╞═════╪═════╡
    │ 2.0 ┆ 0.0 │
    │ 5.5 ┆ 1.0 │
    └─────┴─────┘

    Select all columns *except* for those that are float:

    >>> df.select(~cs.float())
    shape: (2, 2)
    ┌─────┬─────┐
    │ foo ┆ bar │
    │ --- ┆ --- │
    │ str ┆ i64 │
    ╞═════╪═════╡
    │ x   ┆ 123 │
    │ y   ┆ 456 │
    └─────┴─────┘

    See Also
    --------
    integer : Select all integer columns.
    numeric : Select all numeric columns.
    temporal : Select all temporal columns.
    string : Select all string columns.

    """
    return _selector_proxy_(
        F.col(FLOAT_DTYPES),
        name="float",
    )


def integer() -> SelectorType:
    """
    Select all integer columns.

    Examples
    --------
    >>> import polars.selectors as cs
    >>> df = pl.DataFrame(
    ...     {
    ...         "foo": ["x", "y"],
    ...         "bar": [123, 456],
    ...         "baz": [2.0, 5.5],
    ...         "zap": [0, 1],
    ...     }
    ... )

    Select all integer columns:

    >>> df.select(cs.integer())
    shape: (2, 2)
    ┌─────┬─────┐
    │ bar ┆ zap │
    │ --- ┆ --- │
    │ i64 ┆ i64 │
    ╞═════╪═════╡
    │ 123 ┆ 0   │
    │ 456 ┆ 1   │
    └─────┴─────┘

    Select all columns *except* for those that are integer:

    >>> df.select(~cs.integer())
    shape: (2, 2)
    ┌─────┬─────┐
    │ foo ┆ baz │
    │ --- ┆ --- │
    │ str ┆ f64 │
    ╞═════╪═════╡
    │ x   ┆ 2.0 │
    │ y   ┆ 5.5 │
    └─────┴─────┘

    See Also
    --------
    by_dtype : Select columns by dtype.
    float : Select all float columns.
    numeric : Select all numeric columns.
    temporal : Select all temporal columns.
    string : Select all string columns.

    """
    return _selector_proxy_(
        F.col(INTEGER_DTYPES),
        name="integer",
    )


def signed_integer() -> SelectorType:
    """
    Select all signed integer columns.

    Examples
    --------
    >>> import polars.selectors as cs
    >>> df = pl.DataFrame(
    ...     {
    ...         "foo": [-123, -456],
    ...         "bar": [3456, 6789],
    ...         "baz": [7654, 4321],
    ...         "zap": ["ab", "cd"],
    ...     },
    ...     schema_overrides={"bar": pl.UInt32, "baz": pl.UInt64},
    ... )

    Select all signed integer columns:

    >>> df.select(cs.signed_integer())
    shape: (2, 1)
    ┌──────┐
    │ foo  │
    │ ---  │
    │ i64  │
    ╞══════╡
    │ -123 │
    │ -456 │
    └──────┘

    >>> df.select(~cs.signed_integer())
    shape: (2, 3)
    ┌──────┬──────┬─────┐
    │ bar  ┆ baz  ┆ zap │
    │ ---  ┆ ---  ┆ --- │
    │ u32  ┆ u64  ┆ str │
    ╞══════╪══════╪═════╡
    │ 3456 ┆ 7654 ┆ ab  │
    │ 6789 ┆ 4321 ┆ cd  │
    └──────┴──────┴─────┘

    Select all integer columns (both signed and unsigned):

    >>> df.select(cs.integer())
    shape: (2, 3)
    ┌──────┬──────┬──────┐
    │ foo  ┆ bar  ┆ baz  │
    │ ---  ┆ ---  ┆ ---  │
    │ i64  ┆ u32  ┆ u64  │
    ╞══════╪══════╪══════╡
    │ -123 ┆ 3456 ┆ 7654 │
    │ -456 ┆ 6789 ┆ 4321 │
    └──────┴──────┴──────┘

    See Also
    --------
    by_dtype : Select columns by dtype.
    float : Select all float columns.
    integer: Select all integer columns.
    numeric : Select all numeric columns.
    unsigned_integer: Select all unsigned integer columns.

    """
    return _selector_proxy_(
        F.col(SIGNED_INTEGER_DTYPES),
        name="signed_integer",
    )


def unsigned_integer() -> SelectorType:
    """
    Select all unsigned integer columns.

    Examples
    --------
    >>> import polars.selectors as cs
    >>> df = pl.DataFrame(
    ...     {
    ...         "foo": [-123, -456],
    ...         "bar": [3456, 6789],
    ...         "baz": [7654, 4321],
    ...         "zap": ["ab", "cd"],
    ...     },
    ...     schema_overrides={"bar": pl.UInt32, "baz": pl.UInt64},
    ... )

    Select all unsigned integer columns:

    >>> df.select(cs.unsigned_integer())
    shape: (2, 2)
    ┌──────┬──────┐
    │ bar  ┆ baz  │
    │ ---  ┆ ---  │
    │ u32  ┆ u64  │
    ╞══════╪══════╡
    │ 3456 ┆ 7654 │
    │ 6789 ┆ 4321 │
    └──────┴──────┘

    Select all columns *except* for those that are unsigned integers:

    >>> df.select(~cs.unsigned_integer())
    shape: (2, 2)
    ┌──────┬─────┐
    │ foo  ┆ zap │
    │ ---  ┆ --- │
    │ i64  ┆ str │
    ╞══════╪═════╡
    │ -123 ┆ ab  │
    │ -456 ┆ cd  │
    └──────┴─────┘

    Select all integer columns (both signed and unsigned):

    >>> df.select(cs.integer())
    shape: (2, 3)
    ┌──────┬──────┬──────┐
    │ foo  ┆ bar  ┆ baz  │
    │ ---  ┆ ---  ┆ ---  │
    │ i64  ┆ u32  ┆ u64  │
    ╞══════╪══════╪══════╡
    │ -123 ┆ 3456 ┆ 7654 │
    │ -456 ┆ 6789 ┆ 4321 │
    └──────┴──────┴──────┘

    See Also
    --------
    by_dtype : Select columns by dtype.
    float : Select all float columns.
    integer: Select all integer columns.
    numeric : Select all numeric columns.
    signed_integer: Select all signed integer columns.

    """
    return _selector_proxy_(
        F.col(UNSIGNED_INTEGER_DTYPES),
        name="unsigned_integer",
    )


def last() -> SelectorType:
    """
    Select the last column in the current scope.

    Examples
    --------
    >>> import polars.selectors as cs
    >>> df = pl.DataFrame(
    ...     {
    ...         "foo": ["x", "y"],
    ...         "bar": [123, 456],
    ...         "baz": [2.0, 5.5],
    ...         "zap": [0, 1],
    ...     }
    ... )

    Select the last column:

    >>> df.select(cs.last())
    shape: (2, 1)
    ┌─────┐
    │ zap │
    │ --- │
    │ i64 │
    ╞═════╡
    │ 0   │
    │ 1   │
    └─────┘

    Select everything  *except* for the last column:

    >> df.select(~cs.last())
    shape: (2, 3)
    ┌─────┬─────┬─────┐
    │ foo ┆ bar ┆ baz │
    │ --- ┆ --- ┆ --- │
    │ str ┆ i64 ┆ f64 │
    ╞═════╪═════╪═════╡
    │ x   ┆ 123 ┆ 2.0 │
    │ y   ┆ 456 ┆ 5.5 │
    └─────┴─────┴─────┘

    See Also
    --------
    all : Select all columns.
    first : Select the first column in the current scope.

    """
    return _selector_proxy_(F.last(), name="last")


def matches(pattern: str) -> SelectorType:
    """
    Select all columns that match the given regex pattern.

    Parameters
    ----------
    pattern
        A valid regular expression pattern, compatible with the `regex crate
        <https://docs.rs/regex/latest/regex/>`_.

    Examples
    --------
    >>> import polars.selectors as cs
    >>> df = pl.DataFrame(
    ...     {
    ...         "foo": ["x", "y"],
    ...         "bar": [123, 456],
    ...         "baz": [2.0, 5.5],
    ...         "zap": [0, 1],
    ...     }
    ... )

    Match column names containing an 'a', preceded by a character that is not 'z':

    >>> df.select(cs.matches("[^z]a"))
    shape: (2, 2)
    ┌─────┬─────┐
    │ bar ┆ baz │
    │ --- ┆ --- │
    │ i64 ┆ f64 │
    ╞═════╪═════╡
    │ 123 ┆ 2.0 │
    │ 456 ┆ 5.5 │
    └─────┴─────┘

    Do not match column names ending in 'R' or 'z' (case-insensitively):

    >>> df.select(~cs.matches(r"(?i)R|z$"))
    shape: (2, 2)
    ┌─────┬─────┐
    │ foo ┆ zap │
    │ --- ┆ --- │
    │ str ┆ i64 │
    ╞═════╪═════╡
    │ x   ┆ 0   │
    │ y   ┆ 1   │
    └─────┴─────┘

    See Also
    --------
    contains : Select all columns that contain the given substring.
    ends_with : Select all columns that end with the given substring(s).
    starts_with : Select all columns that start with the given substring(s).

    """
    if pattern == ".*":
        return all()
    else:
        if pattern.startswith(".*"):
            pattern = pattern[2:]
        elif pattern.endswith(".*"):
            pattern = pattern[:-2]

        pfx = "^.*" if not pattern.startswith("^") else ""
        sfx = ".*$" if not pattern.endswith("$") else ""
        raw_params = f"{pfx}{pattern}{sfx}"

        return _selector_proxy_(
            F.col(raw_params),
            name="matches",
            parameters={"pattern": pattern},
        )


def numeric() -> SelectorType:
    """
    Select all numeric columns.

    Examples
    --------
    >>> import polars.selectors as cs
    >>> df = pl.DataFrame(
    ...     {
    ...         "foo": ["x", "y"],
    ...         "bar": [123, 456],
    ...         "baz": [2.0, 5.5],
    ...         "zap": [0, 0],
    ...     },
    ...     schema_overrides={"bar": pl.Int16, "baz": pl.Float32, "zap": pl.UInt8},
    ... )

    Match all numeric columns:

    >>> df.select(cs.numeric())
    shape: (2, 3)
    ┌─────┬─────┬─────┐
    │ bar ┆ baz ┆ zap │
    │ --- ┆ --- ┆ --- │
    │ i16 ┆ f32 ┆ u8  │
    ╞═════╪═════╪═════╡
    │ 123 ┆ 2.0 ┆ 0   │
    │ 456 ┆ 5.5 ┆ 0   │
    └─────┴─────┴─────┘

    Match all columns *except* for those that are numeric:

    >>> df.select(~cs.numeric())
    shape: (2, 1)
    ┌─────┐
    │ foo │
    │ --- │
    │ str │
    ╞═════╡
    │ x   │
    │ y   │
    └─────┘

    See Also
    --------
    by_dtype : Select columns by dtype.
    float : Select all float columns.
    integer : Select all integer columns.
    temporal : Select all temporal columns.
    string : Select all string columns.

    """
    return _selector_proxy_(
        F.col(NUMERIC_DTYPES),
        name="numeric",
    )


def starts_with(*prefix: str) -> SelectorType:
    """
    Select columns that start with the given substring(s).

    Parameters
    ----------
    prefix
        Substring(s) that matching column names should start with.

    Examples
    --------
    >>> import polars.selectors as cs
    >>> df = pl.DataFrame(
    ...     {
    ...         "foo": [1.0, 2.0],
    ...         "bar": [3.0, 4.0],
    ...         "baz": [5, 6],
    ...         "zap": [7, 8],
    ...     }
    ... )

    Match columns starting with a 'b':

    >>> df.select(cs.starts_with("b"))
    shape: (2, 2)
    ┌─────┬─────┐
    │ bar ┆ baz │
    │ --- ┆ --- │
    │ f64 ┆ i64 │
    ╞═════╪═════╡
    │ 3.0 ┆ 5   │
    │ 4.0 ┆ 6   │
    └─────┴─────┘

    Match columns starting with *either* the letter 'b' or 'z':

    >>> df.select(cs.starts_with("b", "z"))
    shape: (2, 3)
    ┌─────┬─────┬─────┐
    │ bar ┆ baz ┆ zap │
    │ --- ┆ --- ┆ --- │
    │ f64 ┆ i64 ┆ i64 │
    ╞═════╪═════╪═════╡
    │ 3.0 ┆ 5   ┆ 7   │
    │ 4.0 ┆ 6   ┆ 8   │
    └─────┴─────┴─────┘

    Match all columns *except* for those starting with 'b':

    >>> df.select(~cs.starts_with("b"))
    shape: (2, 2)
    ┌─────┬─────┐
    │ foo ┆ zap │
    │ --- ┆ --- │
    │ f64 ┆ i64 │
    ╞═════╪═════╡
    │ 1.0 ┆ 7   │
    │ 2.0 ┆ 8   │
    └─────┴─────┘

    See Also
    --------
    contains : Select all columns that contain the given substring.
    ends_with : Select all columns that end with the given substring(s).
    matches : Select all columns that match the given regex pattern.

    """
    escaped_prefix = _re_string(prefix)
    raw_params = f"^{escaped_prefix}.*$"

    return _selector_proxy_(
        F.col(raw_params),
        name="starts_with",
        parameters={"*prefix": prefix},
    )


def string(include_categorical: bool = False) -> SelectorType:
    """
    Select all Utf8 (and, optionally, Categorical) string columns.

    Examples
    --------
    >>> import polars.selectors as cs
    >>> df = pl.DataFrame(
    ...     {
    ...         "w": ["xx", "yy", "xx", "yy", "xx"],
    ...         "x": [1, 2, 1, 4, -2],
    ...         "y": [3.0, 4.5, 1.0, 2.5, -2.0],
    ...         "z": ["a", "b", "a", "b", "b"],
    ...     },
    ... ).with_columns(
    ...     z=pl.col("z").cast(pl.Categorical).cat.set_ordering("lexical"),
    ... )

    Group by all string columns, sum the numeric columns, then sort by the string cols:

    >>> df.groupby(cs.string()).agg(cs.numeric().sum()).sort(by=cs.string())
    shape: (2, 3)
    ┌─────┬─────┬─────┐
    │ w   ┆ x   ┆ y   │
    │ --- ┆ --- ┆ --- │
    │ str ┆ i64 ┆ f64 │
    ╞═════╪═════╪═════╡
    │ xx  ┆ 0   ┆ 2.0 │
    │ yy  ┆ 6   ┆ 7.0 │
    └─────┴─────┴─────┘

    Group by all string *and* categorical columns:

    >>> df.groupby(cs.string(True)).agg(cs.numeric().sum()).sort(by=cs.string(True))
    shape: (3, 4)
    ┌─────┬─────┬─────┬──────┐
    │ w   ┆ z   ┆ x   ┆ y    │
    │ --- ┆ --- ┆ --- ┆ ---  │
    │ str ┆ cat ┆ i64 ┆ f64  │
    ╞═════╪═════╪═════╪══════╡
    │ xx  ┆ a   ┆ 2   ┆ 4.0  │
    │ xx  ┆ b   ┆ -2  ┆ -2.0 │
    │ yy  ┆ b   ┆ 6   ┆ 7.0  │
    └─────┴─────┴─────┴──────┘

    See Also
    --------
    by_dtype : Select all columns of a given dtype.
    float : Select all float columns.
    integer : Select all integer columns.
    numeric : Select all numeric columns.
    temporal : Select all temporal columns.

    """
    string_dtypes: list[PolarsDataType] = [Utf8]
    if include_categorical:
        string_dtypes.append(Categorical)

    return _selector_proxy_(
        F.col(string_dtypes),
        name="string",
    )


def temporal() -> SelectorType:
    """
    Select all temporal columns.

    Examples
    --------
    >>> from datetime import date, time
    >>> import polars.selectors as cs
    >>> df = pl.DataFrame(
    ...     {
    ...         "dt": [date(2021, 1, 1), date(2021, 1, 2)],
    ...         "tm": [time(12, 0, 0), time(20, 30, 45)],
    ...         "value": [1.2345, 2.3456],
    ...     }
    ... )

    Match all temporal columns:

    >>> df.select(cs.temporal())
    shape: (2, 2)
    ┌────────────┬──────────┐
    │ dt         ┆ tm       │
    │ ---        ┆ ---      │
    │ date       ┆ time     │
    ╞════════════╪══════════╡
    │ 2021-01-01 ┆ 12:00:00 │
    │ 2021-01-02 ┆ 20:30:45 │
    └────────────┴──────────┘

    Match all temporal columns *except* for `Time` columns:

    >>> df.select(cs.temporal() - cs.by_dtype(pl.Time))
    shape: (2, 1)
    ┌────────────┐
    │ dt         │
    │ ---        │
    │ date       │
    ╞════════════╡
    │ 2021-01-01 │
    │ 2021-01-02 │
    └────────────┘

    Match all columns *except* for temporal columns:

    >>> df.select(~cs.temporal())
    shape: (2, 1)
    ┌────────┐
    │ value  │
    │ ---    │
    │ f64    │
    ╞════════╡
    │ 1.2345 │
    │ 2.3456 │
    └────────┘

    See Also
    --------
    by_dtype : Select all columns of a given dtype.
    float : Select all float columns.
    integer : Select all integer columns.
    numeric : Select all numeric columns.
    string : Select all string columns.

    """
    return _selector_proxy_(
        F.col(TEMPORAL_DTYPES),
        name="temporal",
    )


__all__ = [
    "all",
    "by_dtype",
    "by_name",
    "contains",
    "datetime",
    "duration",
    "ends_with",
    "first",
    "float",
    "integer",
    "last",
    "matches",
    "numeric",
    "starts_with",
    "temporal",
    "string",
    "is_selector",
    "expand_selector",
    "SelectorType",
]
