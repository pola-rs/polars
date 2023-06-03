from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, Collection

from polars import Expr
from polars import functions as F
from polars.datatypes import (
    FLOAT_DTYPES,
    INTEGER_DTYPES,
    NUMERIC_DTYPES,
    TEMPORAL_DTYPES,
    is_polars_dtype,
)

if TYPE_CHECKING:
    import sys

    from polars.datatypes import PolarsDataType

    if sys.version_info >= (3, 11):
        from typing import Self
    else:
        from typing_extensions import Self


class _selector_proxy_(Expr):
    """Base column selector expression/proxy."""

    _attrs: dict[str, Any]
    _inverted: bool

    def __init__(
        self,
        expr: Expr,
        type_: str,
        parameters: dict[str, Any] | None = None,
        raw_parameters: list[Any] | None = None,
        invert: bool = False,
    ):
        self._pyexpr = expr._pyexpr
        self._inverted = invert
        self._attrs = {
            "raw_params": raw_parameters,
            "params": parameters,
            "type": type_,
        }

    def __invert__(self) -> Self:
        """Invert the selector."""
        selector_type = self._attrs["type"]
        if selector_type in ("first", "last"):
            raise ValueError(f"Cannot currently invert {selector_type!r} selector")

        params = self._attrs["params"] or {}
        raw_params = self._attrs["raw_params"] or []
        if not raw_params and params:
            raw_params = list(params.values())

        if selector_type == "all":
            inverted_expr = F.all() if self._inverted else F.col([])
        elif self._inverted:
            inverted_expr = F.col(*raw_params)
        else:
            inverted_expr = F.all().exclude(*raw_params)

        return self.__class__(
            expr=inverted_expr,
            type_=selector_type,
            parameters=params,
            raw_parameters=raw_params,
            invert=not self._inverted,
        )

    def __repr__(self) -> str:
        params = self._attrs["params"]
        not_ = "~" if self._inverted else ""
        str_params = ",".join(f"{k}={v!r}" for k, v in (params or {}).items())
        return f"{not_}s.{self._attrs['type']}({str_params})"

    # --------------------------------------------------------------------------------
    # Note: before offering operator support we need a new first-class expression
    # construct on the Rust side that can represent combinatorial selections, eg:
    # >>> (s.starts_with("foo") | s.ends_with("bar")) & ~s.of_type(pl.Utf8)
    # --------------------------------------------------------------------------------
    # Consequently the following operators are reserved for this future usage.
    # --------------------------------------------------------------------------------

    def __and__(self, other: Any) -> Self:
        raise NotImplementedError("Combining selectors with '&' is not yet supported")

    def __or__(self, other: Any) -> Self:
        raise NotImplementedError("Combining selectors with '|' is not yet supported")

    def __rand__(self, other: Any) -> Self:
        raise NotImplementedError("Combining selectors with '&' is not yet supported")

    def __ror__(self, other: Any) -> Self:
        raise NotImplementedError("Combining selectors with '|' is not yet supported")


def _re_string(string: str | Collection[str]) -> str:
    if isinstance(string, str):
        return re.escape(string)
    else:
        strings: list[str] = []
        for st in string:
            if isinstance(st, Collection):
                strings.extend(st)
            else:
                strings.append(st)
        return "|".join(re.escape(x) for x in strings)


class s:
    """Polars column selectors."""

    @staticmethod
    def all() -> Expr:
        """
        Select all columns.

        See Also
        --------
        first
        last

        Examples
        --------
        >>> from datetime import date
        >>> from polars.selectors import s
        >>> df = pl.DataFrame(
        ...     {
        ...         "dt": [date(1999, 12, 31), date(2024, 1, 1)],
        ...         "value": [1_234_500, 5_000_555],
        ...     },
        ...     schema_overrides={"value": pl.Int32},
        ... )

        Select all columns, casting them to string:

        >>> df.select(s.all().cast(pl.Utf8))
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

        >>> df.select(s.all().exclude(pl.NUMERIC_DTYPES))
        shape: (2, 1)
        ┌────────────┐
        │ dt         │
        │ ---        │
        │ date       │
        ╞════════════╡
        │ 1999-12-31 │
        │ 2024-01-01 │
        └────────────┘

        """
        return _selector_proxy_(F.all(), type_="all")

    @staticmethod
    def by_dtype(
        *dtypes: PolarsDataType | Collection[PolarsDataType],
    ) -> Expr:
        """
        Select all columns matching the given dtypes.

        See Also
        --------
        integer
        float
        numeric
        temporal

        Examples
        --------
        >>> from datetime import date
        >>> from polars.selectors import s
        >>> df = pl.DataFrame(
        ...     {
        ...         "dt": [date(1999, 12, 31), date(2024, 1, 1), date(2010, 7, 5)],
        ...         "value": [1_234_500, 5_000_555, -4_500_000],
        ...         "other": ["foo", "bar", "foo"],
        ...     }
        ... )

        Select all columns with date or integer dtypes:

        >>> df.select(s.by_dtype(pl.Date, pl.INTEGER_DTYPES))
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

        >>> df.select(~s.by_dtype(pl.Date, pl.INTEGER_DTYPES))
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

        >>> df.groupby(s.by_dtype(pl.Utf8)).agg(s.numeric().sum()).sort(by="other")
        shape: (2, 2)
        ┌───────┬──────────┐
        │ other ┆ value    │
        │ ---   ┆ ---      │
        │ str   ┆ i64      │
        ╞═══════╪══════════╡
        │ bar   ┆ 5000555  │
        │ foo   ┆ -3265500 │
        └───────┴──────────┘

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
            F.col(*all_dtypes), type_="by_dtype", parameters={"dtypes": all_dtypes}
        )

    @staticmethod
    def by_name(*names: str | Collection[str]) -> Expr:
        """
        Select all columns matching the given names.

        Parameters
        ----------
        *names
            One or more names of columns to select.

        See Also
        --------
        by_dtype

        Examples
        --------
        >>> from polars.selectors import s
        >>> df = pl.DataFrame(
        ...     {
        ...         "foo": ["x", "y"],
        ...         "bar": [123, 456],
        ...         "baz": [2.0, 5.5],
        ...         "zap": [False, True],
        ...     }
        ... )

        Select columns by name:

        >>> df.select(s.by_name("foo", "bar"))
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

        >>> df.select(~s.by_name("foo", "bar"))
        shape: (2, 2)
        ┌─────┬───────┐
        │ baz ┆ zap   │
        │ --- ┆ ---   │
        │ f64 ┆ bool  │
        ╞═════╪═══════╡
        │ 2.0 ┆ false │
        │ 5.5 ┆ true  │
        └─────┴───────┘

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
            F.col(*all_names), type_="by_dtype", parameters={"names": all_names}
        )

    @staticmethod
    def contains(substring: str | Collection[str]) -> Expr:
        """
        Select columns that contain the given literal substring(s).

        Parameters
        ----------
        substring
            Substring(s) that matching column names should contain.

        See Also
        --------
        matches
        ends_with
        starts_with

        Examples
        --------
        >>> from polars.selectors import s
        >>> df = pl.DataFrame(
        ...     {
        ...         "foo": ["x", "y"],
        ...         "bar": [123, 456],
        ...         "baz": [2.0, 5.5],
        ...         "zap": [False, True],
        ...     }
        ... )

        Select columns that contain the substring 'ba':

        >>> df.select(s.contains("ba"))
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

        >>> df.select(s.contains(("ba", "z")))
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

        >>> df.select(~s.contains("ba"))
        shape: (2, 2)
        ┌─────┬───────┐
        │ foo ┆ zap   │
        │ --- ┆ ---   │
        │ str ┆ bool  │
        ╞═════╪═══════╡
        │ x   ┆ false │
        │ y   ┆ true  │
        └─────┴───────┘

        """
        escaped_substring = _re_string(substring)
        raw_params = f"^.*{escaped_substring}.*$"

        return _selector_proxy_(
            F.col(raw_params),
            type_="contains",
            parameters={"substring": escaped_substring},
            raw_parameters=[raw_params],
        )

    @staticmethod
    def ends_with(*suffix: str) -> Expr:
        """
        Select columns that end with the given substring(s).

        Parameters
        ----------
        suffix
            Substring(s) that matching column names should end with.

        See Also
        --------
        contains
        matches
        starts_with

        Examples
        --------
        >>> from polars.selectors import s
        >>> df = pl.DataFrame(
        ...     {
        ...         "foo": ["x", "y"],
        ...         "bar": [123, 456],
        ...         "baz": [2.0, 5.5],
        ...         "zap": [False, True],
        ...     }
        ... )

        Select columns that end with the substring 'z':

        >>> df.select(s.ends_with("z"))
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

        >>> df.select(s.ends_with("z", "r"))
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

        >>> df.select(~s.ends_with("z"))
        shape: (2, 3)
        ┌─────┬─────┬───────┐
        │ foo ┆ bar ┆ zap   │
        │ --- ┆ --- ┆ ---   │
        │ str ┆ i64 ┆ bool  │
        ╞═════╪═════╪═══════╡
        │ x   ┆ 123 ┆ false │
        │ y   ┆ 456 ┆ true  │
        └─────┴─────┴───────┘

        """
        escaped_suffix = _re_string(suffix)
        raw_params = f"^.*({escaped_suffix})$"

        return _selector_proxy_(
            F.col(raw_params),
            type_="ends_with",
            parameters={"suffix": escaped_suffix},
            raw_parameters=[raw_params],
        )

    @staticmethod
    def first() -> Expr:
        """
        Select the first column in the current scope.

        See Also
        --------
        all
        last

        Examples
        --------
        >>> from polars.selectors import s
        >>> df = pl.DataFrame(
        ...     {
        ...         "foo": ["x", "y"],
        ...         "bar": [123, 456],
        ...         "baz": [2.0, 5.5],
        ...         "zap": [0, 1],
        ...     }
        ... )

        Select the first column:

        >>> df.select(s.first())
        shape: (2, 1)
        ┌─────┐
        │ foo │
        │ --- │
        │ str │
        ╞═════╡
        │ x   │
        │ y   │
        └─────┘

        """
        return _selector_proxy_(F.first(), type_="first")

    @staticmethod
    def float() -> Expr:
        """
        Select all float columns.

        See Also
        --------
        integer
        numeric
        temporal

        Examples
        --------
        >>> from polars.selectors import s
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

        >>> df.select(s.float())
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

        >>> df.select(~s.float())
        shape: (2, 2)
        ┌─────┬─────┐
        │ foo ┆ bar │
        │ --- ┆ --- │
        │ str ┆ i64 │
        ╞═════╪═════╡
        │ x   ┆ 123 │
        │ y   ┆ 456 │
        └─────┴─────┘

        """
        return _selector_proxy_(
            F.col(FLOAT_DTYPES), type_="float", raw_parameters=[FLOAT_DTYPES]
        )

    @staticmethod
    def integer() -> Expr:
        """
        Select all integer columns.

        See Also
        --------
        by_dtype
        float
        numeric
        temporal

        Examples
        --------
        >>> from polars.selectors import s
        >>> df = pl.DataFrame(
        ...     {
        ...         "foo": ["x", "y"],
        ...         "bar": [123, 456],
        ...         "baz": [2.0, 5.5],
        ...         "zap": [0, 1],
        ...     }
        ... )

        Select all integer columns:

        >>> df.select(s.integer())
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

        >>> df.select(~s.integer())
        shape: (2, 2)
        ┌─────┬─────┐
        │ foo ┆ baz │
        │ --- ┆ --- │
        │ str ┆ f64 │
        ╞═════╪═════╡
        │ x   ┆ 2.0 │
        │ y   ┆ 5.5 │
        └─────┴─────┘

        """
        return _selector_proxy_(
            F.col(INTEGER_DTYPES), type_="integer", raw_parameters=[INTEGER_DTYPES]
        )

    @staticmethod
    def last() -> Expr:
        """
        Select the last column in the current scope.

        See Also
        --------
        all
        first

        Examples
        --------
        >>> from polars.selectors import s
        >>> df = pl.DataFrame(
        ...     {
        ...         "foo": ["x", "y"],
        ...         "bar": [123, 456],
        ...         "baz": [2.0, 5.5],
        ...         "zap": [0, 1],
        ...     }
        ... )

        Select the last column:

        >>> df.select(s.last())
        shape: (2, 1)
        ┌─────┐
        │ zap │
        │ --- │
        │ i64 │
        ╞═════╡
        │ 0   │
        │ 1   │
        └─────┘

        """
        return _selector_proxy_(F.last(), type_="first")

    @staticmethod
    def matches(pattern: str) -> Expr:
        """
        Select all columns that match the given regex pattern.

        Parameters
        ----------
        pattern
            A valid regular expression pattern, compatible with the `regex crate
            <https://docs.rs/regex/latest/regex/>`_.

        See Also
        --------
        contains
        ends_with
        starts_with

        Examples
        --------
        >>> from polars.selectors import s
        >>> df = pl.DataFrame(
        ...     {
        ...         "foo": ["x", "y"],
        ...         "bar": [123, 456],
        ...         "baz": [2.0, 5.5],
        ...         "zap": [0, 1],
        ...     }
        ... )

        Match column names containing an 'a', preceded by a character that is not 'z':

        >>> df.select(s.matches("[^z]a"))
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

        >>> df.select(~s.matches(r"(?i)R|z$"))
        shape: (2, 2)
        ┌─────┬─────┐
        │ foo ┆ zap │
        │ --- ┆ --- │
        │ str ┆ i64 │
        ╞═════╪═════╡
        │ x   ┆ 0   │
        │ y   ┆ 1   │
        └─────┴─────┘

        """
        if pattern == ".*":
            return s.all()
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
                type_="matches",
                parameters={"pattern": pattern},
                raw_parameters=[raw_params],
            )

    @staticmethod
    def numeric() -> Expr:
        """
        Select all numeric columns.

        See Also
        --------
        by_dtype
        float
        integer
        temporal

        Examples
        --------
        >>> from polars.selectors import s
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

        >>> df.select(s.numeric())
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

        >>> df.select(~s.numeric())
        shape: (2, 1)
        ┌─────┐
        │ foo │
        │ --- │
        │ str │
        ╞═════╡
        │ x   │
        │ y   │
        └─────┘

        """
        return _selector_proxy_(
            F.col(NUMERIC_DTYPES), type_="numeric", raw_parameters=[NUMERIC_DTYPES]
        )

    @staticmethod
    def starts_with(*prefix: str) -> Expr:
        """
        Select columns that start with the given substring(s).

        Parameters
        ----------
        prefix
            Substring(s) that matching column names should start with.

        See Also
        --------
        contains
        ends_with
        matches

        Examples
        --------
        >>> from polars.selectors import s
        >>> df = pl.DataFrame(
        ...     {
        ...         "foo": [1.0, 2.0],
        ...         "bar": [3.0, 4.0],
        ...         "baz": [5, 6],
        ...         "zap": [7, 8],
        ...     }
        ... )

        Match columns starting with a 'b':

        >>> df.select(s.starts_with("b"))
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

        >>> df.select(s.starts_with("b", "z"))
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

        >>> df.select(~s.starts_with("b"))
        shape: (2, 2)
        ┌─────┬─────┐
        │ foo ┆ zap │
        │ --- ┆ --- │
        │ f64 ┆ i64 │
        ╞═════╪═════╡
        │ 1.0 ┆ 7   │
        │ 2.0 ┆ 8   │
        └─────┴─────┘

        """
        escaped_prefix = _re_string(prefix)
        raw_params = f"^({escaped_prefix}).*$"

        return _selector_proxy_(
            F.col(raw_params),
            type_="starts_with",
            parameters={"prefix": prefix},
            raw_parameters=[raw_params],
        )

    @staticmethod
    def temporal() -> Expr:
        """
        Select all temporal columns.

        See Also
        --------
        by_dtype
        float
        integer
        numeric

        Examples
        --------
        >>> from datetime import date, time
        >>> from polars.selectors import s
        >>> df = pl.DataFrame(
        ...     {
        ...         "dt": [date(2021, 1, 1), date(2021, 1, 2)],
        ...         "tm": [time(12, 0, 0), time(20, 30, 45)],
        ...         "value": [1.2345, 2.3456],
        ...     }
        ... )

        Match all temporal columns:

        >>> df.select(s.temporal())
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

        >>> df.select(s.temporal().exclude(pl.Time))
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

        >>> df.select(~s.temporal())
        shape: (2, 1)
        ┌────────┐
        │ value  │
        │ ---    │
        │ f64    │
        ╞════════╡
        │ 1.2345 │
        │ 2.3456 │
        └────────┘

        """
        return _selector_proxy_(
            F.col(TEMPORAL_DTYPES), type_="temporal", raw_parameters=[TEMPORAL_DTYPES]
        )


__all__ = ["s"]
