from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

from polars.datatypes import Date, Datetime, Time, py_type_to_dtype
from polars.exceptions import ChronoFormatWarning
from polars.utils._parse_expr_input import parse_as_expression
from polars.utils._wrap import wrap_expr
from polars.utils.decorators import deprecated_alias
from polars.utils.various import find_stacklevel

if TYPE_CHECKING:
    from polars import Expr
    from polars.type_aliases import (
        PolarsDataType,
        PolarsTemporalType,
        TimeUnit,
        TransferEncoding,
    )


class ExprStringNameSpace:
    """Namespace for string related expressions."""

    _accessor = "str"

    def __init__(self, expr: Expr):
        self._pyexpr = expr._pyexpr

    def to_date(
        self,
        format: str | None = None,
        *,
        strict: bool = True,
        exact: bool = True,
        cache: bool = True,
    ) -> Expr:
        """
        Convert a Utf8 column into a Date column.

        Parameters
        ----------
        format
            Format to use for conversion. Refer to the `chrono crate documentation
            <https://docs.rs/chrono/latest/chrono/format/strftime/index.html>`_
            for the full specification. Example: ``"%Y-%m-%d"``.
            If set to None (default), the format is inferred from the data.
        strict
            Raise an error if any conversion fails.
        exact
            Require an exact format match. If False, allow the format to match anywhere
            in the target string.
        cache
            Use a cache of unique, converted dates to apply the conversion.

        Examples
        --------
        >>> s = pl.Series(["2020/01/01", "2020/02/01", "2020/03/01"])
        >>> s.str.to_date()
        shape: (3,)
        Series: '' [date]
        [
                2020-01-01
                2020-02-01
                2020-03-01
        ]

        """
        _validate_format_argument(format)
        return wrap_expr(self._pyexpr.str_to_date(format, strict, exact, cache))

    def to_datetime(
        self,
        format: str | None = None,
        *,
        time_unit: TimeUnit | None = None,
        time_zone: str | None = None,
        strict: bool = True,
        exact: bool = True,
        cache: bool = True,
        utc: bool | None = None,
    ) -> Expr:
        """
        Convert a Utf8 column into a Datetime column.

        Parameters
        ----------
        format
            Format to use for conversion. Refer to the `chrono crate documentation
            <https://docs.rs/chrono/latest/chrono/format/strftime/index.html>`_
            for the full specification. Example: ``"%Y-%m-%d %H:%M:%S"``.
            If set to None (default), the format is inferred from the data.
        time_unit : {None, 'us', 'ns', 'ms'}
            Unit of time for the resulting Datetime column. If set to None (default),
            the time unit is inferred from the format string if given, eg:
            ``"%F %T%.3f"`` => ``Datetime("ms")``. If no fractional second component is
            found, the default is ``"us"``.
        time_zone
            Time zone for the resulting Datetime column.
        strict
            Raise an error if any conversion fails.
        exact
            Require an exact format match. If False, allow the format to match anywhere
            in the target string.
        cache
            Use a cache of unique, converted datetimes to apply the conversion.
        utc
            Parse time zone aware datetimes as UTC. This may be useful if you have data
            with mixed offsets.

            .. deprecated:: 0.18.0
                This is now a no-op, you can safely remove it.
                Offset-naive strings are parsed as ``pl.Datetime(time_unit)``,
                and offset-aware strings are converted to
                ``pl.Datetime(time_unit, "UTC")``.

        Examples
        --------
        >>> s = pl.Series(["2020-01-01 01:00Z", "2020-01-01 02:00Z"])
        >>> s.str.to_datetime("%Y-%m-%d %H:%M%#z")
        shape: (2,)
        Series: '' [datetime[μs, UTC]]
        [
                2020-01-01 01:00:00 UTC
                2020-01-01 02:00:00 UTC
        ]
        """
        _validate_format_argument(format)
        if utc is not None:
            warnings.warn(
                "The `utc` argument is now a no-op and has no effect. "
                "You can safely remove it. "
                "Offset-naive strings are parsed as ``pl.Datetime(time_unit)``, "
                "and offset-aware strings are converted to "
                '``pl.Datetime(time_unit, "UTC")``.',
                DeprecationWarning,
                stacklevel=find_stacklevel(),
            )
        return wrap_expr(
            self._pyexpr.str_to_datetime(
                format,
                time_unit,
                time_zone,
                strict,
                exact,
                cache,
            )
        )

    def to_time(
        self,
        format: str | None = None,
        *,
        strict: bool = True,
        cache: bool = True,
    ) -> Expr:
        """
        Convert a Utf8 column into a Time column.

        Parameters
        ----------
        format
            Format to use for conversion. Refer to the `chrono crate documentation
            <https://docs.rs/chrono/latest/chrono/format/strftime/index.html>`_
            for the full specification. Example: ``"%H:%M:%S"``.
            If set to None (default), the format is inferred from the data.
        strict
            Raise an error if any conversion fails.
        cache
            Use a cache of unique, converted times to apply the conversion.

        Examples
        --------
        >>> s = pl.Series(["01:00", "02:00", "03:00"])
        >>> s.str.to_time("%H:%M")
        shape: (3,)
        Series: '' [time]
        [
                01:00:00
                02:00:00
                03:00:00
        ]

        """
        _validate_format_argument(format)
        return wrap_expr(self._pyexpr.str_to_time(format, strict, cache))

    @deprecated_alias(datatype="dtype", fmt="format")
    def strptime(
        self,
        dtype: PolarsTemporalType,
        format: str | None = None,
        *,
        strict: bool = True,
        exact: bool = True,
        cache: bool = True,
        utc: bool | None = None,
    ) -> Expr:
        """
        Convert a Utf8 column into a Date/Datetime/Time column.

        Parameters
        ----------
        dtype
            The data type to convert into. Can be either Date, Datetime, or Time.
        format
            Format to use for conversion. Refer to the `chrono crate documentation
            <https://docs.rs/chrono/latest/chrono/format/strftime/index.html>`_
            for the full specification. Example: ``"%Y-%m-%d %H:%M:%S"``.
            If set to None (default), the format is inferred from the data.
        strict
            Raise an error if any conversion fails.
        exact
            Require an exact format match. If False, allow the format to match anywhere
            in the target string. Conversion to the Time type is always exact.
        cache
            Use a cache of unique, converted dates to apply the datetime conversion.
        utc
            Parse time zone aware datetimes as UTC. This may be useful if you have data
            with mixed offsets.

            .. deprecated:: 0.18.0
                This is now a no-op, you can safely remove it.
                Offset-naive strings are parsed as ``pl.Datetime(time_unit)``,
                and offset-aware strings are converted to
                ``pl.Datetime(time_unit, "UTC")``.

        Notes
        -----
        When converting to a Datetime type, the time unit is inferred from the format
        string if given, eg: ``"%F %T%.3f"`` => ``Datetime("ms")``. If no fractional
        second component is found, the default is ``"us"``.

        Examples
        --------
        Dealing with a consistent format:

        >>> s = pl.Series(["2020-01-01 01:00Z", "2020-01-01 02:00Z"])
        >>> s.str.strptime(pl.Datetime, "%Y-%m-%d %H:%M%#z")
        shape: (2,)
        Series: '' [datetime[μs, UTC]]
        [
                2020-01-01 01:00:00 UTC
                2020-01-01 02:00:00 UTC
        ]

        Dealing with different formats.

        >>> s = pl.Series(
        ...     "date",
        ...     [
        ...         "2021-04-22",
        ...         "2022-01-04 00:00:00",
        ...         "01/31/22",
        ...         "Sun Jul  8 00:34:60 2001",
        ...     ],
        ... )
        >>> s.to_frame().select(
        ...     pl.coalesce(
        ...         pl.col("date").str.strptime(pl.Date, "%F", strict=False),
        ...         pl.col("date").str.strptime(pl.Date, "%F %T", strict=False),
        ...         pl.col("date").str.strptime(pl.Date, "%D", strict=False),
        ...         pl.col("date").str.strptime(pl.Date, "%c", strict=False),
        ...     )
        ... ).to_series()
        shape: (4,)
        Series: 'date' [date]
        [
                2021-04-22
                2022-01-04
                2022-01-31
                2001-07-08
        ]
        """
        _validate_format_argument(format)

        if dtype == Date:
            return self.to_date(format, strict=strict, exact=exact, cache=cache)
        elif dtype == Datetime:
            time_unit = dtype.time_unit  # type: ignore[union-attr]
            time_zone = dtype.time_zone  # type: ignore[union-attr]
            return self.to_datetime(
                format,
                time_unit=time_unit,
                time_zone=time_zone,
                strict=strict,
                exact=exact,
                cache=cache,
                utc=utc,
            )
        elif dtype == Time:
            return self.to_time(format, strict=strict, cache=cache)
        else:
            raise ValueError("dtype should be of type {Date, Datetime, Time}")

    def to_decimal(
        self,
        inference_length: int = 100,
    ) -> Expr:
        """
        Convert a Utf8 column into a Date column.

        This method infers the needed parameters ``precision`` and ``scale``.

        Parameters
        ----------
        inference_length
            Number of elements to parse to determine the `precision` and `scale`

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "numbers": [
        ...             "40.12",
        ...             "3420.13",
        ...             "120134.19",
        ...             "3212.98",
        ...             "12.90",
        ...             "143.09",
        ...             "143.9",
        ...         ]
        ...     }
        ... )
        >>> df.select(pl.col("numbers").str.to_decimal())
        shape: (7, 1)
        ┌──────────────┐
        │ numbers      │
        │ ---          │
        │ decimal[8,2] │
        ╞══════════════╡
        │ 40.12        │
        │ 3420.13      │
        │ 120134.19    │
        │ 3212.98      │
        │ 12.9         │
        │ 143.09       │
        │ 143.9        │
        └──────────────┘

        """
        return wrap_expr(self._pyexpr.str_to_decimal(inference_length))

    def lengths(self) -> Expr:
        """
        Get length of the strings as UInt32 (as number of bytes).

        Notes
        -----
        The returned lengths are equal to the number of bytes in the UTF8 string. If you
        need the length in terms of the number of characters, use ``n_chars`` instead.

        Examples
        --------
        >>> df = pl.DataFrame({"s": ["Café", None, "345", "東京"]}).with_columns(
        ...     [
        ...         pl.col("s").str.lengths().alias("length"),
        ...         pl.col("s").str.n_chars().alias("nchars"),
        ...     ]
        ... )
        >>> df
        shape: (4, 3)
        ┌──────┬────────┬────────┐
        │ s    ┆ length ┆ nchars │
        │ ---  ┆ ---    ┆ ---    │
        │ str  ┆ u32    ┆ u32    │
        ╞══════╪════════╪════════╡
        │ Café ┆ 5      ┆ 4      │
        │ null ┆ null   ┆ null   │
        │ 345  ┆ 3      ┆ 3      │
        │ 東京  ┆ 6      ┆ 2      │
        └──────┴────────┴────────┘

        """
        return wrap_expr(self._pyexpr.str_lengths())

    def n_chars(self) -> Expr:
        """
        Get length of the strings as UInt32 (as number of chars).

        Notes
        -----
        If you know that you are working with ASCII text, ``lengths`` will be
        equivalent, and faster (returns length in terms of the number of bytes).

        Examples
        --------
        >>> df = pl.DataFrame({"s": ["Café", None, "345", "東京"]}).with_columns(
        ...     [
        ...         pl.col("s").str.n_chars().alias("nchars"),
        ...         pl.col("s").str.lengths().alias("length"),
        ...     ]
        ... )
        >>> df
        shape: (4, 3)
        ┌──────┬────────┬────────┐
        │ s    ┆ nchars ┆ length │
        │ ---  ┆ ---    ┆ ---    │
        │ str  ┆ u32    ┆ u32    │
        ╞══════╪════════╪════════╡
        │ Café ┆ 4      ┆ 5      │
        │ null ┆ null   ┆ null   │
        │ 345  ┆ 3      ┆ 3      │
        │ 東京  ┆ 2      ┆ 6      │
        └──────┴────────┴────────┘

        """
        return wrap_expr(self._pyexpr.str_n_chars())

    def concat(self, delimiter: str = "-") -> Expr:
        """
        Vertically concat the values in the Series to a single string value.

        Parameters
        ----------
        delimiter
            The delimiter to insert between consecutive string values.

        Returns
        -------
        Series of dtype Utf8

        Examples
        --------
        >>> df = pl.DataFrame({"foo": [1, None, 2]})
        >>> df.select(pl.col("foo").str.concat("-"))
        shape: (1, 1)
        ┌──────────┐
        │ foo      │
        │ ---      │
        │ str      │
        ╞══════════╡
        │ 1-null-2 │
        └──────────┘

        """
        return wrap_expr(self._pyexpr.str_concat(delimiter))

    def to_uppercase(self) -> Expr:
        """
        Transform to uppercase variant.

        Examples
        --------
        >>> df = pl.DataFrame({"foo": ["cat", "dog"]})
        >>> df.select(pl.col("foo").str.to_uppercase())
        shape: (2, 1)
        ┌─────┐
        │ foo │
        │ --- │
        │ str │
        ╞═════╡
        │ CAT │
        │ DOG │
        └─────┘

        """
        return wrap_expr(self._pyexpr.str_to_uppercase())

    def to_lowercase(self) -> Expr:
        """
        Transform to lowercase variant.

        Examples
        --------
        >>> df = pl.DataFrame({"foo": ["CAT", "DOG"]})
        >>> df.select(pl.col("foo").str.to_lowercase())
        shape: (2, 1)
        ┌─────┐
        │ foo │
        │ --- │
        │ str │
        ╞═════╡
        │ cat │
        │ dog │
        └─────┘

        """
        return wrap_expr(self._pyexpr.str_to_lowercase())

    def strip(self, characters: str | None = None) -> Expr:
        r"""
        Remove leading and trailing characters.

        Parameters
        ----------
        characters
            The set of characters to be removed. All combinations of this set of
            characters will be stripped. If set to None (default), all whitespace is
            removed instead.

        Examples
        --------
        >>> df = pl.DataFrame({"foo": [" hello ", "\tworld"]})
        >>> df.select(pl.col("foo").str.strip())
        shape: (2, 1)
        ┌───────┐
        │ foo   │
        │ ---   │
        │ str   │
        ╞═══════╡
        │ hello │
        │ world │
        └───────┘

        Characters can be stripped by passing a string as argument. Note that whitespace
        will not be stripped automatically when doing so.

        >>> df.select(pl.col("foo").str.strip("od\t"))
        shape: (2, 1)
        ┌─────────┐
        │ foo     │
        │ ---     │
        │ str     │
        ╞═════════╡
        │  hello  │
        │ worl    │
        └─────────┘

        """
        return wrap_expr(self._pyexpr.str_strip(characters))

    def lstrip(self, characters: str | None = None) -> Expr:
        r"""
        Remove leading characters.

        Parameters
        ----------
        characters
            The set of characters to be removed. All combinations of this set of
            characters will be stripped. If set to None (default), all whitespace is
            removed instead.

        Examples
        --------
        >>> df = pl.DataFrame({"foo": [" hello ", "\tworld"]})
        >>> df.select(pl.col("foo").str.lstrip())
        shape: (2, 1)
        ┌────────┐
        │ foo    │
        │ ---    │
        │ str    │
        ╞════════╡
        │ hello  │
        │ world  │
        └────────┘

        Characters can be stripped by passing a string as argument. Note that whitespace
        will not be stripped automatically when doing so.

        >>> df.select(pl.col("foo").str.lstrip("wod\t"))
        shape: (2, 1)
        ┌─────────┐
        │ foo     │
        │ ---     │
        │ str     │
        ╞═════════╡
        │  hello  │
        │ rld     │
        └─────────┘

        """
        return wrap_expr(self._pyexpr.str_lstrip(characters))

    def rstrip(self, characters: str | None = None) -> Expr:
        r"""
        Remove trailing characters.

        Parameters
        ----------
        characters
            The set of characters to be removed. All combinations of this set of
            characters will be stripped. If set to None (default), all whitespace is
            removed instead.

        Examples
        --------
        >>> df = pl.DataFrame({"foo": [" hello ", "world\t"]})
        >>> df.select(pl.col("foo").str.rstrip())
        shape: (2, 1)
        ┌────────┐
        │ foo    │
        │ ---    │
        │ str    │
        ╞════════╡
        │  hello │
        │ world  │
        └────────┘

        Characters can be stripped by passing a string as argument. Note that whitespace
        will not be stripped automatically when doing so.

        >>> df.select(pl.col("foo").str.rstrip("wod\t"))
        shape: (2, 1)
        ┌─────────┐
        │ foo     │
        │ ---     │
        │ str     │
        ╞═════════╡
        │  hello  │
        │ worl    │
        └─────────┘

        """
        return wrap_expr(self._pyexpr.str_rstrip(characters))

    def zfill(self, alignment: int) -> Expr:
        """
        Fills the string with zeroes.

        Return a copy of the string left filled with ASCII '0' digits to make a string
        of length width.

        A leading sign prefix ('+'/'-') is handled by inserting the padding after the
        sign character rather than before. The original string is returned if width is
        less than or equal to ``len(s)``.

        Parameters
        ----------
        alignment
            Fill the value up to this length

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "num": [-10, -1, 0, 1, 10, 100, 1000, 10000, 100000, 1000000, None],
        ...     }
        ... )
        >>> df.with_columns(pl.col("num").cast(str).str.zfill(5))
        shape: (11, 1)
        ┌─────────┐
        │ num     │
        │ ---     │
        │ str     │
        ╞═════════╡
        │ -0010   │
        │ -0001   │
        │ 00000   │
        │ 00001   │
        │ …       │
        │ 10000   │
        │ 100000  │
        │ 1000000 │
        │ null    │
        └─────────┘

        """
        return wrap_expr(self._pyexpr.str_zfill(alignment))

    def ljust(self, width: int, fill_char: str = " ") -> Expr:
        """
        Return the string left justified in a string of length ``width``.

        Padding is done using the specified ``fill_char``.
        The original string is returned if ``width`` is less than or equal to
        ``len(s)``.

        Parameters
        ----------
        width
            Justify left to this length.
        fill_char
            Fill with this ASCII character.

        Examples
        --------
        >>> df = pl.DataFrame({"a": ["cow", "monkey", None, "hippopotamus"]})
        >>> df.select(pl.col("a").str.ljust(8, "*"))
        shape: (4, 1)
        ┌──────────────┐
        │ a            │
        │ ---          │
        │ str          │
        ╞══════════════╡
        │ cow*****     │
        │ monkey**     │
        │ null         │
        │ hippopotamus │
        └──────────────┘

        """
        return wrap_expr(self._pyexpr.str_ljust(width, fill_char))

    def rjust(self, width: int, fill_char: str = " ") -> Expr:
        """
        Return the string right justified in a string of length ``width``.

        Padding is done using the specified ``fill_char``.
        The original string is returned if ``width`` is less than or equal to
        ``len(s)``.

        Parameters
        ----------
        width
            Justify right to this length.
        fill_char
            Fill with this ASCII character.

        Examples
        --------
        >>> df = pl.DataFrame({"a": ["cow", "monkey", None, "hippopotamus"]})
        >>> df.select(pl.col("a").str.rjust(8, "*"))
        shape: (4, 1)
        ┌──────────────┐
        │ a            │
        │ ---          │
        │ str          │
        ╞══════════════╡
        │ *****cow     │
        │ **monkey     │
        │ null         │
        │ hippopotamus │
        └──────────────┘

        """
        return wrap_expr(self._pyexpr.str_rjust(width, fill_char))

    def contains(
        self, pattern: str | Expr, *, literal: bool = False, strict: bool = True
    ) -> Expr:
        """
        Check if string contains a substring that matches a regex.

        Parameters
        ----------
        pattern
            A valid regular expression pattern, compatible with the `regex crate
            <https://docs.rs/regex/latest/regex/>`_.
        literal
            Treat ``pattern`` as a literal string, not as a regular expression.
        strict
            Raise an error if the underlying pattern is not a valid regex,
            otherwise mask out with a null value.

        Notes
        -----
        To modify regular expression behaviour (such as case-sensitivity) with
        flags, use the inline ``(?iLmsuxU)`` syntax. For example:

        >>> pl.DataFrame({"s": ["AAA", "aAa", "aaa"]}).with_columns(
        ...     default_match=pl.col("s").str.contains("AA"),
        ...     insensitive_match=pl.col("s").str.contains("(?i)AA"),
        ... )
        shape: (3, 3)
        ┌─────┬───────────────┬───────────────────┐
        │ s   ┆ default_match ┆ insensitive_match │
        │ --- ┆ ---           ┆ ---               │
        │ str ┆ bool          ┆ bool              │
        ╞═════╪═══════════════╪═══════════════════╡
        │ AAA ┆ true          ┆ true              │
        │ aAa ┆ false         ┆ true              │
        │ aaa ┆ false         ┆ true              │
        └─────┴───────────────┴───────────────────┘

        See the regex crate's section on `grouping and flags
        <https://docs.rs/regex/latest/regex/#grouping-and-flags>`_ for
        additional information about the use of inline expression modifiers.

        Examples
        --------
        >>> df = pl.DataFrame({"a": ["Crab", "cat and dog", "rab$bit", None]})
        >>> df.select(
        ...     pl.col("a"),
        ...     pl.col("a").str.contains("cat|bit").alias("regex"),
        ...     pl.col("a").str.contains("rab$", literal=True).alias("literal"),
        ... )
        shape: (4, 3)
        ┌─────────────┬───────┬─────────┐
        │ a           ┆ regex ┆ literal │
        │ ---         ┆ ---   ┆ ---     │
        │ str         ┆ bool  ┆ bool    │
        ╞═════════════╪═══════╪═════════╡
        │ Crab        ┆ false ┆ false   │
        │ cat and dog ┆ true  ┆ false   │
        │ rab$bit     ┆ true  ┆ true    │
        │ null        ┆ null  ┆ null    │
        └─────────────┴───────┴─────────┘

        See Also
        --------
        starts_with : Check if string values start with a substring.
        ends_with : Check if string values end with a substring.

        """
        pattern = parse_as_expression(pattern, str_as_lit=True)._pyexpr
        return wrap_expr(self._pyexpr.str_contains(pattern, literal, strict))

    def ends_with(self, suffix: str | Expr) -> Expr:
        """
        Check if string values end with a substring.

        Parameters
        ----------
        suffix
            Suffix substring.

        Examples
        --------
        >>> df = pl.DataFrame({"fruits": ["apple", "mango", None]})
        >>> df.with_columns(
        ...     pl.col("fruits").str.ends_with("go").alias("has_suffix"),
        ... )
        shape: (3, 2)
        ┌────────┬────────────┐
        │ fruits ┆ has_suffix │
        │ ---    ┆ ---        │
        │ str    ┆ bool       │
        ╞════════╪════════════╡
        │ apple  ┆ false      │
        │ mango  ┆ true       │
        │ null   ┆ null       │
        └────────┴────────────┘

        Using ``ends_with`` as a filter condition:

        >>> df.filter(pl.col("fruits").str.ends_with("go"))
        shape: (1, 1)
        ┌────────┐
        │ fruits │
        │ ---    │
        │ str    │
        ╞════════╡
        │ mango  │
        └────────┘

        See Also
        --------
        contains : Check if string contains a substring that matches a regex.
        starts_with : Check if string values start with a substring.

        """
        suffix = parse_as_expression(suffix, str_as_lit=True)._pyexpr
        return wrap_expr(self._pyexpr.str_ends_with(suffix))

    def starts_with(self, prefix: str | Expr) -> Expr:
        """
        Check if string values start with a substring.

        Parameters
        ----------
        prefix
            Prefix substring.

        Examples
        --------
        >>> df = pl.DataFrame({"fruits": ["apple", "mango", None]})
        >>> df.with_columns(
        ...     pl.col("fruits").str.starts_with("app").alias("has_prefix"),
        ... )
        shape: (3, 2)
        ┌────────┬────────────┐
        │ fruits ┆ has_prefix │
        │ ---    ┆ ---        │
        │ str    ┆ bool       │
        ╞════════╪════════════╡
        │ apple  ┆ true       │
        │ mango  ┆ false      │
        │ null   ┆ null       │
        └────────┴────────────┘

        Using ``starts_with`` as a filter condition:

        >>> df.filter(pl.col("fruits").str.starts_with("app"))
        shape: (1, 1)
        ┌────────┐
        │ fruits │
        │ ---    │
        │ str    │
        ╞════════╡
        │ apple  │
        └────────┘

        See Also
        --------
        contains : Check if string contains a substring that matches a regex.
        ends_with : Check if string values end with a substring.

        """
        prefix = parse_as_expression(prefix, str_as_lit=True)._pyexpr
        return wrap_expr(self._pyexpr.str_starts_with(prefix))

    def json_extract(self, dtype: PolarsDataType | None = None) -> Expr:
        """
        Parse string values as JSON.

        Throw errors if encounter invalid JSON strings.

        Parameters
        ----------
        dtype
            The dtype to cast the extracted value to. If None, the dtype will be
            inferred from the JSON value.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {"json": ['{"a":1, "b": true}', None, '{"a":2, "b": false}']}
        ... )
        >>> dtype = pl.Struct([pl.Field("a", pl.Int64), pl.Field("b", pl.Boolean)])
        >>> df.select(pl.col("json").str.json_extract(dtype))
        shape: (3, 1)
        ┌─────────────┐
        │ json        │
        │ ---         │
        │ struct[2]   │
        ╞═════════════╡
        │ {1,true}    │
        │ {null,null} │
        │ {2,false}   │
        └─────────────┘

        See Also
        --------
        json_path_match : Extract the first match of json string with provided JSONPath
            expression.

        """
        if dtype is not None:
            dtype = py_type_to_dtype(dtype)
        return wrap_expr(self._pyexpr.str_json_extract(dtype))

    def json_path_match(self, json_path: str) -> Expr:
        """
        Extract the first match of json string with provided JSONPath expression.

        Throw errors if encounter invalid json strings.
        All return value will be casted to Utf8 regardless of the original value.

        Documentation on JSONPath standard can be found
        `here <https://goessner.net/articles/JsonPath/>`_.

        Parameters
        ----------
        json_path
            A valid JSON path query string.

        Returns
        -------
        Utf8 array. Contain null if original value is null or the json_path return
        nothing.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {"json_val": ['{"a":"1"}', None, '{"a":2}', '{"a":2.1}', '{"a":true}']}
        ... )
        >>> df.select(pl.col("json_val").str.json_path_match("$.a"))
        shape: (5, 1)
        ┌──────────┐
        │ json_val │
        │ ---      │
        │ str      │
        ╞══════════╡
        │ 1        │
        │ null     │
        │ 2        │
        │ 2.1      │
        │ true     │
        └──────────┘

        """
        return wrap_expr(self._pyexpr.str_json_path_match(json_path))

    def decode(self, encoding: TransferEncoding, *, strict: bool = True) -> Expr:
        """
        Decode a value using the provided encoding.

        Parameters
        ----------
        encoding : {'hex', 'base64'}
            The encoding to use.
        strict
            Raise an error if the underlying value cannot be decoded,
            otherwise mask out with a null value.

        """
        if encoding == "hex":
            return wrap_expr(self._pyexpr.str_hex_decode(strict))
        elif encoding == "base64":
            return wrap_expr(self._pyexpr.str_base64_decode(strict))
        else:
            raise ValueError(
                f"encoding must be one of {{'hex', 'base64'}}, got {encoding}"
            )

    def encode(self, encoding: TransferEncoding) -> Expr:
        """
        Encode a value using the provided encoding.

        Parameters
        ----------
        encoding : {'hex', 'base64'}
            The encoding to use.

        Returns
        -------
        Utf8 array with values encoded using provided encoding

        Examples
        --------
        >>> df = pl.DataFrame({"strings": ["foo", "bar", None]})
        >>> df.select(pl.col("strings").str.encode("hex"))
        shape: (3, 1)
        ┌─────────┐
        │ strings │
        │ ---     │
        │ str     │
        ╞═════════╡
        │ 666f6f  │
        │ 626172  │
        │ null    │
        └─────────┘

        """
        if encoding == "hex":
            return wrap_expr(self._pyexpr.str_hex_encode())
        elif encoding == "base64":
            return wrap_expr(self._pyexpr.str_base64_encode())
        else:
            raise ValueError(
                f"encoding must be one of {{'hex', 'base64'}}, got {encoding}"
            )

    def extract(self, pattern: str, group_index: int = 1) -> Expr:
        r"""
        Extract the target capture group from provided patterns.

        Parameters
        ----------
        pattern
            A valid regular expression pattern, compatible with the `regex crate
            <https://docs.rs/regex/latest/regex/>`_.
        group_index
            Index of the targeted capture group.
            Group 0 mean the whole pattern, first group begin at index 1
            Default to the first capture group

        Notes
        -----
        To modify regular expression behaviour (such as multi-line matching)
        with flags, use the inline ``(?iLmsuxU)`` syntax. For example:

        >>> df = pl.DataFrame(
        ...     data={
        ...         "lines": [
        ...             "I Like\nThose\nOdds",
        ...             "This is\nThe Way",
        ...         ]
        ...     }
        ... )
        >>> df.select(
        ...     pl.col("lines").str.extract(r"(?m)^(T\w+)", 1).alias("matches"),
        ... )
        shape: (2, 1)
        ┌─────────┐
        │ matches │
        │ ---     │
        │ str     │
        ╞═════════╡
        │ Those   │
        │ This    │
        └─────────┘

        See the regex crate's section on `grouping and flags
        <https://docs.rs/regex/latest/regex/#grouping-and-flags>`_ for
        additional information about the use of inline expression modifiers.

        Returns
        -------
        Utf8 array. Contain null if original value is null or regex capture nothing.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "url": [
        ...             "http://vote.com/ballon_dor?error=404&ref=unknown",
        ...             "http://vote.com/ballon_dor?ref=polars&candidate=messi",
        ...             "http://vote.com/ballon_dor?candidate=ronaldo&ref=polars",
        ...         ]
        ...     }
        ... )
        >>> df.select(
        ...     pl.col("url").str.extract(r"candidate=(\w+)", 1).alias("candidate"),
        ...     pl.col("url").str.extract(r"ref=(\w+)", 1).alias("referer"),
        ...     pl.col("url").str.extract(r"error=(\w+)", 1).alias("error"),
        ... )
        shape: (3, 3)
        ┌───────────┬─────────┬───────┐
        │ candidate ┆ referer ┆ error │
        │ ---       ┆ ---     ┆ ---   │
        │ str       ┆ str     ┆ str   │
        ╞═══════════╪═════════╪═══════╡
        │ null      ┆ unknown ┆ 404   │
        │ messi     ┆ polars  ┆ null  │
        │ ronaldo   ┆ polars  ┆ null  │
        └───────────┴─────────┴───────┘

        """
        return wrap_expr(self._pyexpr.str_extract(pattern, group_index))

    def extract_all(self, pattern: str | Expr) -> Expr:
        r'''
        Extract all matches for the given regex pattern.

        Extract each successive non-overlapping regex match in an individual string
        as a list. Extracted matches contain ``null`` if the original value is null
        or the regex did not capture anything.

        Parameters
        ----------
        pattern
            A valid regular expression pattern, compatible with the `regex crate
            <https://docs.rs/regex/latest/regex/>`_.

        Notes
        -----
        To modify regular expression behaviour (such as "verbose" mode and/or
        case-sensitive matching) with flags, use the inline ``(?iLmsuxU)`` syntax.
        For example:

        >>> df = pl.DataFrame(
        ...     data={
        ...         "email": [
        ...             "real.email@spam.com",
        ...             "some_account@somewhere.net",
        ...             "abc.def.ghi.jkl@uvw.xyz.co.uk",
        ...         ]
        ...     }
        ... )
        >>> # extract name/domain parts from the addresses, using verbose regex
        >>> df.with_columns(
        ...     pl.col("email")
        ...     .str.extract_all(
        ...         r"""(?xi)   # activate 'verbose' and 'case-insensitive' flags
        ...         [           # (start character group)
        ...           A-Z       # letters
        ...           0-9       # digits
        ...           ._%+\-    # special chars
        ...         ]           # (end character group)
        ...         +           # 'one or more' quantifier
        ...         """
        ...     )
        ...     .list.to_struct(fields=["name", "domain"])
        ...     .alias("email_parts")
        ... ).unnest("email_parts")
        shape: (3, 3)
        ┌───────────────────────────────┬─────────────────┬───────────────┐
        │ email                         ┆ name            ┆ domain        │
        │ ---                           ┆ ---             ┆ ---           │
        │ str                           ┆ str             ┆ str           │
        ╞═══════════════════════════════╪═════════════════╪═══════════════╡
        │ real.email@spam.com           ┆ real.email      ┆ spam.com      │
        │ some_account@somewhere.net    ┆ some_account    ┆ somewhere.net │
        │ abc.def.ghi.jkl@uvw.xyz.co.uk ┆ abc.def.ghi.jkl ┆ uvw.xyz.co.uk │
        └───────────────────────────────┴─────────────────┴───────────────┘

        See the regex crate's section on `grouping and flags
        <https://docs.rs/regex/latest/regex/#grouping-and-flags>`_ for
        additional information about the use of inline expression modifiers.

        Returns
        -------
        List[Utf8]

        Examples
        --------
        >>> df = pl.DataFrame({"foo": ["123 bla 45 asd", "xyz 678 910t"]})
        >>> df.select(
        ...     pl.col("foo").str.extract_all(r"\d+").alias("extracted_nrs"),
        ... )
        shape: (2, 1)
        ┌────────────────┐
        │ extracted_nrs  │
        │ ---            │
        │ list[str]      │
        ╞════════════════╡
        │ ["123", "45"]  │
        │ ["678", "910"] │
        └────────────────┘

        '''
        pattern = parse_as_expression(pattern, str_as_lit=True)._pyexpr
        return wrap_expr(self._pyexpr.str_extract_all(pattern))

    def count_match(self, pattern: str) -> Expr:
        r"""
        Count all successive non-overlapping regex matches.

        Parameters
        ----------
        pattern
            A valid regular expression pattern, compatible with the `regex crate
            <https://docs.rs/regex/latest/regex/>`_.

        Returns
        -------
        UInt32 array. Contain null if original value is null or regex capture nothing.

        Examples
        --------
        >>> df = pl.DataFrame({"foo": ["123 bla 45 asd", "xyz 678 910t"]})
        >>> df.select(
        ...     pl.col("foo").str.count_match(r"\d").alias("count_digits"),
        ... )
        shape: (2, 1)
        ┌──────────────┐
        │ count_digits │
        │ ---          │
        │ u32          │
        ╞══════════════╡
        │ 5            │
        │ 6            │
        └──────────────┘

        """
        return wrap_expr(self._pyexpr.str_count_match(pattern))

    def split(self, by: str, *, inclusive: bool = False) -> Expr:
        """
        Split the string by a substring.

        Parameters
        ----------
        by
            Substring to split by.
        inclusive
            If True, include the split character/string in the results.

        Examples
        --------
        >>> df = pl.DataFrame({"s": ["foo bar", "foo-bar", "foo bar baz"]})
        >>> df.select(pl.col("s").str.split(by=" "))
        shape: (3, 1)
        ┌───────────────────────┐
        │ s                     │
        │ ---                   │
        │ list[str]             │
        ╞═══════════════════════╡
        │ ["foo", "bar"]        │
        │ ["foo-bar"]           │
        │ ["foo", "bar", "baz"] │
        └───────────────────────┘

        Returns
        -------
        List of Utf8 type

        """
        if inclusive:
            return wrap_expr(self._pyexpr.str_split_inclusive(by))
        return wrap_expr(self._pyexpr.str_split(by))

    def split_exact(self, by: str, n: int, *, inclusive: bool = False) -> Expr:
        """
        Split the string by a substring using ``n`` splits.

        Results in a struct of ``n+1`` fields.

        If it cannot make ``n`` splits, the remaining field elements will be null.

        Parameters
        ----------
        by
            Substring to split by.
        n
            Number of splits to make.
        inclusive
            If True, include the split character/string in the results.

        Examples
        --------
        >>> df = pl.DataFrame({"x": ["a_1", None, "c", "d_4"]})
        >>> df.select(
        ...     pl.col("x").str.split_exact("_", 1).alias("fields"),
        ... )
        shape: (4, 1)
        ┌─────────────┐
        │ fields      │
        │ ---         │
        │ struct[2]   │
        ╞═════════════╡
        │ {"a","1"}   │
        │ {null,null} │
        │ {"c",null}  │
        │ {"d","4"}   │
        └─────────────┘


        Split string values in column x in exactly 2 parts and assign
        each part to a new column.

        >>> df.with_columns(
        ...     [
        ...         pl.col("x")
        ...         .str.split_exact("_", 1)
        ...         .struct.rename_fields(["first_part", "second_part"])
        ...         .alias("fields"),
        ...     ]
        ... ).unnest("fields")
        shape: (4, 3)
        ┌──────┬────────────┬─────────────┐
        │ x    ┆ first_part ┆ second_part │
        │ ---  ┆ ---        ┆ ---         │
        │ str  ┆ str        ┆ str         │
        ╞══════╪════════════╪═════════════╡
        │ a_1  ┆ a          ┆ 1           │
        │ null ┆ null       ┆ null        │
        │ c    ┆ c          ┆ null        │
        │ d_4  ┆ d          ┆ 4           │
        └──────┴────────────┴─────────────┘

        Returns
        -------
        Struct of Utf8 type

        """
        if inclusive:
            return wrap_expr(self._pyexpr.str_split_exact_inclusive(by, n))
        return wrap_expr(self._pyexpr.str_split_exact(by, n))

    def splitn(self, by: str, n: int) -> Expr:
        """
        Split the string by a substring, restricted to returning at most ``n`` items.

        If the number of possible splits is less than ``n-1``, the remaining field
        elements will be null. If the number of possible splits is ``n-1`` or greater,
        the last (nth) substring will contain the remainder of the string.

        Parameters
        ----------
        by
            Substring to split by.
        n
            Max number of items to return.

        Examples
        --------
        >>> df = pl.DataFrame({"s": ["foo bar", None, "foo-bar", "foo bar baz"]})
        >>> df.select(pl.col("s").str.splitn(" ", 2).alias("fields"))
        shape: (4, 1)
        ┌───────────────────┐
        │ fields            │
        │ ---               │
        │ struct[2]         │
        ╞═══════════════════╡
        │ {"foo","bar"}     │
        │ {null,null}       │
        │ {"foo-bar",null}  │
        │ {"foo","bar baz"} │
        └───────────────────┘

        Split string values in column s in exactly 2 parts and assign
        each part to a new column.

        >>> df.with_columns(
        ...     [
        ...         pl.col("s")
        ...         .str.splitn(" ", 2)
        ...         .struct.rename_fields(["first_part", "second_part"])
        ...         .alias("fields"),
        ...     ]
        ... ).unnest("fields")
        shape: (4, 3)
        ┌─────────────┬────────────┬─────────────┐
        │ s           ┆ first_part ┆ second_part │
        │ ---         ┆ ---        ┆ ---         │
        │ str         ┆ str        ┆ str         │
        ╞═════════════╪════════════╪═════════════╡
        │ foo bar     ┆ foo        ┆ bar         │
        │ null        ┆ null       ┆ null        │
        │ foo-bar     ┆ foo-bar    ┆ null        │
        │ foo bar baz ┆ foo        ┆ bar baz     │
        └─────────────┴────────────┴─────────────┘

        Returns
        -------
        Struct of Utf8 type

        """
        return wrap_expr(self._pyexpr.str_splitn(by, n))

    def replace(
        self,
        pattern: str | Expr,
        value: str | Expr,
        *,
        literal: bool = False,
        n: int = 1,
    ) -> Expr:
        r"""
        Replace first matching regex/literal substring with a new string value.

        Parameters
        ----------
        pattern
            A valid regular expression pattern, compatible with the `regex crate
            <https://docs.rs/regex/latest/regex/>`_.
        value
            String that will replace the matched substring.
        literal
            Treat pattern as a literal string.
        n
            Number of matches to replace.

        Notes
        -----
        To modify regular expression behaviour (such as case-sensitivity) with flags,
        use the inline ``(?iLmsuxU)`` syntax. For example:

        >>> df = pl.DataFrame(
        ...     {
        ...         "city": "Philadelphia",
        ...         "season": ["Spring", "Summer", "Autumn", "Winter"],
        ...         "weather": ["Rainy", "Sunny", "Cloudy", "Snowy"],
        ...     }
        ... )
        >>> df.with_columns(
        ...     # apply case-insensitive string replacement
        ...     pl.col("weather").str.replace(r"(?i)foggy|rainy|cloudy|snowy", "Sunny")
        ... )
        shape: (4, 3)
        ┌──────────────┬────────┬─────────┐
        │ city         ┆ season ┆ weather │
        │ ---          ┆ ---    ┆ ---     │
        │ str          ┆ str    ┆ str     │
        ╞══════════════╪════════╪═════════╡
        │ Philadelphia ┆ Spring ┆ Sunny   │
        │ Philadelphia ┆ Summer ┆ Sunny   │
        │ Philadelphia ┆ Autumn ┆ Sunny   │
        │ Philadelphia ┆ Winter ┆ Sunny   │
        └──────────────┴────────┴─────────┘

        See the regex crate's section on `grouping and flags
        <https://docs.rs/regex/latest/regex/#grouping-and-flags>`_ for
        additional information about the use of inline expression modifiers.

        See Also
        --------
        replace_all : Replace all matching regex/literal substrings.

        Examples
        --------
        >>> df = pl.DataFrame({"id": [1, 2], "text": ["123abc", "abc456"]})
        >>> df.with_columns(
        ...     pl.col("text").str.replace(r"abc\b", "ABC")
        ... )  # doctest: +IGNORE_RESULT
        shape: (2, 2)
        ┌─────┬────────┐
        │ id  ┆ text   │
        │ --- ┆ ---    │
        │ i64 ┆ str    │
        ╞═════╪════════╡
        │ 1   ┆ 123ABC │
        │ 2   ┆ abc456 │
        └─────┴────────┘

        """
        pattern = parse_as_expression(pattern, str_as_lit=True)._pyexpr
        value = parse_as_expression(value, str_as_lit=True)._pyexpr
        return wrap_expr(self._pyexpr.str_replace_n(pattern, value, literal, n))

    def replace_all(
        self, pattern: str | Expr, value: str | Expr, *, literal: bool = False
    ) -> Expr:
        """
        Replace all matching regex/literal substrings with a new string value.

        Parameters
        ----------
        pattern
            A valid regular expression pattern, compatible with the `regex crate
            <https://docs.rs/regex/latest/regex/>`_.
        value
            Replacement string.
        literal
            Treat pattern as a literal string.

        See Also
        --------
        replace : Replace first matching regex/literal substring.

        Examples
        --------
        >>> df = pl.DataFrame({"id": [1, 2], "text": ["abcabc", "123a123"]})
        >>> df.with_columns(pl.col("text").str.replace_all("a", "-"))
        shape: (2, 2)
        ┌─────┬─────────┐
        │ id  ┆ text    │
        │ --- ┆ ---     │
        │ i64 ┆ str     │
        ╞═════╪═════════╡
        │ 1   ┆ -bc-bc  │
        │ 2   ┆ 123-123 │
        └─────┴─────────┘

        """
        pattern = parse_as_expression(pattern, str_as_lit=True)._pyexpr
        value = parse_as_expression(value, str_as_lit=True)._pyexpr
        return wrap_expr(self._pyexpr.str_replace_all(pattern, value, literal))

    def slice(self, offset: int, length: int | None = None) -> Expr:
        """
        Create subslices of the string values of a Utf8 Series.

        Parameters
        ----------
        offset
            Start index. Negative indexing is supported.
        length
            Length of the slice. If set to ``None`` (default), the slice is taken to the
            end of the string.

        Returns
        -------
        Expr
            Series of dtype Utf8.

        Examples
        --------
        >>> df = pl.DataFrame({"s": ["pear", None, "papaya", "dragonfruit"]})
        >>> df.with_columns(
        ...     pl.col("s").str.slice(-3).alias("s_sliced"),
        ... )
        shape: (4, 2)
        ┌─────────────┬──────────┐
        │ s           ┆ s_sliced │
        │ ---         ┆ ---      │
        │ str         ┆ str      │
        ╞═════════════╪══════════╡
        │ pear        ┆ ear      │
        │ null        ┆ null     │
        │ papaya      ┆ aya      │
        │ dragonfruit ┆ uit      │
        └─────────────┴──────────┘

        Using the optional `length` parameter

        >>> df.with_columns(
        ...     pl.col("s").str.slice(4, length=3).alias("s_sliced"),
        ... )
        shape: (4, 2)
        ┌─────────────┬──────────┐
        │ s           ┆ s_sliced │
        │ ---         ┆ ---      │
        │ str         ┆ str      │
        ╞═════════════╪══════════╡
        │ pear        ┆          │
        │ null        ┆ null     │
        │ papaya      ┆ ya       │
        │ dragonfruit ┆ onf      │
        └─────────────┴──────────┘

        """
        return wrap_expr(self._pyexpr.str_slice(offset, length))

    def explode(self) -> Expr:
        """
        Returns a column with a separate row for every string character.

        Returns
        -------
        Exploded column with string datatype.

        Examples
        --------
        >>> df = pl.DataFrame({"a": ["foo", "bar"]})
        >>> df.select(pl.col("a").str.explode())
        shape: (6, 1)
        ┌─────┐
        │ a   │
        │ --- │
        │ str │
        ╞═════╡
        │ f   │
        │ o   │
        │ o   │
        │ b   │
        │ a   │
        │ r   │
        └─────┘

        """
        return wrap_expr(self._pyexpr.str_explode())

    def parse_int(self, radix: int = 2, *, strict: bool = True) -> Expr:
        """
        Parse integers with base radix from strings.

        By default base 2. ParseError/Overflows become Nulls.

        Parameters
        ----------
        radix
            Positive integer which is the base of the string we are parsing.
            Default: 2.

        strict
            Bool, Default=True will raise any ParseError or overflow as ComputeError.
            False silently convert to Null.

        Returns
        -------
        Expr: Series of parsed integers in i32 format

        Examples
        --------
        >>> df = pl.DataFrame({"bin": ["110", "101", "010", "invalid"]})
        >>> df.select(pl.col("bin").str.parse_int(2, strict=False))
        shape: (4, 1)
        ┌──────┐
        │ bin  │
        │ ---  │
        │ i32  │
        ╞══════╡
        │ 6    │
        │ 5    │
        │ 2    │
        │ null │
        └──────┘

        >>> df = pl.DataFrame({"hex": ["fa1e", "ff00", "cafe", None]})
        >>> df.select(pl.col("hex").str.parse_int(16, strict=True))
        shape: (4, 1)
        ┌───────┐
        │ hex   │
        │ ---   │
        │ i32   │
        ╞═══════╡
        │ 64030 │
        │ 65280 │
        │ 51966 │
        │ null  │
        └───────┘

        """
        return wrap_expr(self._pyexpr.str_parse_int(radix, strict))


def _validate_format_argument(format: str | None) -> None:
    if format is not None and ".%f" in format:
        message = (
            "Detected the pattern `.%f` in the chrono format string."
            " This pattern should not be used to parse values after a decimal point."
            " Use `%.f` instead."
            " See the full specification: https://docs.rs/chrono/latest/chrono/format/strftime"
        )
        warnings.warn(
            message, category=ChronoFormatWarning, stacklevel=find_stacklevel()
        )
