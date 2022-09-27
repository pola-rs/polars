from __future__ import annotations

from typing import TYPE_CHECKING

import polars.internals as pli
from polars.datatypes import (
    DataType,
    Date,
    Datetime,
    TemporalDataType,
    Time,
    is_polars_dtype,
)
from polars.utils import deprecated_alias

if TYPE_CHECKING:
    from polars.internals.type_aliases import TransferEncoding


class ExprStringNameSpace:
    """Namespace for string related expressions."""

    _accessor = "str"

    def __init__(self, expr: pli.Expr):
        self._pyexpr = expr._pyexpr

    def strptime(
        self,
        datatype: TemporalDataType,
        fmt: str | None = None,
        strict: bool = True,
        exact: bool = True,
    ) -> pli.Expr:
        """
        Parse a Utf8 expression to a Date/Datetime/Time type.

        Parameters
        ----------
        datatype
            Date | Datetime | Time
        fmt
            Format to use, refer to the `chrono strftime documentation
            <https://docs.rs/chrono/latest/chrono/format/strftime/index.html>`_
            for specification. Example: ``"%y-%m-%d"``.
        strict
            Raise an error if any conversion fails.
        exact
            - If True, require an exact format match.
            - If False, allow the format to match anywhere in the target string.

        Notes
        -----
        When parsing a Datetime the column precision will be inferred from
        the format string, if given, eg: "%F %T%.3f" => Datetime("ms"). If
        no fractional second component is found then the default is "us".

        Examples
        --------
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
        >>> (
        ...     s.to_frame().with_column(
        ...         pl.col("date")
        ...         .str.strptime(pl.Date, "%F", strict=False)
        ...         .fill_null(
        ...             pl.col("date").str.strptime(pl.Date, "%F %T", strict=False)
        ...         )
        ...         .fill_null(pl.col("date").str.strptime(pl.Date, "%D", strict=False))
        ...         .fill_null(pl.col("date").str.strptime(pl.Date, "%c", strict=False))
        ...     )
        ... )
        shape: (4, 1)
        ┌────────────┐
        │ date       │
        │ ---        │
        │ date       │
        ╞════════════╡
        │ 2021-04-22 │
        ├╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 2022-01-04 │
        ├╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 2022-01-31 │
        ├╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 2001-07-08 │
        └────────────┘

        """
        if not is_polars_dtype(datatype):  # pragma: no cover
            raise ValueError(f"expected: {DataType} got: {datatype}")

        if datatype == Date:
            return pli.wrap_expr(self._pyexpr.str_parse_date(fmt, strict, exact))
        elif datatype == Datetime:
            tu = datatype.tu  # type: ignore[union-attr]
            dtcol = pli.wrap_expr(self._pyexpr.str_parse_datetime(fmt, strict, exact))
            return dtcol if (tu is None) else dtcol.dt.cast_time_unit(tu)
        elif datatype == Time:
            return pli.wrap_expr(self._pyexpr.str_parse_time(fmt, strict, exact))
        else:  # pragma: no cover
            raise ValueError("dtype should be of type {Date, Datetime, Time}")

    def lengths(self) -> pli.Expr:
        """
        Get the length of the Strings as UInt32.

        Examples
        --------
        >>> df = pl.DataFrame({"s": [None, "bears", "110"]})
        >>> df.select(["s", pl.col("s").str.lengths().alias("len")])
        shape: (3, 2)
        ┌───────┬──────┐
        │ s     ┆ len  │
        │ ---   ┆ ---  │
        │ str   ┆ u32  │
        ╞═══════╪══════╡
        │ null  ┆ null │
        ├╌╌╌╌╌╌╌┼╌╌╌╌╌╌┤
        │ bears ┆ 5    │
        ├╌╌╌╌╌╌╌┼╌╌╌╌╌╌┤
        │ 110   ┆ 3    │
        └───────┴──────┘

        """
        return pli.wrap_expr(self._pyexpr.str_lengths())

    def concat(self, delimiter: str = "-") -> pli.Expr:
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
        return pli.wrap_expr(self._pyexpr.str_concat(delimiter))

    def to_uppercase(self) -> pli.Expr:
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
        ├╌╌╌╌╌┤
        │ DOG │
        └─────┘

        """
        return pli.wrap_expr(self._pyexpr.str_to_uppercase())

    def to_lowercase(self) -> pli.Expr:
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
        ├╌╌╌╌╌┤
        │ dog │
        └─────┘

        """
        return pli.wrap_expr(self._pyexpr.str_to_lowercase())

    def strip(self, matches: None | str = None) -> pli.Expr:
        """
        Remove leading and trailing whitespace.

        Parameters
        ----------
        matches
            An optional single character that should be trimmed

        Examples
        --------
        >>> df = pl.DataFrame({"foo": [" lead", "trail ", " both "]})
        >>> df.select(pl.col("foo").str.strip())
        shape: (3, 1)
        ┌───────┐
        │ foo   │
        │ ---   │
        │ str   │
        ╞═══════╡
        │ lead  │
        ├╌╌╌╌╌╌╌┤
        │ trail │
        ├╌╌╌╌╌╌╌┤
        │ both  │
        └───────┘

        """
        if matches is not None and len(matches) > 1:
            raise ValueError("matches should contain a single character")
        return pli.wrap_expr(self._pyexpr.str_strip(matches))

    def lstrip(self, matches: None | str = None) -> pli.Expr:
        """
        Remove leading whitespace.

        Examples
        --------
        >>> df = pl.DataFrame({"foo": [" lead", "trail ", " both "]})
        >>> df.select(pl.col("foo").str.lstrip())
        shape: (3, 1)
        ┌────────┐
        │ foo    │
        │ ---    │
        │ str    │
        ╞════════╡
        │ lead   │
        ├╌╌╌╌╌╌╌╌┤
        │ trail  │
        ├╌╌╌╌╌╌╌╌┤
        │ both   │
        └────────┘

        """
        if matches is not None and len(matches) > 1:
            raise ValueError("matches should contain a single character")
        return pli.wrap_expr(self._pyexpr.str_lstrip(matches))

    def rstrip(self, matches: None | str = None) -> pli.Expr:
        """
        Remove trailing whitespace.

        Parameters
        ----------
        matches
            An optional single character that should be trimmed

        Examples
        --------
        >>> df = pl.DataFrame({"foo": [" lead", "trail ", " both "]})
        >>> df.select(pl.col("foo").str.rstrip())
        shape: (3, 1)
        ┌───────┐
        │ foo   │
        │ ---   │
        │ str   │
        ╞═══════╡
        │  lead │
        ├╌╌╌╌╌╌╌┤
        │ trail │
        ├╌╌╌╌╌╌╌┤
        │  both │
        └───────┘

        """
        if matches is not None and len(matches) > 1:
            raise ValueError("matches should contain a single character")
        return pli.wrap_expr(self._pyexpr.str_rstrip(matches))

    def zfill(self, alignment: int) -> pli.Expr:
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
        >>> df.with_column(pl.col("num").cast(str).str.zfill(5))
        shape: (11, 1)
        ┌─────────┐
        │ num     │
        │ ---     │
        │ str     │
        ╞═════════╡
        │ -0010   │
        ├╌╌╌╌╌╌╌╌╌┤
        │ -0001   │
        ├╌╌╌╌╌╌╌╌╌┤
        │ 00000   │
        ├╌╌╌╌╌╌╌╌╌┤
        │ 00001   │
        ├╌╌╌╌╌╌╌╌╌┤
        │ ...     │
        ├╌╌╌╌╌╌╌╌╌┤
        │ 10000   │
        ├╌╌╌╌╌╌╌╌╌┤
        │ 100000  │
        ├╌╌╌╌╌╌╌╌╌┤
        │ 1000000 │
        ├╌╌╌╌╌╌╌╌╌┤
        │ null    │
        └─────────┘

        """
        return pli.wrap_expr(self._pyexpr.str_zfill(alignment))

    def ljust(self, width: int, fillchar: str = " ") -> pli.Expr:
        """
        Return the string left justified in a string of length ``width``.

        Padding is done using the specified ``fillchar``.
        The original string is returned if ``width`` is less than or equal to
        ``len(s)``.

        Parameters
        ----------
        width
            Justify left to this length.
        fillchar
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
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ monkey**     │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ null         │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ hippopotamus │
        └──────────────┘

        """
        return pli.wrap_expr(self._pyexpr.str_ljust(width, fillchar))

    def rjust(self, width: int, fillchar: str = " ") -> pli.Expr:
        """
        Return the string right justified in a string of length ``width``.

        Padding is done using the specified ``fillchar``.
        The original string is returned if ``width`` is less than or equal to
        ``len(s)``.

        Parameters
        ----------
        width
            Justify right to this length.
        fillchar
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
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ **monkey     │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ null         │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ hippopotamus │
        └──────────────┘

        """
        return pli.wrap_expr(self._pyexpr.str_rjust(width, fillchar))

    def contains(self, pattern: str, literal: bool = False) -> pli.Expr:
        """
        Check if string contains a substring that matches a regex.

        Parameters
        ----------
        pattern
            A valid regex pattern.
        literal
            Treat pattern as a literal string.

        Examples
        --------
        >>> df = pl.DataFrame({"a": ["Crab", "cat and dog", "rab$bit", None]})
        >>> df.select(
        ...     [
        ...         pl.col("a"),
        ...         pl.col("a").str.contains("cat|bit").alias("regex"),
        ...         pl.col("a").str.contains("rab$", literal=True).alias("literal"),
        ...     ]
        ... )
        shape: (4, 3)
        ┌─────────────┬───────┬─────────┐
        │ a           ┆ regex ┆ literal │
        │ ---         ┆ ---   ┆ ---     │
        │ str         ┆ bool  ┆ bool    │
        ╞═════════════╪═══════╪═════════╡
        │ Crab        ┆ false ┆ false   │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌┤
        │ cat and dog ┆ true  ┆ false   │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌┤
        │ rab$bit     ┆ true  ┆ true    │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌┤
        │ null        ┆ null  ┆ null    │
        └─────────────┴───────┴─────────┘

        See Also
        --------
        starts_with : Check if string values start with a substring.
        ends_with : Check if string values end with a substring.

        """
        return pli.wrap_expr(self._pyexpr.str_contains(pattern, literal))

    def ends_with(self, sub: str) -> pli.Expr:
        """
        Check if string values end with a substring.

        Parameters
        ----------
        sub
            Suffix substring.

        Examples
        --------
        >>> df = pl.DataFrame({"fruits": ["apple", "mango", None]})
        >>> df.with_column(
        ...     pl.col("fruits").str.ends_with("go").alias("has_suffix"),
        ... )
        shape: (3, 2)
        ┌────────┬────────────┐
        │ fruits ┆ has_suffix │
        │ ---    ┆ ---        │
        │ str    ┆ bool       │
        ╞════════╪════════════╡
        │ apple  ┆ false      │
        ├╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ mango  ┆ true       │
        ├╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┤
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
        return pli.wrap_expr(self._pyexpr.str_ends_with(sub))

    def starts_with(self, sub: str) -> pli.Expr:
        """
        Check if string values start with a substring.

        Parameters
        ----------
        sub
            Prefix substring.

        Examples
        --------
        >>> df = pl.DataFrame({"fruits": ["apple", "mango", None]})
        >>> df.with_column(
        ...     pl.col("fruits").str.starts_with("app").alias("has_prefix"),
        ... )
        shape: (3, 2)
        ┌────────┬────────────┐
        │ fruits ┆ has_prefix │
        │ ---    ┆ ---        │
        │ str    ┆ bool       │
        ╞════════╪════════════╡
        │ apple  ┆ true       │
        ├╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ mango  ┆ false      │
        ├╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┤
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
        return pli.wrap_expr(self._pyexpr.str_starts_with(sub))

    def json_path_match(self, json_path: str) -> pli.Expr:
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
        ├╌╌╌╌╌╌╌╌╌╌┤
        │ null     │
        ├╌╌╌╌╌╌╌╌╌╌┤
        │ 2        │
        ├╌╌╌╌╌╌╌╌╌╌┤
        │ 2.1      │
        ├╌╌╌╌╌╌╌╌╌╌┤
        │ true     │
        └──────────┘

        """
        return pli.wrap_expr(self._pyexpr.str_json_path_match(json_path))

    def decode(self, encoding: TransferEncoding, strict: bool = False) -> pli.Expr:
        """
        Decode a value using the provided encoding.

        Parameters
        ----------
        encoding : {'hex', 'base64'}
            The encoding to use.
        strict
            How to handle invalid inputs:

            - ``True``: An error will be thrown if unable to decode a value.
            - ``False``: Unhandled values will be replaced with `None`.

        Examples
        --------
        >>> df = pl.DataFrame({"encoded": ["666f6f", "626172", None]})
        >>> df.select(pl.col("encoded").str.decode("hex"))
        shape: (3, 1)
        ┌─────────┐
        │ encoded │
        │ ---     │
        │ str     │
        ╞═════════╡
        │ foo     │
        ├╌╌╌╌╌╌╌╌╌┤
        │ bar     │
        ├╌╌╌╌╌╌╌╌╌┤
        │ null    │
        └─────────┘

        """
        if encoding == "hex":
            return pli.wrap_expr(self._pyexpr.str_hex_decode(strict))
        elif encoding == "base64":
            return pli.wrap_expr(self._pyexpr.str_base64_decode(strict))
        else:
            raise ValueError(
                f"encoding must be one of {{'hex', 'base64'}}, got {encoding}"
            )

    def encode(self, encoding: TransferEncoding) -> pli.Expr:
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
        ├╌╌╌╌╌╌╌╌╌┤
        │ 626172  │
        ├╌╌╌╌╌╌╌╌╌┤
        │ null    │
        └─────────┘

        """
        if encoding == "hex":
            return pli.wrap_expr(self._pyexpr.str_hex_encode())
        elif encoding == "base64":
            return pli.wrap_expr(self._pyexpr.str_base64_encode())
        else:
            raise ValueError(
                f"encoding must be one of {{'hex', 'base64'}}, got {encoding}"
            )

    def extract(self, pattern: str, group_index: int = 1) -> pli.Expr:
        r"""
        Extract the target capture group from provided patterns.

        Parameters
        ----------
        pattern
            A valid regex pattern
        group_index
            Index of the targeted capture group.
            Group 0 mean the whole pattern, first group begin at index 1
            Default to the first capture group

        Returns
        -------
        Utf8 array. Contain null if original value is null or regex capture nothing.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "a": [
        ...             "http://vote.com/ballon_dor?candidate=messi&ref=polars",
        ...             "http://vote.com/ballon_dor?candidat=jorginho&ref=polars",
        ...             "http://vote.com/ballon_dor?candidate=ronaldo&ref=polars",
        ...         ]
        ...     }
        ... )
        >>> df.select(
        ...     [
        ...         pl.col("a").str.extract(r"candidate=(\w+)", 1),
        ...     ]
        ... )
        shape: (3, 1)
        ┌─────────┐
        │ a       │
        │ ---     │
        │ str     │
        ╞═════════╡
        │ messi   │
        ├╌╌╌╌╌╌╌╌╌┤
        │ null    │
        ├╌╌╌╌╌╌╌╌╌┤
        │ ronaldo │
        └─────────┘

        """
        return pli.wrap_expr(self._pyexpr.str_extract(pattern, group_index))

    def extract_all(self, pattern: str) -> pli.Expr:
        r"""
        Extracts all matches for the given regex pattern.

        Extracts each successive non-overlapping regex match in an individual string as
        an array.

        Parameters
        ----------
        pattern
            A valid regex pattern

        Returns
        -------
        List[Utf8] array. Contain null if original value is null or regex capture
        nothing.

        Examples
        --------
        >>> df = pl.DataFrame({"foo": ["123 bla 45 asd", "xyz 678 910t"]})
        >>> df.select(
        ...     [
        ...         pl.col("foo").str.extract_all(r"(\d+)").alias("extracted_nrs"),
        ...     ]
        ... )
        shape: (2, 1)
        ┌────────────────┐
        │ extracted_nrs  │
        │ ---            │
        │ list[str]      │
        ╞════════════════╡
        │ ["123", "45"]  │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ ["678", "910"] │
        └────────────────┘

        """
        return pli.wrap_expr(self._pyexpr.str_extract_all(pattern))

    def count_match(self, pattern: str) -> pli.Expr:
        r"""
        Count all successive non-overlapping regex matches.

        Parameters
        ----------
        pattern
            A valid regex pattern

        Returns
        -------
        UInt32 array. Contain null if original value is null or regex capture nothing.

        Examples
        --------
        >>> df = pl.DataFrame({"foo": ["123 bla 45 asd", "xyz 678 910t"]})
        >>> df.select(
        ...     [
        ...         pl.col("foo").str.count_match(r"\d").alias("count_digits"),
        ...     ]
        ... )
        shape: (2, 1)
        ┌──────────────┐
        │ count_digits │
        │ ---          │
        │ u32          │
        ╞══════════════╡
        │ 5            │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 6            │
        └──────────────┘

        """
        return pli.wrap_expr(self._pyexpr.count_match(pattern))

    def split(self, by: str, inclusive: bool = False) -> pli.Expr:
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
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ ["foo-bar"]           │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ ["foo", "bar", "baz"] │
        └───────────────────────┘

        Returns
        -------
        List of Utf8 type

        """
        if inclusive:
            return pli.wrap_expr(self._pyexpr.str_split_inclusive(by))
        return pli.wrap_expr(self._pyexpr.str_split(by))

    def split_exact(self, by: str, n: int, inclusive: bool = False) -> pli.Expr:
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
        ...     [
        ...         pl.col("x").str.split_exact("_", 1).alias("fields"),
        ...     ]
        ... )
        shape: (4, 1)
        ┌─────────────┐
        │ fields      │
        │ ---         │
        │ struct[2]   │
        ╞═════════════╡
        │ {"a","1"}   │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ {null,null} │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ {"c",null}  │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌┤
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
        ├╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ null ┆ null       ┆ null        │
        ├╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ c    ┆ c          ┆ null        │
        ├╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ d_4  ┆ d          ┆ 4           │
        └──────┴────────────┴─────────────┘

        Returns
        -------
        Struct of Utf8 type

        """
        if inclusive:
            return pli.wrap_expr(self._pyexpr.str_split_exact_inclusive(by, n))
        return pli.wrap_expr(self._pyexpr.str_split_exact(by, n))

    def splitn(self, by: str, n: int) -> pli.Expr:
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
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ {null,null}       │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ {"foo-bar",null}  │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
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
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ null        ┆ null       ┆ null        │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ foo-bar     ┆ foo-bar    ┆ null        │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ foo bar baz ┆ foo        ┆ bar baz     │
        └─────────────┴────────────┴─────────────┘

        Returns
        -------
        Struct of Utf8 type

        """
        return pli.wrap_expr(self._pyexpr.str_splitn(by, n))

    def replace(
        self, pattern: str | pli.Expr, value: str | pli.Expr, literal: bool = False
    ) -> pli.Expr:
        r"""
        Replace first matching regex/literal substring with a new string value.

        Parameters
        ----------
        pattern
            Regex pattern.
        value
            Replacement string.
        literal
             Treat pattern as a literal string.

        See Also
        --------
        replace_all : Replace all matching regex/literal substrings.

        Examples
        --------
        >>> df = pl.DataFrame({"id": [1, 2], "text": ["123abc", "abc456"]})
        >>> df.with_column(
        ...     pl.col("text").str.replace(r"abc\b", "ABC")
        ... )  # doctest: +IGNORE_RESULT
        shape: (2, 2)
        ┌─────┬────────┐
        │ id  ┆ text   │
        │ --- ┆ ---    │
        │ i64 ┆ str    │
        ╞═════╪════════╡
        │ 1   ┆ 123ABC │
        ├╌╌╌╌╌┼╌╌╌╌╌╌╌╌┤
        │ 2   ┆ abc456 │
        └─────┴────────┘

        """
        pattern = pli.expr_to_lit_or_expr(pattern, str_to_lit=True)
        value = pli.expr_to_lit_or_expr(value, str_to_lit=True)
        return pli.wrap_expr(
            self._pyexpr.str_replace(pattern._pyexpr, value._pyexpr, literal)
        )

    def replace_all(
        self, pattern: str | pli.Expr, value: str | pli.Expr, literal: bool = False
    ) -> pli.Expr:
        """
        Replace all matching regex/literal substrings with a new string value.

        Parameters
        ----------
        pattern
            Regex pattern.
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
        >>> df.with_column(pl.col("text").str.replace_all("a", "-"))
        shape: (2, 2)
        ┌─────┬─────────┐
        │ id  ┆ text    │
        │ --- ┆ ---     │
        │ i64 ┆ str     │
        ╞═════╪═════════╡
        │ 1   ┆ -bc-bc  │
        ├╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌┤
        │ 2   ┆ 123-123 │
        └─────┴─────────┘

        """
        pattern = pli.expr_to_lit_or_expr(pattern, str_to_lit=True)
        value = pli.expr_to_lit_or_expr(value, str_to_lit=True)
        return pli.wrap_expr(
            self._pyexpr.str_replace_all(pattern._pyexpr, value._pyexpr, literal)
        )

    @deprecated_alias(start="offset")
    def slice(self, offset: int, length: int | None = None) -> pli.Expr:
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
        >>> df.with_column(
        ...     pl.col("s").str.slice(-3).alias("s_sliced"),
        ... )
        shape: (4, 2)
        ┌─────────────┬──────────┐
        │ s           ┆ s_sliced │
        │ ---         ┆ ---      │
        │ str         ┆ str      │
        ╞═════════════╪══════════╡
        │ pear        ┆ ear      │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌┤
        │ null        ┆ null     │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌┤
        │ papaya      ┆ aya      │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌┤
        │ dragonfruit ┆ uit      │
        └─────────────┴──────────┘

        Using the optional `length` parameter

        >>> df.with_column(
        ...     pl.col("s").str.slice(4, length=3).alias("s_sliced"),
        ... )
        shape: (4, 2)
        ┌─────────────┬──────────┐
        │ s           ┆ s_sliced │
        │ ---         ┆ ---      │
        │ str         ┆ str      │
        ╞═════════════╪══════════╡
        │ pear        ┆          │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌┤
        │ null        ┆ null     │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌┤
        │ papaya      ┆ ya       │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌┤
        │ dragonfruit ┆ onf      │
        └─────────────┴──────────┘

        """
        return pli.wrap_expr(self._pyexpr.str_slice(offset, length))
