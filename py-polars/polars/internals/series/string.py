from __future__ import annotations

from typing import TYPE_CHECKING

import polars.internals as pli
from polars.datatypes import PolarsTemporalType
from polars.internals.series.utils import expr_dispatch
from polars.utils import deprecated_alias

if TYPE_CHECKING:
    from polars.internals.type_aliases import TransferEncoding
    from polars.polars import PySeries


@expr_dispatch
class StringNameSpace:
    """Series.str namespace."""

    _accessor = "str"

    def __init__(self, series: pli.Series):
        self._s: PySeries = series._s

    def strptime(
        self,
        datatype: PolarsTemporalType,
        fmt: str | None = None,
        strict: bool = True,
        exact: bool = True,
        cache: bool = True,
        tz_aware: bool = False,
    ) -> pli.Series:
        """
        Parse a Series of dtype Utf8 to a Date/Datetime Series.

        Parameters
        ----------
        datatype
            Date, Datetime or Time.
        fmt
            Format to use, refer to the
            `chrono strftime documentation
            <https://docs.rs/chrono/latest/chrono/format/strftime/index.html>`_
            for specification. Example: ``"%y-%m-%d"``.
        strict
            Raise an error if any conversion fails.
        exact
            - If True, require an exact format match.
            - If False, allow the format to match anywhere in the target string.
        cache
            Use a cache of unique, converted dates to apply the datetime conversion.
        tz_aware
            Parse timezone aware datetimes. This may be automatically toggled by the
            'fmt' given.

        Returns
        -------
        A Date / Datetime / Time Series

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
        │ 2022-01-04 │
        │ 2022-01-31 │
        │ 2001-07-08 │
        └────────────┘

        """

    def lengths(self) -> pli.Series:
        """
        Get length of the string values in the Series (as number of bytes).

        Notes
        -----
        The returned lengths are equal to the number of bytes in the UTF8 string. If you
        need the length in terms of the number of characters, use ``n_chars`` instead.

        Returns
        -------
        Series[u32]

        Examples
        --------
        >>> s = pl.Series(["Café", None, "345", "東京"])
        >>> s.str.lengths()
        shape: (4,)
        Series: '' [u32]
        [
            5
            null
            3
            6
        ]

        """

    def n_chars(self) -> pli.Series:
        """
        Get length of the string values in the Series (as number of chars).

        Returns
        -------
        Series[u32]

        Notes
        -----
        If you know that you are working with ASCII text, ``lengths`` will be
        equivalent, and faster (returns length in terms of the number of bytes).

        Examples
        --------
        >>> s = pl.Series(["Café", None, "345", "東京"])
        >>> s.str.n_chars()
        shape: (4,)
        Series: '' [u32]
        [
            4
            null
            3
            2
        ]

        """

    def concat(self, delimiter: str = "-") -> pli.Series:
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
        >>> pl.Series([1, None, 2]).str.concat("-")[0]
        '1-null-2'

        """

    def contains(self, pattern: str, literal: bool = False) -> pli.Series:
        """
        Check if strings in Series contain a substring that matches a regex.

        Parameters
        ----------
        pattern
            A valid regex pattern.
        literal
            Treat pattern as a literal string.

        Returns
        -------
        Boolean mask

        Examples
        --------
        >>> s = pl.Series(["Crab", "cat and dog", "rab$bit", None])
        >>> s.str.contains("cat|bit")
        shape: (4,)
        Series: '' [bool]
        [
            false
            true
            true
            null
        ]
        >>> s.str.contains("rab$", literal=True)
        shape: (4,)
        Series: '' [bool]
        [
            false
            false
            true
            null
        ]

        """

    def ends_with(self, sub: str) -> pli.Series:
        """
        Check if string values end with a substring.

        Parameters
        ----------
        sub
            Suffix substring.

        Examples
        --------
        >>> s = pl.Series("fruits", ["apple", "mango", None])
        >>> s.str.ends_with("go")
        shape: (3,)
        Series: 'fruits' [bool]
        [
            false
            true
            null
        ]

        See Also
        --------
        contains : Check if string contains a substring that matches a regex.
        starts_with : Check if string values start with a substring.

        """

    def starts_with(self, sub: str) -> pli.Series:
        """
        Check if string values start with a substring.

        Parameters
        ----------
        sub
            Prefix substring.

        Examples
        --------
        >>> s = pl.Series("fruits", ["apple", "mango", None])
        >>> s.str.starts_with("app")
        shape: (3,)
        Series: 'fruits' [bool]
        [
            true
            false
            null
        ]

        See Also
        --------
        contains : Check if string contains a substring that matches a regex.
        ends_with : Check if string values end with a substring.

        """

    def decode(self, encoding: TransferEncoding, *, strict: bool = True) -> pli.Series:
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

    def encode(self, encoding: TransferEncoding) -> pli.Series:
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
        >>> s = pl.Series(["foo", "bar", None])
        >>> s.str.encode("hex")
        shape: (3,)
        Series: '' [str]
        [
            "666f6f"
            "626172"
            null
        ]

        """

    def json_path_match(self, json_path: str) -> pli.Series:
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
        >>> df.select(pl.col("json_val").str.json_path_match("$.a"))[:, 0]
        shape: (5,)
        Series: 'json_val' [str]
        [
            "1"
            null
            "2"
            "2.1"
            "true"
        ]

        """

    def extract(self, pattern: str, group_index: int = 1) -> pli.Series:
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
        >>> df.select([pl.col("a").str.extract(r"candidate=(\w+)", 1)])
        shape: (3, 1)
        ┌─────────┐
        │ a       │
        │ ---     │
        │ str     │
        ╞═════════╡
        │ messi   │
        │ null    │
        │ ronaldo │
        └─────────┘

        """

    def extract_all(self, pattern: str | pli.Series) -> pli.Series:
        r"""
        Extracts all matches for the given regex pattern.

        Extract each successive non-overlapping regex match in an individual string as
        an array

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
        >>> s = pl.Series("foo", ["123 bla 45 asd", "xyz 678 910t"])
        >>> s.str.extract_all(r"(\d+)")
        shape: (2,)
        Series: 'foo' [list[str]]
        [
            ["123", "45"]
            ["678", "910"]
        ]

        """

    def count_match(self, pattern: str) -> pli.Series:
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
        >>> s = pl.Series("foo", ["123 bla 45 asd", "xyz 678 910t"])
        >>> # count digits
        >>> s.str.count_match(r"\d")
        shape: (2,)
        Series: 'foo' [u32]
        [
            5
            6
        ]

        """

    def split(self, by: str, inclusive: bool = False) -> pli.Series:
        """
        Split the string by a substring.

        Parameters
        ----------
        by
            Substring to split by.
        inclusive
            If True, include the split character/string in the results.

        Returns
        -------
        List of Utf8 type

        """

    def split_exact(self, by: str, n: int, inclusive: bool = False) -> pli.Series:
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
        >>> df["x"].str.split_exact("_", 1).alias("fields")
        shape: (4,)
        Series: 'fields' [struct[2]]
        [
                {"a","1"}
                {null,null}
                {"c",null}
                {"d","4"}
        ]

        Split string values in column x in exactly 2 parts and assign
        each part to a new column.

        >>> (
        ...     df["x"]
        ...     .str.split_exact("_", 1)
        ...     .struct.rename_fields(["first_part", "second_part"])
        ...     .alias("fields")
        ...     .to_frame()
        ...     .unnest("fields")
        ... )
        shape: (4, 2)
        ┌────────────┬─────────────┐
        │ first_part ┆ second_part │
        │ ---        ┆ ---         │
        │ str        ┆ str         │
        ╞════════════╪═════════════╡
        │ a          ┆ 1           │
        │ null       ┆ null        │
        │ c          ┆ null        │
        │ d          ┆ 4           │
        └────────────┴─────────────┘

        Returns
        -------
        Struct of Utf8 type

        """

    def splitn(self, by: str, n: int) -> pli.Series:
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
        >>> df["s"].str.splitn(" ", 2).alias("fields")
        shape: (4,)
        Series: 'fields' [struct[2]]
        [
                {"foo","bar"}
                {null,null}
                {"foo-bar",null}
                {"foo","bar baz"}
        ]

        Split string values in column s in exactly 2 parts and assign
        each part to a new column.

        >>> (
        ...     df["s"]
        ...     .str.splitn(" ", 2)
        ...     .struct.rename_fields(["first_part", "second_part"])
        ...     .alias("fields")
        ...     .to_frame()
        ...     .unnest("fields")
        ... )
        shape: (4, 2)
        ┌────────────┬─────────────┐
        │ first_part ┆ second_part │
        │ ---        ┆ ---         │
        │ str        ┆ str         │
        ╞════════════╪═════════════╡
        │ foo        ┆ bar         │
        │ null       ┆ null        │
        │ foo-bar    ┆ null        │
        │ foo        ┆ bar baz     │
        └────────────┴─────────────┘

        Returns
        -------
        Struct of Utf8 type

        """
        s = pli.wrap_s(self._s)
        return s.to_frame().select(pli.col(s.name).str.splitn(by, n)).to_series()

    def replace(self, pattern: str, value: str, literal: bool = False) -> pli.Series:
        r"""
        Replace first matching regex/literal substring with a new string value.

        Parameters
        ----------
        pattern
            A valid regex pattern.
        value
            Substring to replace.
        literal
             Treat pattern as a literal string.

        See Also
        --------
        replace_all : Replace all matching regex/literal substrings.

        Examples
        --------
        >>> s = pl.Series(["123abc", "abc456"])
        >>> s.str.replace(r"abc\b", "ABC")  # doctest: +IGNORE_RESULT
        shape: (2,)
        Series: '' [str]
        [
            "123ABC"
            "abc456"
        ]

        """

    def replace_all(
        self, pattern: str, value: str, literal: bool = False
    ) -> pli.Series:
        """
        Replace all matching regex/literal substrings with a new string value.

        Parameters
        ----------
        pattern
            A valid regex pattern.
        value
            Substring to replace.
        literal
             Treat pattern as a literal string.

        See Also
        --------
        replace : Replace first matching regex/literal substring.

        Examples
        --------
        >>> df = pl.Series(["abcabc", "123a123"])
        >>> df.str.replace_all("a", "-")
        shape: (2,)
        Series: '' [str]
        [
            "-bc-bc"
            "123-123"
        ]

        """

    def strip(self, matches: str | None = None) -> pli.Series:
        r"""
        Remove leading and trailing characters.

        Parameters
        ----------
        matches
            The set of characters to be removed. All combinations of this set of
            characters will be stripped. If set to None (default), all whitespace is
            removed instead.

        Examples
        --------
        >>> s = pl.Series([" hello ", "\tworld"])
        >>> s.str.strip()
        shape: (2,)
        Series: '' [str]
        [
                "hello"
                "world"
        ]

        Characters can be stripped by passing a string as argument. Note that whitespace
        will not be stripped automatically when doing so.

        >>> s.str.strip("od\t")
        shape: (2,)
        Series: '' [str]
        [
                " hello "
                "worl"
        ]

        """

    def lstrip(self, matches: str | None = None) -> pli.Series:
        r"""
        Remove leading characters.

        Parameters
        ----------
        matches
            The set of characters to be removed. All combinations of this set of
            characters will be stripped. If set to None (default), all whitespace is
            removed instead.

        Examples
        --------
        >>> s = pl.Series([" hello ", "\tworld"])
        >>> s.str.lstrip()
        shape: (2,)
        Series: '' [str]
        [
                "hello "
                "world"
        ]

        Characters can be stripped by passing a string as argument. Note that whitespace
        will not be stripped automatically when doing so.

        >>> s.str.lstrip("wod\t")
        shape: (2,)
        Series: '' [str]
        [
                " hello "
                "rld"
        ]

        """

    def rstrip(self, matches: str | None = None) -> pli.Series:
        r"""
        Remove trailing characters.

        Parameters
        ----------
        matches
            The set of characters to be removed. All combinations of this set of
            characters will be stripped. If set to None (default), all whitespace is
            removed instead.

        Examples
        --------
        >>> s = pl.Series([" hello ", "world\t"])
        >>> s.str.rstrip()
        shape: (2,)
        Series: '' [str]
        [
                " hello"
                "world"
        ]

        Characters can be stripped by passing a string as argument. Note that whitespace
        will not be stripped automatically when doing so.

        >>> s.str.rstrip("wod\t")
        shape: (2,)
        Series: '' [str]
        [
                " hello "
                "worl"
        ]

        """

    def zfill(self, alignment: int) -> pli.Series:
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
            Fill the value up to this length.

        """

    def ljust(self, width: int, fillchar: str = " ") -> pli.Series:
        """
        Return the string left justified in a string of length ``width``.

        Padding is done using the specified ``fillchar``. The original string is
        returned if ``width`` is less than or equal to``len(s)``.

        Parameters
        ----------
        width
            Justify left to this length.
        fillchar
            Fill with this ASCII character.

        Examples
        --------
        >>> s = pl.Series("a", ["cow", "monkey", None, "hippopotamus"])
        >>> s.str.ljust(8, "*")
        shape: (4,)
        Series: 'a' [str]
        [
            "cow*****"
            "monkey**"
            null
            "hippopotamus"
        ]

        """

    def rjust(self, width: int, fillchar: str = " ") -> pli.Series:
        """
        Return the string right justified in a string of length ``width``.

        Padding is done using the specified ``fillchar``. The original string is
        returned if ``width`` is less than or equal to ``len(s)``.

        Parameters
        ----------
        width
            Justify right to this length.
        fillchar
            Fill with this ASCII character.

        Examples
        --------
        >>> s = pl.Series("a", ["cow", "monkey", None, "hippopotamus"])
        >>> s.str.rjust(8, "*")
        shape: (4,)
        Series: 'a' [str]
        [
            "*****cow"
            "**monkey"
            null
            "hippopotamus"
        ]

        """

    def to_lowercase(self) -> pli.Series:
        """Modify the strings to their lowercase equivalent."""

    def to_uppercase(self) -> pli.Series:
        """Modify the strings to their uppercase equivalent."""

    @deprecated_alias(start="offset")
    def slice(self, offset: int, length: int | None = None) -> pli.Series:
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
        Series
            Series of dtype Utf8.

        Examples
        --------
        >>> s = pl.Series("s", ["pear", None, "papaya", "dragonfruit"])
        >>> s.str.slice(-3)
        shape: (4,)
        Series: 's' [str]
        [
            "ear"
            null
            "aya"
            "uit"
        ]

        Using the optional `length` parameter

        >>> s.str.slice(4, length=3)
        shape: (4,)
        Series: 's' [str]
        [
            ""
            null
            "ya"
            "onf"
        ]

        """
        s = pli.wrap_s(self._s)
        return (
            s.to_frame().select(pli.col(s.name).str.slice(offset, length)).to_series()
        )
