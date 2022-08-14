from __future__ import annotations

from typing import TYPE_CHECKING

import polars.internals as pli
from polars.datatypes import Date, Datetime, Time

if TYPE_CHECKING:
    from polars.internals.type_aliases import TransferEncoding


class StringNameSpace:
    """Series.str namespace."""

    def __init__(self, series: pli.Series):
        self._s = series._s

    def strptime(
        self,
        datatype: type[Date] | type[Datetime] | type[Time],
        fmt: str | None = None,
        strict: bool = True,
        exact: bool = True,
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
        ├╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 2022-01-04 │
        ├╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 2022-01-31 │
        ├╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ 2001-07-08 │
        └────────────┘

        """
        s = pli.wrap_s(self._s)
        return (
            s.to_frame()
            .select(pli.col(s.name).str.strptime(datatype, fmt, strict, exact))
            .to_series()
        )

    def lengths(self) -> pli.Series:
        """
        Get length of the string values in the Series.

        Returns
        -------
        Series[u32]

        Examples
        --------
        >>> s = pl.Series(["foo", None, "hello", "world"])
        >>> s.str.lengths()
        shape: (4,)
        Series: '' [u32]
        [
            3
            null
            5
            5
        ]

        """
        return pli.wrap_s(self._s.str_lengths())

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
        s = pli.wrap_s(self._s)
        return s.to_frame().select(pli.col(s.name).str.concat(delimiter)).to_series()

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
        return pli.wrap_s(self._s.str_contains(pattern, literal))

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
        s = pli.wrap_s(self._s)
        return s.to_frame().select(pli.col(s.name).str.ends_with(sub)).to_series()

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
        s = pli.wrap_s(self._s)
        return s.to_frame().select(pli.col(s.name).str.starts_with(sub)).to_series()

    def decode(self, encoding: TransferEncoding, strict: bool = False) -> pli.Series:
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
        >>> s = pl.Series(["666f6f", "626172", None])
        >>> s.str.decode("hex")
        shape: (3,)
        Series: '' [str]
        [
            "foo"
            "bar"
            null
        ]

        """
        if encoding == "hex":
            return pli.wrap_s(self._s.str_hex_decode(strict))
        elif encoding == "base64":
            return pli.wrap_s(self._s.str_base64_decode(strict))
        else:
            raise ValueError(
                f"encoding must be one of {{'hex', 'base64'}}, got {encoding}"
            )

    def encode(self, encoding: TransferEncoding) -> pli.Series:
        """
        Encode a value using the provided encoding

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
        if encoding == "hex":
            return pli.wrap_s(self._s.str_hex_encode())
        elif encoding == "base64":
            return pli.wrap_s(self._s.str_base64_encode())
        else:
            raise ValueError(
                f"encoding must be one of {{'hex', 'base64'}}, got {encoding}"
            )

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
        return pli.wrap_s(self._s.str_json_path_match(json_path))

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
        ├╌╌╌╌╌╌╌╌╌┤
        │ null    │
        ├╌╌╌╌╌╌╌╌╌┤
        │ ronaldo │
        └─────────┘

        """
        return pli.wrap_s(self._s.str_extract(pattern, group_index))

    def extract_all(self, pattern: str) -> pli.Series:
        r"""
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
        Series: 'foo' [list]
        [
            ["123", "45"]
            ["678", "910"]
        ]

        """
        s = pli.wrap_s(self._s)
        return s.to_frame().select(pli.col(s.name).str.extract_all(pattern)).to_series()

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
        s = pli.wrap_s(self._s)
        return s.to_frame().select(pli.col(s.name).str.count_match(pattern)).to_series()

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
        s = pli.wrap_s(self._s)
        return s.to_frame().select(pli.col(s.name).str.split(by, inclusive)).to_series()

    def split_exact(self, by: str, n: int, inclusive: bool = False) -> pli.Series:
        """
        Split the string by a substring into a struct of ``n`` fields.

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

        Split column in ``n`` fields, give them a proper name in the struct and add them
        as columns.

        >>> df.select(
        ...     [
        ...         pl.col("x")
        ...         .str.split_exact("_", 1)
        ...         .struct.rename_fields(["first_part", "second_part"])
        ...         .alias("fields"),
        ...     ]
        ... ).unnest("fields")
        shape: (4, 2)
        ┌────────────┬─────────────┐
        │ first_part ┆ second_part │
        │ ---        ┆ ---         │
        │ str        ┆ str         │
        ╞════════════╪═════════════╡
        │ a          ┆ 1           │
        ├╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ null       ┆ null        │
        ├╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ c          ┆ null        │
        ├╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ d          ┆ 4           │
        └────────────┴─────────────┘

        Returns
        -------
        Struct of Utf8 type

        """
        s = pli.wrap_s(self._s)
        return (
            s.to_frame()
            .select(pli.col(s.name).str.split_exact(by, n, inclusive))
            .to_series()
        )

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
        return pli.wrap_s(self._s.str_replace(pattern, value, literal))

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
        return pli.wrap_s(self._s.str_replace_all(pattern, value, literal))

    def strip(self) -> pli.Series:
        """Remove leading and trailing whitespace."""
        s = pli.wrap_s(self._s)
        return s.to_frame().select(pli.col(s.name).str.strip()).to_series()

    def lstrip(self) -> pli.Series:
        """Remove leading whitespace."""
        s = pli.wrap_s(self._s)
        return s.to_frame().select(pli.col(s.name).str.lstrip()).to_series()

    def rstrip(self) -> pli.Series:
        """Remove trailing whitespace."""
        s = pli.wrap_s(self._s)
        return s.to_frame().select(pli.col(s.name).str.rstrip()).to_series()

    def zfill(self, alignment: int) -> pli.Series:
        """
        Return a copy of the string left filled with ASCII '0' digits to make a string
        of length width. A leading sign prefix ('+'/'-') is handled by inserting the
        padding after the sign character rather than before.
        The original string is returned if width is less than or equal to ``len(s)``.

        Parameters
        ----------
        alignment
            Fill the value up to this length.

        """
        s = pli.wrap_s(self._s)
        return s.to_frame().select(pli.col(s.name).str.zfill(alignment)).to_series()

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
        s = pli.wrap_s(self._s)
        return (
            s.to_frame().select(pli.col(s.name).str.ljust(width, fillchar)).to_series()
        )

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
        s = pli.wrap_s(self._s)
        return (
            s.to_frame().select(pli.col(s.name).str.rjust(width, fillchar)).to_series()
        )

    def to_lowercase(self) -> pli.Series:
        """Modify the strings to their lowercase equivalent."""
        return pli.wrap_s(self._s.str_to_lowercase())

    def to_uppercase(self) -> pli.Series:
        """Modify the strings to their uppercase equivalent."""
        return pli.wrap_s(self._s.str_to_uppercase())

    def slice(self, start: int, length: int | None = None) -> pli.Series:
        """
        Create subslices of the string values of a Utf8 Series.

        Parameters
        ----------
        start
            Starting index of the slice (zero-indexed). Negative indexing
            may be used.
        length
            Optional length of the slice. If None (default), the slice is taken to the
            end of the string.

        Returns
        -------
        Series of Utf8 type

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
        return pli.wrap_s(self._s.str_slice(start, length))
