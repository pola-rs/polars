from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

from polars import functions as F
from polars.series.utils import expr_dispatch
from polars.utils._wrap import wrap_s
from polars.utils.decorators import deprecated_alias
from polars.utils.various import find_stacklevel

if TYPE_CHECKING:
    from polars import Expr, Series
    from polars.polars import PySeries
    from polars.type_aliases import (
        PolarsDataType,
        PolarsTemporalType,
        TimeUnit,
        TransferEncoding,
    )


@expr_dispatch
class StringNameSpace:
    """Series.str namespace."""

    _accessor = "str"

    def __init__(self, series: Series):
        self._s: PySeries = series._s

    def to_date(
        self,
        format: str | None = None,
        *,
        strict: bool = True,
        exact: bool = True,
        cache: bool = True,
    ) -> Series:
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
    ) -> Series:
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

    def to_time(
        self,
        format: str | None = None,
        *,
        strict: bool = True,
        cache: bool = True,
    ) -> Series:
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
    ) -> Series:
        """
        Convert a Utf8 column into a Date/Datetime/Time column.

        Parameters
        ----------
        dtype
            The data type to convert to. Can be either Date, Datetime, or Time.
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
        s = wrap_s(self._s)
        return (
            s.to_frame()
            .select(
                F.col(s.name).str.strptime(
                    dtype,
                    format,
                    strict=strict,
                    exact=exact,
                    cache=cache,
                )
            )
            .to_series()
        )

    def to_decimal(
        self,
        inference_length: int = 100,
    ) -> Series:
        """
        Convert a Utf8 column into a Decimal column.

        This method infers the needed parameters ``precision`` and ``scale``.

        Parameters
        ----------
        inference_length
            Number of elements to parse to determine the `precision` and `scale`

        Examples
        --------
        >>> s = pl.Series(
        ...     ["40.12", "3420.13", "120134.19", "3212.98", "12.90", "143.09", "143.9"]
        ... )
        >>> s.str.to_decimal()
        shape: (7,)
        Series: '' [decimal[8,2]]
        [
            40.12
            3420.13
            120134.19
            3212.98
            12.9
            143.09
            143.9
        ]

        """

    def lengths(self) -> Series:
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

    def n_chars(self) -> Series:
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

    def concat(self, delimiter: str = "-") -> Series:
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

    def contains(
        self, pattern: str | Expr, *, literal: bool = False, strict: bool = True
    ) -> Series:
        """
        Check if strings in Series contain a substring that matches a regex.

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

        Default (case-sensitive) match:

        >>> s = pl.Series("s", ["AAA", "aAa", "aaa"])
        >>> s.str.contains("AA").to_list()
        [True, False, False]

        Case-insensitive match, using an inline flag:

        >>> s = pl.Series("s", ["AAA", "aAa", "aaa"])
        >>> s.str.contains("(?i)AA").to_list()
        [True, True, True]

        See the regex crate's section on `grouping and flags
        <https://docs.rs/regex/latest/regex/#grouping-and-flags>`_ for
        additional information about the use of inline expression modifiers.

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

    def ends_with(self, suffix: str | Expr) -> Series:
        """
        Check if string values end with a substring.

        Parameters
        ----------
        suffix
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

    def starts_with(self, prefix: str | Expr) -> Series:
        """
        Check if string values start with a substring.

        Parameters
        ----------
        prefix
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

    def decode(self, encoding: TransferEncoding, *, strict: bool = True) -> Series:
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

    def encode(self, encoding: TransferEncoding) -> Series:
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

    def json_extract(self, dtype: PolarsDataType | None = None) -> Series:
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
        >>> s = pl.Series("json", ['{"a":1, "b": true}', None, '{"a":2, "b": false}'])
        >>> s.str.json_extract()
        shape: (3,)
        Series: 'json' [struct[2]]
        [
                {1,true}
                {null,null}
                {2,false}
        ]

        See Also
        --------
        json_path_match : Extract the first match of json string with provided JSONPath
            expression.

        """

    def json_path_match(self, json_path: str) -> Series:
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

    def extract(self, pattern: str, group_index: int = 1) -> Series:
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

        >>> s = pl.Series(
        ...     name="lines",
        ...     values=[
        ...         "I Like\nThose\nOdds",
        ...         "This is\nThe Way",
        ...     ],
        ... )
        >>> s.str.extract(r"(?m)^(T\w+)", 1).alias("matches")
        shape: (2,)
        Series: 'matches' [str]
        [
            "Those"
            "This"
        ]

        See the regex crate's section on `grouping and flags
        <https://docs.rs/regex/latest/regex/#grouping-and-flags>`_ for
        additional information about the use of inline expression modifiers.

        Returns
        -------
        Utf8 array. Contain null if original value is null or regex capture nothing.

        Examples
        --------
        >>> s = pl.Series(
        ...     name="url",
        ...     values=[
        ...         "http://vote.com/ballon_dor?ref=polars&candidate=messi",
        ...         "http://vote.com/ballon_dor?candidate=ronaldo&ref=polars",
        ...         "http://vote.com/ballon_dor?error=404&ref=unknown",
        ...     ],
        ... )
        >>> s.str.extract(r"candidate=(\w+)", 1).alias("candidate")
        shape: (3,)
        Series: 'candidate' [str]
        [
            "messi"
            "ronaldo"
            null
        ]

        """

    def extract_all(self, pattern: str | Series) -> Series:
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

        >>> s = pl.Series(
        ...     name="email",
        ...     values=[
        ...         "real.email@spam.com",
        ...         "some_account@somewhere.net",
        ...         "abc.def.ghi.jkl@uvw.xyz.co.uk",
        ...     ],
        ... )
        >>> # extract name/domain parts from email, using verbose regex
        >>> s.str.extract_all(
        ...     r"""(?xi)   # activate 'verbose' and 'case-insensitive' flags
        ...       [         # (start character group)
        ...         A-Z     # letters
        ...         0-9     # digits
        ...         ._%+\-  # special chars
        ...       ]         # (end character group)
        ...       +         # 'one or more' quantifier
        ...     """
        ... ).alias("email_parts")
        shape: (3,)
        Series: 'email_parts' [list[str]]
        [
            ["real.email", "spam.com"]
            ["some_account", "somewhere.net"]
            ["abc.def.ghi.jkl", "uvw.xyz.co.uk"]
        ]

        See the regex crate's section on `grouping and flags
        <https://docs.rs/regex/latest/regex/#grouping-and-flags>`_ for
        additional information about the use of inline expression modifiers.

        Returns
        -------
        List[Utf8]

        Examples
        --------
        >>> s = pl.Series("foo", ["123 bla 45 asd", "xyz 678 910t"])
        >>> s.str.extract_all(r"\d+")
        shape: (2,)
        Series: 'foo' [list[str]]
        [
            ["123", "45"]
            ["678", "910"]
        ]

        '''

    def count_match(self, pattern: str) -> Series:
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

    def split(self, by: str, *, inclusive: bool = False) -> Series:
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

    def split_exact(self, by: str, n: int, *, inclusive: bool = False) -> Series:
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

    def splitn(self, by: str, n: int) -> Series:
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
        s = wrap_s(self._s)
        return s.to_frame().select(F.col(s.name).str.splitn(by, n)).to_series()

    def replace(
        self, pattern: str, value: str, *, literal: bool = False, n: int = 1
    ) -> Series:
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

        >>> s = pl.Series(
        ...     name="weather",
        ...     values=[
        ...         "Foggy",
        ...         "Rainy",
        ...         "Sunny",
        ...     ],
        ... )
        >>> # apply case-insensitive string replacement
        >>> s.str.replace(r"(?i)foggy|rainy", "Sunny")
        shape: (3,)
        Series: 'weather' [str]
        [
            "Sunny"
            "Sunny"
            "Sunny"
        ]

        See the regex crate's section on `grouping and flags
        <https://docs.rs/regex/latest/regex/#grouping-and-flags>`_ for
        additional information about the use of inline expression modifiers.

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

    def replace_all(self, pattern: str, value: str, *, literal: bool = False) -> Series:
        """
        Replace all matching regex/literal substrings with a new string value.

        Parameters
        ----------
        pattern
            A valid regular expression pattern, compatible with the `regex crate
            <https://docs.rs/regex/latest/regex/>`_.
        value
            String that will replace the matches.
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

    def strip(self, characters: str | None = None) -> Series:
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

    def lstrip(self, characters: str | None = None) -> Series:
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

    def rstrip(self, characters: str | None = None) -> Series:
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

    def zfill(self, alignment: int) -> Series:
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

    def ljust(self, width: int, fill_char: str = " ") -> Series:
        """
        Return the string left justified in a string of length ``width``.

        Padding is done using the specified ``fill_char``. The original string is
        returned if ``width`` is less than or equal to``len(s)``.

        Parameters
        ----------
        width
            Justify left to this length.
        fill_char
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

    def rjust(self, width: int, fill_char: str = " ") -> Series:
        """
        Return the string right justified in a string of length ``width``.

        Padding is done using the specified ``fill_char``. The original string is
        returned if ``width`` is less than or equal to ``len(s)``.

        Parameters
        ----------
        width
            Justify right to this length.
        fill_char
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

    def to_lowercase(self) -> Series:
        """Modify the strings to their lowercase equivalent."""

    def to_uppercase(self) -> Series:
        """Modify the strings to their uppercase equivalent."""

    def slice(self, offset: int, length: int | None = None) -> Series:
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

    def explode(self) -> Series:
        """
        Returns a column with a separate row for every string character.

        Returns
        -------
        Exploded column with string datatype.

        Examples
        --------
        >>> s = pl.Series("a", ["foo", "bar"])
        >>> s.str.explode()
        shape: (6,)
        Series: 'a' [str]
        [
                "f"
                "o"
                "o"
                "b"
                "a"
                "r"
        ]

        """

    def parse_int(self, radix: int = 2, *, strict: bool = True) -> Series:
        r"""
        Parse integers with base radix from strings.

        By default base 2. ParseError/Overflows become Nulls.

        Parameters
        ----------
        radix
            Positive integer which is the base of the string we are parsing.
            Default: 2

        strict
            Bool, Default=True will raise any ParseError or overflow as ComputeError.
            False silently convert to Null.

        Returns
        -------
        Series of parsed integers in i32 format

        Examples
        --------
        >>> s = pl.Series("bin", ["110", "101", "010", "invalid"])
        >>> s.str.parse_int(2, strict=False)
        shape: (4,)
        Series: 'bin' [i32]
        [
                6
                5
                2
                null
        ]

        >>> s = pl.Series("hex", ["fa1e", "ff00", "cafe", None])
        >>> s.str.parse_int(16)
        shape: (4,)
        Series: 'hex' [i32]
        [
                64030
                65280
                51966
                null
        ]

        """
