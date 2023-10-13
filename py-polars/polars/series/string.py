from __future__ import annotations

from typing import TYPE_CHECKING

from polars.series.utils import expr_dispatch
from polars.utils.deprecation import deprecate_renamed_function

if TYPE_CHECKING:
    from polars import Expr, Series
    from polars.polars import PySeries
    from polars.type_aliases import (
        Ambiguous,
        IntoExpr,
        IntoExprColumn,
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

            .. note::
                Using ``exact=False`` introduces a performance penalty - cleaning your
                data beforehand will almost certainly be more performant.
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
        use_earliest: bool | None = None,
        ambiguous: Ambiguous | Series = "raise",
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

            .. note::
                Using ``exact=False`` introduces a performance penalty - cleaning your
                data beforehand will almost certainly be more performant.
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
        use_earliest
            Determine how to deal with ambiguous datetimes:

            - ``None`` (default): raise
            - ``True``: use the earliest datetime
            - ``False``: use the latest datetime

            .. deprecated:: 0.19.0
                Use `ambiguous` instead
        ambiguous
            Determine how to deal with ambiguous datetimes:

            - ``'raise'`` (default): raise
            - ``'earliest'``: use the earliest datetime
            - ``'latest'``: use the latest datetime

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

    def strptime(
        self,
        dtype: PolarsTemporalType,
        format: str | None = None,
        *,
        strict: bool = True,
        exact: bool = True,
        cache: bool = True,
        use_earliest: bool | None = None,
        ambiguous: Ambiguous | Series = "raise",
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

            .. note::
                Using ``exact=False`` introduces a performance penalty - cleaning your
                data beforehand will almost certainly be more performant.
        cache
            Use a cache of unique, converted dates to apply the datetime conversion.
        use_earliest
            Determine how to deal with ambiguous datetimes:

            - ``None`` (default): raise
            - ``True``: use the earliest datetime
            - ``False``: use the latest datetime

            .. deprecated:: 0.19.0
                Use `ambiguous` instead
        ambiguous
            Determine how to deal with ambiguous datetimes:

            - ``'raise'`` (default): raise
            - ``'earliest'``: use the earliest datetime
            - ``'latest'``: use the latest datetime

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
        Series: '' [decimal[2]]
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

    def len_bytes(self) -> Series:
        """
        Return the length of each string as the number of bytes.

        Returns
        -------
        Series
            Series of data type :class:`UInt32`.

        See Also
        --------
        len_chars

        Notes
        -----
        When working with non-ASCII text, the length in bytes is not the same as the
        length in characters. You may want to use :func:`len_chars` instead.
        Note that :func:`len_bytes` is much more performant (_O(1)_) than
        :func:`len_chars` (_O(n)_).

        Examples
        --------
        >>> s = pl.Series(["Café", "345", "東京", None])
        >>> s.str.len_bytes()
        shape: (4,)
        Series: '' [u32]
        [
            5
            3
            6
            null
        ]

        """

    def len_chars(self) -> Series:
        """
        Return the length of each string as the number of characters.

        Returns
        -------
        Series
            Series of data type :class:`UInt32`.

        See Also
        --------
        len_bytes

        Notes
        -----
        When working with ASCII text, use :func:`len_bytes` instead to achieve
        equivalent output with much better performance:
        :func:`len_bytes` runs in _O(1)_, while :func:`len_chars` runs in (_O(n)_).

        Examples
        --------
        >>> s = pl.Series(["Café", "345", "東京", None])
        >>> s.str.len_chars()
        shape: (4,)
        Series: '' [u32]
        [
            4
            3
            2
            null
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
        Series
            Series of data type :class:`Utf8`.

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
        Series
            Series of data type :class:`Boolean`.

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

        See Also
        --------
        contains : Check if string contains a substring that matches a regex.
        starts_with : Check if string values start with a substring.

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

        """

    def starts_with(self, prefix: str | Expr) -> Series:
        """
        Check if string values start with a substring.

        Parameters
        ----------
        prefix
            Prefix substring.

        See Also
        --------
        contains : Check if string contains a substring that matches a regex.
        ends_with : Check if string values end with a substring.

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
        Series
            Series of data type :class:`Utf8`.

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

    def json_extract(
        self, dtype: PolarsDataType | None = None, infer_schema_length: int | None = 100
    ) -> Series:
        """
        Parse string values as JSON.

        Throw errors if encounter invalid JSON strings.

        Parameters
        ----------
        dtype
            The dtype to cast the extracted value to. If None, the dtype will be
            inferred from the JSON value.
        infer_schema_length
            How many rows to parse to determine the schema.
            If ``None`` all rows are used.

        See Also
        --------
        json_path_match : Extract the first match of json string with provided JSONPath
            expression.

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
        Series
            Series of data type :class:`Utf8`. Contains null values if the original
            value is null or the json_path returns nothing.

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
            Group 0 means the whole pattern, the first group begin at index 1.
            Defaults to the first capture group.

        Returns
        -------
        Series
            Series of data type :class:`Utf8`. Contains null values if the original
            value is null or regex captures nothing.

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
        as a list. If the haystack string is ``null``, ``null`` is returned.

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
        Series
            Series of data type ``List(Utf8)``.

        Examples
        --------
        >>> s = pl.Series("foo", ["123 bla 45 asd", "xyz 678 910t", "bar", None])
        >>> s.str.extract_all(r"\d+")
        shape: (4,)
        Series: 'foo' [list[str]]
        [
            ["123", "45"]
            ["678", "910"]
            []
            null
        ]

        '''

    def extract_groups(self, pattern: str) -> Series:
        r"""
        Extract all capture groups for the given regex pattern.

        Parameters
        ----------
        pattern
            A valid regular expression pattern, compatible with the `regex crate
            <https://docs.rs/regex/latest/regex/>`_.

        Notes
        -----
        All group names are **strings**.

        If your pattern contains unnamed groups, their numerical position is converted
        to a string.

        For example, we can access the first group via the string `"1"`::

            >>> (
            ...     pl.Series(["foo bar baz"])
            ...     .str.extract_groups(r"(\w+) (.+) (\w+)")
            ...     .struct["1"]
            ... )
            shape: (1,)
            Series: '1' [str]
            [
                "foo"
            ]

        Returns
        -------
        Series
            Series of data type :class:`Struct` with fields of data type :class:`Utf8`.

        Examples
        --------
        >>> s = pl.Series(
        ...     name="url",
        ...     values=[
        ...         "http://vote.com/ballon_dor?candidate=messi&ref=python",
        ...         "http://vote.com/ballon_dor?candidate=weghorst&ref=polars",
        ...         "http://vote.com/ballon_dor?error=404&ref=rust",
        ...     ],
        ... )
        >>> s.str.extract_groups(r"candidate=(?<candidate>\w+)&ref=(?<ref>\w+)")
        shape: (3,)
        Series: 'url' [struct[2]]
        [
            {"messi","python"}
            {"weghorst","polars"}
            {null,null}
        ]

        """

    def count_matches(self, pattern: str | Series, *, literal: bool = False) -> Series:
        r"""
        Count all successive non-overlapping regex matches.

        Parameters
        ----------
        pattern
            A valid regular expression pattern, compatible with the `regex crate
            <https://docs.rs/regex/latest/regex/>`_. Can also be a :class:`Series` of
            regular expressions.
        literal
            Treat ``pattern`` as a literal string, not as a regular expression.

        Returns
        -------
        Series
            Series of data type :class:`UInt32`. Returns null if the original
            value is null.

        Examples
        --------
        >>> s = pl.Series("foo", ["123 bla 45 asd", "xyz 678 910t", "bar", None])
        >>> # count digits
        >>> s.str.count_matches(r"\d")
        shape: (4,)
        Series: 'foo' [u32]
        [
            5
            6
            0
            null
        ]

        >>> s = pl.Series("bar", ["12 dbc 3xy", "cat\\w", "1zy3\\d\\d", None])
        >>> s.str.count_matches(r"\d", literal=True)
        shape: (4,)
        Series: 'bar' [u32]
        [
            0
            0
            2
            null
        ]

        """

    def split(self, by: IntoExpr, *, inclusive: bool = False) -> Series:
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
        Series
            Series of data type ``List(Utf8)``.

        """

    def split_exact(self, by: IntoExpr, n: int, *, inclusive: bool = False) -> Series:
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
        Series
            Series of data type :class:`Struct` with fields of data type :class:`Utf8`.

        """

    def splitn(self, by: IntoExpr, n: int) -> Series:
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
        Series
            Series of data type :class:`Struct` with fields of data type :class:`Utf8`.

        """

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

    def strip_chars(self, characters: IntoExprColumn | None = None) -> Series:
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
        >>> s.str.strip_chars()
        shape: (2,)
        Series: '' [str]
        [
                "hello"
                "world"
        ]

        Characters can be stripped by passing a string as argument. Note that whitespace
        will not be stripped automatically when doing so, unless that whitespace is
        also included in the string.

        >>> s.str.strip_chars("o ")
        shape: (2,)
        Series: '' [str]
        [
            "hell"
            "	world"
        ]

        """

    def strip_chars_start(self, characters: IntoExprColumn | None = None) -> Series:
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
        >>> s.str.strip_chars_start()
        shape: (2,)
        Series: '' [str]
        [
                "hello "
                "world"
        ]

        Characters can be stripped by passing a string as argument. Note that whitespace
        will not be stripped automatically when doing so.

        >>> s.str.strip_chars_start("wod\t")
        shape: (2,)
        Series: '' [str]
        [
                " hello "
                "rld"
        ]

        """

    def strip_chars_end(self, characters: IntoExprColumn | None = None) -> Series:
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
        >>> s.str.strip_chars_end()
        shape: (2,)
        Series: '' [str]
        [
                " hello"
                "world"
        ]

        Characters can be stripped by passing a string as argument. Note that whitespace
        will not be stripped automatically when doing so.

        >>> s.str.strip_chars_end("orld\t")
        shape: (2,)
        Series: '' [str]
        [
            " hello "
            "w"
        ]

        """

    def strip_prefix(self, prefix: IntoExpr) -> Series:
        """
        Remove prefix.

        The prefix will be removed from the string exactly once, if found.

        Parameters
        ----------
        prefix
            The prefix to be removed.

        Examples
        --------
        >>> s = pl.Series(["foobar", "foofoobar", "foo", "bar"])
        >>> s.str.strip_prefix("foo")
        shape: (4,)
        Series: '' [str]
        [
                "bar"
                "foobar"
                ""
                "bar"
        ]

        """

    def strip_suffix(self, suffix: IntoExpr) -> Series:
        """
        Remove suffix.

        The suffix will be removed from the string exactly once, if found.

        Parameters
        ----------
        suffix
            The suffix to be removed.

        Examples
        --------
        >>> s = pl.Series(["foobar", "foobarbar", "foo", "bar"])
        >>> s.str.strip_suffix("bar")
        shape: (4,)
        Series: '' [str]
        [
                "foo"
                "foobar"
                "foo"
                ""
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
        """
        Modify the strings to their lowercase equivalent.

        Examples
        --------
        >>> s = pl.Series("foo", ["CAT", "DOG"])
        >>> s.str.to_lowercase()
        shape: (2,)
        Series: 'foo' [str]
        [
            "cat"
            "dog"
        ]

        """

    def to_uppercase(self) -> Series:
        """
        Modify the strings to their uppercase equivalent.

        Examples
        --------
        >>> s = pl.Series("foo", ["cat", "dog"])
        >>> s.str.to_uppercase()
        shape: (2,)
        Series: 'foo' [str]
        [
            "CAT"
            "DOG"
        ]

        """

    def to_titlecase(self) -> Series:
        """
        Modify the strings to their titlecase equivalent.

        Examples
        --------
        >>> s = pl.Series("sing", ["welcome to my world", "THERE'S NO TURNING BACK"])
        >>> s.str.to_titlecase()
        shape: (2,)
        Series: 'sing' [str]
        [
            "Welcome To My …
            "There's No Tur…
        ]

        """

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
            Series of data type :class:`Struct` with fields of data type :class:`Utf8`.

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
        Series
            Series of data type :class:`Utf8`.

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

    def parse_int(self, radix: int | None = None, *, strict: bool = True) -> Series:
        """
        Parse integers with base radix from strings.

        ParseError/Overflows become Nulls.

        Parameters
        ----------
        radix
            Positive integer which is the base of the string we are parsing.
        strict
            Bool, Default=True will raise any ParseError or overflow as ComputeError.
            False silently convert to Null.

        Returns
        -------
        Series
            Series of data type :class:`Int32`.

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

    @deprecate_renamed_function("strip_chars", version="0.19.3")
    def strip(self, characters: str | None = None) -> Series:
        """
        Remove leading and trailing characters.

        .. deprecated:: 0.19.3
            This method has been renamed to :func:`strip_chars`.

        Parameters
        ----------
        characters
            The set of characters to be removed. All combinations of this set of
            characters will be stripped. If set to None (default), all whitespace is
            removed instead.

        """

    @deprecate_renamed_function("strip_chars_start", version="0.19.3")
    def lstrip(self, characters: str | None = None) -> Series:
        """
        Remove leading characters.

        .. deprecated:: 0.19.3
            This method has been renamed to :func:`strip_chars_start`.

        Parameters
        ----------
        characters
            The set of characters to be removed. All combinations of this set of
            characters will be stripped. If set to None (default), all whitespace is
            removed instead.

        """

    @deprecate_renamed_function("strip_chars_end", version="0.19.3")
    def rstrip(self, characters: str | None = None) -> Series:
        """
        Remove trailing characters.

        .. deprecated:: 0.19.3
            This method has been renamed to :func:`Series.strip_chars_end`.

        Parameters
        ----------
        characters
            The set of characters to be removed. All combinations of this set of
            characters will be stripped. If set to None (default), all whitespace is
            removed instead.

        """

    @deprecate_renamed_function("count_matches", version="0.19.3")
    def count_match(self, pattern: str | Series) -> Series:
        """
        Count all successive non-overlapping regex matches.

        .. deprecated:: 0.19.3
            This method has been renamed to :func:`count_matches`.

        Parameters
        ----------
        pattern
            A valid regular expression pattern, compatible with the `regex crate
            <https://docs.rs/regex/latest/regex/>`_. Can also be a :class:`Series` of
            regular expressions.

        Returns
        -------
        Series
            Series of data type :class:`UInt32`. Returns null if the original
            value is null.

        """

    @deprecate_renamed_function("len_bytes", version="0.19.8")
    def lengths(self) -> Series:
        """
        Return the number of bytes in each string.

        .. deprecated:: 0.19.8
            This method has been renamed to :func:`len_bytes`.

        """

    @deprecate_renamed_function("len_chars", version="0.19.8")
    def n_chars(self) -> Series:
        """
        Return the length of each string as the number of characters.

        .. deprecated:: 0.19.8
            This method has been renamed to :func:`len_chars`.

        """
