from __future__ import annotations

from typing import TYPE_CHECKING

from polars._utils.unstable import unstable
from polars._utils.wrap import wrap_s
from polars.series.utils import expr_dispatch

if TYPE_CHECKING:
    from polars import Series
    from polars.polars import PySeries


@expr_dispatch
class CatNameSpace:
    """Namespace for categorical related series."""

    _accessor = "cat"

    def __init__(self, series: Series) -> None:
        self._s: PySeries = series._s

    def get_categories(self) -> Series:
        """
        Get the categories stored in this data type.

        Examples
        --------
        >>> s = pl.Series(["foo", "bar", "foo", "foo", "ham"], dtype=pl.Categorical)
        >>> s.cat.get_categories()
        shape: (3,)
        Series: '' [str]
        [
            "foo"
            "bar"
            "ham"
        ]
        """

    def is_local(self) -> bool:
        """
        Return whether or not the column is a local categorical.

        Examples
        --------
        Categoricals constructed without a string cache are considered local.

        >>> s = pl.Series(["a", "b", "a"], dtype=pl.Categorical)
        >>> s.cat.is_local()
        True

        Categoricals constructed with a string cache are considered global.

        >>> with pl.StringCache():
        ...     s = pl.Series(["a", "b", "a"], dtype=pl.Categorical)
        >>> s.cat.is_local()
        False
        """
        return self._s.cat_is_local()

    def to_local(self) -> Series:
        """
        Convert a categorical column to its local representation.

        This may change the underlying physical representation of the column.

        See the documentation of :func:`StringCache` for more information on the
        difference between local and global categoricals.

        Examples
        --------
        Compare the global and local representations of a categorical.

        >>> with pl.StringCache():
        ...     _ = pl.Series("x", ["a", "b", "a"], dtype=pl.Categorical)
        ...     s = pl.Series("y", ["c", "b", "d"], dtype=pl.Categorical)
        >>> s.to_physical()
        shape: (3,)
        Series: 'y' [u32]
        [
                2
                1
                3
        ]
        >>> s.cat.to_local().to_physical()
        shape: (3,)
        Series: 'y' [u32]
        [
                0
                1
                2
        ]
        """
        return wrap_s(self._s.cat_to_local())

    @unstable()
    def uses_lexical_ordering(self) -> bool:
        """
        Indicate whether the Series uses lexical ordering.

        .. warning::
            This functionality is considered **unstable**. It may be changed
            at any point without it being considered a breaking change.

        Examples
        --------
        >>> s = pl.Series(["b", "a", "b"]).cast(pl.Categorical)
        >>> s.cat.uses_lexical_ordering()
        False
        >>> s = s.cast(pl.Categorical("lexical"))
        >>> s.cat.uses_lexical_ordering()
        True
        """
        return self._s.cat_uses_lexical_ordering()

    def len_bytes(self) -> Series:
        """
        Return the byte-length of the string representation of each value.

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
        >>> s = pl.Series(["Café", "345", "東京", None], dtype=pl.Categorical)
        >>> s.cat.len_bytes()
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
        Return the number of characters of the string representation of each value.

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

        A character is defined as a `Unicode scalar value`_. A single character is
        represented by a single byte when working with ASCII text, and a maximum of
        4 bytes otherwise.

        .. _Unicode scalar value: https://www.unicode.org/glossary/#unicode_scalar_value

        Examples
        --------
        >>> s = pl.Series(["Café", "345", "東京", None], dtype=pl.Categorical)
        >>> s.cat.len_chars()
        shape: (4,)
        Series: '' [u32]
        [
            4
            3
            2
            null
        ]
        """

    def starts_with(self, prefix: str) -> Series:
        """
        Check if string representations of values start with a substring.

        Parameters
        ----------
        prefix
            Prefix substring.

        See Also
        --------
        contains : Check if the string repr contains a substring that matches a pattern.
        ends_with : Check if string repr ends with a substring.

        Examples
        --------
        >>> s = pl.Series("fruits", ["apple", "mango", None], dtype=pl.Categorical)
        >>> s.cat.starts_with("app")
        shape: (3,)
        Series: 'fruits' [bool]
        [
            true
            false
            null
        ]
        """

    def ends_with(self, suffix: str) -> Series:
        """
        Check if string representations of values end with a substring.

        Parameters
        ----------
        suffix
            Suffix substring.

        See Also
        --------
        contains : Check if the string repr contains a substring that matches a pattern.
        starts_with : Check if string repr starts with a substring.

        Examples
        --------
        >>> s = pl.Series("fruits", ["apple", "mango", None], dtype=pl.Categorical)
        >>> s.cat.ends_with("go")
        shape: (3,)
        Series: 'fruits' [bool]
        [
            false
            true
            null
        ]
        """

    def contains(
        self, pattern: str, *, literal: bool = False, strict: bool = True
    ) -> Series:
        """
        Check if the string representation contains a substring that matches a pattern.

        Parameters
        ----------
        pattern
            A valid regular expression pattern, compatible with the `regex crate
            <https://docs.rs/regex/latest/regex/>`_.
        literal
            Treat `pattern` as a literal string, not as a regular expression.
        strict
            Raise an error if the underlying pattern is not a valid regex,
            otherwise mask out with a null value.

        Notes
        -----
        To modify regular expression behaviour (such as case-sensitivity) with
        flags, use the inline `(?iLmsuxU)` syntax. For example:

        Default (case-sensitive) match:

        >>> s = pl.Series("s", ["AAA", "aAa", "aaa"], dtype=pl.Categorical)
        >>> s.cat.contains("AA").to_list()
        [True, False, False]

        Case-insensitive match, using an inline flag:

        >>> s = pl.Series("s", ["AAA", "aAa", "aaa"], dtype=pl.Categorical)
        >>> s.cat.contains("(?i)AA").to_list()
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
        >>> s = pl.Series(
        ...     ["Crab", "cat and dog", "rab$bit", None],
        ...     dtype=pl.Categorical,
        ... )
        >>> s.cat.contains("cat|bit")
        shape: (4,)
        Series: '' [bool]
        [
            false
            true
            true
            null
        ]
        >>> s.cat.contains("rab$", literal=True)
        shape: (4,)
        Series: '' [bool]
        [
            false
            false
            true
            null
        ]
        """

    def contains_any(
        self, patterns: Series | list[str], *, ascii_case_insensitive: bool = False
    ) -> Series:
        """
        Use the Aho-Corasick algorithm to find matches.

        Determines if any of the patterns are contained in the string representation.

        Parameters
        ----------
        patterns
            String patterns to search.
        ascii_case_insensitive
            Enable ASCII-aware case-insensitive matching.
            When this option is enabled, searching will be performed without respect
            to case for ASCII letters (a-z and A-Z) only.

        Notes
        -----
        This method supports matching on string literals only, and does not support
        regular expression matching.

        Examples
        --------
        >>> _ = pl.Config.set_fmt_str_lengths(100)
        >>> s = pl.Series(
        ...     "lyrics",
        ...     [
        ...         "Everybody wants to rule the world",
        ...         "Tell me what you want, what you really really want",
        ...         "Can you feel the love tonight",
        ...     ],
        ...     dtype=pl.Categorical,
        ... )
        >>> s.cat.contains_any(["you", "me"])
        shape: (3,)
        Series: 'lyrics' [bool]
        [
            false
            true
            true
        ]
        """
