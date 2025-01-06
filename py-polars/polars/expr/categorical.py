from __future__ import annotations

from typing import TYPE_CHECKING

from polars._utils.parse import parse_into_expression
from polars._utils.wrap import wrap_expr

if TYPE_CHECKING:
    from polars import Expr
    from polars._typing import IntoExpr


class ExprCatNameSpace:
    """Namespace for categorical related expressions."""

    _accessor = "cat"

    def __init__(self, expr: Expr) -> None:
        self._pyexpr = expr._pyexpr

    def get_categories(self) -> Expr:
        """
        Get the categories stored in this data type.

        Examples
        --------
        >>> df = pl.Series(
        ...     "cats", ["foo", "bar", "foo", "foo", "ham"], dtype=pl.Categorical
        ... ).to_frame()
        >>> df.select(pl.col("cats").cat.get_categories())
        shape: (3, 1)
        ┌──────┐
        │ cats │
        │ ---  │
        │ str  │
        ╞══════╡
        │ foo  │
        │ bar  │
        │ ham  │
        └──────┘
        """
        return wrap_expr(self._pyexpr.cat_get_categories())

    def len_bytes(self) -> Expr:
        """
        Return the byte-length of the string representation of each value.

        Returns
        -------
        Expr
            Expression of data type :class:`UInt32`.

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
        >>> df = pl.DataFrame(
        ...     {"a": pl.Series(["Café", "345", "東京", None], dtype=pl.Categorical)}
        ... )
        >>> df.with_columns(
        ...     pl.col("a").cat.len_bytes().alias("n_bytes"),
        ...     pl.col("a").cat.len_chars().alias("n_chars"),
        ... )
        shape: (4, 3)
        ┌──────┬─────────┬─────────┐
        │ a    ┆ n_bytes ┆ n_chars │
        │ ---  ┆ ---     ┆ ---     │
        │ cat  ┆ u32     ┆ u32     │
        ╞══════╪═════════╪═════════╡
        │ Café ┆ 5       ┆ 4       │
        │ 345  ┆ 3       ┆ 3       │
        │ 東京 ┆ 6       ┆ 2       │
        │ null ┆ null    ┆ null    │
        └──────┴─────────┴─────────┘
        """
        return wrap_expr(self._pyexpr.cat_len_bytes())

    def len_chars(self) -> Expr:
        """
        Return the number of characters of the string representation of each value.

        Returns
        -------
        Expr
            Expression of data type :class:`UInt32`.

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
        >>> df = pl.DataFrame(
        ...     {"a": pl.Series(["Café", "345", "東京", None], dtype=pl.Categorical)}
        ... )
        >>> df.with_columns(
        ...     pl.col("a").cat.len_chars().alias("n_chars"),
        ...     pl.col("a").cat.len_bytes().alias("n_bytes"),
        ... )
        shape: (4, 3)
        ┌──────┬─────────┬─────────┐
        │ a    ┆ n_chars ┆ n_bytes │
        │ ---  ┆ ---     ┆ ---     │
        │ cat  ┆ u32     ┆ u32     │
        ╞══════╪═════════╪═════════╡
        │ Café ┆ 4       ┆ 5       │
        │ 345  ┆ 3       ┆ 3       │
        │ 東京 ┆ 2       ┆ 6       │
        │ null ┆ null    ┆ null    │
        └──────┴─────────┴─────────┘
        """
        return wrap_expr(self._pyexpr.cat_len_chars())

    def starts_with(self, prefix: str) -> Expr:
        """
        Check if string representations of values start with a substring.

        Parameters
        ----------
        prefix
            Prefix substring.

        See Also
        --------
        contains : Check if string repr contains a substring that matches a pattern.
        ends_with : Check if string repr end with a substring.

        Notes
        -----
        Whereas `str.starts_with` allows expression inputs, `cat.starts_with` requires
        a literal string value.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {"fruits": pl.Series(["apple", "mango", None], dtype=pl.Categorical)}
        ... )
        >>> df.with_columns(
        ...     pl.col("fruits").cat.starts_with("app").alias("has_prefix"),
        ... )
        shape: (3, 2)
        ┌────────┬────────────┐
        │ fruits ┆ has_prefix │
        │ ---    ┆ ---        │
        │ cat    ┆ bool       │
        ╞════════╪════════════╡
        │ apple  ┆ true       │
        │ mango  ┆ false      │
        │ null   ┆ null       │
        └────────┴────────────┘

        Using `starts_with` as a filter condition:

        >>> df.filter(pl.col("fruits").cat.starts_with("app"))
        shape: (1, 1)
        ┌────────┐
        │ fruits │
        │ ---    │
        │ cat    │
        ╞════════╡
        │ apple  │
        └────────┘
        """
        if not isinstance(prefix, str):
            msg = f"'prefix' must be a string; found {type(prefix)!r}"
            raise TypeError(msg)
        return wrap_expr(self._pyexpr.cat_starts_with(prefix))

    def ends_with(self, suffix: str) -> Expr:
        """
        Check if string representations of values end with a substring.

        Parameters
        ----------
        suffix
            Suffix substring.

        See Also
        --------
        contains : Check if string reprs contains a substring that matches a pattern.
        starts_with : Check if string reprs start with a substring.

        Notes
        -----
        Whereas `str.ends_with` allows expression inputs, `cat.ends_with` requires a
        literal string value.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {"fruits": pl.Series(["apple", "mango", None], dtype=pl.Categorical)}
        ... )
        >>> df.with_columns(pl.col("fruits").cat.ends_with("go").alias("has_suffix"))
        shape: (3, 2)
        ┌────────┬────────────┐
        │ fruits ┆ has_suffix │
        │ ---    ┆ ---        │
        │ cat    ┆ bool       │
        ╞════════╪════════════╡
        │ apple  ┆ false      │
        │ mango  ┆ true       │
        │ null   ┆ null       │
        └────────┴────────────┘

        Using `ends_with` as a filter condition:

        >>> df.filter(pl.col("fruits").cat.ends_with("go"))
        shape: (1, 1)
        ┌────────┐
        │ fruits │
        │ ---    │
        │ cat    │
        ╞════════╡
        │ mango  │
        └────────┘
        """
        if not isinstance(suffix, str):
            msg = f"'suffix' must be a string; found {type(suffix)!r}"
            raise TypeError(msg)
        return wrap_expr(self._pyexpr.cat_ends_with(suffix))

    def contains(
        self, pattern: str, *, literal: bool = False, strict: bool = True
    ) -> Expr:
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

        >>> pl.DataFrame({"s": ["AAA", "aAa", "aaa"]}).with_columns(
        ...     pl.col("s").cast(pl.Categorical)
        ... ).with_columns(
        ...     default_match=pl.col("s").cat.contains("AA"),
        ...     insensitive_match=pl.col("s").cat.contains("(?i)AA"),
        ... )
        shape: (3, 3)
        ┌─────┬───────────────┬───────────────────┐
        │ s   ┆ default_match ┆ insensitive_match │
        │ --- ┆ ---           ┆ ---               │
        │ cat ┆ bool          ┆ bool              │
        ╞═════╪═══════════════╪═══════════════════╡
        │ AAA ┆ true          ┆ true              │
        │ aAa ┆ false         ┆ true              │
        │ aaa ┆ false         ┆ true              │
        └─────┴───────────────┴───────────────────┘

        See the regex crate's section on `grouping and flags
        <https://docs.rs/regex/latest/regex/#grouping-and-flags>`_ for
        additional information about the use of inline expression modifiers.

        See Also
        --------
        starts_with : Check if string values start with a substring.
        ends_with : Check if string values end with a substring.
        find: Return the index of the first substring matching a pattern.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "txt": pl.Series(
        ...             ["Crab", "cat and dog", "rab$bit", None],
        ...             dtype=pl.Categorical,
        ...         )
        ...     }
        ... )
        >>> df.select(
        ...     pl.col("txt"),
        ...     pl.col("txt").cat.contains("cat|bit").alias("regex"),
        ...     pl.col("txt").cat.contains("rab$", literal=True).alias("literal"),
        ... )
        shape: (4, 3)
        ┌─────────────┬───────┬─────────┐
        │ txt         ┆ regex ┆ literal │
        │ ---         ┆ ---   ┆ ---     │
        │ cat         ┆ bool  ┆ bool    │
        ╞═════════════╪═══════╪═════════╡
        │ Crab        ┆ false ┆ false   │
        │ cat and dog ┆ true  ┆ false   │
        │ rab$bit     ┆ true  ┆ true    │
        │ null        ┆ null  ┆ null    │
        └─────────────┴───────┴─────────┘
        """
        return wrap_expr(self._pyexpr.cat_contains(pattern, literal, strict))

    def contains_any(
        self, patterns: IntoExpr, *, ascii_case_insensitive: bool = False
    ) -> Expr:
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
        >>> df = pl.DataFrame(
        ...     {
        ...         "lyrics": pl.Series(
        ...             [
        ...                 "Everybody wants to rule the world",
        ...                 "Tell me what you want, what you really really want",
        ...                 "Can you feel the love tonight",
        ...             ],
        ...             dtype=pl.Categorical,
        ...         )
        ...     }
        ... )
        >>> df.with_columns(
        ...     pl.col("lyrics").cat.contains_any(["you", "me"]).alias("contains_any")
        ... )
        shape: (3, 2)
        ┌────────────────────────────────────────────────────┬──────────────┐
        │ lyrics                                             ┆ contains_any │
        │ ---                                                ┆ ---          │
        │ cat                                                ┆ bool         │
        ╞════════════════════════════════════════════════════╪══════════════╡
        │ Everybody wants to rule the world                  ┆ false        │
        │ Tell me what you want, what you really really want ┆ true         │
        │ Can you feel the love tonight                      ┆ true         │
        └────────────────────────────────────────────────────┴──────────────┘
        """
        patterns = parse_into_expression(
            patterns, str_as_lit=False, list_as_series=True
        )
        return wrap_expr(
            self._pyexpr.cat_contains_any(patterns, ascii_case_insensitive)
        )
