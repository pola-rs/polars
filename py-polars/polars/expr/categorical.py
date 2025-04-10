from __future__ import annotations

from typing import TYPE_CHECKING

from polars._utils.wrap import wrap_expr

if TYPE_CHECKING:
    from polars import Expr


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

    def slice(self, offset: int, length: int | None = None) -> Expr:
        """
        Extract a substring from the string representation of each value.

        Parameters
        ----------
        offset
            Start index. Negative indexing is supported.
        length
            Length of the slice. If set to `None` (default), the slice is taken to the
            end of the string.

        Returns
        -------
        Expr
            Expression of data type :class:`String`.

        Notes
        -----
        Both the `offset` and `length` inputs are defined in terms of the number
        of characters in the (UTF8) string. A character is defined as a
        `Unicode scalar value`_. A single character is represented by a single byte
        when working with ASCII text, and a maximum of 4 bytes otherwise.

        .. _Unicode scalar value: https://www.unicode.org/glossary/#unicode_scalar_value

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "s": pl.Series(
        ...             ["pear", None, "papaya", "dragonfruit"],
        ...             dtype=pl.Categorical,
        ...         )
        ...     }
        ... )
        >>> df.with_columns(pl.col("s").cat.slice(-3).alias("slice"))
        shape: (4, 2)
        ┌─────────────┬───────┐
        │ s           ┆ slice │
        │ ---         ┆ ---   │
        │ cat         ┆ str   │
        ╞═════════════╪═══════╡
        │ pear        ┆ ear   │
        │ null        ┆ null  │
        │ papaya      ┆ aya   │
        │ dragonfruit ┆ uit   │
        └─────────────┴───────┘

        Using the optional `length` parameter

        >>> df.with_columns(pl.col("s").cat.slice(4, length=3).alias("slice"))
        shape: (4, 2)
        ┌─────────────┬───────┐
        │ s           ┆ slice │
        │ ---         ┆ ---   │
        │ cat         ┆ str   │
        ╞═════════════╪═══════╡
        │ pear        ┆       │
        │ null        ┆ null  │
        │ papaya      ┆ ya    │
        │ dragonfruit ┆ onf   │
        └─────────────┴───────┘
        """
        return wrap_expr(self._pyexpr.cat_slice(offset, length))

    def to_uppercase(self) -> Expr:
        """
        Modify and return string representations as their uppercase equivalent.

        Examples
        --------
        >>> df = pl.DataFrame({"foo": pl.Series(["cat", "dog"], dtype=pl.Categorical)})
        >>> df.with_columns(foo_upper=pl.col("foo").cat.to_uppercase())
        shape: (2, 2)
        ┌─────┬───────────┐
        │ foo ┆ foo_upper │
        │ --- ┆ ---       │
        │ cat ┆ str       │
        ╞═════╪═══════════╡
        │ cat ┆ CAT       │
        │ dog ┆ DOG       │
        └─────┴───────────┘
        """
        return wrap_expr(self._pyexpr.cat_to_uppercase())

    def to_lowercase(self) -> Expr:
        """
        Modify and return string representations as their lowercase equivalent.

        Examples
        --------
        >>> df = pl.DataFrame({"foo": pl.Series(["CAT", "DOG"], dtype=pl.Categorical)})
        >>> df.with_columns(foo_lower=pl.col("foo").cat.to_lowercase())
        shape: (2, 2)
        ┌─────┬───────────┐
        │ foo ┆ foo_lower │
        │ --- ┆ ---       │
        │ cat ┆ str       │
        ╞═════╪═══════════╡
        │ CAT ┆ cat       │
        │ DOG ┆ dog       │
        └─────┴───────────┘
        """
        return wrap_expr(self._pyexpr.cat_to_lowercase())

    def to_titlecase(self) -> Expr:
        """
        Modify and return string representations as their titlecase equivalent.

        Notes
        -----
        This is a form of case transform where the first letter of each word is
        capitalized, with the rest of the word in lowercase. Non-alphanumeric
        characters define the word boundaries.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "quotes": pl.Series(
        ...             [
        ...                 "'e.t. phone home'",
        ...                 "you talkin' to me?",
        ...                 "to infinity,and BEYOND!",
        ...             ],
        ...             dtype=pl.Categorical,
        ...         )
        ...     }
        ... )
        >>> df.with_columns(
        ...     quotes_title=pl.col("quotes").cat.to_titlecase(),
        ... )
        shape: (3, 2)
        ┌─────────────────────────┬─────────────────────────┐
        │ quotes                  ┆ quotes_title            │
        │ ---                     ┆ ---                     │
        │ cat                     ┆ str                     │
        ╞═════════════════════════╪═════════════════════════╡
        │ 'e.t. phone home'       ┆ 'E.T. Phone Home'       │
        │ you talkin' to me?      ┆ You Talkin' To Me?      │
        │ to infinity,and BEYOND! ┆ To Infinity,And Beyond! │
        └─────────────────────────┴─────────────────────────┘
        """
        return wrap_expr(self._pyexpr.cat_to_titlecase())
