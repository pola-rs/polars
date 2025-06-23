from __future__ import annotations

import polars._reexport as pl


class DataTypeExprEnumNameSpace:
    """Namespace for enum datatype expressions."""

    _accessor = "enum"

    def __init__(self, expr: pl.DataTypeExpr) -> None:
        self._pydatatype_expr = expr._pydatatype_expr

    def num_categories(self) -> pl.Expr:
        """
        Get the number of enum categories.

        Examples
        --------
        >>> country_codes = pl.Enum([ 'US', 'NL', 'CN', 'IN', 'FR' ])
        >>> df = pl.DataFrame({
        ...     'country': ['NL', 'FR', 'CN'],
        ... }, schema={ 'country': country_codes })
        >>> df.select(
        ...     num_country_codes = pl.dtype_of('country').enum.num_categories(),
        ... )
        shape: (1, 1)
        ┌───────────────────┐
        │ num_country_codes │
        │ ---               │
        │ u32               │
        ╞═══════════════════╡
        │ 5                 │
        └───────────────────┘
        """
        return pl.Expr._from_pyexpr(self._pydatatype_expr.enum_num_categories())

    def categories(self) -> pl.Expr:
        """
        Get the enum categories as a list.

        Examples
        --------
        >>> country_codes = pl.Enum([ 'US', 'NL', 'CN', 'IN', 'FR' ])
        >>> df = pl.DataFrame({
        ...     'country': ['NL', 'FR', 'CN'],
        ... }, schema={ 'country': country_codes })
        >>> df.select(
        ...     all_country_codes = pl.dtype_of('country').enum.categories(),
        ... )
        shape: (1, 1)
        ┌──────────────────────┐
        │ all_country_codes    │
        │ ---                  │
        │ list[str]            │
        ╞══════════════════════╡
        │ ["US", "NL", … "FR"] │
        └──────────────────────┘
        """
        return pl.Expr._from_pyexpr(self._pydatatype_expr.enum_categories())

    def get_category(self, index: int, *, raise_on_oob: bool = True) -> pl.Expr:
        """
        Get the enum category at a specific index.

        Parameters
        ----------
        index
            The category index to get.
        raise_on_oob
            If the index is greater than the maximum index, should an exception be raised or
            should a missing value be inserted.

        Examples
        --------
        >>> country_codes = pl.Enum(["US", "NL", "CN", "IN", "FR"])
        >>> df = pl.DataFrame(
        ...     {
        ...         "country": ["NL", "FR", "CN"],
        ...     },
        ...     schema={"country": country_codes},
        ... )
        >>> df.select(
        ...     first_country_code=pl.dtype_of("country").enum.get_category(0),
        ... )
        shape: (1, 1)
        ┌────────────────────┐
        │ first_country_code │
        │ ---                │
        │ str                │
        ╞════════════════════╡
        │ US                 │
        └────────────────────┘
        """
        return pl.Expr._from_pyexpr(
            self._pydatatype_expr.enum_get_category(index, raise_on_oob)
        )

    def index_of_category(
        self, category: str, *, raise_on_missing: bool = True
    ) -> pl.Expr:
        """
        Get the index of a specific enum category.

        Parameters
        ----------
        category
            Category name to search for.
        raise_on_missing
            If the category cannot be found, should an exception be raised or
            should a missing value be inserted.

        Examples
        --------
        >>> country_codes = pl.Enum([ 'US', 'NL', 'CN', 'IN', 'FR' ])
        >>> df = pl.DataFrame({
        ...     'country': ['NL', 'FR', 'CN'],
        ... }, schema={ 'country': country_codes })
        >>> df.select(
        ...     india_enum_index = pl.dtype_of('country').enum.index_of_category('IN'),
        ... )
        shape: (1, 1)
        ┌──────────────────┐
        │ india_enum_index │
        │ ---              │
        │ u32              │
        ╞══════════════════╡
        │ 3                │
        └──────────────────┘
        """
        return pl.Expr._from_pyexpr(
            self._pydatatype_expr.enum_index_of_category(category, raise_on_missing)
        )
