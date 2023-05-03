from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

from polars.utils._wrap import wrap_expr

if TYPE_CHECKING:
    from polars import Expr


class ExprStructNameSpace:
    """Namespace for struct related expressions."""

    _accessor = "struct"

    def __init__(self, expr: Expr):
        self._pyexpr = expr._pyexpr

    def __getitem__(self, item: str | int) -> Expr:
        if isinstance(item, str):
            return self.field(item)
        elif isinstance(item, int):
            return wrap_expr(self._pyexpr.struct_field_by_index(item))
        else:
            raise ValueError(
                f"expected type 'int | str', got {type(item).__name__} ({item!r})"
            )

    def field(self, name: str) -> Expr:
        """
        Retrieve a ``Struct`` field as a new Series.

        Parameters
        ----------
        name
            Name of the struct field to retrieve.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "aaa": [1, 2],
        ...         "bbb": ["ab", "cd"],
        ...         "ccc": [True, None],
        ...         "ddd": [[1, 2], [3]],
        ...     }
        ... ).select(pl.struct(["aaa", "bbb", "ccc", "ddd"]).alias("struct_col"))
        >>> df
        shape: (2, 1)
        ┌──────────────────────┐
        │ struct_col           │
        │ ---                  │
        │ struct[4]            │
        ╞══════════════════════╡
        │ {1,"ab",true,[1, 2]} │
        │ {2,"cd",null,[3]}    │
        └──────────────────────┘

        Retrieve struct field(s) as Series:

        >>> df.select(pl.col("struct_col").struct.field("bbb"))
        shape: (2, 1)
        ┌─────┐
        │ bbb │
        │ --- │
        │ str │
        ╞═════╡
        │ ab  │
        │ cd  │
        └─────┘

        >>> df.select(
        ...     pl.col("struct_col").struct.field("bbb"),
        ...     pl.col("struct_col").struct.field("ddd"),
        ... )
        shape: (2, 2)
        ┌─────┬───────────┐
        │ bbb ┆ ddd       │
        │ --- ┆ ---       │
        │ str ┆ list[i64] │
        ╞═════╪═══════════╡
        │ ab  ┆ [1, 2]    │
        │ cd  ┆ [3]       │
        └─────┴───────────┘

        """
        return wrap_expr(self._pyexpr.struct_field_by_name(name))

    def rename_fields(self, names: Sequence[str]) -> Expr:
        """
        Rename the fields of the struct.

        Parameters
        ----------
        names
            New names, given in the same order as the struct's fields.

        Examples
        --------
        >>> df = pl.DataFrame(
        ...     {
        ...         "aaa": [1, 2],
        ...         "bbb": ["ab", "cd"],
        ...         "ccc": [True, None],
        ...         "ddd": [[1, 2], [3]],
        ...     }
        ... ).select(pl.struct(["aaa", "bbb", "ccc", "ddd"]).alias("struct_col"))
        >>> df
        shape: (2, 1)
        ┌──────────────────────┐
        │ struct_col           │
        │ ---                  │
        │ struct[4]            │
        ╞══════════════════════╡
        │ {1,"ab",true,[1, 2]} │
        │ {2,"cd",null,[3]}    │
        └──────────────────────┘

        >>> df.unnest("struct_col")
        shape: (2, 4)
        ┌─────┬─────┬──────┬───────────┐
        │ aaa ┆ bbb ┆ ccc  ┆ ddd       │
        │ --- ┆ --- ┆ ---  ┆ ---       │
        │ i64 ┆ str ┆ bool ┆ list[i64] │
        ╞═════╪═════╪══════╪═══════════╡
        │ 1   ┆ ab  ┆ true ┆ [1, 2]    │
        │ 2   ┆ cd  ┆ null ┆ [3]       │
        └─────┴─────┴──────┴───────────┘

        Rename fields:

        >>> df = df.select(
        ...     pl.col("struct_col").struct.rename_fields(["www", "xxx", "yyy", "zzz"])
        ... )
        >>> df.unnest("struct_col")
        shape: (2, 4)
        ┌─────┬─────┬──────┬───────────┐
        │ www ┆ xxx ┆ yyy  ┆ zzz       │
        │ --- ┆ --- ┆ ---  ┆ ---       │
        │ i64 ┆ str ┆ bool ┆ list[i64] │
        ╞═════╪═════╪══════╪═══════════╡
        │ 1   ┆ ab  ┆ true ┆ [1, 2]    │
        │ 2   ┆ cd  ┆ null ┆ [3]       │
        └─────┴─────┴──────┴───────────┘

        Following a rename, the previous field names (obviously) cannot be referenced:

        >>> df.select(pl.col("struct_col").struct.field("aaa"))  # doctest: +SKIP
        StructFieldNotFoundError: aaa

        """
        return wrap_expr(self._pyexpr.struct_rename_fields(names))
