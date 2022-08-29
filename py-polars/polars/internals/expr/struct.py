from __future__ import annotations

import polars.internals as pli


class ExprStructNameSpace:
    """Namespace for struct related expressions."""

    _accessor = "struct"

    def __init__(self, expr: pli.Expr):
        self._pyexpr = expr._pyexpr

    def __getitem__(self, item: str | int) -> pli.Expr:
        if isinstance(item, str):
            return self.field(item)
        elif isinstance(item, int):
            return pli.wrap_expr(self._pyexpr.struct_field_by_index(item))
        else:
            raise ValueError(f"expected type 'int | str', got {type(item)}")

    def field(self, name: str) -> pli.Expr:
        """
        Retrieve one of the fields of this `Struct` as a new Series.

        Parameters
        ----------
        name
            Name of the field

        Examples
        --------
        >>> df = (
        ...     pl.DataFrame(
        ...         {
        ...             "int": [1, 2],
        ...             "str": ["a", "b"],
        ...             "bool": [True, None],
        ...             "list": [[1, 2], [3]],
        ...         }
        ...     )
        ...     .to_struct("my_struct")
        ...     .to_frame()
        ... )
        >>> df.select(pl.col("my_struct").struct.field("str"))
        shape: (2, 1)
        ┌─────┐
        │ str │
        │ --- │
        │ str │
        ╞═════╡
        │ a   │
        ├╌╌╌╌╌┤
        │ b   │
        └─────┘

        """
        return pli.wrap_expr(self._pyexpr.struct_field_by_name(name))

    def rename_fields(self, names: list[str]) -> pli.Expr:
        """
        Rename the fields of the struct.

        Parameters
        ----------
        names
            New names in the order of the struct's fields

        Examples
        --------
        >>> df = (
        ...     pl.DataFrame(
        ...         {
        ...             "int": [1, 2],
        ...             "str": ["a", "b"],
        ...             "bool": [True, None],
        ...             "list": [[1, 2], [3]],
        ...         }
        ...     )
        ...     .to_struct("my_struct")
        ...     .to_frame()
        ... )
        >>> df = df.with_column(
        ...     pl.col("my_struct").struct.rename_fields(["INT", "STR", "BOOL", "LIST"])
        ... )

        Does NOT work anymore:
        # df.select(pl.col("my_struct").struct.field("int"))
        #               PanicException: int not found ^^^

        >>> df.select(pl.col("my_struct").struct.field("INT"))
        shape: (2, 1)
        ┌─────┐
        │ INT │
        │ --- │
        │ i64 │
        ╞═════╡
        │ 1   │
        ├╌╌╌╌╌┤
        │ 2   │
        └─────┘

        """
        return pli.wrap_expr(self._pyexpr.struct_rename_fields(names))
