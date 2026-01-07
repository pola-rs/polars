from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from typing import TypeAlias

    import polars as pl

    IntoExprColumn: TypeAlias = Union[pl.Expr, str, pl.Series]
