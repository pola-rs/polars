from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import TypeAlias

    import polars as pl

    IntoExprColumn: TypeAlias = pl.Expr | str | pl.Series
