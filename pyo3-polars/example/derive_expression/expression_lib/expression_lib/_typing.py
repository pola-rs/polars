from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    import sys

    import polars as pl

    if sys.version_info >= (3, 10):
        from typing import TypeAlias
    else:
        from typing_extensions import TypeAlias

    IntoExprColumn: TypeAlias = Union[pl.Expr, str, pl.Series]
