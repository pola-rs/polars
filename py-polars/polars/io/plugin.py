from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Iterator

import polars._reexport as pl
from polars._utils.unstable import unstable

if TYPE_CHECKING:
    from polars import DataFrame, Expr, LazyFrame
    from polars._typing import SchemaDict


@unstable()
def register_io_source(
    callable: Callable[
        [list[str] | None, Expr | None, int | None, int | None], Iterator[DataFrame]
    ],
    schema: SchemaDict,
) -> LazyFrame:
    """
    Register your IO plugin and initialize a LazyFrame.

    Parameters
    ----------
    callable
        Function that accepts the following arguments:
        `with_columns`
            Columns that are projected. The reader must
            project these columns if applied
        predicate
            Polars expression. The reader must filter
            there rows accordingly.
        n_rows:
            Materialize only n rows from the source.
            The reader can stop when `n_rows` are read.
        batch_size
            A hint of the ideal batch size the readers
            generator must produce.
    schema
        Schema that the reader will produce before projection pushdown.

    """
    return pl.LazyFrame._scan_python_function(
        schema=schema, scan_fn=callable, pyarrow=False
    )
