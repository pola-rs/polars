from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
from polars.io.plugins import register_io_source

if TYPE_CHECKING:
    from collections.abc import Iterator


def test_io_plugin_predicate_no_serialization_21130() -> None:
    def custom_io() -> pl.LazyFrame:
        def source_generator(
            with_columns: list[str] | None,
            predicate: pl.Expr | None,
            n_rows: int | None,
            batch_size: int | None,
        ) -> Iterator[pl.DataFrame]:
            df = pl.DataFrame(
                {"json_val": ['{"a":"1"}', None, '{"a":2}', '{"a":2.1}', '{"a":true}']}
            )
            print(predicate)
            if predicate is not None:
                df = df.filter(predicate)
            if batch_size and df.height > batch_size:
                yield from df.iter_slices(n_rows=batch_size)
            else:
                yield df

        return register_io_source(
            io_source=source_generator, schema={"json_val": pl.String}
        )

    lf = custom_io()
    assert lf.filter(
        pl.col("json_val").str.json_path_match("$.a").is_in(["1"])
    ).collect().to_dict(as_series=False) == {"json_val": ['{"a":"1"}']}
