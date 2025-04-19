from __future__ import annotations

import datetime
import io
from typing import TYPE_CHECKING

import numpy as np
import pytest

import polars as pl
from polars.io.plugins import register_io_source
from polars.testing import assert_series_equal

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


def test_defer() -> None:
    lf = pl.defer(
        lambda: pl.DataFrame({"a": np.ones(3)}),
        schema={"a": pl.Boolean},
        validate_schema=False,
    )
    assert lf.collect().to_dict(as_series=False) == {"a": [1.0, 1.0, 1.0]}
    lf = pl.defer(
        lambda: pl.DataFrame({"a": np.ones(3)}),
        schema={"a": pl.Boolean},
        validate_schema=True,
    )
    with pytest.raises(pl.exceptions.SchemaError):
        lf.collect()


def test_empty_iterator_io_plugin() -> None:
    def _io_source(
        with_columns: list[str] | None,
        predicate: pl.Expr | None,
        n_rows: int | None,
        batch_size: int | None,
    ) -> Iterator[pl.DataFrame]:
        yield from []

    schema = pl.Schema([("a", pl.Int64)])
    df = register_io_source(_io_source, schema=schema)
    assert df.collect().schema == schema


def test_scan_lines() -> None:
    def scan_lines(f: io.BytesIO) -> pl.LazyFrame:
        schema = pl.Schema({"lines": pl.String()})

        def generator(
            with_columns: list[str] | None,
            predicate: pl.Expr | None,
            n_rows: int | None,
            batch_size: int | None,
        ) -> Iterator[pl.DataFrame]:
            x = f
            if batch_size is None:
                batch_size = 100_000

            batch_lines: list[str] = []
            while n_rows != 0:
                batch_lines.clear()
                remaining_rows = batch_size
                if n_rows is not None:
                    remaining_rows = min(remaining_rows, n_rows)
                    n_rows -= remaining_rows

                while remaining_rows != 0 and (line := x.readline().rstrip()):
                    if isinstance(line, str):
                        batch_lines += [batch_lines]
                    else:
                        batch_lines += [line.decode()]
                    remaining_rows -= 1

                df = pl.Series("lines", batch_lines, pl.String()).to_frame()

                if with_columns is not None:
                    df = df.select(with_columns)
                if predicate is not None:
                    df = df.filter(predicate)

                yield df

                if remaining_rows != 0:
                    break

        return register_io_source(io_source=generator, schema=schema)

    text = """
Hello
This is some text
It is spread over multiple lines
This allows it to read into multiple rows.
    """.strip()
    f = io.BytesIO(bytes(text, encoding="utf-8"))

    assert_series_equal(
        scan_lines(f).collect().to_series(),
        pl.Series("lines", text.splitlines(), pl.String()),
    )


def test_datetime_io_predicate_pushdown_21790() -> None:
    recorded: dict[str, pl.Expr | None] = {"predicate": None}
    df = pl.DataFrame(
        {
            "timestamp": [
                datetime.datetime(2024, 1, 1, 0),
                datetime.datetime(2024, 1, 3, 0),
            ]
        }
    )

    def _source(
        with_columns: list[str] | None,
        predicate: pl.Expr | None,
        n_rows: int | None,
        batch_size: int | None,
    ) -> Iterator[pl.DataFrame]:
        # capture the predicate passed in
        recorded["predicate"] = predicate
        inner_df = df.clone()
        if with_columns is not None:
            inner_df = inner_df.select(with_columns)
        if predicate is not None:
            inner_df = inner_df.filter(predicate)

        yield inner_df

    schema = {"timestamp": pl.Datetime(time_unit="ns")}
    lf = register_io_source(io_source=_source, schema=schema)

    cutoff = datetime.datetime(2024, 1, 4)
    expr = pl.col("timestamp") < cutoff
    filtered_df = lf.filter(expr).collect()

    pushed_predicate = recorded["predicate"]
    assert pushed_predicate is not None
    assert_series_equal(filtered_df.to_series(), df.filter(expr).to_series())

    # check the expression directly
    dt_val, column_cast = pushed_predicate.meta.pop()
    # Extract the datetime value from the expression
    assert pl.DataFrame({}).select(dt_val).item() == cutoff

    column = column_cast.meta.pop()[0]
    assert column.meta == pl.col("timestamp")
