from __future__ import annotations

import datetime
import io
import subprocess
import sys
from typing import TYPE_CHECKING

import numpy as np
import pytest

import polars as pl
from polars.io.plugins import register_io_source
from polars.testing import assert_frame_equal, assert_series_equal

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


def test_defer_validate_true() -> None:
    lf = pl.defer(
        lambda: pl.DataFrame({"a": np.ones(3)}),
        schema={"a": pl.Boolean},
        validate_schema=True,
    )
    with pytest.raises(pl.exceptions.SchemaError):
        lf.collect()


@pytest.mark.may_fail_cloud
@pytest.mark.may_fail_auto_streaming  # IO plugin validate=False schema mismatch
def test_defer_validate_false() -> None:
    lf = pl.defer(
        lambda: pl.DataFrame({"a": np.ones(3)}),
        schema={"a": pl.Boolean},
        validate_schema=False,
    )
    assert lf.collect().to_dict(as_series=False) == {"a": [1.0, 1.0, 1.0]}


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


@pytest.mark.may_fail_cloud
@pytest.mark.may_fail_auto_streaming  # IO plugin validate=False schema mismatch
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


@pytest.mark.parametrize(("validate"), [(True), (False)])
def test_reordered_columns_22731(validate: bool) -> None:
    def my_scan() -> pl.LazyFrame:
        schema = pl.Schema({"a": pl.Int64, "b": pl.Int64})

        def source_generator(
            with_columns: list[str] | None,
            predicate: pl.Expr | None,
            n_rows: int | None,
            batch_size: int | None,
        ) -> Iterator[pl.DataFrame]:
            df = pl.DataFrame({"a": [1, 2, 3], "b": [42, 13, 37]})

            if n_rows is not None:
                df = df.head(min(n_rows, df.height))

            maxrows = 1
            if batch_size is not None:
                maxrows = batch_size

            while df.height > 0:
                maxrows = min(maxrows, df.height)
                cur = df.head(maxrows)
                df = df.slice(maxrows)

                if predicate is not None:
                    cur = cur.filter(predicate)
                if with_columns is not None:
                    cur = cur.select(with_columns)

                yield cur

        return register_io_source(
            io_source=source_generator, schema=schema, validate_schema=validate
        )

    expected_select = pl.DataFrame({"b": [42, 13, 37], "a": [1, 2, 3]})
    assert_frame_equal(my_scan().select("b", "a").collect(), expected_select)

    expected_ri = pl.DataFrame({"b": [42, 13, 37], "a": [1, 2, 3]}).with_row_index()
    assert_frame_equal(
        my_scan().select("b", "a").with_row_index().collect(),
        expected_ri,
    )

    expected_with_columns = pl.DataFrame({"a": [1, 2, 3], "b": [42, 13, 37]})
    assert_frame_equal(
        my_scan().with_columns("b", "a").collect(), expected_with_columns
    )


def test_io_plugin_reentrant_deadlock() -> None:
    out = subprocess.check_output(
        [
            sys.executable,
            "-c",
            """\
from __future__ import annotations

import os
import sys

os.environ["POLARS_MAX_THREADS"] = "1"

import polars as pl
from polars.io.plugins import register_io_source

assert pl.thread_pool_size() == 1

n = 3
i = 0


def reentrant(
    with_columns: list[str] | None,
    predicate: pl.Expr | None,
    n_rows: int | None,
    batch_size: int | None,
):
    global i

    df = pl.DataFrame({"x": 1})

    if i < n:
        i += 1
        yield register_io_source(io_source=reentrant, schema={"x": pl.Int64}).collect()

    yield df


register_io_source(io_source=reentrant, schema={"x": pl.Int64}).collect()

print("OK", end="", file=sys.stderr)
""",
        ],
        stderr=subprocess.STDOUT,
        timeout=7,
    )

    assert out == b"OK"


def test_io_plugin_categorical_24172() -> None:
    schema = {"cat": pl.Categorical}

    df = pl.concat(
        [
            pl.DataFrame({"cat": ["X", "Y"]}, schema=schema),
            pl.DataFrame({"cat": ["X", "Y"]}, schema=schema),
        ],
        rechunk=False,
    )

    assert df.n_chunks() == 2

    assert_frame_equal(
        register_io_source(lambda *_: iter([df]), schema=df.schema).collect(),
        df,
    )
