"""Struct-field projection pushdown at the IPC and NDJSON sources."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pyarrow as pa
import pytest

import polars as pl
from polars.exceptions import PolarsError
from polars.testing import assert_frame_equal
from tests.unit.io.test_struct_field_pruning import (
    NO_PUSHDOWN,
    QUERIES,
    assert_same_as_without_pushdown,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

s = pl.col("s")

STRUCT_DF = pl.DataFrame(
    {
        "id": [1, 2, 3, 4],
        "s": [
            {"a": 1, "b": "x", "c": {"x": 1.5, "y": [1, 2]}},
            None,
            {"a": 3, "b": None, "c": None},
            {"a": None, "b": "z", "c": {"x": None, "y": None}},
        ],
    },
    schema={
        "id": pl.Int64,
        "s": pl.Struct(
            {
                "a": pl.Int64,
                "b": pl.String,
                "c": pl.Struct({"x": pl.Float64, "y": pl.List(pl.Int64)}),
            }
        ),
    },
)

FORMATS = ["ipc", "ipc-zstd", "ndjson"]


def _write(df: pl.DataFrame, path: Path, fmt: str) -> None:
    if fmt == "ipc":
        df.write_ipc(path, compression="uncompressed")
    elif fmt == "ipc-zstd":
        df.write_ipc(path, compression="zstd")
    else:
        df.write_ndjson(path)


def _scan(path: Path | list[Path], fmt: str) -> pl.LazyFrame:
    if fmt.startswith("ipc"):
        return pl.scan_ipc(path)
    return pl.scan_ndjson(path)


@pytest.mark.parametrize("fmt", FORMATS)
@pytest.mark.parametrize("query", QUERIES)
def test_struct_field_pruning_sources(
    query: Callable[[pl.LazyFrame], pl.LazyFrame], fmt: str, tmp_path: Path
) -> None:
    path = tmp_path / "s"
    _write(STRUCT_DF, path, fmt)
    assert_same_as_without_pushdown(query(_scan(path, fmt)))


@pytest.mark.parametrize("fmt", FORMATS)
def test_struct_field_pruning_sources_explain(fmt: str, tmp_path: Path) -> None:
    path = tmp_path / "s"
    _write(STRUCT_DF, path, fmt)
    lf = _scan(path, fmt)

    assert "PROJECT FIELDS: s{a}" in lf.select(s.struct.field("a")).explain()
    assert (
        "PROJECT FIELDS: s{c{x}}"
        in lf.select(s.struct.field("c").struct.field("x")).explain()
    )
    assert "PROJECT FIELDS: s{}" in lf.select(s.is_null()).explain()
    assert "PROJECT FIELDS" not in lf.select(s).explain()


@pytest.mark.parametrize("fmt", FORMATS)
def test_struct_field_pruning_sources_validity_only(fmt: str, tmp_path: Path) -> None:
    df = pl.DataFrame({"s": [{"a": 1}, None, {"a": None}, None, {"a": 5}] * 1000})
    path = tmp_path / "s"
    _write(df, path, fmt)
    lf = _scan(path, fmt)

    out = assert_same_as_without_pushdown(
        lf.select(s.is_null().alias("n"), s.is_not_null().sum().alias("v"))
    )
    assert out["n"].to_list() == [False, True, False, True, False] * 1000
    assert out["v"].to_list() == [3000] * 5000
    assert_same_as_without_pushdown(lf.select(pl.len()))
    assert_same_as_without_pushdown(lf.select(s.null_count()))


@pytest.mark.parametrize("fmt", FORMATS)
def test_struct_field_pruning_sources_multiple_files(fmt: str, tmp_path: Path) -> None:
    for i in range(3):
        _write(
            pl.DataFrame(
                {
                    "s": [{"a": i, "b": str(i), "c": {"x": i, "y": "y"}}, None],
                    "i": [i, i],
                }
            ),
            tmp_path / f"{i}",
            fmt,
        )

    lf = _scan([tmp_path / f"{i}" for i in range(3)], fmt)
    assert_same_as_without_pushdown(lf.select(s.struct.field("b"), "i"))
    assert_same_as_without_pushdown(
        lf.filter(pl.col("i") > 0).select(s.struct.field("c").struct.field("y"))
    )
    assert_same_as_without_pushdown(lf.head(3).select(s.struct.field("a")))
    assert_same_as_without_pushdown(lf.tail(3).select(s.struct.field("a")))
    assert_same_as_without_pushdown(
        lf.with_row_index().select("index", s.struct.field("a"))
    )
    assert_same_as_without_pushdown(lf.select(s.struct.field("a"), s.struct.field("c")))


def _assert_same_or_both_raise(lf: pl.LazyFrame) -> None:
    """Without pushdown, either the same result or the same error."""
    try:
        lf.collect(optimizations=NO_PUSHDOWN)
    except PolarsError as e:
        for engine in ("in-memory", "streaming"):
            with pytest.raises(type(e)):
                lf.collect(engine=engine)  # type: ignore[call-overload]
    else:
        assert_same_as_without_pushdown(lf)


@pytest.mark.parametrize("fmt", ["ipc", "ipc-zstd"])
def test_struct_field_pruning_ipc_schema_evolution(fmt: str, tmp_path: Path) -> None:
    _write(
        pl.DataFrame({"s": [{"a": 1, "b": "x"}, None, {"a": 3, "b": None}]}),
        tmp_path / "1",
        fmt,
    )
    # Struct fields missing from, or extra in, the second file: whatever the cast
    # policy makes of them without pushdown must not change with it.
    _write(pl.DataFrame({"s": [None, {"a": 5}, {"a": None}]}), tmp_path / "2", fmt)
    _write(pl.DataFrame({"s": [{"z": 0, "b": "y", "a": 3}]}), tmp_path / "3", fmt)

    for other in ["2", "3"]:
        lf = pl.scan_ipc([tmp_path / "1", tmp_path / other])
        for q in [
            lf.select(s.struct.field("b"), s.is_null().alias("is_null")),
            lf.select(s.struct.field("a")),
            lf.select(s.is_null()),
            lf.select(s),
        ]:
            _assert_same_or_both_raise(q)


@pytest.mark.parametrize("fmt", ["ipc", "ipc-zstd"])
def test_struct_field_pruning_ipc_nested_and_dictionaries(
    fmt: str, tmp_path: Path
) -> None:
    df = pl.DataFrame(
        {
            "s": [
                {
                    "cat": "u",
                    "enum": "p",
                    "inner": {"x": 1, "cat": "v", "l": [{"q": 1}]},
                    "arr": [1, 2],
                },
                None,
                {"cat": None, "enum": "q", "inner": None, "arr": None},
            ]
            * 100,
        },
        schema={
            "s": pl.Struct(
                {
                    "cat": pl.Categorical(),
                    "enum": pl.Enum(["p", "q"]),
                    "inner": pl.Struct(
                        {
                            "x": pl.Int32,
                            "cat": pl.Categorical(),
                            "l": pl.List(pl.Struct({"q": pl.Int8})),
                        }
                    ),
                    "arr": pl.Array(pl.Int64, 2),
                }
            )
        },
    )
    path = tmp_path / "s"
    _write(df, path, fmt)
    lf = pl.scan_ipc(path)

    for q in [
        lf.select(s.struct.field("enum")),
        lf.select(s.struct.field("cat")),
        lf.select(s.struct.field("arr")),
        lf.select(s.struct.field("inner").struct.field("cat")),
        lf.select(s.struct.field("inner").struct.field("l")),
        lf.select(s.struct.field("inner").struct.field("x"), s.struct.field("arr")),
        lf.select(s.struct.field("inner").is_null()),
        lf.select(s.struct.field("inner").struct.field("x"), s.is_null()),
    ]:
        assert_same_as_without_pushdown(q)


def test_struct_field_pruning_ipc_skips_unselected_fields(tmp_path: Path) -> None:
    marker = "QZQZQZQZ"
    n = 1000
    table = pa.table(
        {
            "s": pa.array(
                [
                    {"a": i, "b": marker * 4, "c": {"x": i, "y": marker}}
                    for i in range(n)
                ],
                type=pa.struct(
                    [
                        ("a", pa.int64()),
                        ("b", pa.string()),
                        ("c", pa.struct([("x", pa.int64()), ("y", pa.string())])),
                    ]
                ),
            )
        }
    )
    path = tmp_path / "corrupt.arrow"
    with pa.ipc.new_file(path, table.schema) as writer:
        writer.write_table(table, max_chunksize=300)

    # Make the string values of `b` and `c.y` invalid UTF-8.
    data = path.read_bytes()
    assert marker.encode() in data
    path.write_bytes(data.replace(marker.encode(), b"\xff" * len(marker)))

    lf = pl.scan_ipc(path)

    # Reading the corrupted fields fails, so these only pass if they are not read.
    with pytest.raises(PolarsError):
        lf.select(s.struct.field("a")).collect(optimizations=NO_PUSHDOWN)

    expected = pl.DataFrame(
        {"a": range(n), "x": range(n)}, schema={"a": pl.Int64, "x": pl.Int64}
    )
    for engine in ("in-memory", "streaming"):
        assert_frame_equal(
            lf.select(
                s.struct.field("a"), s.struct.field("c").struct.field("x")
            ).collect(engine=engine),  # type: ignore[call-overload]
            expected,
        )
        out = lf.select(s.is_null().sum()).collect(engine=engine)  # type: ignore[call-overload]
        assert out.item() == 0


def test_struct_field_pruning_ndjson_skips_unselected_fields(tmp_path: Path) -> None:
    path = tmp_path / "s.ndjson"
    path.write_text(
        '{"s": {"a": 1, "b": {"not": "an int"}}}\n'
        '{"s": null}\n'
        '{"s": {"a": 3, "b": "nor this"}}\n'
    )
    lf = pl.scan_ndjson(path, schema={"s": pl.Struct({"a": pl.Int64, "b": pl.Int64})})

    # Parsing `b` fails, so these only pass if it is not parsed.
    with pytest.raises(PolarsError):
        lf.select(s.struct.field("a")).collect(optimizations=NO_PUSHDOWN)

    for engine in ("in-memory", "streaming"):
        assert_frame_equal(
            lf.select(s.struct.field("a"), s.is_null().alias("n")).collect(
                engine=engine  # type: ignore[call-overload]
            ),
            pl.DataFrame({"a": [1, None, 3], "n": [False, True, False]}),
        )


def test_struct_field_pruning_ndjson_schema(tmp_path: Path) -> None:
    path = tmp_path / "s.ndjson"
    path.write_text(
        '{"s": {"a": 1, "b": "x", "extra": [1]}, "t": 1}\n'
        '{"s": {"b": "y"}, "t": 2}\n'
        '{"s": null, "t": 3}\n'
        '{"t": 4}\n'
    )
    schemas: list[pl.Schema | None] = [
        None,
        pl.Schema({"s": pl.Struct({"a": pl.Int64, "b": pl.String}), "t": pl.Int64()}),
        pl.Schema({"s": pl.Struct({"b": pl.String, "a": pl.Int32, "z": pl.Float64})}),
    ]
    for schema in schemas:
        for ignore_errors in [False, True]:
            lf = pl.scan_ndjson(path, schema=schema, ignore_errors=ignore_errors)
            assert_same_as_without_pushdown(lf.select(s.struct.field("b")))
            assert_same_as_without_pushdown(lf.select(s.struct.field("a"), s.is_null()))
            assert_same_as_without_pushdown(lf.select(s.is_null()))
