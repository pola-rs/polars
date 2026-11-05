"""Struct-field projection pushdown through lists and arrays."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import polars as pl
from polars.exceptions import PolarsError
from polars.testing import assert_frame_equal
from tests.unit.io.test_struct_field_pruning import (
    NO_PUSHDOWN,
    assert_same_as_without_pushdown,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

l = pl.col("l")  # noqa: E741
s = pl.col("s")
arr = pl.col("arr")
e = pl.element()

ELEMENT = pl.Struct(
    {
        "a": pl.Int64,
        "b": pl.String,
        "c": pl.Struct({"x": pl.Float64, "y": pl.List(pl.Int64)}),
    }
)

LIST_DF = pl.DataFrame(
    {
        "id": [1, 2, 3, 4, 5],
        "l": [
            [
                {"a": 1, "b": "x", "c": {"x": 1.5, "y": [1, 2]}},
                None,
                {"a": 3, "b": None, "c": None},
            ],
            None,
            [],
            [{"a": None, "b": "z", "c": {"x": None, "y": None}}],
            [
                {"a": 5, "b": "w", "c": {"x": 2.5, "y": []}},
                {"a": 6, "b": "v", "c": None},
            ],
        ],
        "s": [
            {"m": [{"p": 1, "q": "a"}], "n": 1},
            None,
            {"m": None, "n": 3},
            {"m": [], "n": None},
            {"m": [None, {"p": 5, "q": None}], "n": 5},
        ],
    },
    schema={
        "id": pl.Int64,
        "l": pl.List(ELEMENT),
        "s": pl.Struct(
            {"m": pl.List(pl.Struct({"p": pl.Int64, "q": pl.String})), "n": pl.Int64}
        ),
    },
)

ARRAY_DF = pl.DataFrame(
    {
        "id": [1, 2, 3],
        "arr": [
            [{"a": 1, "b": "x"}, None],
            None,
            [{"a": 3, "b": None}, {"a": None, "b": "z"}],
        ],
    },
    schema={
        "id": pl.Int64,
        "arr": pl.Array(pl.Struct({"a": pl.Int64, "b": pl.String}), 2),
    },
)

FORMATS = ["parquet", "ipc", "ndjson", "memory"]


def _scan(df: pl.DataFrame, fmt: str, tmp_path: Path) -> pl.LazyFrame:
    path = tmp_path / f"data.{fmt}"
    if fmt == "parquet":
        df.write_parquet(path)
        return pl.scan_parquet(path)
    if fmt == "ipc":
        df.write_ipc(path)
        return pl.scan_ipc(path)
    if fmt == "ndjson":
        df.write_ndjson(path)
        return pl.scan_ndjson(path, schema=df.schema)
    return df.lazy()


LIST_QUERIES: list[Callable[[pl.LazyFrame], pl.LazyFrame]] = [
    lambda lf: lf.select(l.list.eval(e.struct.field("a"))),
    lambda lf: lf.select(l.list.eval(e.struct.field("c").struct.field("x"))),
    lambda lf: lf.select(l.list.eval(e.struct.field("c").struct.field("y"))),
    lambda lf: lf.select(l.list.get(0, null_on_oob=True).struct.field("b")),
    lambda lf: lf.select(l.list.first().struct.field("a")),
    lambda lf: lf.select(l.list.last().struct.field("c").struct.field("x")),
    lambda lf: lf.select(l.list.first().is_null()),
    lambda lf: lf.select(l.list.first()),
    lambda lf: lf.select(l.explode().struct.field("a")),
    lambda lf: lf.select(l.explode().is_null()),
    lambda lf: lf.select(l.explode(keep_nulls=False).struct.field("b")),
    lambda lf: lf.explode("l").select(l.struct.field("a"), "id"),
    lambda lf: lf.explode("l").select("id"),
    lambda lf: lf.explode("l").select(l.is_null()),
    lambda lf: (
        lf.explode("l").filter(l.struct.field("a") > 1).select(l.struct.field("b"))
    ),
    lambda lf: lf.select(
        l.list.eval(e.struct.field("a")).alias("x"),
        l.list.first().struct.field("b"),
    ),
    lambda lf: lf.select(l.list.eval(e.struct.field("a")).alias("x"), l),
    lambda lf: lf.select(l, l.list.first().struct.field("b")),
    lambda lf: lf.select(l.list.len()),
    lambda lf: lf.select(l.is_null()),
    # Validity only of the list: one leaf under it.
    lambda lf: lf.select(
        l.is_not_null().sum().alias("a"), l.null_count().alias("b"), l.count()
    ),
    lambda lf: lf.select(l.has_nulls()),
    lambda lf: lf.filter(l.is_not_null()).select("id"),
    lambda lf: lf.select(s.struct.field("m").is_null()),
    # The row count of an explode.
    lambda lf: lf.explode("l").select(pl.len()),
    lambda lf: lf.select("l").explode("l").select(pl.len()),
    lambda lf: lf.select("l").explode("l", keep_nulls=False).select(pl.len()),
    lambda lf: lf.filter(l.list.first().struct.field("a") > 1).select(
        l.list.last().struct.field("b")
    ),
    lambda lf: lf.select(s.struct.field("m").list.eval(e.struct.field("q"))),
    lambda lf: lf.select(s.struct.field("m").list.first().struct.field("p")),
    lambda lf: lf.select(s.struct.field("m").explode().struct.field("p")),
    lambda lf: lf.select(
        s.struct.field("m").list.first().struct.field("p"), s.struct.field("n")
    ),
    # Not pruned: not a plain field chain or a constant index.
    lambda lf: lf.select(l.list.eval(e.struct.field("a") + 1)),
    lambda lf: lf.select(l.list.eval(e.struct.field("a").sum())),
    lambda lf: lf.select(
        l.list.get(pl.col("id") % 2, null_on_oob=True).struct.field("a")
    ),
    lambda lf: lf.select(l.list.eval(e.struct.json_encode())),
    # Rewritten mid-plan, then pruned through the list.
    lambda lf: lf.with_columns(t=l).select(pl.col("t").list.first().struct.field("a")),
    lambda lf: lf.group_by("id", maintain_order=True).agg(
        l.list.first().struct.field("a")
    ),
]


@pytest.mark.parametrize("fmt", FORMATS)
@pytest.mark.parametrize("query", LIST_QUERIES)
def test_struct_field_pruning_lists(
    query: Callable[[pl.LazyFrame], pl.LazyFrame], fmt: str, tmp_path: Path
) -> None:
    assert_same_as_without_pushdown(query(_scan(LIST_DF, fmt, tmp_path)))


ARRAY_QUERIES: list[Callable[[pl.LazyFrame], pl.LazyFrame]] = [
    lambda lf: lf.select(arr.arr.get(1, null_on_oob=True).struct.field("a")),
    lambda lf: lf.select(arr.arr.first().struct.field("b")),
    lambda lf: lf.select(arr.arr.last().is_null()),
    lambda lf: lf.select(arr.arr.eval(e.struct.field("b"))),
    lambda lf: lf.select(arr.arr.eval(e.struct.field("b"), as_list=True)),
    lambda lf: lf.select(arr.explode().struct.field("a")),
    lambda lf: lf.select(arr, arr.arr.first().struct.field("a").alias("x")),
    lambda lf: lf.select(arr.is_null()),
    lambda lf: lf.select("arr").explode("arr").select(pl.len()),
]


@pytest.mark.parametrize("fmt", ["parquet", "ipc", "memory"])
@pytest.mark.parametrize("query", ARRAY_QUERIES)
def test_struct_field_pruning_arrays(
    query: Callable[[pl.LazyFrame], pl.LazyFrame], fmt: str, tmp_path: Path
) -> None:
    assert_same_as_without_pushdown(query(_scan(ARRAY_DF, fmt, tmp_path)))


@pytest.mark.parametrize("fmt", ["parquet", "ipc", "ndjson"])
def test_struct_field_pruning_lists_explain(fmt: str, tmp_path: Path) -> None:
    lf = _scan(LIST_DF, fmt, tmp_path)

    def fields(q: pl.LazyFrame) -> str | None:
        for line in q.explain().splitlines():
            if "PROJECT FIELDS:" in line:
                return line.split("PROJECT FIELDS:")[1].strip()
        return None

    assert fields(lf.select(l.list.eval(e.struct.field("a")))) == "l[]{a}"
    assert fields(lf.select(l.list.first().struct.field("c").struct.field("x"))) == (
        "l[]{c{x}}"
    )
    assert fields(lf.select(l.explode().is_null())) == "l[]{}"
    assert fields(lf.explode("l").select("id")) == "l[]{}"
    assert fields(lf.select(l.is_null())) == "l[]{}"
    assert fields(lf.select(s.struct.field("m").is_null())) == "s{m[]{}}"
    # Counted with the exploded column, also when it is the only one.
    assert fields(lf.explode("l").select(pl.len())) == "l[]{}"
    assert fields(lf.select("l").explode("l").select(pl.len())) == "l[]{}"
    assert fields(lf.select(s.struct.field("m").list.last().struct.field("q"))) == (
        "s{m[]{q}}"
    )
    plan = fields(
        lf.select(l.list.first().struct.field("a"), l.list.last().struct.field("b"))
    )
    assert plan in ("l[]{a, b}", "l[]{b, a}")

    # Not pruned.
    for q in [
        lf.select(l.list.eval(e.struct.field("a") + 1)),
        lf.select(l.list.get(pl.col("id"), null_on_oob=True).struct.field("a")),
        lf.select(l.list.len()),
        lf.select(l, l.list.first().struct.field("a").alias("x")),
    ]:
        assert fields(q) is None, q.explain()


def test_struct_field_pruning_lists_concat(tmp_path: Path) -> None:
    path = tmp_path / "l.parquet"
    LIST_DF.write_parquet(path)
    # The scan is pruned, the in-memory frame is not: the concatenation aligns them.
    lf = pl.concat([pl.scan_parquet(path), LIST_DF.lazy()])
    assert_same_as_without_pushdown(lf.select(l.list.first().struct.field("a")))
    assert_same_as_without_pushdown(lf.select(l.explode().struct.field("b")))
    assert_same_as_without_pushdown(lf.select(s.struct.field("m").list.first()))


def _corrupt_chunk(path: Path, leaf: str, row_groups: list[int] | None = None) -> None:
    """Overwrite the column chunk of `leaf` (a dotted path) with garbage."""
    md = pq.ParquetFile(path).metadata
    data = bytearray(path.read_bytes())
    for rg in range(md.num_row_groups):
        if row_groups is not None and rg not in row_groups:
            continue
        chunks = [md.row_group(rg).column(i) for i in range(md.num_columns)]
        starts = sorted(
            min(c.data_page_offset, c.dictionary_page_offset or c.data_page_offset)
            for c in chunks
        )
        found = False
        for col in chunks:
            if col.path_in_schema != leaf:
                continue
            found = True
            start = min(
                col.data_page_offset, col.dictionary_page_offset or col.data_page_offset
            )
            end = min(
                [start + col.total_compressed_size] + [s for s in starts if s > start]
            )
            data[start:end] = b"\xff" * (end - start)
        assert found, leaf
    path.write_bytes(bytes(data))


def test_struct_field_pruning_lists_skip_unselected_leaves(tmp_path: Path) -> None:
    n = 1000
    df = pl.DataFrame(
        {
            "l": [
                [{"a": i, "b": f"long string {i}" * 10}, {"a": -i, "b": "x" * 30}]
                for i in range(n)
            ]
        }
    )
    path = tmp_path / "corrupt.parquet"
    pq.write_table(df.to_arrow(), path, use_dictionary=False, write_statistics=False)
    leaf = next(
        c
        for c in (
            pq.ParquetFile(path).metadata.row_group(0).column(i) for i in range(2)
        )
        if c.path_in_schema.endswith(".b")
    ).path_in_schema
    _corrupt_chunk(path, leaf)

    lf = pl.scan_parquet(path)
    with pytest.raises(PolarsError):
        lf.select(l.list.first().struct.field("a")).collect(optimizations=NO_PUSHDOWN)

    for engine in ("in-memory", "streaming"):
        for q, expected in [
            (
                lf.select(l.list.eval(e.struct.field("a"))),
                df.select(l.list.eval(e.struct.field("a"))),
            ),
            (
                lf.select(l.list.last().struct.field("a")),
                df.select(l.list.last().struct.field("a")),
            ),
            (lf.select(l.explode().is_null().sum()), pl.DataFrame({"l": [0]})),
            (lf.select(l.is_null().sum()), pl.DataFrame({"l": [0]})),
            # `l` is the file's only column.
            (lf.explode("l").select(pl.len()), pl.DataFrame({"len": [2 * n]})),
        ]:
            assert_frame_equal(
                q.collect(engine=engine),  # type: ignore[call-overload]
                expected,
                check_dtypes=False,
            )


def test_struct_field_pruning_validity_leaf_per_row_group(tmp_path: Path) -> None:
    # In the first row group `a` is the cheapest leaf of `s`, in the second `b`.
    n = 2000
    rg0 = {"s": [{"a": 0, "b": f"{i:>40}"} for i in range(n)]}
    rg1 = {"s": [{"a": i * 7919, "b": "x"} for i in range(n)]}
    t0 = pa.table(rg0)
    t1 = pa.table(rg1, schema=t0.schema)
    path = tmp_path / "rg.parquet"
    with pq.ParquetWriter(path, t0.schema, use_dictionary=False) as w:
        w.write_table(t0)
        w.write_table(t1)
    md = pq.ParquetFile(path).metadata
    assert md.num_row_groups == 2
    sizes = [
        {
            md.row_group(rg).column(i).path_in_schema: md.row_group(rg)
            .column(i)
            .total_compressed_size
            for i in range(2)
        }
        for rg in range(2)
    ]
    assert sizes[0]["s.a"] < sizes[0]["s.b"]
    assert sizes[1]["s.b"] < sizes[1]["s.a"]

    # Only `b` is readable in the second row group: reading `a` there fails.
    _corrupt_chunk(path, "s.a", row_groups=[1])

    lf = pl.scan_parquet(path)
    with pytest.raises(PolarsError):
        lf.select(s.is_null().sum()).collect(optimizations=NO_PUSHDOWN)
    for engine in ("in-memory", "streaming"):
        out = lf.select(s.is_null().sum()).collect(engine=engine)  # type: ignore[call-overload]
        assert out.item() == 0
        out = lf.select(pl.len()).collect(engine=engine)  # type: ignore[call-overload]
        assert out.item() == 2 * n
