"""Struct-field projection pushdown through lists and arrays."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pyarrow as pa
import pyarrow.feather as feather
import pyarrow.parquet as pq
import pytest

import polars as pl
from polars.exceptions import PolarsError
from tests.unit.io.test_struct_field_pruning import (
    NO_PUSHDOWN,
    assert_engines_equal,
    assert_raises_with_and_without_pushdown,
    assert_same_as_without_pushdown,
    corrupt_chunk,
    project_fields,
    scan,
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

m = s.struct.field("m")

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
    lambda lf: lf.select(m.is_null()),
    # The row count of an explode.
    lambda lf: lf.explode("l").select(pl.len()),
    lambda lf: lf.select("l").explode("l").select(pl.len()),
    lambda lf: lf.select("l").explode("l", keep_nulls=False).select(pl.len()),
    lambda lf: lf.filter(l.list.first().struct.field("a") > 1).select(
        l.list.last().struct.field("b")
    ),
    lambda lf: lf.select(m.list.eval(e.struct.field("q"))),
    lambda lf: lf.select(m.list.first().struct.field("p")),
    lambda lf: lf.select(m.explode().struct.field("p")),
    lambda lf: lf.select(m.list.first().struct.field("p"), s.struct.field("n")),
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


@pytest.mark.parametrize("fmt", ["parquet", "ipc", "ndjson", "memory"])
@pytest.mark.parametrize("query", LIST_QUERIES)
def test_struct_field_pruning_lists(
    query: Callable[[pl.LazyFrame], pl.LazyFrame], fmt: str, tmp_path: Path
) -> None:
    assert_same_as_without_pushdown(query(scan(LIST_DF, fmt, tmp_path)))


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
    assert_same_as_without_pushdown(query(scan(ARRAY_DF, fmt, tmp_path)))


@pytest.mark.parametrize("fmt", ["parquet", "ipc", "ndjson"])
def test_struct_field_pruning_lists_explain(fmt: str, tmp_path: Path) -> None:
    lf = scan(LIST_DF, fmt, tmp_path)
    first, last = l.list.first(), l.list.last()

    for q, fields in [
        (lf.select(l.list.eval(e.struct.field("a"))), "l[]{a}"),
        (lf.select(first.struct.field("c").struct.field("x")), "l[]{c{x}}"),
        (lf.select(l.explode().is_null()), "l[]{}"),
        (lf.explode("l").select("id"), "l[]{}"),
        (lf.select(l.is_null()), "l[]{}"),
        (lf.select(m.is_null()), "s{m[]{}}"),
        # Counted with the exploded column, also when it is the only one.
        (lf.explode("l").select(pl.len()), "l[]{}"),
        (lf.select("l").explode("l").select(pl.len()), "l[]{}"),
        (lf.select(m.list.last().struct.field("q")), "s{m[]{q}}"),
    ]:
        assert project_fields(q) == fields
    q = lf.select(first.struct.field("a"), last.struct.field("b"))
    assert project_fields(q) in ("l[]{a, b}", "l[]{b, a}")

    # Not pruned.
    for q in [
        lf.select(l.list.eval(e.struct.field("a") + 1)),
        lf.select(l.list.get(pl.col("id"), null_on_oob=True).struct.field("a")),
        lf.select(l.list.len()),
        lf.select(l, first.struct.field("a").alias("x")),
    ]:
        assert project_fields(q) is None, q.explain()


def test_struct_field_pruning_lists_concat(tmp_path: Path) -> None:
    # The scan is pruned, the in-memory frame is not: the concatenation aligns them.
    lf = pl.concat([scan(LIST_DF, "parquet", tmp_path), LIST_DF.lazy()])
    assert_same_as_without_pushdown(lf.select(l.list.first().struct.field("a")))
    assert_same_as_without_pushdown(lf.select(l.explode().struct.field("b")))
    assert_same_as_without_pushdown(lf.select(m.list.first()))


NESTED = pl.DataFrame(
    {
        "ll": [
            [[{"a": 1, "b": "p"}], None, []],
            None,
            [],
            [[None, {"a": 2, "b": "q"}]],
        ],
        "al": [[[{"a": 1, "b": "p"}]], None, [[]], [None]],
        "s": [
            {"m": [[{"p": 1, "q": "x"}]], "k": [{"p": 2, "q": "y"}]},
            None,
            {"m": None, "k": []},
            {"m": [[]], "k": None},
        ],
    },
    schema={
        "ll": pl.List(pl.List(pl.Struct({"a": pl.Int64, "b": pl.String}))),
        "al": pl.Array(pl.List(pl.Struct({"a": pl.Int64, "b": pl.String})), 1),
        "s": pl.Struct(
            {
                "m": pl.List(pl.List(pl.Struct({"p": pl.Int64, "q": pl.String}))),
                "k": pl.List(pl.Struct({"p": pl.Int64, "q": pl.String})),
            }
        ),
    },
)


@pytest.mark.parametrize("fmt", ["parquet", "ipc"])
@pytest.mark.parametrize("how", ["vertical", "diagonal"])
def test_struct_field_pruning_concat_aligns_nested_lists(
    how: str, fmt: str, tmp_path: Path
) -> None:
    lf = pl.concat([scan(NESTED, fmt, tmp_path), NESTED.lazy()], how=how)  # type: ignore[arg-type]
    ll, al = pl.col("ll"), pl.col("al")
    for q in [
        lf.select(ll.list.first().list.first().struct.field("a")),
        lf.select(ll.explode().explode().is_null()),
        lf.select(al.arr.first().list.first().struct.field("a")),
        lf.select(al.explode().explode().is_null()),
        lf.select(m.list.first().list.first().struct.field("q")),
        lf.select(
            s.struct.field("k").list.eval(e.struct.field("p")),
            m.list.first().list.len(),
        ),
        lf.select(m.explode().explode().is_null()),
    ]:
        assert_same_as_without_pushdown(q)


@pytest.mark.parametrize("policy", ["insert", "raise"])
def test_struct_field_pruning_missing_field_under_list(
    policy: str, tmp_path: Path
) -> None:
    full = pl.List(pl.Struct({"a": pl.Int64, "b": pl.String, "c": pl.Int64}))
    pl.DataFrame(
        {"l": [[{"a": 9, "b": "n", "c": 9}]]}, schema={"l": full}
    ).write_parquet(tmp_path / "1.parquet")
    pl.DataFrame({"l": [[{"a": 1, "b": "x"}, None], None, []]}).write_parquet(
        tmp_path / "2.parquet"
    )
    lf = pl.scan_parquet(
        [tmp_path / "1.parquet", tmp_path / "2.parquet"],
        cast_options=pl.ScanCastOptions(missing_struct_fields=policy),  # type: ignore[arg-type]
    )
    for q in [
        lf.select(l.list.eval(e.struct.field("c"))),
        lf.select(l.list.eval(e.struct.field("a"))),
        lf.select(l.explode().is_null()),
    ]:
        if policy == "raise":
            assert_raises_with_and_without_pushdown(q)
        else:
            assert_same_as_without_pushdown(q)


@pytest.mark.parametrize("fmt", ["parquet", "ipc", "ndjson"])
def test_struct_field_pruning_zero_rows_and_null_fields(
    fmt: str, tmp_path: Path
) -> None:
    el = pl.Struct({"n": pl.Null, "a": pl.String})
    df = pl.DataFrame(
        {
            "s": [{"n": None, "a": "x"}, None, {"n": None, "a": None}],
            "l": [[{"n": None, "a": "y" * 50}, None], None, []],
        },
        schema={"s": el, "l": pl.List(el)},
    )
    for frame, name in [(df, "full"), (df.clear(), "empty")]:
        lf = scan(frame, fmt, tmp_path, name)
        for q in [
            s.is_null(),
            l.explode().is_null(),
            s.struct.field("n"),
            l.list.first().struct.field("a"),
        ]:
            assert_same_as_without_pushdown(lf.select(q))


def test_struct_field_pruning_ipc_views_compression_batches(tmp_path: Path) -> None:
    # String views with variadic buffers before and after the read field, in several
    # record batches, compressed or not.
    df = pl.DataFrame(
        {
            "s": [
                {
                    "a": f"long string value {i}" * (i % 3),
                    "b": i,
                    "c": f"another long string {i}" * 2,
                    "d": ["x" * 20] * (i % 4),
                }
                if i % 7
                else None
                for i in range(3000)
            ],
            "l": [
                [{"p": [{"u": i, "v": f"{i}" * 15}], "q": f"q{i}" * 10}, None]
                if i % 5
                else None
                for i in range(3000)
            ],
        }
    )
    for compression in ["uncompressed", "lz4", "zstd"]:
        path = tmp_path / f"{compression}.ipc"
        df.write_ipc(path, compression=compression, record_batch_size=700)  # type: ignore[call-overload]
        for lf in [pl.scan_ipc(path), pl.scan_ipc(path, n_rows=1000)]:
            for q in [
                lf.select(s.struct.field("c")),
                lf.select(s.struct.field("d")),
                lf.select(s.is_null()),
                lf.select(
                    l.list.eval(e.struct.field("p").list.eval(e.struct.field("v")))
                ),
                lf.select(l.explode().struct.field("p").explode().struct.field("u")),
                lf.slice(1500, 800).select(
                    s.struct.field("c"), l.list.last().struct.field("q")
                ),
            ]:
                assert_same_as_without_pushdown(q)


@pytest.mark.parametrize("fmt", ["parquet", "ipc"])
def test_struct_field_pruning_pyarrow_written_lists(fmt: str, tmp_path: Path) -> None:
    # LargeList, FixedSizeList, large strings and dictionaries as pyarrow writes them.
    element = pa.struct(
        [("a", pa.int64()), ("b", pa.large_string()), ("c", pa.string())]
    )
    x, y = {"a": 1, "b": "x", "c": "y"}, {"a": None, "b": None, "c": "q"}
    table = pa.table(
        {
            "ll": pa.array([[x, None], None, [], [y]], type=pa.large_list(element)),
            "fl": pa.array(
                [[x, None], None, [None, y], [None, None]], type=pa.list_(element, 2)
            ),
            "d": pa.array(
                [{"cat": "u", "v": 1}, {"cat": "w", "v": None}, None, {"v": 4}],
                type=pa.struct(
                    [("cat", pa.dictionary(pa.int32(), pa.string())), ("v", pa.int64())]
                ),
            ),
        }
    )
    path = tmp_path / f"pa.{fmt}"
    if fmt == "ipc":
        feather.write_feather(table, path, compression="lz4")
        lf = pl.scan_ipc(path)
    else:
        pq.write_table(table, path)
        lf = pl.scan_parquet(path)
    ll, fl, d = pl.col("ll"), pl.col("fl"), pl.col("d")
    for q in [
        lf.select(ll.list.eval(e.struct.field("c"))),
        lf.select(ll.explode().is_null()),
        lf.select(fl.arr.eval(e.struct.field("c"))),
        lf.select(fl.explode().is_null()),
        lf.select(d.struct.field("v")),
        lf.select(d.struct.field("cat")),
        lf.slice(1, 2).select(fl.arr.last().struct.field("c")),
    ]:
        assert_same_as_without_pushdown(q)


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
    corrupt_chunk(path, "l.list.element.b")

    lf = pl.scan_parquet(path)
    with pytest.raises(PolarsError):
        lf.select(l.list.first().struct.field("a")).collect(optimizations=NO_PUSHDOWN)

    for q, expected in [
        (l.list.eval(e.struct.field("a")), df.select(l.list.eval(e.struct.field("a")))),
        (l.list.last().struct.field("a"), df.select(l.list.last().struct.field("a"))),
        (l.explode().is_null().sum(), pl.DataFrame({"l": [0]})),
        (l.is_null().sum(), pl.DataFrame({"l": [0]})),
    ]:
        assert_engines_equal(lf.select(q), expected, check_dtypes=False)
    # `l` is the file's only column.
    assert_engines_equal(
        lf.explode("l").select(pl.len()),
        pl.DataFrame({"len": [2 * n]}),
        check_dtypes=False,
    )


def _two_row_groups(path: Path, wrap: Callable[[dict[str, Any]], Any]) -> None:
    """Row group 0: leaf `a` is the cheapest of the struct; row group 1: leaf `b`."""
    n = 2000
    groups = [[(0, f"{i:>40}") for i in range(n)], [(i * 7919, "x") for i in range(n)]]
    tables = [
        pa.table(
            {
                "id": list(range(k * n, (k + 1) * n)),
                "v": [wrap({"a": a, "b": b}) for a, b in rows],
            }
        )
        for k, rows in enumerate(groups)
    ]
    with pq.ParquetWriter(path, tables[0].schema, use_dictionary=False) as w:
        for t in tables:
            w.write_table(t)


def test_struct_field_pruning_validity_leaf_per_row_group(tmp_path: Path) -> None:
    path = tmp_path / "rg.parquet"
    _two_row_groups(path, lambda v: v)
    md = pq.ParquetFile(path).metadata
    sizes = [
        {
            md.row_group(rg).column(i).path_in_schema: md.row_group(rg)
            .column(i)
            .total_compressed_size
            for i in range(md.num_columns)
        }
        for rg in range(md.num_row_groups)
    ]
    assert sizes[0]["v.a"] < sizes[0]["v.b"]
    assert sizes[1]["v.b"] < sizes[1]["v.a"]

    # Only `b` is readable in the second row group: reading `a` there fails.
    corrupt_chunk(path, "v.a", row_groups=[1])
    v = pl.col("v")
    with pytest.raises(PolarsError):
        pl.scan_parquet(path).select(v.is_null().sum()).collect(
            optimizations=NO_PUSHDOWN
        )

    # The decode of a row group must be the one chosen for it, also when earlier row
    # groups are skipped (slice, statistics) or the column is decoded prefiltered.
    for parallel in ["auto", "prefiltered", "row_groups", "columns"]:
        lf = pl.scan_parquet(path, parallel=parallel)  # type: ignore[arg-type]
        for q, expected in [
            (lf.select(v.is_null().sum()), 0),
            (lf.select(pl.len()), 4000),
            (lf.slice(2005, 10).select(v.is_null().sum()), 0),
            (lf.tail(10).select(v.is_null().sum()), 0),
            (lf.filter(pl.col("id") >= 2003).select(v.is_null().sum()), 0),
            (lf.filter(v.is_not_null()).select(pl.len()), 4000),
        ]:
            for engine in ["in-memory", "streaming"]:
                assert q.collect(engine=engine).item() == expected  # type: ignore[call-overload]


def test_struct_field_pruning_validity_leaf_per_row_group_under_list(
    tmp_path: Path,
) -> None:
    path = tmp_path / "rgl.parquet"
    _two_row_groups(path, lambda v: [v])
    corrupt_chunk(path, "v.list.element.a", row_groups=[1])
    v = pl.col("v")
    lf = pl.scan_parquet(path)
    with pytest.raises(PolarsError):
        lf.select(v.explode().is_null().sum()).collect(optimizations=NO_PUSHDOWN)
    for q in [
        lf.select(v.explode().is_null().sum()),
        lf.slice(2001, 3).select(v.list.first().is_null().sum()),
    ]:
        for engine in ["in-memory", "streaming"]:
            assert q.collect(engine=engine).item() == 0  # type: ignore[call-overload]
