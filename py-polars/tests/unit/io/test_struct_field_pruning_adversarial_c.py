"""Adversarial cases for struct-field pruning through lists, IPC, NDJSON and Iceberg.

Written while reviewing stage C; none of them failed. Kept as regression tests for edge
cases the other test files do not cover.
"""

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
    assert_same_as_without_pushdown,
)
from tests.unit.io.test_struct_field_pruning_lists import _corrupt_chunk

if TYPE_CHECKING:
    from pathlib import Path

l = pl.col("l")  # noqa: E741
s = pl.col("s")
e = pl.element()


def _two_row_groups(path: Path, make: object, n: int = 2000) -> None:
    """Row group 0: leaf `a` is the cheapest of the struct; row group 1: leaf `b`."""
    rg0 = make([(0, f"{i:>40}") for i in range(n)])  # type: ignore[operator]
    rg1 = make([(i * 7919, "x") for i in range(n)])  # type: ignore[operator]
    t0 = pa.table(rg0)
    t1 = pa.table(rg1, schema=t0.schema)
    with pq.ParquetWriter(path, t0.schema, use_dictionary=False) as w:
        w.write_table(t0)
        w.write_table(t1)


@pytest.mark.parametrize(
    "query",
    [
        lambda lf: lf.slice(2005, 10).select(s.is_null().sum()),
        lambda lf: lf.tail(10).select(s.is_null().sum()),
        lambda lf: lf.filter(pl.col("id") >= 2003).select(s.is_null().sum()),
        lambda lf: lf.filter(s.is_not_null()).select(pl.len()),
    ],
)
@pytest.mark.parametrize("parallel", ["auto", "prefiltered", "row_groups", "columns"])
def test_validity_leaf_per_row_group_with_skipped_row_groups(
    query: object, parallel: str, tmp_path: Path
) -> None:
    # The decode of a row group must be the one chosen for it, also when earlier row
    # groups are skipped (slice, statistics) or the column is decoded prefiltered.
    path = tmp_path / "rg.parquet"
    _two_row_groups(
        path,
        lambda rows: {
            "id": list(range(len(rows))),
            "s": [{"a": a, "b": b} for a, b in rows],
        },
    )
    tbl = pq.read_table(path)
    ids = pa.chunked_array([pa.array(range(4000))])
    pq.write_table(
        tbl.set_column(0, "id", ids), path, use_dictionary=False, row_group_size=2000
    )
    _corrupt_chunk(path, "s.a", row_groups=[1])

    lf = pl.scan_parquet(path, parallel=parallel)  # type: ignore[arg-type]
    for engine in ("in-memory", "streaming"):
        query(lf).collect(engine=engine)  # type: ignore[operator]


def test_validity_leaf_per_row_group_under_list(tmp_path: Path) -> None:
    path = tmp_path / "rgl.parquet"
    _two_row_groups(path, lambda rows: {"l": [[{"a": a, "b": b}] for a, b in rows]})
    _corrupt_chunk(path, "l.list.element.a", row_groups=[1])
    lf = pl.scan_parquet(path)
    with pytest.raises(PolarsError):
        lf.select(l.explode().is_null().sum()).collect(optimizations=NO_PUSHDOWN)
    for engine in ("in-memory", "streaming"):
        for q in [
            lf.select(l.explode().is_null().sum()),
            lf.slice(2001, 3).select(l.list.first().is_null().sum()),
        ]:
            assert q.collect(engine=engine).item() == 0  # type: ignore[call-overload]


BIG = pl.DataFrame(
    {
        "id": range(3000),
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


@pytest.mark.parametrize("compression", ["uncompressed", "lz4", "zstd"])
@pytest.mark.parametrize(
    "query",
    [
        lambda lf: lf.select(s.struct.field("c")),
        lambda lf: lf.select(s.struct.field("d")),
        lambda lf: lf.select(s.is_null()),
        lambda lf: lf.select(
            l.list.eval(e.struct.field("p").list.eval(e.struct.field("v")))
        ),
        lambda lf: lf.select(l.explode().struct.field("p").explode().struct.field("u")),
        lambda lf: lf.slice(1500, 800).select(
            s.struct.field("c"), l.list.last().struct.field("q")
        ),
    ],
)
def test_ipc_views_compression_batches(
    query: object, compression: str, tmp_path: Path
) -> None:
    # String views with variadic buffers before and after the read field, in several
    # record batches, compressed or not.
    path = tmp_path / "big.ipc"
    BIG.write_ipc(path, compression=compression, record_batch_size=700)  # type: ignore[call-overload]
    assert_same_as_without_pushdown(query(pl.scan_ipc(path)))  # type: ignore[operator]
    assert_same_as_without_pushdown(query(pl.scan_ipc(path, n_rows=1000)))  # type: ignore[operator]


PA_ELEMENT = pa.struct(
    [("a", pa.int64()), ("b", pa.large_string()), ("c", pa.string())]
)
PA_TABLE = pa.table(
    {
        "ll": pa.array(
            [
                [{"a": 1, "b": "x" * 20, "c": "y"}, None],
                None,
                [],
                [{"a": None, "b": None, "c": "q"}],
            ],
            type=pa.large_list(PA_ELEMENT),
        ),
        "fl": pa.array(
            [
                [{"a": 1, "b": "x", "c": "y"}, None],
                None,
                [None, {"a": 3, "b": "w", "c": "v"}],
                [None, None],
            ],
            type=pa.list_(PA_ELEMENT, 2),
        ),
        "d": pa.array(
            [
                {"cat": "u", "v": 1},
                {"cat": "w", "v": None},
                None,
                {"cat": None, "v": 4},
            ],
            type=pa.struct(
                [("cat", pa.dictionary(pa.int32(), pa.string())), ("v", pa.int64())]
            ),
        ),
    }
)


@pytest.mark.parametrize("fmt", ["parquet", "ipc"])
@pytest.mark.parametrize(
    "query",
    [
        lambda lf: lf.select(pl.col("ll").list.eval(e.struct.field("c"))),
        lambda lf: lf.select(pl.col("ll").explode().is_null()),
        lambda lf: lf.select(pl.col("fl").arr.eval(e.struct.field("c"))),
        lambda lf: lf.select(pl.col("fl").explode().is_null()),
        lambda lf: lf.select(pl.col("d").struct.field("v")),
        lambda lf: lf.select(pl.col("d").struct.field("cat")),
        lambda lf: lf.slice(1, 2).select(pl.col("fl").arr.last().struct.field("c")),
    ],
)
def test_pyarrow_written_lists(query: object, fmt: str, tmp_path: Path) -> None:
    # LargeList, FixedSizeList, large strings and dictionaries as pyarrow writes them.
    path = tmp_path / f"pa.{fmt}"
    if fmt == "ipc":
        feather.write_feather(PA_TABLE, path, compression="lz4")
        lf = pl.scan_ipc(path)
    else:
        pq.write_table(PA_TABLE, path)
        lf = pl.scan_parquet(path)
    assert_same_as_without_pushdown(query(lf))  # type: ignore[operator]


def _f(name: str, type_: pa.DataType, field_id: int) -> pa.Field:
    return pa.field(name, type_, metadata={b"PARQUET:field_id": str(field_id).encode()})


def _ice_schema(names: dict[int, str], order: tuple[int, ...] = (3, 4)) -> pa.Schema:
    sub = {
        3: _f(names.get(3, "a"), pa.int64(), 3),
        4: _f(names.get(4, "b"), pa.string(), 4),
    }
    lsub = {
        10: _f(names.get(10, "p"), pa.int64(), 10),
        11: _f(names.get(11, "q"), pa.string(), 11),
    }
    return pa.schema(
        [
            _f("id", pa.int64(), 1),
            _f("s", pa.struct([sub[i] for i in order]), 2),
            _f("l", pa.list_(_f("element", pa.struct([lsub[10], lsub[11]]), 9)), 8),
        ]
    )


@pytest.mark.parametrize(
    "file_names",
    [
        {3: "b", 4: "a"},  # field names swapped between versions
        {10: "q", 11: "p"},  # the same, in the list element
        {3: "a_old", 4: "b_old", 10: "p_old"},
    ],
)
@pytest.mark.parametrize(
    "query",
    [
        lambda lf: lf.select(s.struct.field("a")),
        lambda lf: lf.select(s.struct.field("b"), "id"),
        lambda lf: lf.select(s.is_null()),
        lambda lf: lf.select(l.list.eval(e.struct.field("q"))),
        lambda lf: lf.select(l.list.first().struct.field("p")),
        lambda lf: lf.filter(s.struct.field("a") > 0).select(s.struct.field("b")),
    ],
)
def test_iceberg_names_swapped(
    query: object, file_names: dict[int, str], tmp_path: Path
) -> None:
    table = _ice_schema({})
    old = _ice_schema(file_names, order=(4, 3))
    rows = {
        "id": [1, 2, 3],
        "s": [{"a": 1, "b": "x"}, None, {"a": 3, "b": None}],
        "l": [[{"p": 1, "q": "u"}, None], None, []],
    }
    pq.write_table(pa.table(rows, schema=table), tmp_path / "1.parquet")

    def rename(v: Any, ids: dict[str, int]) -> Any:
        return (
            None if v is None else {file_names.get(ids[k], k): x for k, x in v.items()}
        )

    old_rows: dict[str, list[Any]] = {
        "id": [11, 12, 13],
        "s": [
            rename(v, {"a": 3, "b": 4})
            for v in [{"a": 11, "b": "y"}, {"a": None, "b": "z"}, None]
        ],
        "l": [None, [rename({"p": 12, "q": "v"}, {"p": 10, "q": 11})], [None]],
    }
    old_rows["s"] = [
        None if v is None else {f.name: v[f.name] for f in old.field("s").type}
        for v in old_rows["s"]
    ]
    pq.write_table(pa.table(old_rows, schema=old), tmp_path / "2.parquet")

    lf = pl.scan_parquet(
        [tmp_path / "1.parquet", tmp_path / "2.parquet"],
        schema=pl.from_arrow(table.empty_table()).schema,  # type: ignore[union-attr]
        _column_mapping=("iceberg-column-mapping", table),
    )
    assert_same_as_without_pushdown(query(lf))  # type: ignore[operator]


def test_ndjson_empty_object_vs_null(tmp_path: Path) -> None:
    # `{}` is a valid struct with null fields, `null` a null struct: validity-only reads
    # must tell them apart, also for list elements.
    path = tmp_path / "x.ndjson"
    path.write_text(
        '{"s": {}, "l": [{}, null, {"a": 1}]}\n'
        '{"s": null, "l": null}\n'
        '{"s": {"a": 2}, "l": []}\n'
    )
    el = pl.Struct({"a": pl.Int64, "b": pl.String})
    lf = pl.scan_ndjson(path, schema={"s": el, "l": pl.List(el)})
    assert_same_as_without_pushdown(lf.select(s.is_null()))
    assert_same_as_without_pushdown(lf.select(l.explode().is_null()))
    assert_same_as_without_pushdown(lf.select(l.list.first().is_null()))


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
@pytest.mark.parametrize(
    "query",
    [
        lambda lf: lf.select(pl.col("ll").list.first().list.first().struct.field("a")),
        lambda lf: lf.select(pl.col("ll").explode().explode().is_null()),
        lambda lf: lf.select(pl.col("al").arr.first().list.first().struct.field("a")),
        lambda lf: lf.select(pl.col("al").explode().explode().is_null()),
        lambda lf: lf.select(
            s.struct.field("m").list.first().list.first().struct.field("q")
        ),
        lambda lf: lf.select(
            s.struct.field("k").list.eval(e.struct.field("p")),
            s.struct.field("m").list.first().list.len(),
        ),
        lambda lf: lf.select(s.struct.field("m").explode().explode().is_null()),
    ],
)
def test_concat_aligns_nested_lists(
    query: object, how: str, fmt: str, tmp_path: Path
) -> None:
    path = tmp_path / f"n.{fmt}"
    getattr(NESTED, f"write_{fmt}")(path)
    scan = getattr(pl, f"scan_{fmt}")(path)
    lf = pl.concat([scan, NESTED.lazy()], how=how)  # type: ignore[arg-type]
    assert_same_as_without_pushdown(query(lf))  # type: ignore[operator]


@pytest.mark.parametrize("policy", ["insert", "raise"])
@pytest.mark.parametrize(
    "query",
    [
        lambda lf: lf.select(l.list.eval(e.struct.field("c"))),
        lambda lf: lf.select(l.list.eval(e.struct.field("a"))),
        lambda lf: lf.select(l.explode().is_null()),
    ],
)
def test_missing_struct_field_under_list(
    query: object, policy: str, tmp_path: Path
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
    q = query(lf)  # type: ignore[operator]
    if policy == "raise":
        with pytest.raises(PolarsError):
            q.collect(optimizations=NO_PUSHDOWN)
        for engine in ("in-memory", "streaming"):
            with pytest.raises(PolarsError):
                q.collect(engine=engine)
    else:
        assert_same_as_without_pushdown(q)


@pytest.mark.parametrize("fmt", ["parquet", "ipc", "ndjson"])
def test_zero_rows_and_null_fields(fmt: str, tmp_path: Path) -> None:
    el = pl.Struct({"n": pl.Null, "a": pl.String})
    df = pl.DataFrame(
        {
            "s": [{"n": None, "a": "x"}, None, {"n": None, "a": None}],
            "l": [[{"n": None, "a": "y" * 50}, None], None, []],
        },
        schema={"s": el, "l": pl.List(el)},
    )
    for frame, name in [(df, "full"), (df.clear(), "empty")]:
        path = tmp_path / f"{name}.{fmt}"
        getattr(frame, f"write_{fmt}")(path)
        scan = (
            pl.scan_ndjson(path, schema=df.schema)
            if fmt == "ndjson"
            else getattr(pl, f"scan_{fmt}")(path)
        )
        assert_same_as_without_pushdown(scan.select(s.is_null()))
        assert_same_as_without_pushdown(scan.select(l.explode().is_null()))
        assert_same_as_without_pushdown(scan.select(s.struct.field("n")))
        assert_same_as_without_pushdown(scan.select(l.list.first().struct.field("a")))
