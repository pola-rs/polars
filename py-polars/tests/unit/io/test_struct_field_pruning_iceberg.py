"""Struct-field projection pushdown with an Iceberg column mapping (by field ID)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pyarrow as pa
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
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


def _f(name: str, type_: pa.DataType, field_id: int) -> pa.Field:
    return pa.field(name, type_, metadata={b"PARQUET:field_id": str(field_id).encode()})


def _schema(names: dict[int, str] | None = None, *, reverse: bool = False) -> pa.Schema:
    """The schema with field IDs, under the names given by ID (default: the table's)."""

    def f(i: int, name: str, type_: pa.DataType) -> pa.Field:
        return _f((names or {}).get(i, name), type_, i)

    # With the fields of each struct in reverse order if `reverse`.
    def struct(*fields: pa.Field) -> pa.DataType:
        return pa.struct(fields[::-1] if reverse else fields)

    return pa.schema(
        [
            f(1, "id", pa.int64()),
            f(
                2,
                "s",
                struct(
                    f(3, "a", pa.int64()),
                    f(4, "b", pa.string()),
                    f(5, "c", struct(f(6, "x", pa.float64()), f(7, "y", pa.string()))),
                ),
            ),
            f(
                8,
                "l",
                pa.list_(
                    f(
                        9,
                        "element",
                        struct(f(10, "p", pa.int64()), f(11, "q", pa.string())),
                    )
                ),
            ),
        ]
    )


TABLE_SCHEMA = _schema()
POLARS_SCHEMA = pl.from_arrow(TABLE_SCHEMA.empty_table()).schema  # type: ignore[union-attr]


def _field_id(f: pa.Field) -> int:
    return int(f.metadata[b"PARQUET:field_id"])


def _convert(v: Any, src: Any, dst: Any) -> Any:
    """A value of type `src` as one of `dst`, its struct fields matched by ID."""
    if v is None:
        return None
    if pa.types.is_struct(src):
        by_id = {_field_id(f): f for f in src}
        return {
            f.name: _convert(
                v[by_id[_field_id(f)].name], by_id[_field_id(f)].type, f.type
            )
            for f in dst
        }
    if pa.types.is_list(src):
        return [_convert(x, src.value_type, dst.value_type) for x in v]
    return v


def _write_file(
    path: Path, offset: int, names: dict[int, str] | None = None, reverse: bool = False
) -> None:
    """A data file, with the names (and field order) it had when it was written."""
    rows: dict[str, list[Any]] = {
        "id": [offset + i for i in range(4)],
        "s": [
            {"a": offset, "b": "x", "c": {"x": 1.5, "y": "y"}},
            None,
            {"a": None, "b": None, "c": None},
            {"a": offset + 3, "b": "w", "c": {"x": None, "y": "v"}},
        ],
        "l": [[{"p": offset, "q": "a"}, None], None, [], [{"p": None, "q": "b"}]],
    }
    schema = _schema(names, reverse=reverse)
    columns = {
        dst.name: [_convert(v, src.type, dst.type) for v in rows[src.name]]
        for src, dst in zip(TABLE_SCHEMA, schema, strict=True)
    }
    table = pa.table(columns, schema=schema)
    pq.write_table(table, path, use_dictionary=False, write_statistics=False)


def _scan(paths: list[Path], **kwargs: Any) -> pl.LazyFrame:
    return pl.scan_parquet(
        paths,
        schema=POLARS_SCHEMA,
        _column_mapping=("iceberg-column-mapping", TABLE_SCHEMA),
        **kwargs,
    )


# Older versions of the table: names by field ID, and whether struct fields reverse.
RENAMED = {2: "st", 3: "alpha", 6: "cx"}
OLD_VERSIONS: list[tuple[dict[int, str], bool]] = [
    (RENAMED, False),
    ({3: "b", 4: "a"}, True),  # field names swapped
    ({10: "q", 11: "p"}, True),  # the same, in the list element
    ({3: "a_old", 4: "b_old", 10: "p_old"}, True),
]


@pytest.fixture
def iceberg_scan(tmp_path: Path) -> pl.LazyFrame:
    # One file written under the table's names, one under each older version.
    paths = [tmp_path / f"{i}.parquet" for i in range(len(OLD_VERSIONS) + 1)]
    _write_file(paths[0], 0)
    for i, (names, reverse) in enumerate(OLD_VERSIONS, 1):
        _write_file(paths[i], 10 * i, names, reverse)
    return _scan(paths)


s = pl.col("s")
l = pl.col("l")  # noqa: E741
a, b, c = (s.struct.field(f) for f in "abc")

QUERIES: list[Callable[[pl.LazyFrame], pl.LazyFrame]] = [
    lambda lf: lf.select(a),
    lambda lf: lf.select(b, "id"),
    lambda lf: lf.select(c.struct.field("x")),
    lambda lf: lf.select(c.struct.field("y")),
    lambda lf: lf.select(a, c),
    lambda lf: lf.select(s.is_null()),
    lambda lf: lf.select(c.is_null()),
    lambda lf: lf.select(s, a.alias("x")),
    lambda lf: lf.filter(a > 0).select(b),
    lambda lf: lf.select(l.list.eval(pl.element().struct.field("q"))),
    lambda lf: lf.select(l.list.first().struct.field("p")),
    lambda lf: lf.select(l.explode().is_null()),
    lambda lf: lf.select(a, l.list.last().struct.field("q"), "id"),
    lambda lf: lf.head(5).select(c.struct.field("x")),
]


@pytest.mark.parametrize("query", QUERIES)
def test_struct_field_pruning_iceberg(
    query: Callable[[pl.LazyFrame], pl.LazyFrame], iceberg_scan: pl.LazyFrame
) -> None:
    assert_same_as_without_pushdown(query(iceberg_scan))


def test_struct_field_pruning_iceberg_explain(tmp_path: Path) -> None:
    _write_file(tmp_path / "1.parquet", 0, RENAMED)
    lf = _scan([tmp_path / "1.parquet"])
    assert project_fields(lf.select(c.struct.field("x"))) == "s{c{x}}"
    assert project_fields(lf.select(l.list.first().struct.field("q"))) == "l[]{q}"


def test_struct_field_pruning_iceberg_skips_unselected_leaves(tmp_path: Path) -> None:
    _write_file(tmp_path / "1.parquet", 0)
    _write_file(tmp_path / "2.parquet", 10, RENAMED)
    # Leaves never read below, in both files (under their names there).
    corrupt_chunk(tmp_path / "1.parquet", "s.b")
    corrupt_chunk(tmp_path / "1.parquet", "l.list.element.q")
    corrupt_chunk(tmp_path / "2.parquet", "st.c.y")

    lf = _scan([tmp_path / "1.parquet", tmp_path / "2.parquet"])
    with pytest.raises(PolarsError):
        lf.select(a).collect(optimizations=NO_PUSHDOWN)

    q = lf.select("id", a, c.struct.field("x"), l.list.first().struct.field("p"))
    expected = pl.DataFrame(
        {
            "id": [0, 1, 2, 3, 10, 11, 12, 13],
            "a": [0, None, None, 3, 10, None, None, 13],
            "x": [1.5, None, None, None, 1.5, None, None, None],
            "p": [0, None, None, None, 10, None, None, None],
        }
    )
    assert_engines_equal(q, expected)


@pytest.mark.parametrize("missing_struct_fields", ["insert", "raise"])
def test_struct_field_pruning_iceberg_missing_field(
    missing_struct_fields: str, tmp_path: Path
) -> None:
    _write_file(tmp_path / "1.parquet", 0)
    # A file written before `s.c` existed.
    old = pa.schema(
        [
            _f("id", pa.int64(), 1),
            _f("s", pa.struct([_f("a", pa.int64(), 3), _f("b", pa.string(), 4)]), 2),
            TABLE_SCHEMA.field("l"),
        ]
    )
    rows = {
        "id": [20, 21],
        "s": [{"a": 20, "b": "z"}, None],
        "l": [None, [{"p": 1, "q": "c"}]],
    }
    pq.write_table(pa.table(rows, schema=old), tmp_path / "2.parquet")

    lf = _scan(
        [tmp_path / "1.parquet", tmp_path / "2.parquet"],
        cast_options=pl.ScanCastOptions(
            missing_struct_fields=missing_struct_fields  # type: ignore[arg-type]
        ),
    )
    for q in [
        lf.select(c.struct.field("x"), s.is_null()),
        lf.select(a),
        lf.select(s.is_null()),
    ]:
        if missing_struct_fields == "raise":
            # Raises as without pushdown, whether the missing field is read or not.
            assert_raises_with_and_without_pushdown(q)
        else:
            assert_same_as_without_pushdown(q)


# Dataset scans (`pl.scan_delta`, `pl.scan_iceberg`), expanded after projection
# pushdown.

DATASET_DF = pl.DataFrame(
    {
        "id": list(range(50)),
        "s": [
            None if i % 7 == 3 else {"a": i, "b": f"b{i}" * 9, "c": {"x": i, "y": "y"}}
            for i in range(50)
        ],
        "l": [None if i % 9 == 4 else [{"p": i, "q": f"q{i}" * 9}] for i in range(50)],
    }
)

DATASET_QUERIES: list[Callable[[pl.LazyFrame], pl.LazyFrame]] = [
    lambda lf: lf.select(a),
    lambda lf: lf.select(s.is_null(), "id"),
    lambda lf: lf.select(c.struct.field("x")),
    lambda lf: lf.filter(a > 5).select("id"),
    lambda lf: lf.select(l.list.first().struct.field("p")),
    lambda lf: lf.select(s),
    lambda lf: lf.head(3).select(b),
    lambda lf: lf.select(pl.len()),
]


def _leaf(path: Path, name: str) -> str:
    """The leaf of `path` whose dotted path starts and ends like `name` (`l.q`)."""
    md = pq.ParquetFile(path).metadata
    first, last = name.split(".")[0], name.split(".")[-1]
    leaves = [md.schema.column(i).path for i in range(md.num_columns)]
    (leaf,) = (x for x in leaves if x.startswith(first) and x.endswith("." + last))
    return str(leaf)


def _corrupt_unselected(paths: list[Path]) -> None:
    for path in paths:
        corrupt_chunk(path, "s.b")
        corrupt_chunk(path, _leaf(path, "l.q"))


@pytest.fixture
def delta_path(tmp_path: Path) -> Path:
    pytest.importorskip("deltalake")
    path = tmp_path / "delta"
    DATASET_DF.write_delta(path)
    return path


@pytest.mark.parametrize("query", DATASET_QUERIES)
def test_struct_field_pruning_delta(
    query: Callable[[pl.LazyFrame], pl.LazyFrame], delta_path: Path
) -> None:
    assert_same_as_without_pushdown(query(pl.scan_delta(delta_path)))


def test_struct_field_pruning_delta_skips_unselected_leaves(delta_path: Path) -> None:
    _corrupt_unselected(list(delta_path.glob("*.parquet")))
    q = pl.scan_delta(delta_path).select(
        a, s.is_null().alias("n"), l.list.first().struct.field("p")
    )
    with pytest.raises(PolarsError):
        q.collect(optimizations=NO_PUSHDOWN)
    assert_engines_equal(
        q,
        DATASET_DF.select(a, s.is_null().alias("n"), l.list.first().struct.field("p")),
    )
    fields = project_fields(q)
    assert fields is not None
    assert "s{a}" in fields
    assert "l[]{p}" in fields


def _iceberg_table(tmp_path: Path) -> Any:
    pytest.importorskip("pyiceberg")
    from pyiceberg.catalog.sql import SqlCatalog

    from tests.unit.io.test_iceberg import format_file_uri_iceberg

    catalog = SqlCatalog(
        "default",
        uri=f"sqlite:///{tmp_path / 'catalog.sqlite'}",
        warehouse=format_file_uri_iceberg(tmp_path),
    )
    catalog.create_namespace("ns")
    data = DATASET_DF.to_arrow(compat_level=pl.CompatLevel.oldest())
    table = catalog.create_table("ns.t", schema=data.schema)
    table.append(data)
    return table


@pytest.mark.filterwarnings("ignore::polars.exceptions.PerformanceWarning")
@pytest.mark.parametrize("reader_override", ["native", "pyiceberg"])
@pytest.mark.parametrize("query", DATASET_QUERIES)
def test_struct_field_pruning_scan_iceberg(
    query: Callable[[pl.LazyFrame], pl.LazyFrame], reader_override: Any, tmp_path: Path
) -> None:
    # The pyiceberg reader is a Python scan, which can't prune: its output is narrowed.
    lf = pl.scan_iceberg(_iceberg_table(tmp_path), reader_override=reader_override)
    assert_same_as_without_pushdown(query(lf))


@pytest.mark.filterwarnings("ignore::polars.exceptions.PerformanceWarning")
def test_struct_field_pruning_scan_iceberg_skips_unselected_leaves(
    tmp_path: Path,
) -> None:
    table = _iceberg_table(tmp_path)
    _corrupt_unselected(list(tmp_path.rglob("*.parquet")))
    q = pl.scan_iceberg(table).select(a, l.list.first().struct.field("p"))
    with pytest.raises(PolarsError):
        q.collect(optimizations=NO_PUSHDOWN)
    assert_engines_equal(q, DATASET_DF.select(a, l.list.first().struct.field("p")))
