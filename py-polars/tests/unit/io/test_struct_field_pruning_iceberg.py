"""Struct-field projection pushdown with an Iceberg column mapping (by field ID)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

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
from tests.unit.io.test_struct_field_pruning_lists import _corrupt_chunk

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


def _f(name: str, type_: pa.DataType, field_id: int) -> pa.Field:
    return pa.field(name, type_, metadata={b"PARQUET:field_id": str(field_id).encode()})


def _table_schema(names: dict[int, str]) -> pa.Schema:
    """The schema with field IDs, under the names given by ID (default: the table's)."""

    def n(i: int, default: str) -> str:
        return names.get(i, default)

    return pa.schema(
        [
            _f(n(1, "id"), pa.int64(), 1),
            _f(
                n(2, "s"),
                pa.struct(
                    [
                        _f(n(3, "a"), pa.int64(), 3),
                        _f(n(4, "b"), pa.string(), 4),
                        _f(
                            n(5, "c"),
                            pa.struct(
                                [
                                    _f(n(6, "x"), pa.float64(), 6),
                                    _f(n(7, "y"), pa.string(), 7),
                                ]
                            ),
                            5,
                        ),
                    ]
                ),
                2,
            ),
            _f(
                n(8, "l"),
                pa.list_(
                    _f(
                        "element",
                        pa.struct(
                            [
                                _f(n(10, "p"), pa.int64(), 10),
                                _f(n(11, "q"), pa.string(), 11),
                            ]
                        ),
                        9,
                    )
                ),
                8,
            ),
        ]
    )


# The table renames `s` to `st`, `s.a` to `alpha` and `s.c.x` to `cx`.
TABLE_NAMES = {2: "st", 3: "alpha", 6: "cx"}
TABLE_SCHEMA = _table_schema(TABLE_NAMES)
POLARS_SCHEMA = pl.from_arrow(TABLE_SCHEMA.empty_table()).schema  # type: ignore[union-attr]


def _rows(offset: int) -> dict[str, list[Any]]:
    return {
        "id": [offset + i for i in range(4)],
        "s": [
            {"a": offset, "b": "x", "c": {"x": 1.5, "y": "y"}},
            None,
            {"a": None, "b": None, "c": None},
            {"a": offset + 3, "b": "w", "c": {"x": None, "y": "v"}},
        ],
        "l": [
            [{"p": offset, "q": "a"}, None],
            None,
            [],
            [{"p": None, "q": "b"}],
        ],
    }


def _write_file(path: Path, offset: int, file_names: dict[int, str]) -> None:
    """A data file, with the names it had when it was written."""
    rows = _rows(offset)
    schema = _table_schema(file_names)
    names = {
        "s": file_names.get(2, "s"),
        "a": file_names.get(3, "a"),
        "x": file_names.get(6, "x"),
    }
    rows["s"] = [
        None
        if v is None
        else {
            names["a"]: v["a"],
            "b": v["b"],
            "c": None
            if v["c"] is None
            else {names["x"]: v["c"]["x"], "y": v["c"]["y"]},
        }
        for v in rows["s"]
    ]
    table = pa.table(
        {"id": rows["id"], names["s"]: rows["s"], "l": rows["l"]}, schema=schema
    )
    pq.write_table(table, path, use_dictionary=False, write_statistics=False)


@pytest.fixture
def iceberg_scan(tmp_path: Path) -> pl.LazyFrame:
    # One file written under the table's names, one under older ones.
    _write_file(tmp_path / "1.parquet", 0, TABLE_NAMES)
    _write_file(tmp_path / "2.parquet", 10, {})
    return pl.scan_parquet(
        [tmp_path / "1.parquet", tmp_path / "2.parquet"],
        schema=POLARS_SCHEMA,
        _column_mapping=("iceberg-column-mapping", TABLE_SCHEMA),
    )


st = pl.col("st")
l = pl.col("l")  # noqa: E741

QUERIES: list[Callable[[pl.LazyFrame], pl.LazyFrame]] = [
    lambda lf: lf.select(st.struct.field("alpha")),
    lambda lf: lf.select(st.struct.field("b"), "id"),
    lambda lf: lf.select(st.struct.field("c").struct.field("cx")),
    lambda lf: lf.select(st.struct.field("c").struct.field("y")),
    lambda lf: lf.select(st.struct.field("alpha"), st.struct.field("c")),
    lambda lf: lf.select(st.is_null()),
    lambda lf: lf.select(st.struct.field("c").is_null()),
    lambda lf: lf.select(st, st.struct.field("alpha").alias("x")),
    lambda lf: lf.filter(st.struct.field("alpha") > 0).select(st.struct.field("b")),
    lambda lf: lf.select(l.list.eval(pl.element().struct.field("q"))),
    lambda lf: lf.select(l.list.first().struct.field("p")),
    lambda lf: lf.select(l.explode().is_null()),
    lambda lf: lf.select(
        st.struct.field("alpha"), l.list.last().struct.field("q"), "id"
    ),
    lambda lf: lf.head(5).select(st.struct.field("c").struct.field("cx")),
]


@pytest.mark.parametrize("query", QUERIES)
def test_struct_field_pruning_iceberg(
    query: Callable[[pl.LazyFrame], pl.LazyFrame], iceberg_scan: pl.LazyFrame
) -> None:
    assert_same_as_without_pushdown(query(iceberg_scan))


def test_struct_field_pruning_iceberg_explain(iceberg_scan: pl.LazyFrame) -> None:
    plan = iceberg_scan.select(st.struct.field("c").struct.field("cx")).explain()
    assert "PROJECT FIELDS: st{c{cx}}" in plan
    plan = iceberg_scan.select(l.list.first().struct.field("q")).explain()
    assert "PROJECT FIELDS: l[]{q}" in plan


def test_struct_field_pruning_iceberg_values(iceberg_scan: pl.LazyFrame) -> None:
    out = iceberg_scan.select(
        "id",
        st.struct.field("alpha"),
        st.struct.field("c").struct.field("cx"),
        l.list.first().struct.field("p"),
    ).collect()
    assert out.to_dict(as_series=False) == {
        "id": [0, 1, 2, 3, 10, 11, 12, 13],
        "alpha": [0, None, None, 3, 10, None, None, 13],
        "cx": [1.5, None, None, None, 1.5, None, None, None],
        "p": [0, None, None, None, 10, None, None, None],
    }


def test_struct_field_pruning_iceberg_skips_unselected_leaves(tmp_path: Path) -> None:
    _write_file(tmp_path / "1.parquet", 0, TABLE_NAMES)
    _write_file(tmp_path / "2.parquet", 10, {})
    # Leaves never read below, in both files (under their names there).
    _corrupt_chunk(tmp_path / "1.parquet", "st.b")
    _corrupt_chunk(tmp_path / "1.parquet", "l.list.element.q")
    _corrupt_chunk(tmp_path / "2.parquet", "s.c.y")

    lf = pl.scan_parquet(
        [tmp_path / "1.parquet", tmp_path / "2.parquet"],
        schema=POLARS_SCHEMA,
        _column_mapping=("iceberg-column-mapping", TABLE_SCHEMA),
    )
    with pytest.raises(PolarsError):
        lf.select(st.struct.field("alpha")).collect(optimizations=NO_PUSHDOWN)

    expected = pl.DataFrame(
        {
            "alpha": [0, None, None, 3, 10, None, None, 13],
            "cx": [1.5, None, None, None, 1.5, None, None, None],
            "p": [0, None, None, None, 10, None, None, None],
        }
    )
    for engine in ("in-memory", "streaming"):
        assert_frame_equal(
            lf.select(
                st.struct.field("alpha"),
                st.struct.field("c").struct.field("cx"),
                l.list.first().struct.field("p"),
            ).collect(engine=engine),  # type: ignore[call-overload]
            expected,
        )


def test_struct_field_pruning_iceberg_missing_field(tmp_path: Path) -> None:
    _write_file(tmp_path / "1.parquet", 0, TABLE_NAMES)
    # A file written before `s.c` existed.
    old = pa.schema(
        [
            _f("id", pa.int64(), 1),
            _f(
                "s",
                pa.struct([_f("a", pa.int64(), 3), _f("b", pa.string(), 4)]),
                2,
            ),
            TABLE_SCHEMA.field("l"),
        ]
    )
    pq.write_table(
        pa.table(
            {
                "id": [20, 21],
                "s": [{"a": 20, "b": "z"}, None],
                "l": [None, [{"p": 1, "q": "c"}]],
            },
            schema=old,
        ),
        tmp_path / "2.parquet",
    )

    for missing_struct_fields in ["insert", "raise"]:
        lf = pl.scan_parquet(
            [tmp_path / "1.parquet", tmp_path / "2.parquet"],
            schema=POLARS_SCHEMA,
            _column_mapping=("iceberg-column-mapping", TABLE_SCHEMA),
            cast_options=pl.ScanCastOptions(
                missing_struct_fields=missing_struct_fields  # type: ignore[arg-type]
            ),
        )
        for q in [
            lf.select(st.struct.field("c").struct.field("cx"), st.is_null()),
            lf.select(st.struct.field("alpha")),
            lf.select(st.is_null()),
        ]:
            if missing_struct_fields == "raise":
                # Raises as without pushdown, whether the missing field is read or not.
                with pytest.raises(PolarsError):
                    q.collect(optimizations=NO_PUSHDOWN)
                for engine in ("in-memory", "streaming"):
                    with pytest.raises(PolarsError):
                        q.collect(engine=engine)  # type: ignore[call-overload]
            else:
                assert_same_as_without_pushdown(q)
