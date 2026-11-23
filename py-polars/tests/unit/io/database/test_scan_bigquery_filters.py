from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from itertools import islice
from json import dumps
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest

import polars as pl

if TYPE_CHECKING:
    from collections.abc import Iterator

pytest.importorskip("polars_database")
pytest.importorskip("sqlglot")


@dataclass
class _Field:
    name: str
    field_type: str
    mode: str = "NULLABLE"
    precision: int | None = None
    scale: int | None = None
    fields: tuple[_Field, ...] = ()


class _Cursor:
    __module__ = "google.cloud.bigquery.dbapi.cursor"

    def __init__(
        self,
        fields: list[_Field],
        rows: list[tuple[Any, ...]],
        storage_rows: list[tuple[Any, ...]] | None = None,
    ) -> None:
        self._query_rows = SimpleNamespace(schema=fields)
        self.description = [
            (f.name, f.field_type, None, None, None, None, True) for f in fields
        ]
        self._rows = rows
        self._storage_rows = storage_rows
        self._remaining: Iterator[tuple[Any, ...]] | None = None
        self._sql = ""

    def execute(self, sql: str) -> None:
        self._sql = sql
        self._remaining = None

    def _bqstorage_fetch(self) -> Iterator[tuple[Any, ...]]:
        assert self._storage_rows is not None
        return iter(self._storage_rows)

    def fetchmany(self, size: int) -> list[tuple[Any, ...]]:
        if self._remaining is None:
            if "WHERE 1 = 0" in self._sql:
                self._remaining = iter(())
            elif "LIMIT 1000" in self._sql:
                self._remaining = islice(self._rows, 1000)
            elif self._storage_rows is not None:
                self._remaining = self._bqstorage_fetch()
            else:
                self._remaining = iter(self._rows)
        return list(islice(self._remaining, size))

    def close(self) -> None:
        pass


class _Connection:
    __module__ = "google.cloud.bigquery.dbapi.connection"

    def __init__(
        self,
        fields: list[_Field],
        rows: list[tuple[Any, ...]],
        storage_rows: list[tuple[Any, ...]] | None = None,
    ) -> None:
        self.fields = fields
        self.rows = rows
        self.storage_rows = storage_rows

    def cursor(self) -> _Cursor:
        return _Cursor(self.fields, self.rows, self.storage_rows)


def test_bigquery_only_pushes_equality_on_native_string() -> None:
    conn = _Connection(
        [
            _Field("label", "STRING"),
            _Field("amount", "BIGNUMERIC"),
            _Field("geo", "GEOGRAPHY"),
        ],
        [("a", Decimal("1.25"), "POINT(1 2)")],
    )
    lf = pl.scan_database("data", conn)
    assert "filters 1/1 (0 exact)" in lf.filter(pl.col("label") == "a").explain()
    for name, value in [("amount", "1.25"), ("geo", "POINT(1 2)")]:
        filtered = lf.filter(pl.col(name) == value)
        assert "filters 0/1" in filtered.explain()
        assert filtered.collect()[name].item() == value

    lf = pl.scan_database(
        "data", conn, schema={"label": pl.String, "amount": pl.String, "geo": pl.String}
    )
    assert "filters 0/1" in lf.filter(pl.col("amount") == "1.25").explain()


def test_bigquery_sampled_json_filters_stay_local() -> None:
    # The REST driver decodes JSON 1 as an integer and JSON null as Python None.
    conn = _Connection([_Field("value", "JSON")], [(1,), (None,)])
    lf = pl.scan_database("data", conn)
    assert lf.collect_schema() == {"value": pl.Int64}
    assert "filters 0/1" in lf.filter(pl.col("value") == 1).explain()
    filtered = lf.filter(pl.col("value").is_null()).head(1)
    plan = filtered.explain()
    assert "filters 0/1" in plan
    assert "limit: no" in plan
    assert filtered.collect()["value"].to_list() == [None]


@pytest.mark.parametrize("overrides", [False, True])
def test_bigquery_supplied_json_type_filters_stay_local(overrides: bool) -> None:
    conn = _Connection([_Field("value", "JSON")], [(1,), (None,)])
    schema = pl.scan_database("data", conn).collect_schema()
    lf = pl.scan_database(
        "data",
        conn,
        schema=schema if not overrides else None,
        schema_overrides=schema if overrides else None,
    )
    filtered = lf.filter(pl.col("value").is_null()).head(1)
    assert "filters 0/1" in filtered.explain()
    assert filtered.collect()["value"].to_list() == [None]


def test_bigquery_json_rest_sample_and_storage_rows_match() -> None:
    fields = [
        _Field("j", "JSON"),
        _Field("s", "JSON"),
        _Field("label", "STRING"),
        _Field(
            "entries",
            "RECORD",
            "REPEATED",
            fields=(_Field("n", "INTEGER"), _Field("j", "JSON", "REPEATED")),
        ),
    ]
    rows = [
        ({"x": 1}, "null", "null", [{"n": 5, "j": [1, 2]}]),
        ({"x": 2}, '"123"', '"123"', [{"n": None, "j": []}]),
    ] * 500 + [(None, None, "[123]", [])]
    storage_rows = [
        (
            dumps(j),
            dumps(s),
            label,
            [{"n": row["n"], "j": [dumps(v) for v in row["j"]]} for row in entries],
        )
        for j, s, label, entries in rows
    ]
    rest = pl.scan_database("data", _Connection(fields, rows)).collect()
    conn = _Connection(fields, rows, storage_rows)
    mixed = pl.scan_database("data", conn, batch_size=1000).collect()
    assert mixed.equals(rest)
    assert mixed.rows() == rows


def test_bigquery_json_resolves_unsampled_container_leaves() -> None:
    rows = [([], {"a": None})] * 1000 + [([123], {"a": 456})]
    conn = _Connection([_Field("j", "JSON"), _Field("obj", "JSON")], rows)
    lf = pl.scan_database("data", conn, batch_size=1000)
    assert lf.collect_schema() == {
        "j": pl.List(pl.String),
        "obj": pl.Struct({"a": pl.String}),
    }
    assert lf.collect().tail(1).rows() == [(["123"], {"a": "456"})]


@pytest.mark.parametrize("batch_size", [1, 1000])
@pytest.mark.parametrize(
    "values",
    [
        [9007199254740993, 0.5],
        [True, 2],
        ["2", 2],
        [{"x": [9007199254740993, 0.5]}],
    ],
)
def test_bigquery_json_rejects_lossy_inference(
    batch_size: int, values: list[Any]
) -> None:
    conn = _Connection([_Field("j", "JSON")], [(value,) for value in values])
    with pytest.raises(
        pl.exceptions.ComputeError, match="cannot be inferred losslessly"
    ):
        pl.scan_database("data", conn, batch_size=batch_size).collect()


@pytest.mark.parametrize(
    ("first", "late"), [(True, 2), (2, True), ("two", 2), (2, "2")]
)
def test_bigquery_json_rejects_late_mixed_types(first: Any, late: Any) -> None:
    conn = _Connection(
        [_Field("j", "JSON")], [({"x": [first]},)] * 1000 + [({"x": [late]},)]
    )
    lf = pl.scan_database("data", conn, batch_size=1000)
    with pytest.raises(
        pl.exceptions.ComputeError, match="cannot be inferred losslessly"
    ):
        lf.collect()


@pytest.mark.parametrize(("first", "late"), [(True, 2), (2, True)])
def test_bigquery_json_rejects_boolean_numeric_cast_after_sample(
    first: Any, late: Any
) -> None:
    conn = _Connection([_Field("j", "JSON")], [([first],)] * 1000, [(dumps([late]),)])
    lf = pl.scan_database("data", conn, batch_size=1000)
    with pytest.raises(pl.exceptions.SchemaError, match="type changed"):
        lf.collect()


def test_bigquery_json_preserves_exact_integer_promotion() -> None:
    values = [9007199254740994, 0.5]
    conn = _Connection([_Field("j", "JSON")], [(value,) for value in values])
    actual = pl.scan_database("data", conn, batch_size=1).collect()
    assert actual.schema == {"j": pl.Float64}
    assert actual["j"].to_list() == values


@pytest.mark.parametrize("populated", [True, False])
def test_bigquery_samples_only_unknown_leaves(populated: bool) -> None:
    fields = (
        _Field("z", "NUMERIC"),
        _Field("json", "JSON"),
        _Field("a", "STRING"),
    )
    entries = [
        [{"z": Decimal("1.25"), "json": None, "a": None}],
        [{"z": None, "json": 3, "a": "a"}],
    ]
    conn = _Connection(
        [_Field("entries", "RECORD", "REPEATED", fields=fields)],
        [(entry,) for entry in entries] if populated else [],
    )
    lf = pl.scan_database("data", conn, batch_size=1)
    expected = pl.List(
        pl.Struct(
            {
                "z": pl.Decimal(38, 9),
                "json": pl.Int64 if populated else pl.String,
                "a": pl.String,
            }
        )
    )
    assert lf.collect_schema() == {"entries": expected}
    assert lf.collect()["entries"].to_list() == (entries if populated else [])

    supplied = pl.List(
        pl.Struct({"z": pl.Decimal(38, 9), "json": pl.Int32, "a": pl.String})
    )
    lf = pl.scan_database("data", conn, schema={"entries": supplied}, batch_size=1)
    assert lf.collect().schema == {"entries": supplied}


@pytest.mark.parametrize("supplied_schema", [True, False])
def test_bigquery_array_null_filters_stay_local(supplied_schema: bool) -> None:
    # The database sees a NULL during the query and emits an empty array in its result.
    conn = _Connection([_Field("xs", "INTEGER", "REPEATED")], [([],)])
    lf = pl.scan_database(
        "SELECT CAST(NULL AS ARRAY<INT64>) AS xs",
        conn,
        schema={"xs": pl.List(pl.Int64)} if supplied_schema else None,
    )
    is_null = lf.filter(pl.col("xs").is_null())
    is_not_null = lf.filter(pl.col("xs").is_not_null())
    assert "filters 0/1" in is_null.explain()
    assert "filters 0/1" in is_not_null.explain()
    assert is_null.collect().is_empty()
    assert is_not_null.collect()["xs"].to_list() == [[]]
