from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from itertools import islice
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

    def __init__(self, fields: list[_Field], rows: list[tuple[Any, ...]]) -> None:
        self._query_rows = SimpleNamespace(schema=fields)
        self.description = [
            (f.name, f.field_type, None, None, None, None, True) for f in fields
        ]
        self._rows = rows
        self._remaining: Iterator[tuple[Any, ...]] = iter(())

    def execute(self, sql: str) -> None:
        self._remaining = iter([] if "WHERE 1 = 0" in sql else self._rows)

    def fetchmany(self, size: int) -> list[tuple[Any, ...]]:
        return list(islice(self._remaining, size))

    def close(self) -> None:
        pass


class _Connection:
    __module__ = "google.cloud.bigquery.dbapi.connection"

    def __init__(self, fields: list[_Field], rows: list[tuple[Any, ...]]) -> None:
        self.fields = fields
        self.rows = rows

    def cursor(self) -> _Cursor:
        return _Cursor(self.fields, self.rows)


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
