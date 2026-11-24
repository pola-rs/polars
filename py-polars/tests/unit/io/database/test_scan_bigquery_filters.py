from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from io import BytesIO
from itertools import islice
from json import dumps
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest

import polars as pl

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from polars._typing import PolarsDataType

bridge = pytest.importorskip("polars_database._bridge")
pytest.importorskip("sqlglot")


@pytest.fixture(autouse=True)
def _mock_bigquery_driver_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    probe: Callable[[str], tuple[str | None, str | None] | None] = bridge.probe

    def mock_probe(kind: str) -> tuple[str | None, str | None] | None:
        if kind == "python:google.cloud.bigquery":
            return None, None
        return probe(kind)

    monkeypatch.setattr(bridge, "probe", mock_probe)


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
        storage_rows: list[tuple[Any, ...]] | None,
        sqls: list[str],
    ) -> None:
        self._query_rows = SimpleNamespace(schema=fields)
        self.description = [
            (f.name, f.field_type, None, None, None, None, True) for f in fields
        ]
        self._rows = rows
        self._storage_rows = storage_rows
        self._sqls = sqls
        self._remaining: Iterator[tuple[Any, ...]] | None = None
        self._sql = ""

    def execute(self, sql: str) -> None:
        self._sqls.append(sql)
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
        self.sqls: list[str] = []

    def cursor(self) -> _Cursor:
        return _Cursor(self.fields, self.rows, self.storage_rows, self.sqls)


@pytest.mark.parametrize("batch_size", [1, 2])
def test_bigquery_json_object_key_order_is_stable(batch_size: int) -> None:
    rows = [
        ({"z": 1, "a": [{"y": 2, "x": "first"}]},),
        ({"a": [{"x": "second", "y": 3}], "z": 4},),
    ]
    conn = _Connection([_Field("j", "JSON")], rows)
    expected = pl.Struct(
        {"a": pl.List(pl.Struct({"x": pl.String, "y": pl.Int64})), "z": pl.Int64}
    )
    scan = pl.scan_database("SELECT j FROM fixture", conn, batch_size=batch_size)
    assert scan.collect_schema() == {"j": expected}
    assert scan.collect().rows() == rows


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


def test_bigquery_double_percent_filter_stays_local() -> None:
    conn = _Connection([_Field("label", "STRING")], [("a%%b",), ("a%b",)])
    scan = pl.scan_database("data", conn).filter(pl.col("label") == "a%%b")
    assert scan.collect()["label"].to_list() == ["a%%b"]
    assert conn.sqls
    assert all("a%%b" not in sql for sql in conn.sqls)


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
    rows: list[tuple[Any, ...]] = [
        ({"x": 1}, "null", "null", [{"n": 5, "j": [1, 2]}]),
        ({"x": 2}, '"123"', '"123"', [{"n": None, "j": []}]),
    ]
    rows = rows * 500 + [(None, None, "[123]", [])]
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
    rows: list[tuple[Any, ...]] = [([], {"a": None})]
    rows = rows * 1000 + [([123], {"a": 456})]
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


@pytest.mark.parametrize("batch_size", [1, 400, 1000])
def test_bigquery_json_tracks_keys_introduced_within_first_batch(
    batch_size: int,
) -> None:
    conn = _Connection(
        [_Field("j", "JSON")],
        [({"a": 0},)] + [({"a": 1, "b": "2"},)] * 999 + [({"a": 1, "b": 2},)],
    )
    lf = pl.scan_database("data", conn, batch_size=batch_size)
    if batch_size == 1:
        # Its first data batch lacks a field seen during sample inference.
        with pytest.raises(pl.exceptions.SchemaError, match="struct"):
            lf.collect()
    else:
        # The new key must remain tracked even across an intervening batch (size 400).
        with pytest.raises(
            pl.exceptions.ComputeError, match="cannot be inferred losslessly"
        ):
            lf.collect()


@pytest.mark.parametrize(("first", "late"), [(True, 2), (2, True)])
def test_bigquery_json_rejects_boolean_numeric_cast_after_sample(
    first: Any, late: Any
) -> None:
    conn = _Connection([_Field("j", "JSON")], [([first],)], [(dumps([late]),)])
    lf = pl.scan_database("data", conn)
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


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize(
    ("values", "dtype"), [([2, 0.5], pl.Float64), ([-1, 2**63 + 1], pl.Int128)]
)
def test_bigquery_json_numeric_array_inference_is_order_independent(
    values: list[float | int], dtype: type[pl.DataType], reverse: bool
) -> None:
    values = values[::-1] if reverse else values
    row = (values, {"a": [{"b": values}]}, values, [{"n": v} for v in values])
    conn = _Connection(
        [
            _Field("a", "JSON"),
            _Field("b", "JSON"),
            _Field("c", "JSON", "REPEATED"),
            _Field("d", "JSON"),
        ],
        [row],
    )
    lf = pl.scan_database("data", conn)
    assert lf.collect_schema() == {
        "a": pl.List(dtype),
        "b": pl.Struct({"a": pl.List(pl.Struct({"b": pl.List(dtype)}))}),
        "c": pl.List(dtype),
        "d": pl.List(pl.Struct({"n": dtype})),
    }
    assert lf.collect().rows() == [row]


@pytest.mark.parametrize(
    ("sample", "late", "path"),
    [
        ({"a": 1}, 2, "j"),
        ({"a": [{"b": 1}]}, {"a": [2]}, r"j\.a\[\]"),
        ("1", 2, "j"),
        (1, "2", "j"),
        ([{"a": "1"}], [{"a": 2}], r"j\[\]\.a"),
    ],
)
def test_bigquery_json_checks_sampled_kind_after_pushed_filter_and_limit(
    sample: Any, late: Any, path: str
) -> None:
    conn = _Connection(
        [_Field("id", "INTEGER"), _Field("j", "JSON")],
        [(1, sample)],
        [(2, dumps(late))],
    )
    # The data job sees only the last row after SQL filter pushdown. There are no
    # earlier rows in that cursor's history that could reveal the type changed.
    lf = pl.scan_database("SELECT id, j FROM data", conn)
    filtered = lf.filter(pl.col("id") == 2).head(1)
    plan = filtered.explain()
    assert "filters 1/1 (1 exact)" in plan
    with pytest.raises(
        pl.exceptions.SchemaError, match=rf"field '{path}'.*type changed"
    ):
        filtered.collect()


@pytest.mark.parametrize(
    ("sample", "late", "dtype", "path"),
    [
        (None, {"b": 3}, pl.String, "j"),
        ({"a": None}, {"a": {"b": 3}}, pl.Struct({"a": pl.String}), r"j\.a"),
        ([], [[1]], pl.List(pl.String), r"j\[\]"),
    ],
)
@pytest.mark.parametrize("push_filter", [False, True])
def test_bigquery_json_rejects_structure_at_unsampled_leaves(
    sample: Any,
    late: Any,
    dtype: PolarsDataType,
    path: str,
    push_filter: bool,
) -> None:
    conn = _Connection(
        [_Field("id", "INTEGER"), _Field("j", "JSON")],
        [(i, sample) for i in range(1, 1001)] + [(1001, late)],
        [(1001, dumps(late))] if push_filter else None,
    )
    scan = pl.scan_database("SELECT id, j FROM data", conn, batch_size=1000)
    assert scan.collect_schema() == {"id": pl.Int64, "j": dtype}
    if push_filter:
        scan = scan.filter(pl.col("id") == 1001).head(1)
        assert "filters 1/1 (1 exact)" in scan.explain()
    with pytest.raises(
        pl.exceptions.SchemaError, match=rf"field '{path}'.*use TO_JSON_STRING"
    ):
        scan.collect()


def test_bigquery_json_checks_sampled_kind_after_serialize(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from polars_database import _bridge

    conn = _Connection(
        [_Field("id", "INTEGER"), _Field("j", "JSON")],
        [(i, {"a": 1}) for i in range(1, 1001)] + [(1001, 2)],
        [(1001, "2")],
    )
    monkeypatch.setattr(_bridge, "_connect_bigquery", lambda *_: (conn, lambda: None))
    lf = pl.scan_database("SELECT id, j FROM data", "bigquery://", driver="bigquery")
    filtered = lf.filter(pl.col("id") == 1001).head(1)
    serialized = filtered.serialize()
    with pytest.raises(
        pl.exceptions.SchemaError, match=r"JSON field 'j'.*type changed"
    ):
        pl.LazyFrame.deserialize(BytesIO(serialized)).collect()


@pytest.mark.parametrize("overrides", [False, True])
def test_bigquery_json_allows_user_requested_cast(overrides: bool) -> None:
    conn = _Connection([_Field("j", "JSON")], [("1",)], [("2",)])
    schema = {"j": pl.String}
    lf = pl.scan_database(
        "data",
        conn,
        schema_overrides=schema if overrides else None,
        schema=None if overrides else schema,
    )
    assert lf.collect().rows() == [("2",)]


def test_bigquery_json_checks_sampled_kind_after_pushed_limit() -> None:
    conn = _Connection([_Field("j", "JSON")], [("1",)], [("2",)])
    lf = pl.scan_database("data", conn).head(1)
    assert "limit: 1" in lf.explain()
    with pytest.raises(
        pl.exceptions.SchemaError, match=r"JSON field 'j'.*type changed"
    ):
        lf.collect()


@pytest.mark.parametrize("late", [None, 2**53 + 1, 2**53 + 2])
def test_bigquery_json_allows_only_exact_casts_to_sampled_float(
    late: int | None,
) -> None:
    conn = _Connection([_Field("j", "JSON")], [([0.5],)], [(dumps([late]),)])
    lf = pl.scan_database("data", conn).head(1)
    assert lf.collect_schema() == {"j": pl.List(pl.Float64)}
    if late == 2**53 + 1:
        with pytest.raises(pl.exceptions.SchemaError, match="would round"):
            lf.collect()
    else:
        assert lf.collect().rows() == [([late],)]
