from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from enum import Enum
from itertools import islice
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest

import polars as pl
from polars.testing import assert_frame_equal

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

pytest.importorskip("polars_database")


class _TimestampPrecision(Enum):
    MICROSECOND = None
    PICOSECOND = 12


@dataclass(frozen=True)
class _Field:
    name: str
    field_type: str
    mode: str = "NULLABLE"
    precision: int | None = None
    scale: int | None = None
    fields: tuple[_Field, ...] = ()
    timestamp_precision: _TimestampPrecision | None = None


class _Cursor:
    __module__ = "google.cloud.bigquery.dbapi.cursor"

    def __init__(
        self, fields: tuple[_Field, ...], rows: Iterable[tuple[Any, ...]]
    ) -> None:
        self._query_rows = SimpleNamespace(schema=fields)
        self.rows: Iterator[tuple[Any, ...]] = iter(rows)
        # The driver's DB-API description drops decimal parameters and the shape
        # of both repeated and record fields.
        self.description = [(f.name, f.field_type, *[None] * 5) for f in fields]
        self.fetches: list[int] = []
        self.closed = False

    def execute(self, _sql: str) -> None:
        pass

    def fetchmany(self, size: int) -> list[tuple[Any, ...]]:
        self.fetches.append(size)
        return list(islice(self.rows, size))

    def close(self) -> None:
        self.closed = True

    @property
    def query_job(self) -> Any:
        msg = "BigQuery schema discovery should not request the query job"
        raise AssertionError(msg)


class _LegacyCursor(_Cursor):
    def __init__(
        self, fields: tuple[_Field, ...], rows: Iterable[tuple[Any, ...]]
    ) -> None:
        super().__init__(fields, rows)
        self._query_job = SimpleNamespace(_query_results=self._query_rows)
        del self._query_rows


class _Connection:
    __module__ = "google.cloud.bigquery.dbapi.connection"

    def __init__(self, cursor: _Cursor) -> None:
        self._cursor = cursor

    def cursor(self) -> _Cursor:
        return self._cursor


@pytest.mark.parametrize("with_rows", [False, True])
@pytest.mark.parametrize("cursor_cls", [_Cursor, _LegacyCursor])
def test_bigquery_schema_stays_typed_per_batch(
    with_rows: bool, cursor_cls: type[_Cursor]
) -> None:
    from polars_database._bridge import _Session

    children = (
        _Field("z", "INTEGER"),
        _Field("a", "BIGNUMERIC"),
        _Field("t", "TIMESTAMP"),
    )
    fields = (
        _Field("n", "NUMERIC"),
        _Field("narrow", "BIGNUMERIC", precision=38, scale=5),
        _Field("b", "BIGNUMERIC"),
        _Field("wide", "BIGNUMERIC", precision=42, scale=5),
        _Field("xs", "INTEGER", "REPEATED"),
        _Field("ss", "RECORD", "REPEATED", fields=children),
        _Field("s", "RECORD", fields=children),
        _Field("ts", "TIMESTAMP", timestamp_precision=_TimestampPrecision.MICROSECOND),
        _Field("wall", "DATETIME"),
        _Field("text", "STRING"),
        _Field("texts", "STRING", "REPEATED"),
        _Field("point", "GEOGRAPHY"),
    )
    utc = datetime(2025, 1, 2, 3, 4, 5, 123456, tzinfo=UTC)
    wall = utc.replace(tzinfo=None)
    exact = Decimal("12345678901234567890123456789.123456789")
    wide = Decimal("123456789012345678901.123456789012345678")
    padded = Decimal("1.00000000000000000000000000000000000000")
    null_children = {"z": None, "a": None, "t": None}
    rows = [
        (None, None, None, None, [], [], None, None, None, None, [], None),
        (
            exact,
            padded,
            padded,
            Decimal("1.1"),
            [1, 2],
            [null_children, {"z": 5, "a": wide, "t": utc}],
            null_children,
            utc,
            wall,
            "hello",
            ["x"],
            "POINT(0 0)",
        ),
    ]
    record = pl.Struct({"z": pl.Int64, "a": pl.String, "t": pl.Datetime("us", "UTC")})
    schema = pl.Schema(
        {
            "n": pl.Decimal(38, 9),
            "narrow": pl.Decimal(38, 5),
            "b": pl.String,
            "wide": pl.String,
            "xs": pl.List(pl.Int64),
            "ss": pl.List(record),
            "s": record,
            "ts": pl.Datetime("us", "UTC"),
            "wall": pl.Datetime("us"),
            "text": pl.String,
            "texts": pl.List(pl.String),
            "point": pl.String,
        }
    )
    cursor = cursor_cls(fields, rows if with_rows else [])
    session = _Session(_Connection(cursor), owned=False, execute_options={})
    batches = list(session.execute("SELECT * FROM t", batch_size=1))
    assert len(batches) == (2 if with_rows else 1)
    assert all(df.schema == schema for df in batches)
    assert session.bigquery_text_columns == ["text"]
    assert cursor.fetches == ([1, 1, 1] if with_rows else [1])
    assert cursor.closed
    if with_rows:
        result = batches[1].row(0, named=True)
        assert result == {
            "n": exact,
            "narrow": Decimal("1.00000"),
            "b": "1",
            "wide": "1.1",
            "xs": [1, 2],
            "ss": [null_children, {"z": 5, "a": format(wide, "f"), "t": utc}],
            "s": null_children,
            "ts": utc,
            "wall": wall,
            "text": "hello",
            "texts": ["x"],
            "point": "POINT(0 0)",
        }
        assert batches[0].row(0) == rows[0]


def test_bigquery_bignumeric_text_is_independent_of_transport_padding() -> None:
    from polars_database._bigquery import result_frames

    pairs = [
        ("1", "1.00000000000000000000000000000000000000", "1"),
        ("0.1", "0.10000000000000000000000000000000000000", "0.1"),
        ("0", "-0.00000000000000000000000000000000000000", "0"),
        ("-1E-38", "-1E-38", "-0.00000000000000000000000000000000000001"),
        (
            "123456789012345678901.123456789012345678",
            "123456789012345678901.12345678901234567800000000000000000000",
            "123456789012345678901.123456789012345678",
        ),
    ]
    fields = (_Field("b", "BIGNUMERIC"),)
    empty, rest, _ = result_frames(
        _Cursor(fields, [(Decimal(r),) for r, _, _ in pairs]), 1
    )
    _, storage, _ = result_frames(
        _Cursor(fields, [(Decimal(s),) for _, s, _ in pairs]), 2
    )
    expected = pl.DataFrame({"b": [v for _, _, v in pairs]})
    assert empty.schema == expected.schema
    assert_frame_equal(pl.concat(rest), expected)
    assert_frame_equal(pl.concat(storage), expected)


def test_bigquery_json_inference_does_not_claim_a_declared_dtype() -> None:
    from polars_database._bigquery import result_frames

    fields = (_Field("j", "JSON"),)
    # REST decodes both SQL NULL and JSON null as None, and numbers as Python numbers.
    empty, batches, text_columns = result_frames(_Cursor(fields, [(None,), (3,)]), 2)
    assert empty.schema == pl.Schema({"j": pl.Null})
    assert text_columns == []
    assert_frame_equal(next(batches), pl.DataFrame({"j": [None, 3]}))


def test_bigquery_nested_json_inference_preserves_declared_siblings() -> None:
    from polars_database._bigquery import result_frames

    children = (
        _Field("id", "INTEGER"),
        _Field("j", "JSON"),
        _Field("ts", "TIMESTAMP"),
    )
    fields = (
        _Field("s", "RECORD", fields=children),
        _Field("ss", "RECORD", "REPEATED", fields=children),
    )
    utc = datetime(2025, 1, 2, 3, 4, 5, 123456, tzinfo=UTC)
    rows = [
        (None, None),
        ({"id": None, "j": {"a": 1}, "ts": utc}, []),
        (
            {"id": None, "j": {"a": 2, "b": "x"}, "ts": None},
            [{"id": None, "j": 3, "ts": None}],
        ),
    ]

    def record(json_dtype: pl.DataType | type[pl.DataType]) -> pl.Struct:
        return pl.Struct(
            {"id": pl.Int64, "j": json_dtype, "ts": pl.Datetime("us", "UTC")}
        )

    empty, batches, _ = result_frames(_Cursor(fields, rows), 1)
    no_json = pl.Schema({"s": record(pl.Null), "ss": pl.List(record(pl.Null))})
    assert empty.schema == no_json
    assert next(batches).schema == no_json
    second = next(batches)
    assert second.schema == pl.Schema(
        {"s": record(pl.Struct({"a": pl.Int64})), "ss": pl.List(record(pl.Null))}
    )
    assert second.row(0) == rows[1]
    third = next(batches)
    assert third.schema == pl.Schema(
        {
            "s": record(pl.Struct({"a": pl.Int64, "b": pl.String})),
            "ss": pl.List(record(pl.Int64)),
        }
    )
    assert third.row(0) == rows[2]

    _, together, _ = result_frames(_Cursor(fields, rows), 3)
    whole = next(together)
    assert whole.schema == third.schema
    assert whole["s"].struct.field("j").to_list() == [
        None,
        {"a": 1, "b": None},
        {"a": 2, "b": "x"},
    ]


@pytest.mark.parametrize("rows", [[], [("2025-01-02 03:04:05.123456789012+00",)]])
def test_bigquery_rejects_picosecond_timestamps(rows: list[tuple[str]]) -> None:
    from polars_database._bigquery import result_frames

    fields = (
        _Field("ts", "TIMESTAMP", timestamp_precision=_TimestampPrecision.PICOSECOND),
    )
    cursor = _Cursor(fields, rows)
    with pytest.raises(
        pl.exceptions.ComputeError,
        match=r"field 'ts'.*TIMESTAMP precision.*CAST\(<timestamp> AS STRING\)",
    ):
        result_frames(cursor, 1)
    assert cursor.fetches == []


@pytest.mark.parametrize("value", ["0.001", "1000", "NaN"])
def test_bigquery_decimal_conversion_rejects_rounding_or_overflow(value: str) -> None:
    from polars_database._bigquery import result_frames

    fields = (_Field("n", "NUMERIC", precision=5, scale=2),)
    cursor = _Cursor(fields, [(Decimal(value),)])
    empty, batches, _ = result_frames(cursor, 1)
    assert empty.schema == pl.Schema({"n": pl.Decimal(5, 2)})
    assert cursor.fetches == []
    with pytest.raises(pl.exceptions.ComputeError, match=r"field 'n'.*Decimal\(5, 2\)"):
        next(batches)
