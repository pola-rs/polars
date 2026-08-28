from __future__ import annotations

from datetime import datetime, timezone
from io import BytesIO, StringIO
from typing import TYPE_CHECKING
from uuid import UUID

import pytest

import polars as pl
from polars.testing import assert_frame_equal, assert_series_equal

if TYPE_CHECKING:
    from pathlib import Path

V4_A = UUID("a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11")
V4_B = UUID("f47ac10b-58cc-4372-a567-0e02b2c3d479")
V7 = UUID("019482e4-1441-7aad-8127-eec99573b0a0")


def test_uuid_construction_and_python_roundtrip() -> None:
    inferred = pl.Series("id", [V4_A, None, V4_B])
    explicit = pl.Series(
        "id",
        [str(V4_A).upper(), None, V4_B.bytes],
        dtype=pl.UUID,
    )

    assert inferred.dtype == pl.UUID
    assert inferred.to_list() == [V4_A, None, V4_B]
    assert_series_equal(inferred, explicit)
    assert inferred[0] == V4_A

    rows = pl.DataFrame([[V4_A], [V4_B]], orient="row")
    assert rows.schema == {"column_0": pl.UUID}
    assert rows["column_0"].to_list() == [V4_A, V4_B]


def test_uuid_postgresql_text_forms_and_strict_cast() -> None:
    values = pl.Series(
        [
            "a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11",
            "A0EEBC999C0B4EF8BB6D6BB9BD380A11",
            "{a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11}",
            "urn:uuid:a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11",
            "a0ee-bc99-9c0b-4ef8-bb6d-6bb9-bd38-0a11",
        ]
    ).cast(pl.UUID)
    assert values.to_list() == [V4_A] * 5
    assert values.cast(pl.String).to_list() == [str(V4_A)] * 5

    with pytest.raises(pl.exceptions.InvalidOperationError):
        pl.Series(["not-a-uuid"]).cast(pl.UUID)
    assert pl.Series(["not-a-uuid"]).cast(pl.UUID, strict=False).to_list() == [None]


def test_uuid_binary_and_integer_roundtrip() -> None:
    values = pl.Series("id", [V4_A, None, V4_B])
    assert values.cast(pl.Binary).to_list() == [V4_A.bytes, None, V4_B.bytes]
    assert_series_equal(values.cast(pl.Binary).cast(pl.UUID), values)
    assert values.cast(pl.UInt128).to_list() == [V4_A.int, None, V4_B.int]
    assert_series_equal(values.cast(pl.UInt128).cast(pl.UUID), values)


def test_uuid_sort_unique_group_and_join() -> None:
    values = pl.Series("id", [V4_B, V4_A, V4_B, None])
    assert values.sort(nulls_last=True).to_list() == [V4_A, V4_B, V4_B, None]
    assert values.n_unique() == 3

    counts = (
        pl.DataFrame({"id": values}).group_by("id").len().sort("id", nulls_last=True)
    )
    assert counts["len"].to_list() == [1, 2, 1]

    left = pl.DataFrame({"id": [V4_A, V4_B], "left": [1, 2]})
    right = pl.DataFrame({"id": [V4_B], "right": [3]})
    assert left.join(right, on="id", how="left")["right"].to_list() == [None, 3]
    assert values.min() == V4_A
    assert values.max() == V4_B
    assert values.arg_min() == 1
    assert values.arg_max() == 0
    assert values.is_in([V4_A]).to_list() == [False, True, False, None]


def test_uuid_scalar_comparison() -> None:
    values = pl.Series("id", [V4_B, V4_A, None])
    assert (values == V4_A).to_list() == [False, True, None]
    assert (values != V4_A).to_list() == [True, False, None]
    assert (values < V4_B).to_list() == [False, True, None]

    assert pl.DataFrame({"id": values}).filter(pl.col("id") == V4_A).to_dicts() == [
        {"id": V4_A}
    ]


def test_uuid_namespace() -> None:
    values = pl.Series("id", [V7, V4_A, None])
    assert values.uuid.version().to_list() == [7, 4, None]
    assert values.uuid.timestamp(strict=False).to_list() == [
        datetime.fromtimestamp(1737362773.057, tz=timezone.utc),
        None,
        None,
    ]
    with pytest.raises(pl.exceptions.ComputeError, match="UUIDv7"):
        values.uuid.timestamp()


@pytest.mark.parametrize(("function", "version"), [(pl.uuid4, 4), (pl.uuid7, 7)])
def test_uuid_generation(function: object, version: int) -> None:
    generated = function(128, eager=True)  # type: ignore[operator]
    assert generated.dtype == pl.UUID
    assert generated.len() == generated.n_unique() == 128
    assert generated.uuid.version().to_list() == [version] * 128

    generated_expr = pl.select(function(pl.lit(5)))  # type: ignore[operator]
    assert generated_expr.height == 5
    if version == 7:
        assert generated.is_sorted()


def test_uuid_arrow_parquet_roundtrip() -> None:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    values = pl.Series("id", [V4_A, None, V7])
    arrow = values.to_arrow()
    assert arrow.type.extension_name == "arrow.uuid"
    assert arrow.type.storage_type == pa.binary(16)
    assert_series_equal(pl.from_arrow(arrow).rename("id"), values)

    buffer = BytesIO()
    frame = values.to_frame()
    frame.write_parquet(buffer)
    buffer.seek(0)
    assert_frame_equal(pl.read_parquet(buffer), frame)

    parquet_schema = pq.read_schema(BytesIO(buffer.getvalue()))
    assert parquet_schema.field("id").type.extension_name == "arrow.uuid"

    ipc = BytesIO()
    frame.write_ipc(ipc)
    assert_frame_equal(pl.read_ipc(BytesIO(ipc.getvalue())), frame)


def test_nested_uuid_roundtrip() -> None:
    pa = pytest.importorskip("pyarrow")
    frame = pl.DataFrame(
        {"ids": [[V4_A, None, V4_B], None]},
        schema={"ids": pl.List(pl.UUID)},
    )
    assert frame.schema == {"ids": pl.List(pl.UUID)}
    assert frame.to_dicts() == [{"ids": [V4_A, None, V4_B]}, {"ids": None}]

    arrow = frame.to_arrow()
    assert arrow.schema.field("ids").type.value_type.extension_name == "arrow.uuid"
    assert arrow.schema.field("ids").type.value_type.storage_type == pa.binary(16)
    assert_frame_equal(pl.from_arrow(arrow), frame)

    ndjson = StringIO(f'{{"ids":["{V4_A}",null]}}\n{{"ids":null}}\n')
    assert_frame_equal(
        pl.read_ndjson(ndjson, schema={"ids": pl.List(pl.UUID)}),
        pl.DataFrame(
            {"ids": [[V4_A, None], None]},
            schema={"ids": pl.List(pl.UUID)},
        ),
    )


def test_uuid_lazy_parquet_predicate(tmp_path: Path) -> None:
    path = tmp_path / "uuid.parquet"
    frame = pl.DataFrame({"id": [V4_A, None, V4_B]})
    frame.write_parquet(path)
    result = pl.scan_parquet(path).filter(pl.col("id") == V4_B).collect()
    assert_frame_equal(result, pl.DataFrame({"id": [V4_B]}))


def test_uuid_csv_and_json_are_canonical_strings() -> None:
    frame = pl.DataFrame({"id": [V4_A, None]})
    csv = StringIO()
    frame.write_csv(csv)
    assert csv.getvalue() == f"id\n{V4_A}\n\n"

    json = StringIO()
    frame.write_json(json)
    assert json.getvalue() == f'[{{"id":"{V4_A}"}},{{"id":null}}]'


def test_uuid_csv_and_ndjson_read_schema_override() -> None:
    csv = StringIO(f"id\n{V4_A}\n")
    assert_frame_equal(
        pl.read_csv(csv, schema_overrides={"id": pl.UUID}),
        pl.DataFrame({"id": [V4_A]}),
    )

    ndjson = StringIO(f'{{"id":"{V4_A}"}}\n{{"id":null}}\n')
    assert_frame_equal(
        pl.read_ndjson(ndjson, schema={"id": pl.UUID}),
        pl.DataFrame({"id": [V4_A, None]}),
    )

    invalid = StringIO('{"id":"not-a-uuid"}\n')
    with pytest.raises(pl.exceptions.ComputeError, match="UUID"):
        pl.read_ndjson(invalid, schema={"id": pl.UUID})

    invalid = StringIO('{"id":"not-a-uuid"}\n')
    assert (
        pl.read_ndjson(
            invalid,
            schema={"id": pl.UUID},
            ignore_errors=True,
        ).item()
        is None
    )


def test_uuid_sql_cast() -> None:
    frame = pl.DataFrame({"id": [str(V4_A)]})
    out = frame.sql("SELECT CAST(id AS UUID) AS id FROM self")
    assert out.schema == {"id": pl.UUID}
    assert out.item() == V4_A
