from __future__ import annotations

import io
import json
import struct
from typing import TYPE_CHECKING, Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import polars as pl

if TYPE_CHECKING:
    from polars._typing import PolarsDataType
    from tests.conftest import PlMonkeyPatch

POINT_WKB = struct.pack("<bIdd", 1, 1, 1.0, 2.0)
PROJJSON = {"type": "GeographicCRS", "name": "WGS 84"}
XY = pl.Struct({"x": pl.Float64, "y": pl.Float64})
XYZ = pl.Struct({"x": pl.Float64, "y": pl.Float64, "z": pl.Float64})


def _write_geo(table: pa.Table, version: str, columns: dict[str, Any]) -> io.BytesIO:
    geo = {"version": version, "primary_column": "a", "columns": columns}
    buf = io.BytesIO()
    pq.write_table(table.replace_schema_metadata({b"geo": json.dumps(geo)}), buf)
    buf.seek(0)
    return buf


def _ext(dtype: PolarsDataType) -> tuple[str, PolarsDataType, Any]:
    assert isinstance(dtype, pl.Extension)
    md = dtype.ext_metadata()
    return dtype.ext_name(), dtype.ext_storage(), None if md is None else json.loads(md)


def _geo_metadata(df: pl.DataFrame) -> Any:
    buf = io.BytesIO()
    df.write_parquet(buf)
    metadata = pq.read_metadata(pa.BufferReader(buf.getvalue())).metadata
    return json.loads(metadata[b"geo"]) if b"geo" in metadata else None


def test_read_geoparquet_1_0(plmonkeypatch: PlMonkeyPatch) -> None:
    table = pa.table({k: pa.array([POINT_WKB], pa.binary()) for k in "abc"})
    wkb = {"encoding": "WKB", "geometry_types": []}
    columns = {
        "a": wkb,
        "b": {**wkb, "crs": None},
        "c": {**wkb, "crs": PROJJSON, "edges": "spherical", "epoch": 2021.5},
    }
    buf = _write_geo(table, "1.0.0", columns)

    assert pl.read_parquet(buf).schema == dict.fromkeys("abc", pl.Binary)

    plmonkeypatch.setenv("POLARS_GEOPARQUET_READ", "1")
    df = pl.read_parquet(buf)
    assert _ext(df.schema["a"]) == ("geoarrow.wkb", pl.Binary, None)
    assert _ext(df.schema["b"]) == ("geoarrow.wkb", pl.Binary, {"crs": None})
    assert _ext(df.schema["c"]) == (
        "geoarrow.wkb",
        pl.Binary,
        {"crs": PROJJSON, "edges": "spherical", "epoch": 2021.5},
    )
    assert df["c"].ext.storage().to_list() == [POINT_WKB]


def test_read_geoparquet_1_1_native(plmonkeypatch: PlMonkeyPatch) -> None:
    point = {"x": 1.0, "y": 2.0}
    table = pa.table({"a": [point], "b": [[{**point, "z": 3.0}]]})
    columns = {
        "a": {"encoding": "point", "geometry_types": ["Point"]},
        "b": {"encoding": "linestring", "geometry_types": ["LineString Z"]},
    }
    plmonkeypatch.setenv("POLARS_GEOPARQUET_READ", "1")
    df = pl.read_parquet(_write_geo(table, "1.1.0", columns))
    assert _ext(df.schema["a"]) == ("geoarrow.point", XY, None)
    assert _ext(df.schema["b"]) == ("geoarrow.linestring", pl.List(XYZ), None)
    assert df["a"].ext.storage().to_list() == [point]


def test_write_geoparquet(plmonkeypatch: PlMonkeyPatch) -> None:
    md = json.dumps({"crs": PROJJSON, "edges": "spherical", "epoch": 2021.5})
    df = pl.DataFrame(
        {"a": [POINT_WKB], "b": [POINT_WKB], "c": [[{"x": 1.0, "y": 2.0, "z": 3.0}]]},
        schema={
            "a": pl.Extension("geoarrow.wkb", pl.Binary, md),
            "b": pl.Extension("geoarrow.wkb", pl.Binary, '{"crs": null}'),
            "c": pl.Extension("geoarrow.multipoint", pl.List(XYZ), None),
        },
    )
    assert _geo_metadata(df) is None

    wkb = {"encoding": "WKB", "geometry_types": []}
    a = {**wkb, "crs": PROJJSON, "edges": "spherical", "epoch": 2021.5}
    plmonkeypatch.setenv("POLARS_GEOPARQUET_WRITE", "1.0")
    assert _geo_metadata(df.drop("c")) == {
        "version": "1.0.0",
        "primary_column": "a",
        "columns": {"a": a, "b": {**wkb, "crs": None}},
    }

    plmonkeypatch.setenv("POLARS_GEOPARQUET_WRITE", "1.1")
    assert _geo_metadata(df) == {
        "version": "1.1.0",
        "primary_column": "a",
        "columns": {
            "a": a,
            "b": {**wkb, "crs": None},
            "c": {"encoding": "multipoint", "geometry_types": ["MultiPoint Z"]},
        },
    }

    buf = io.BytesIO()
    df.write_parquet(buf)
    plmonkeypatch.setenv("POLARS_GEOPARQUET_READ", "1")
    out = pl.read_parquet(buf.getvalue())
    for name in df.columns:
        assert _ext(out.schema[name]) == _ext(df.schema[name])
    assert out.select(pl.all().ext.storage()).rows() == [
        (POINT_WKB, POINT_WKB, [{"x": 1.0, "y": 2.0, "z": 3.0}])
    ]


@pytest.mark.parametrize(
    ("version", "dtype", "match"),
    [
        ("2.0", pl.Extension("geoarrow.wkb", pl.Binary), "POLARS_GEOPARQUET_WRITE"),
        ("1.0", pl.Extension("geoarrow.point", XY), "geoarrow.point"),
        ("1.1", pl.Extension("geoarrow.wkt", pl.String), "geoarrow.wkt"),
        (
            "1.1",
            pl.Extension("geoarrow.wkb", pl.Binary, '{"crs": "OGC:CRS84"}'),
            "CRS",
        ),
    ],
)
def test_write_geoparquet_errors(
    plmonkeypatch: PlMonkeyPatch, version: str, dtype: PolarsDataType, match: str
) -> None:
    plmonkeypatch.setenv("POLARS_GEOPARQUET_WRITE", version)
    df = pl.DataFrame({"a": [None]}, schema={"a": dtype})
    with pytest.raises(pl.exceptions.InvalidOperationError, match=match):
        df.write_parquet(io.BytesIO())
