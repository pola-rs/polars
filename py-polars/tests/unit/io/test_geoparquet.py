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
    from pathlib import Path

    from polars._typing import PolarsDataType
    from tests.conftest import PlMonkeyPatch

POINT_WKB = struct.pack("<bIdd", 1, 1, 1.0, 2.0)
PROJJSON = {"type": "GeographicCRS", "name": "WGS 84"}
CRS84 = {"crs": "OGC:CRS84"}
XY = pl.Struct({"x": pl.Float64, "y": pl.Float64})
XYZ = pl.Struct({"x": pl.Float64, "y": pl.Float64, "z": pl.Float64})
XYM = pl.Struct({"x": pl.Float64, "y": pl.Float64, "m": pl.Float64})
XYZM = pl.Struct(dict.fromkeys(["x", "y", "z", "m"], pl.Float64))


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


def _metadata(source: io.BytesIO | Path) -> dict[bytes, bytes]:
    where = (
        pa.BufferReader(source.getvalue()) if isinstance(source, io.BytesIO) else source
    )
    metadata: dict[bytes, bytes] = pq.read_metadata(where).metadata
    return metadata


def _read_geo(source: io.BytesIO | Path) -> Any:
    metadata = _metadata(source)
    return json.loads(metadata[b"geo"]) if b"geo" in metadata else None


def _geo_metadata(df: pl.DataFrame, **kwargs: Any) -> Any:
    buf = io.BytesIO()
    df.write_parquet(buf, **kwargs)
    return _read_geo(buf)


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
    assert _ext(df.schema["a"]) == ("geoarrow.wkb", pl.Binary, CRS84)
    assert _ext(df.schema["b"]) == ("geoarrow.wkb", pl.Binary, None)
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
    assert _ext(df.schema["a"]) == ("geoarrow.point", XY, CRS84)
    assert _ext(df.schema["b"]) == ("geoarrow.linestring", pl.List(XYZ), CRS84)
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
            "c": {
                "encoding": "multipoint",
                "geometry_types": ["MultiPoint Z"],
                "crs": None,
                "bbox": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
            },
        },
    }

    buf = io.BytesIO()
    df.write_parquet(buf)
    plmonkeypatch.setenv("POLARS_GEOPARQUET_READ", "1")
    out = pl.read_parquet(buf.getvalue())
    # A `null` CRS reads back as no CRS at all, both meaning it is unknown.
    for name in "ac":
        assert _ext(out.schema[name]) == _ext(df.schema[name])
    assert _ext(out.schema["b"]) == ("geoarrow.wkb", pl.Binary, None)
    assert out.select(pl.all().ext.storage()).rows() == [
        (POINT_WKB, POINT_WKB, [{"x": 1.0, "y": 2.0, "z": 3.0}])
    ]


def _bbox_frame() -> pl.DataFrame:
    nan = float("nan")
    return pl.DataFrame(
        {
            "a": [
                {"x": 1.0, "y": 2.0},
                None,
                {"x": nan, "y": -4.0},
                {"x": 3.0, "y": 0.0},
            ],
            "b": [
                [{"x": 0.0, "y": 0.0, "z": 0.0}, {"x": 1.0, "y": 1.0, "z": 1.0}],
                [{"x": -1.0, "y": 5.0, "z": 2.0}],
                [],
                None,
            ],
            "geometry": [POINT_WKB] * 4,
            "c": [POINT_WKB] * 4,
        },
        schema={
            "a": pl.Extension("geoarrow.point", XY, None),
            "b": pl.Extension("geoarrow.linestring", pl.List(XYZ), None),
            "geometry": pl.Extension("geoarrow.wkb", pl.Binary, None),
            "c": pl.Extension("geoarrow.wkb", pl.Binary, None),
        },
    )


def test_write_geoparquet_bbox(plmonkeypatch: PlMonkeyPatch, tmp_path: Path) -> None:
    df = _bbox_frame()
    plmonkeypatch.setenv("POLARS_GEOPARQUET_WRITE", "1.1")
    expected = {
        # Nulls and NaNs are left out of the bounds.
        "a": [1.0, -4.0, 3.0, 2.0],
        "b": [-1.0, 0.0, 0.0, 1.0, 5.0, 2.0],
    }
    columns = _geo_metadata(df, row_group_size=1)["columns"]
    assert {name: c["bbox"] for name, c in columns.items() if "bbox" in c} == expected

    path = tmp_path / "a.parquet"
    df.lazy().sink_parquet(path, row_group_size=1)
    assert pq.read_metadata(path).num_row_groups == 4
    columns = _read_geo(path)["columns"]
    assert {name: c["bbox"] for name, c in columns.items() if "bbox" in c} == expected


def _spherical(df: pl.DataFrame) -> pl.DataFrame:
    def dtype(name: str) -> pl.Extension:
        ext = df.schema[name]
        assert isinstance(ext, pl.Extension)
        md = '{"edges": "spherical"}'
        return pl.Extension(ext.ext_name(), ext.ext_storage(), md)

    return df.with_columns(
        pl.col(name).ext.storage().ext.to(dtype(name)) for name in ["a", "b"]
    )


@pytest.mark.parametrize(
    ("df", "kwargs"),
    [
        (_bbox_frame(), {"statistics": False}),
        (_bbox_frame(), {"statistics": {"min": False}}),
        (_bbox_frame().clear(4), {}),
        (_spherical(_bbox_frame()), {}),
    ],
    ids=["no-statistics", "no-min", "all-null", "spherical"],
)
def test_write_geoparquet_no_bbox(
    plmonkeypatch: PlMonkeyPatch, df: pl.DataFrame, kwargs: dict[str, Any]
) -> None:
    plmonkeypatch.setenv("POLARS_GEOPARQUET_WRITE", "1.1")
    columns = _geo_metadata(df, row_group_size=1, **kwargs)["columns"]
    assert set(columns) == {"a", "b", "geometry", "c"}
    assert all("bbox" not in c for c in columns.values())


def test_write_geoparquet_no_bbox_infinite(plmonkeypatch: PlMonkeyPatch) -> None:
    inf = float("inf")
    df = pl.DataFrame(
        {
            "a": [{"x": inf, "y": 1.0}, {"x": 1.0, "y": 2.0}],
            "b": [{"x": 1.0, "y": 2.0}, {"x": 3.0, "y": 4.0}],
        },
        schema={
            "a": pl.Extension("geoarrow.point", XY, None),
            "b": pl.Extension("geoarrow.point", XY, None),
        },
    )
    plmonkeypatch.setenv("POLARS_GEOPARQUET_WRITE", "1.1")
    columns = _geo_metadata(df)["columns"]
    # JSON can't hold an infinite bound.
    assert "bbox" not in columns["a"]
    assert columns["b"]["bbox"] == [1.0, 2.0, 3.0, 4.0]


def test_write_geoparquet_bbox_encrypted(
    plmonkeypatch: PlMonkeyPatch, tmp_path: Path
) -> None:
    plmonkeypatch.setenv("POLARS_GEOPARQUET_WRITE", "1.1")
    df = _bbox_frame().select("a")
    path = tmp_path / "a.parquet"
    df.write_parquet(path)
    assert b'"bbox"' in path.read_bytes()

    # The plaintext footer holds the `geo` metadata, but not the encrypted statistics.
    encryption = pl.ParquetEncryptionProperties(
        footer_key=b"0123456789012345", plaintext_footer=True
    )
    df.write_parquet(path, encryption=encryption)
    data = path.read_bytes()
    assert b'"encoding":"point"' in data
    assert b'"bbox"' not in data


def test_write_geoparquet_bbox_column_key_encrypted(
    plmonkeypatch: PlMonkeyPatch, tmp_path: Path
) -> None:
    plmonkeypatch.setenv("POLARS_GEOPARQUET_WRITE", "1.1")
    df = _bbox_frame().select(
        "a",
        b=pl.lit({"x": 3.0, "y": 4.0}).ext.to(pl.Extension("geoarrow.point", XY, None)),
    )
    encryption = pl.ParquetEncryptionProperties(
        footer_key=b"0123456789012345",
        column_keys={"a.y": b"1234567890123450"},
        plaintext_footer=True,
    )
    path = tmp_path / "a.parquet"
    df.write_parquet(path, encryption=encryption)
    columns = _read_geo(path)["columns"]
    # One encrypted leaf hides the whole `bbox` of its column, but no other.
    assert "bbox" not in columns["a"]
    assert columns["b"]["bbox"] == [3.0, 4.0, 3.0, 4.0]


@pytest.mark.parametrize(
    ("metadata", "written", "read"),
    [
        (None, {"crs": None}, None),
        ({"crs": None}, {"crs": None}, None),
        (CRS84, {}, CRS84),
        ({**CRS84, "crs_type": "authority_code"}, {}, CRS84),
        ({"crs": {**PROJJSON, "id": {"authority": "OGC", "code": "CRS84"}}}, {}, CRS84),
        ({"crs": PROJJSON}, {"crs": PROJJSON}, {"crs": PROJJSON}),
    ],
)
def test_geoparquet_crs_round_trip(
    plmonkeypatch: PlMonkeyPatch,
    metadata: dict[str, Any] | None,
    written: dict[str, Any],
    read: dict[str, Any] | None,
) -> None:
    md = None if metadata is None else json.dumps(metadata)
    df = pl.DataFrame(
        {"a": [POINT_WKB]}, schema={"a": pl.Extension("geoarrow.wkb", pl.Binary, md)}
    )
    plmonkeypatch.setenv("POLARS_GEOPARQUET_WRITE", "1.1")
    assert _geo_metadata(df)["columns"]["a"] == {
        "encoding": "WKB",
        "geometry_types": [],
        **written,
    }

    buf = io.BytesIO()
    df.write_parquet(buf)
    plmonkeypatch.setenv("POLARS_GEOPARQUET_READ", "1")
    out = pl.read_parquet(buf.getvalue())
    assert _ext(out.schema["a"]) == ("geoarrow.wkb", pl.Binary, read)


@pytest.mark.parametrize(
    ("version", "dtype", "match"),
    [
        ("2.0", pl.Extension("geoarrow.wkb", pl.Binary), "POLARS_GEOPARQUET_WRITE"),
        ("1.0", pl.Extension("geoarrow.point", XY), "geoarrow.point"),
        ("1.1", pl.Extension("geoarrow.wkt", pl.String), "geoarrow.wkt"),
        (
            "1.1",
            pl.Extension("geoarrow.point", XYM),
            r"GeoParquet 1\.1\.0: it cannot store M values",
        ),
        (
            "1.1",
            pl.Extension("geoarrow.polygon", pl.List(pl.List(XYZM))),
            r"GeoParquet 1\.1\.0: it cannot store M values",
        ),
        (
            "1.1",
            pl.Extension("geoarrow.wkb", pl.Binary, '{"crs": "EPSG:4326"}'),
            r"CRS \"EPSG:4326\" must be .*convert it to PROJJSON, or pass your own 'geo' key",
        ),
        (
            "1.1",
            pl.Extension(
                "geoarrow.wkb", pl.Binary, '{"crs": "OGC:CRS84", "crs_type": "srid"}'
            ),
            r"CRS \"OGC:CRS84\" of type 'srid' must be .*or the authority code 'OGC:CRS84'",
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


# Metadata without a `geo` key must not skip the early check.
@pytest.mark.parametrize("metadata", [None, {"k": "v"}])
def test_write_geoparquet_error_before_file_is_opened(
    plmonkeypatch: PlMonkeyPatch, tmp_path: Path, metadata: dict[str, str] | None
) -> None:
    plmonkeypatch.setenv("POLARS_GEOPARQUET_WRITE", "1.1")
    dtype = pl.Extension("geoarrow.wkb", pl.Binary, '{"crs": "EPSG:4326"}')
    df = pl.DataFrame({"a": [POINT_WKB] * 2, "p": [1, 2]}).with_columns(
        pl.col("a").ext.to(dtype)
    )

    path = tmp_path / "a.parquet"
    with pytest.raises(pl.exceptions.InvalidOperationError, match="CRS"):
        df.write_parquet(path, metadata=metadata)
    assert not path.exists()

    with pytest.raises(pl.exceptions.InvalidOperationError, match="CRS"):
        df.lazy().sink_parquet(
            pl.PartitionBy(tmp_path / "p", key="p"), mkdir=True, metadata=metadata
        )
    assert list(tmp_path.rglob("*.parquet")) == []


@pytest.mark.parametrize("as_callable", [False, True])
def test_write_geoparquet_explicit_geo_key(
    plmonkeypatch: PlMonkeyPatch, as_callable: bool
) -> None:
    # Polars refuses to derive the `geo` key for a string CRS such as "EPSG:4326",
    # but one passed in by the user is written as is.
    plmonkeypatch.setenv("POLARS_GEOPARQUET_WRITE", "1.1")
    dtype = pl.Extension("geoarrow.wkb", pl.Binary, '{"crs": "EPSG:4326"}')
    df = pl.DataFrame({"a": [POINT_WKB]}, schema={"a": dtype})
    column = {"encoding": "WKB", "geometry_types": [], "crs": PROJJSON}
    geo = {"version": "1.1.0", "primary_column": "a", "columns": {"a": column}}
    metadata = {"geo": json.dumps(geo)}

    buf = io.BytesIO()
    df.write_parquet(buf, metadata=(lambda _: metadata) if as_callable else metadata)
    written = _metadata(buf)
    assert json.loads(written[b"geo"]) == geo


def test_write_geoparquet_callable_metadata_derives_geo(
    plmonkeypatch: PlMonkeyPatch,
) -> None:
    plmonkeypatch.setenv("POLARS_GEOPARQUET_WRITE", "1.1")
    df = pl.DataFrame(
        {"a": [{"x": 1.0, "y": 2.0}]},
        schema={"a": pl.Extension("geoarrow.point", XY, None)},
    )
    buf = io.BytesIO()
    df.write_parquet(buf, metadata=lambda _: {"k": "v"})
    written = _metadata(buf)
    assert written[b"k"] == b"v"
    assert json.loads(written[b"geo"])["columns"]["a"]["bbox"] == [1.0, 2.0, 1.0, 2.0]
