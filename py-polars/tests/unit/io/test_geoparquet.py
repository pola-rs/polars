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
BOX_XY = pl.Struct(dict.fromkeys(["xmin", "ymin", "xmax", "ymax"], pl.Float64))
BOX_XYZ = pl.Struct(
    dict.fromkeys(["xmin", "ymin", "zmin", "xmax", "ymax", "zmax"], pl.Float64)
)
BOX_XYM = pl.Struct(
    dict.fromkeys(["xmin", "ymin", "mmin", "xmax", "ymax", "mmax"], pl.Float64)
)


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


def _covering(box_name: str, box: pl.Struct) -> dict[str, Any]:
    return {"bbox": {f.name: [box_name, f.name] for f in box.fields}}


def _geo_metadata(df: pl.DataFrame, **kwargs: Any) -> Any:
    buf = io.BytesIO()
    df.write_parquet(buf, **kwargs)
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


@pytest.mark.parametrize(
    ("box", "column", "metadata"),
    [
        (BOX_XY, {}, CRS84),
        (
            BOX_XYZ,
            {"crs": PROJJSON, "edges": "spherical"},
            {"crs": PROJJSON, "edges": "spherical"},
        ),
    ],
)
def test_read_geoparquet_covering(
    plmonkeypatch: PlMonkeyPatch,
    box: pl.Struct,
    column: dict[str, Any],
    metadata: dict[str, Any],
) -> None:
    bounds = {f.name: float(i) for i, f in enumerate(box.fields)}
    table = pa.table({"a": pa.array([POINT_WKB], pa.binary()), "bbox": [bounds]})
    wkb = {"encoding": "WKB", "geometry_types": [], **column}
    buf = _write_geo(table, "1.1.0", {"a": {**wkb, "covering": _covering("bbox", box)}})
    assert pl.read_parquet(buf).schema["bbox"] == box

    plmonkeypatch.setenv("POLARS_GEOPARQUET_READ", "1")
    df = pl.read_parquet(buf)
    # The box gets the same metadata as the geometry it covers.
    assert _ext(df.schema["a"]) == ("geoarrow.wkb", pl.Binary, metadata)
    assert _ext(df.schema["bbox"]) == ("geoarrow.box", box, metadata)
    assert df["bbox"].ext.storage().to_list() == [bounds]


XY_BOUNDS = {"xmin": 1.0, "ymin": 2.0, "xmax": 1.0, "ymax": 2.0}


@pytest.mark.parametrize(
    ("bbox", "covering", "match"),
    [
        (
            pa.array([XY_BOUNDS]),
            {"bbox": {**_covering("bbox", BOX_XY)["bbox"], "ymax": ["b", "ymax"]}},
            "invalid bbox paths",
        ),
        (
            pa.array([XY_BOUNDS]),
            {"bbox": {**_covering("bbox", BOX_XY)["bbox"], "xmin": ["bbox", "ymin"]}},
            "invalid bbox paths",
        ),
        (pa.array([XY_BOUNDS]), _covering("b", BOX_XY), "column 'b' not found"),
        (
            pa.array(
                [XY_BOUNDS], pa.struct([(name, pa.float32()) for name in XY_BOUNDS])
            ),
            _covering("bbox", BOX_XY),
            "not a struct of Float64 fields",
        ),
        (
            pa.array([{f.name: 1.0 for f in BOX_XYM.fields}]),
            _covering("bbox", BOX_XY),
            "not a struct of Float64 fields",
        ),
    ],
)
def test_read_geoparquet_invalid_covering(
    plmonkeypatch: PlMonkeyPatch, bbox: pa.Array, covering: dict[str, Any], match: str
) -> None:
    table = pa.table({"a": pa.array([POINT_WKB], pa.binary()), "bbox": bbox})
    wkb = {"encoding": "WKB", "geometry_types": [], "covering": covering}
    buf = _write_geo(table, "1.1.0", {"a": wkb})

    plmonkeypatch.setenv("POLARS_GEOPARQUET_READ", "1")
    with pytest.warns(UserWarning, match=match):
        df = pl.read_parquet(buf)
    assert _ext(df.schema["a"])[0] == "geoarrow.wkb"
    assert not isinstance(df.schema["bbox"], pl.Extension)


def test_write_geoparquet(plmonkeypatch: PlMonkeyPatch) -> None:
    md = json.dumps({"crs": PROJJSON, "edges": "spherical", "epoch": 2021.5})
    df = pl.DataFrame(
        {
            "a": [POINT_WKB],
            "b": [POINT_WKB],
            "c": [[{"x": 1.0, "y": 2.0, "z": 3.0}]],
            "d": [POINT_WKB],
        },
        schema={
            "a": pl.Extension("geoarrow.wkb", pl.Binary, md),
            "b": pl.Extension("geoarrow.wkb", pl.Binary, '{"crs": null}'),
            "c": pl.Extension("geoarrow.multipoint", pl.List(XYZ), None),
            "d": pl.Extension("geoarrow.wkb", pl.Binary, json.dumps(CRS84)),
        },
    )
    assert _geo_metadata(df) is None

    wkb = {"encoding": "WKB", "geometry_types": []}
    a = {**wkb, "crs": PROJJSON, "edges": "spherical", "epoch": 2021.5}
    plmonkeypatch.setenv("POLARS_GEOPARQUET_WRITE", "1.0")
    assert _geo_metadata(df.drop("c")) == {
        "version": "1.0.0",
        "primary_column": "a",
        "columns": {"a": a, "b": {**wkb, "crs": None}, "d": wkb},
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
            "d": wkb,
        },
    }

    buf = io.BytesIO()
    df.write_parquet(buf)
    plmonkeypatch.setenv("POLARS_GEOPARQUET_READ", "1")
    out = pl.read_parquet(buf.getvalue())
    # A `null` CRS reads back as no CRS at all, both meaning it is unknown.
    for name in "acd":
        assert _ext(out.schema[name]) == _ext(df.schema[name])
    assert _ext(out.schema["b"]) == ("geoarrow.wkb", pl.Binary, None)
    assert out.select(pl.all().ext.storage()).rows() == [
        (POINT_WKB, POINT_WKB, [{"x": 1.0, "y": 2.0, "z": 3.0}], POINT_WKB)
    ]


@pytest.mark.parametrize("version", ["1.0", "1.1"])
def test_write_geoparquet_skips_box(plmonkeypatch: PlMonkeyPatch, version: str) -> None:
    bounds = {"xmin": 1.0, "ymin": 2.0, "xmax": 1.0, "ymax": 2.0}
    df = pl.DataFrame(
        {"a": [POINT_WKB], "bbox": [bounds]},
        schema={
            "a": pl.Extension("geoarrow.wkb", pl.Binary, '{"crs": null}'),
            "bbox": pl.Extension("geoarrow.box", BOX_XY, '{"crs": null}'),
        },
    )
    plmonkeypatch.setenv("POLARS_GEOPARQUET_WRITE", version)
    geo = _geo_metadata(df)
    assert geo["primary_column"] == "a"
    assert list(geo["columns"]) == ["a"]
    assert "covering" not in geo["columns"]["a"]
    assert _geo_metadata(df.select("bbox")) is None

    buf = io.BytesIO()
    df.write_parquet(buf)
    plmonkeypatch.setenv("POLARS_GEOPARQUET_READ", "1")
    out = pl.read_parquet(buf.getvalue())
    assert _ext(out.schema["bbox"]) == _ext(df.schema["bbox"])
    assert out["bbox"].ext.storage().to_list() == [bounds]


def test_write_geoparquet_covering(plmonkeypatch: PlMonkeyPatch) -> None:
    xy = {"xmin": 1.0, "ymin": 2.0, "xmax": 1.0, "ymax": 2.0}
    xyz = {"xmin": 1.0, "ymin": 2.0, "zmin": 3.0, "xmax": 1.0, "ymax": 2.0, "zmax": 3.0}
    df = pl.DataFrame(
        {
            "geometry": [POINT_WKB],
            "bbox": [xy],
            "a": [{"x": 1.0, "y": 2.0, "z": 3.0}],
            "a_bbox": [xyz],
        },
        schema={
            "geometry": pl.Extension("geoarrow.wkb", pl.Binary, json.dumps(CRS84)),
            "bbox": pl.Extension("geoarrow.box", BOX_XY, json.dumps(CRS84)),
            "a": pl.Extension("geoarrow.point", XYZ, None),
            # A box without a CRS may cover a geometry with one.
            "a_bbox": pl.Extension("geoarrow.box", BOX_XYZ, None),
        },
    )
    assert _geo_metadata(df) is None

    plmonkeypatch.setenv("POLARS_GEOPARQUET_WRITE", "1.1")
    columns = _geo_metadata(df)["columns"]
    assert set(columns) == {"geometry", "a"}
    assert columns["geometry"]["covering"] == _covering("bbox", BOX_XY)
    assert columns["a"]["covering"] == _covering("a_bbox", BOX_XYZ)

    # Only a `geoarrow.box` is a covering, not a plain struct with the same fields.
    plain = df.with_columns(pl.col("bbox").ext.storage())
    assert "covering" not in _geo_metadata(plain)["columns"]["geometry"]

    buf = io.BytesIO()
    df.write_parquet(buf)
    plmonkeypatch.setenv("POLARS_GEOPARQUET_READ", "1")
    out = pl.read_parquet(buf.getvalue())
    for name in df.columns:
        assert _ext(out.schema[name]) == _ext(df.schema[name])
    assert out.select(pl.all().ext.storage()).rows() == [
        (POINT_WKB, xy, {"x": 1.0, "y": 2.0, "z": 3.0}, xyz)
    ]

    plmonkeypatch.setenv("POLARS_GEOPARQUET_WRITE", "1.0")
    geometry = _geo_metadata(df.select("geometry", "bbox"))["columns"]["geometry"]
    assert "covering" not in geometry


@pytest.mark.parametrize(
    ("schema", "match"),
    [
        (
            {
                "geometry": pl.Extension("geoarrow.wkb", pl.Binary, '{"crs": null}'),
                "bbox": pl.Extension("geoarrow.box", BOX_XYM, '{"crs": null}'),
            },
            "must have Float64 fields xmin, ymin",
        ),
        (
            {
                "geometry": pl.Extension("geoarrow.wkb", pl.Binary, json.dumps(CRS84)),
                "bbox": pl.Extension(
                    "geoarrow.box", BOX_XY, json.dumps({"crs": PROJJSON})
                ),
            },
            "differs from the geometry's",
        ),
        (
            {
                "geometry": pl.Extension("geoarrow.wkb", pl.Binary, '{"crs": null}'),
                "geography": pl.Extension("geoarrow.wkb", pl.Binary, '{"crs": null}'),
                "bbox": pl.Extension("geoarrow.box", BOX_XY, '{"crs": null}'),
            },
            "'bbox' already covers another column",
        ),
    ],
)
def test_write_geoparquet_covering_errors(
    plmonkeypatch: PlMonkeyPatch, schema: dict[str, PolarsDataType], match: str
) -> None:
    df = pl.DataFrame({name: [None] for name in schema}, schema=schema)
    assert _geo_metadata(df) is None

    plmonkeypatch.setenv("POLARS_GEOPARQUET_WRITE", "1.1")
    with pytest.raises(pl.exceptions.InvalidOperationError, match=match):
        df.write_parquet(io.BytesIO())


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
            "bbox": [
                {"xmin": 0.0, "ymin": 0.0, "xmax": 1.0, "ymax": 1.0},
                {"xmin": -2.0, "ymin": 1.0, "xmax": 0.0, "ymax": 3.0},
                None,
                None,
            ],
            "c": [POINT_WKB] * 4,
        },
        schema={
            "a": pl.Extension("geoarrow.point", XY, None),
            "b": pl.Extension("geoarrow.linestring", pl.List(XYZ), None),
            "geometry": pl.Extension("geoarrow.wkb", pl.Binary, None),
            "bbox": pl.Extension("geoarrow.box", BOX_XY, None),
            "c": pl.Extension("geoarrow.wkb", pl.Binary, None),
        },
    )


def test_write_geoparquet_bbox(plmonkeypatch: PlMonkeyPatch, tmp_path: Path) -> None:
    df = _bbox_frame()
    assert _geo_metadata(df) is None

    plmonkeypatch.setenv("POLARS_GEOPARQUET_WRITE", "1.1")
    expected = {
        # Nulls and NaNs are left out of the bounds.
        "a": [1.0, -4.0, 3.0, 2.0],
        "b": [-1.0, 0.0, 0.0, 1.0, 5.0, 2.0],
    }
    columns = _geo_metadata(df, row_group_size=1)["columns"]
    assert {name: c["bbox"] for name, c in columns.items() if "bbox" in c} == expected
    # WKB gets no `bbox`, not even from the box that covers it.
    assert "covering" in columns["geometry"]

    path = tmp_path / "a.parquet"
    df.lazy().sink_parquet(path, row_group_size=1)
    columns = json.loads(pq.read_metadata(path).metadata[b"geo"])["columns"]
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


def test_write_geoparquet_no_bbox_from_covering_over_antimeridian(
    plmonkeypatch: PlMonkeyPatch,
) -> None:
    # The minimum `xmin` and maximum `xmax` of these boxes are 0 and 10,
    # which leave out the first box: it crosses the antimeridian from 170 to -170.
    df = pl.DataFrame(
        {
            "geometry": [POINT_WKB] * 2,
            "bbox": [
                {"xmin": 170.0, "ymin": 0.0, "xmax": -170.0, "ymax": 1.0},
                {"xmin": 0.0, "ymin": 0.0, "xmax": 10.0, "ymax": 1.0},
            ],
        },
        schema={
            "geometry": pl.Extension("geoarrow.wkb", pl.Binary, json.dumps(CRS84)),
            "bbox": pl.Extension("geoarrow.box", BOX_XY, json.dumps(CRS84)),
        },
    )
    plmonkeypatch.setenv("POLARS_GEOPARQUET_WRITE", "1.1")
    geometry = _geo_metadata(df)["columns"]["geometry"]
    assert geometry["covering"] == _covering("bbox", BOX_XY)
    assert "bbox" not in geometry


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
            pl.Extension("geoarrow.wkb", pl.Binary, '{"crs": "EPSG:4326"}'),
            r"CRS \"EPSG:4326\" must be .*convert it to PROJJSON, or pass your own 'geo' key",
        ),
        (
            "1.1",
            pl.Extension(
                "geoarrow.wkb", pl.Binary, '{"crs": "OGC:CRS84", "crs_type": "srid"}'
            ),
            r"CRS \"OGC:CRS84\" of type 'srid' must be",
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


def test_write_geoparquet_error_before_file_is_opened(
    plmonkeypatch: PlMonkeyPatch, tmp_path: Path
) -> None:
    plmonkeypatch.setenv("POLARS_GEOPARQUET_WRITE", "1.1")
    dtype = pl.Extension("geoarrow.wkb", pl.Binary, '{"crs": "EPSG:4326"}')
    df = pl.DataFrame({"a": [POINT_WKB] * 2, "p": [1, 2]}).with_columns(
        pl.col("a").ext.to(dtype)
    )

    path = tmp_path / "a.parquet"
    with pytest.raises(pl.exceptions.InvalidOperationError, match="CRS"):
        df.write_parquet(path)
    assert not path.exists()

    with pytest.raises(pl.exceptions.InvalidOperationError, match="CRS"):
        df.lazy().sink_parquet(pl.PartitionBy(tmp_path / "p", key="p"), mkdir=True)
    assert list(tmp_path.rglob("*.parquet")) == []


@pytest.mark.parametrize("as_callable", [False, True])
def test_write_geoparquet_explicit_geo_key(
    plmonkeypatch: PlMonkeyPatch, as_callable: bool
) -> None:
    # Polars refuses to derive the `geo` key for a string CRS,
    # but one passed in by the user is written as is.
    plmonkeypatch.setenv("POLARS_GEOPARQUET_WRITE", "1.1")
    dtype = pl.Extension("geoarrow.wkb", pl.Binary, '{"crs": "EPSG:4326"}')
    df = pl.DataFrame({"a": [POINT_WKB]}, schema={"a": dtype})
    column = {"encoding": "WKB", "geometry_types": [], "crs": PROJJSON}
    geo = {"version": "1.1.0", "primary_column": "a", "columns": {"a": column}}
    metadata = {"geo": json.dumps(geo)}

    buf = io.BytesIO()
    df.write_parquet(buf, metadata=(lambda _: metadata) if as_callable else metadata)
    written = pq.read_metadata(pa.BufferReader(buf.getvalue())).metadata
    assert json.loads(written[b"geo"]) == geo
