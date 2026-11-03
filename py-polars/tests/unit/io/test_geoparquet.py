from __future__ import annotations

import io
import json
import struct
import warnings
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
XY_BOUNDS = {"xmin": 1.0, "ymin": 2.0, "xmax": 1.0, "ymax": 2.0}


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


def _metadata(source: io.BytesIO | Path) -> dict[bytes, bytes]:
    if isinstance(source, io.BytesIO):
        return pq.read_metadata(pa.BufferReader(source.getvalue())).metadata
    return pq.read_metadata(source).metadata


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


@pytest.mark.parametrize(
    ("bbox", "covering", "match"),
    [
        (
            pa.array([XY_BOUNDS]),
            {"bbox": {**_covering("bbox", BOX_XY)["bbox"], "xmin": ["bbox", "ymin"]}},
            "invalid bbox paths",
        ),
        (pa.array([XY_BOUNDS]), _covering("b", BOX_XY), "column 'b' not found"),
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


def test_read_geoparquet_covering_claimed_twice(plmonkeypatch: PlMonkeyPatch) -> None:
    table = pa.table(
        {
            "a": pa.array([POINT_WKB], pa.binary()),
            "b": pa.array([POINT_WKB], pa.binary()),
            "bbox": [XY_BOUNDS],
        }
    )
    wkb = {
        "encoding": "WKB",
        "geometry_types": [],
        "covering": _covering("bbox", BOX_XY),
    }
    buf = _write_geo(table, "1.1.0", {"a": {**wkb, "crs": PROJJSON}, "b": wkb})

    plmonkeypatch.setenv("POLARS_GEOPARQUET_READ", "1")
    with pytest.warns(UserWarning, match="'bbox' already covers another column"):
        df = pl.read_parquet(buf)
    # The first claim wins; the second geometry only loses its covering.
    assert _ext(df.schema["bbox"]) == ("geoarrow.box", BOX_XY, {"crs": PROJJSON})
    assert _ext(df.schema["b"]) == ("geoarrow.wkb", pl.Binary, CRS84)


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


def test_write_geoparquet_warns_unclaimed_box(
    plmonkeypatch: PlMonkeyPatch, tmp_path: Path
) -> None:
    # GeoPandas names its covering `bbox` whatever the geometry is called,
    # but Polars only writes `bbox` as the covering of `geometry` or `geography`.
    table = pa.table({"geom": pa.array([POINT_WKB], pa.binary()), "bbox": [XY_BOUNDS]})
    wkb = {
        "encoding": "WKB",
        "geometry_types": [],
        "covering": _covering("bbox", BOX_XY),
    }
    buf = _write_geo(table, "1.1.0", {"geom": wkb})

    plmonkeypatch.setenv("POLARS_GEOPARQUET_READ", "1")
    plmonkeypatch.setenv("POLARS_GEOPARQUET_WRITE", "1.1")
    df = pl.read_parquet(buf)
    assert _ext(df.schema["bbox"])[0] == "geoarrow.box"

    match = (
        "'bbox' as a plain struct, not as a GeoParquet covering: "
        "no geometry column has it as its covering name; name it '<geometry>_bbox'"
    )
    out = io.BytesIO()
    with pytest.warns(UserWarning, match=match):
        df.write_parquet(out)
    geo = _read_geo(out)
    assert "covering" not in geo["columns"]["geom"]
    assert pl.read_parquet(out.getvalue())["bbox"].ext.storage().to_list() == [
        XY_BOUNDS
    ]

    # Once per write, not once per file.
    partitioned = pl.concat([df, df]).with_columns(p=pl.Series([1, 2]))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        partitioned.lazy().sink_parquet(pl.PartitionBy(tmp_path, key="p"), mkdir=True)
    assert len(list(tmp_path.rglob("*.parquet"))) == 2
    assert sum("plain struct" in str(w.message) for w in caught) == 1


@pytest.mark.parametrize("box_name", ["bbox", "other"])
def test_write_geoparquet_box_not_a_covering(
    plmonkeypatch: PlMonkeyPatch, box_name: str
) -> None:
    # A valid `geoarrow.box`, but GeoParquet coverings have no M bounds.
    bounds = {f.name: 1.0 for f in BOX_XYM.fields}
    df = pl.DataFrame(
        {"geometry": [POINT_WKB], box_name: [bounds]},
        schema={
            "geometry": pl.Extension("geoarrow.wkb", pl.Binary, '{"crs": null}'),
            box_name: pl.Extension("geoarrow.box", BOX_XYM, '{"crs": null}'),
        },
    )
    plmonkeypatch.setenv("POLARS_GEOPARQUET_WRITE", "1.1")

    buf = io.BytesIO()
    match = f"'{box_name}' as a plain struct.*must be Float64 xmin, ymin"
    with pytest.warns(UserWarning, match=match):
        df.write_parquet(buf)
    geo = _read_geo(buf)
    assert list(geo["columns"]) == ["geometry"]
    assert "covering" not in geo["columns"]["geometry"]

    plmonkeypatch.setenv("POLARS_GEOPARQUET_READ", "1")
    out = pl.read_parquet(buf.getvalue())
    assert _ext(out.schema[box_name]) == _ext(df.schema[box_name])
    assert out[box_name].ext.storage().to_list() == [bounds]


def test_write_geoparquet_ambiguous_bbox(plmonkeypatch: PlMonkeyPatch) -> None:
    # Only `geometry` has a covering here,
    # but on write `bbox` could cover either column.
    wkb_array = pa.array([POINT_WKB], pa.binary())
    table = pa.table(
        {"geometry": wkb_array, "geography": wkb_array, "bbox": [XY_BOUNDS]}
    )
    wkb = {"encoding": "WKB", "geometry_types": []}
    columns = {
        "geometry": {**wkb, "covering": _covering("bbox", BOX_XY)},
        "geography": wkb,
    }
    buf = _write_geo(table, "1.1.0", columns)

    plmonkeypatch.setenv("POLARS_GEOPARQUET_READ", "1")
    plmonkeypatch.setenv("POLARS_GEOPARQUET_WRITE", "1.1")
    df = pl.read_parquet(buf)
    assert _ext(df.schema["bbox"])[0] == "geoarrow.box"

    out = io.BytesIO()
    match = "'bbox' as a plain struct.*could cover both 'geometry' and 'geography'"
    with pytest.warns(UserWarning, match=match):
        df.write_parquet(out)
    geo = _read_geo(out)
    assert all("covering" not in c for c in geo["columns"].values())
    assert pl.read_parquet(out.getvalue())["bbox"].ext.storage().to_list() == [
        XY_BOUNDS
    ]


def test_write_geoparquet_covering(plmonkeypatch: PlMonkeyPatch) -> None:
    xyz = {"xmin": 1.0, "ymin": 2.0, "zmin": 3.0, "xmax": 1.0, "ymax": 2.0, "zmax": 3.0}
    df = pl.DataFrame(
        {
            "geometry": [POINT_WKB],
            "bbox": [XY_BOUNDS],
            "a": [{"x": 1.0, "y": 2.0, "z": 3.0}],
            "a_bbox": [xyz],
        },
        schema={
            "geometry": pl.Extension("geoarrow.wkb", pl.Binary, json.dumps(CRS84)),
            "bbox": pl.Extension("geoarrow.box", BOX_XY, json.dumps(CRS84)),
            "a": pl.Extension("geoarrow.point", XYZ, json.dumps({"crs": PROJJSON})),
            # A box without a CRS may cover a geometry with one.
            "a_bbox": pl.Extension("geoarrow.box", BOX_XYZ, None),
        },
    )
    plmonkeypatch.setenv("POLARS_GEOPARQUET_WRITE", "1.1")
    columns = _geo_metadata(df)["columns"]
    assert set(columns) == {"geometry", "a"}
    assert columns["geometry"]["covering"] == _covering("bbox", BOX_XY)
    assert columns["a"]["covering"] == _covering("a_bbox", BOX_XYZ)

    # Only a `geoarrow.box` is a covering, not a plain struct with the same fields.
    plain = df.with_columns(pl.col("bbox").ext.storage())
    assert "covering" not in _geo_metadata(plain)["columns"]["geometry"]

    null_crs = pl.Extension("geoarrow.box", BOX_XYZ, '{"crs": null}')
    explicit = df.with_columns(pl.col("a_bbox").ext.storage().ext.to(null_crs))
    covering = _geo_metadata(explicit)["columns"]["a"]["covering"]
    assert covering == _covering("a_bbox", BOX_XYZ)

    buf = io.BytesIO()
    df.write_parquet(buf)
    plmonkeypatch.setenv("POLARS_GEOPARQUET_READ", "1")
    out = pl.read_parquet(buf.getvalue())
    for name in ["geometry", "bbox", "a"]:
        assert _ext(out.schema[name]) == _ext(df.schema[name])
    # A box reads back with the metadata of the geometry it covers.
    assert _ext(out.schema["a_bbox"]) == ("geoarrow.box", BOX_XYZ, {"crs": PROJJSON})
    assert out.select(pl.all().ext.storage()).rows() == [
        (POINT_WKB, XY_BOUNDS, {"x": 1.0, "y": 2.0, "z": 3.0}, xyz)
    ]

    plmonkeypatch.setenv("POLARS_GEOPARQUET_WRITE", "1.0")
    geometry = _geo_metadata(df.select("geometry", "bbox"))["columns"]["geometry"]
    assert "covering" not in geometry
    assert _geo_metadata(df.select("bbox")) is None


@pytest.mark.parametrize(
    ("schema", "match"),
    [
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
                "geometry": pl.Extension("geoarrow.wkb", pl.Binary, json.dumps(CRS84)),
                "bbox": pl.Extension(
                    "geoarrow.box", BOX_XY, json.dumps({**CRS84, "crs_type": "srid"})
                ),
            },
            "'bbox' as GeoParquet: CRS \"OGC:CRS84\" of type 'srid' must be",
        ),
    ],
)
def test_write_geoparquet_covering_errors(
    plmonkeypatch: PlMonkeyPatch, schema: dict[str, PolarsDataType], match: str
) -> None:
    df = pl.DataFrame({name: [None] for name in schema}, schema=schema)
    plmonkeypatch.setenv("POLARS_GEOPARQUET_WRITE", "1.1")
    with pytest.raises(pl.exceptions.InvalidOperationError, match=match):
        df.write_parquet(io.BytesIO())


CRS84_PROJJSON = {"crs": {**PROJJSON, "id": {"authority": "OGC", "code": "CRS84"}}}


@pytest.mark.parametrize(
    ("geometry_crs", "box_crs"), [(CRS84_PROJJSON, CRS84), (CRS84, CRS84_PROJJSON)]
)
def test_write_geoparquet_covering_crs84_spellings(
    plmonkeypatch: PlMonkeyPatch,
    geometry_crs: dict[str, Any],
    box_crs: dict[str, Any],
) -> None:
    # Both spellings are written as OGC:CRS84, so the box covers the geometry.
    df = pl.DataFrame(
        {"geometry": [POINT_WKB], "bbox": [XY_BOUNDS]},
        schema={
            "geometry": pl.Extension(
                "geoarrow.wkb", pl.Binary, json.dumps(geometry_crs)
            ),
            "bbox": pl.Extension("geoarrow.box", BOX_XY, json.dumps(box_crs)),
        },
    )
    plmonkeypatch.setenv("POLARS_GEOPARQUET_WRITE", "1.1")
    geometry = _geo_metadata(df)["columns"]["geometry"]
    assert geometry["covering"] == _covering("bbox", BOX_XY)
    assert "crs" not in geometry


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
