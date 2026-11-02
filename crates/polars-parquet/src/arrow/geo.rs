//! Conversion between GeoParquet `geo` file metadata and GeoArrow extension types.
use polars_arrow::datatypes::{ArrowDataType, ArrowSchema, ExtensionType, Metadata};
use polars_error::{PolarsResult, polars_bail, polars_err, polars_warn};
use polars_parquet_format::RowGroup;
use polars_utils::pl_str::PlSmallStr;
use serde_json::{Map, Value, json};

use crate::parquet::metadata::{ColumnDescriptor, KeyValue};
use crate::parquet::types::decode;

pub const GEO_KEY: &str = "geo";

/// The geoparquet spec states that something that has no `crs` key should be considered "OGC: CRS84".
const DEFAULT_CRS: &str = "OGC:CRS84";

const BOX_EXTENSION_NAME: &str = "geoarrow.box";

/// The field layouts of a box that GeoParquet accepts as a bbox covering.
const BOX_LAYOUTS: [&[&str]; 2] = [
    &["xmin", "ymin", "xmax", "ymax"],
    &["xmin", "ymin", "zmin", "xmax", "ymax", "zmax"],
];

struct Encoding {
    geoparquet: &'static str,
    geoarrow: &'static str,
    geometry_type: &'static str,
    /// Number of list levels around the coordinates, `None` for WKB.
    list_depth: Option<usize>,
}

const fn native(
    geoparquet: &'static str,
    geoarrow: &'static str,
    geometry_type: &'static str,
    list_depth: usize,
) -> Encoding {
    Encoding {
        geoparquet,
        geoarrow,
        geometry_type,
        list_depth: Some(list_depth),
    }
}

const ENCODINGS: [Encoding; 7] = [
    Encoding {
        geoparquet: "WKB",
        geoarrow: "geoarrow.wkb",
        geometry_type: "",
        list_depth: None,
    },
    native("point", "geoarrow.point", "Point", 0),
    native("linestring", "geoarrow.linestring", "LineString", 1),
    native("polygon", "geoarrow.polygon", "Polygon", 2),
    native("multipoint", "geoarrow.multipoint", "MultiPoint", 1),
    native(
        "multilinestring",
        "geoarrow.multilinestring",
        "MultiLineString",
        2,
    ),
    native("multipolygon", "geoarrow.multipolygon", "MultiPolygon", 3),
];

/// Returns `Some(has_z)` if `dtype` is a valid storage type for `encoding`.
fn check_storage(encoding: &Encoding, mut dtype: &ArrowDataType) -> Option<bool> {
    let Some(list_depth) = encoding.list_depth else {
        return matches!(
            dtype,
            ArrowDataType::Binary | ArrowDataType::LargeBinary | ArrowDataType::BinaryView
        )
        .then_some(false);
    };
    for _ in 0..list_depth {
        dtype = match dtype {
            ArrowDataType::List(f) | ArrowDataType::LargeList(f) => &f.dtype,
            _ => return None,
        };
    }
    let ArrowDataType::Struct(fields) = dtype else {
        return None;
    };
    let valid = (2..=3).contains(&fields.len())
        && fields
            .iter()
            .zip(["x", "y", "z"])
            .all(|(f, name)| f.name == name && f.dtype == ArrowDataType::Float64);
    valid.then_some(fields.len() == 3)
}

/// Turns the columns listed in the GeoParquet `geo` metadata into GeoArrow extension types if
/// `POLARS_GEOPARQUET_READ=1`.
pub fn apply_geo_metadata(metadata: &Metadata, schema: &mut ArrowSchema) {
    if std::env::var("POLARS_GEOPARQUET_READ").as_deref() != Ok("1") {
        return;
    }
    let Some(geo) = metadata.get(GEO_KEY) else {
        return;
    };
    let Ok(geo) = serde_json::from_str::<Value>(geo) else {
        polars_warn!("ignoring invalid GeoParquet metadata");
        return;
    };
    let version = geo["version"].as_str().unwrap_or_default();
    if version.split('.').next() != Some("1") {
        polars_warn!(
            "ignoring GeoParquet metadata with unsupported version '{}'",
            version
        );
        return;
    }
    let Some(columns) = geo["columns"].as_object() else {
        polars_warn!("ignoring invalid GeoParquet metadata");
        return;
    };

    let mut covering_columns = Vec::new();
    for (name, column) in columns {
        let Some(field) = schema.get_mut(name) else {
            polars_warn!("ignoring GeoParquet column '{}': column not found", name);
            continue;
        };
        let encoding_name = column["encoding"].as_str().unwrap_or_default();
        let Some(encoding) = ENCODINGS.iter().find(|e| e.geoparquet == encoding_name) else {
            polars_warn!(
                "ignoring GeoParquet column '{}': unknown encoding '{}'",
                name,
                encoding_name
            );
            continue;
        };
        let storage = match &field.dtype {
            ArrowDataType::Extension(ext) => &ext.inner,
            dtype => dtype,
        };
        if check_storage(encoding, storage).is_none() {
            polars_warn!(
                "ignoring GeoParquet column '{}': type {:?} does not match encoding '{}'",
                name,
                storage,
                encoding_name
            );
            continue;
        }

        let mut ext_metadata = Map::new();
        match column.get("crs") {
            None => {
                ext_metadata.insert("crs".into(), DEFAULT_CRS.into());
            },
            // Geoarrow designates an 'unknown' CRS by leaving `crs` out,
            // while GeoParquet explicitly writes it as `Null`.
            Some(Value::Null) => {},
            Some(crs) => {
                ext_metadata.insert("crs".into(), crs.clone());
            },
        }
        for key in ["edges", "epoch"] {
            if let Some(value) = column.get(key) {
                ext_metadata.insert(key.into(), value.clone());
            }
        }

        let metadata: Option<PlSmallStr> =
            (!ext_metadata.is_empty()).then(|| Value::Object(ext_metadata).to_string().into());
        field.dtype = ArrowDataType::Extension(Box::new(ExtensionType {
            name: encoding.geoarrow.into(),
            inner: storage.clone(),
            metadata: metadata.clone(),
        }));

        if let Some(covering) = column.get("covering") {
            apply_bbox_covering(schema, name, covering, metadata, &mut covering_columns);
        }
    }
}

/// Turns the column that a `bbox` covering of `geometry` points to into a `geoarrow.box`,
/// with the same extension `metadata` as `geometry`.
fn apply_bbox_covering(
    schema: &mut ArrowSchema,
    geometry: &str,
    covering: &Value,
    metadata: Option<PlSmallStr>,
    covering_columns: &mut Vec<String>,
) {
    // The spec only defines `bbox`, so other coverings are left for future versions.
    let Some(bbox) = covering.get("bbox") else {
        return;
    };
    let Some((box_name, layout)) = parse_bbox_paths(bbox) else {
        polars_warn!(
            "ignoring GeoParquet covering of '{}': invalid bbox paths",
            geometry
        );
        return;
    };
    if covering_columns.iter().any(|c| c == box_name) {
        polars_warn!(
            "ignoring GeoParquet covering of '{}': column '{}' already covers another column",
            geometry,
            box_name
        );
        return;
    }
    let Some(field) = schema.get_mut(box_name) else {
        polars_warn!(
            "ignoring GeoParquet covering of '{}': column '{}' not found",
            geometry,
            box_name
        );
        return;
    };
    let storage = match &field.dtype {
        ArrowDataType::Extension(ext) => &ext.inner,
        dtype => dtype,
    };
    if box_layout(storage) != Some(layout) {
        polars_warn!(
            "ignoring GeoParquet covering of '{}': type {:?} of column '{}' is not a struct of Float64 fields matching its bbox paths",
            geometry,
            storage,
            box_name
        );
        return;
    }

    field.dtype = ArrowDataType::Extension(Box::new(ExtensionType {
        name: BOX_EXTENSION_NAME.into(),
        inner: storage.clone(),
        metadata,
    }));
    covering_columns.push(box_name.to_string());
}

/// Returns the column that a GeoParquet `bbox` covering points to, and the layout of its fields.
fn parse_bbox_paths(bbox: &Value) -> Option<(&str, &'static [&'static str])> {
    let bbox = bbox.as_object()?;
    let box_name = bbox.get("xmin")?.get(0)?.as_str()?;
    let layout = BOX_LAYOUTS.into_iter().find(|layout| {
        bbox.len() == layout.len()
            && layout
                .iter()
                .all(|key| bbox.get(*key) == Some(&json!([box_name, key])))
    })?;
    Some((box_name, layout))
}

/// Returns the layout of `dtype` if it is a box that GeoParquet accepts as a bbox covering.
fn box_layout(dtype: &ArrowDataType) -> Option<&'static [&'static str]> {
    let ArrowDataType::Struct(fields) = dtype else {
        return None;
    };
    BOX_LAYOUTS.into_iter().find(|layout| {
        fields.len() == layout.len()
            && fields
                .iter()
                .zip(*layout)
                .all(|(f, name)| f.name == *name && f.dtype == ArrowDataType::Float64)
    })
}

/// The statistics of a written file, to fill in the `bbox` of each geometry column.
pub struct FileStatistics<'a> {
    pub columns: &'a [ColumnDescriptor],
    pub row_groups: &'a [RowGroup],
}

/// Builds the GeoParquet `geo` metadata for the GeoArrow columns in `schema` if
/// `POLARS_GEOPARQUET_WRITE` is set to `1.0` or `1.1`.
/// The `bbox` of each geometry column is only filled in if `statistics` are given.
pub fn geo_key_value(
    schema: &ArrowSchema,
    statistics: Option<&FileStatistics<'_>>,
) -> PolarsResult<Option<KeyValue>> {
    let version = match std::env::var("POLARS_GEOPARQUET_WRITE").as_deref() {
        Err(_) | Ok("") => return Ok(None),
        Ok("1.0") => "1.0.0",
        Ok("1.1") => "1.1.0",
        Ok(v) => polars_bail!(
            InvalidOperation: "invalid POLARS_GEOPARQUET_WRITE value '{}', expected '1.0' or '1.1'", v
        ),
    };

    let mut primary_column = None;
    let mut columns = Map::new();
    let mut covering_columns = Vec::new();
    for field in schema.iter_values() {
        let ArrowDataType::Extension(ext) = &field.dtype else {
            continue;
        };
        // GeoParquet has no box encoding, so a box is written as a plain struct column.
        if !ext.name.starts_with("geoarrow.") || ext.name == BOX_EXTENSION_NAME {
            continue;
        }
        let name = &field.name;
        let encoding = ENCODINGS
            .iter()
            .find(|e| ext.name == e.geoarrow && (e.list_depth.is_none() || version != "1.0.0"))
            .ok_or_else(|| {
                polars_err!(
                    InvalidOperation: "cannot write column '{}' of type '{}' as GeoParquet {}",
                    name, ext.name, version
                )
            })?;
        let has_z = check_storage(encoding, &ext.inner).ok_or_else(|| {
            polars_err!(
                InvalidOperation: "cannot write column '{}' as GeoParquet: storage type {:?} does not match '{}'",
                name, ext.inner, ext.name
            )
        })?;

        let ext_metadata = extension_metadata(name, ext)?;

        let geometry_types = match encoding.list_depth {
            None => vec![],
            Some(_) if has_z => vec![format!("{} Z", encoding.geometry_type)],
            Some(_) => vec![encoding.geometry_type.to_string()],
        };
        let mut column = Map::new();
        column.insert("encoding".into(), encoding.geoparquet.into());
        column.insert("geometry_types".into(), geometry_types.into());
        let crs_type = ext_metadata.get("crs_type").and_then(Value::as_str);
        match ext_metadata.get("crs") {
            // Geoarrow designates an 'unknown' CRS by leaving `crs` out,
            // while GeoParquet explicitly writes it as `Null`.
            // Leaving the key out would claim the data is in OGC:CRS84.
            None | Some(Value::Null) => {
                column.insert("crs".into(), Value::Null);
            },
            // The default is left out, as it is the only string CRS we can write without PROJ.
            // An SRID that reads "OGC:CRS84" means whatever its producer says, so it is refused.
            Some(Value::String(crs))
                if crs == DEFAULT_CRS && matches!(crs_type, None | Some("authority_code")) => {},
            Some(Value::Object(projjson)) if is_crs84(projjson) => {},
            Some(crs @ Value::Object(_)) => {
                column.insert("crs".into(), crs.clone());
            },
            Some(crs) => {
                let of_type = crs_type
                    .map(|t| format!(" of type '{t}'"))
                    .unwrap_or_default();
                polars_bail!(
                    InvalidOperation: "cannot write column '{}' as GeoParquet: CRS {}{} must be a PROJJSON object, null or '{}'; \
                    convert it to PROJJSON, or pass your own '{}' key in the file metadata",
                    name, crs, of_type, DEFAULT_CRS, GEO_KEY
                )
            },
        }
        match ext_metadata.get("edges") {
            None => {},
            Some(edges) if edges == "planar" => {},
            Some(edges) if edges == "spherical" => {
                column.insert("edges".into(), edges.clone());
            },
            Some(edges) => polars_bail!(
                InvalidOperation: "cannot write column '{}' as GeoParquet: unsupported edges {}", name, edges
            ),
        }
        if let Some(epoch) = ext_metadata.get("epoch") {
            column.insert("epoch".into(), epoch.clone());
        }

        if version != "1.0.0"
            && let Some((box_name, covering)) =
                bbox_covering(schema, name, ext_metadata.get("crs"))?
        {
            if covering_columns.contains(&box_name) {
                polars_bail!(
                    InvalidOperation: "cannot write column '{}' as GeoParquet: its covering '{}' already covers another column",
                    name, box_name
                );
            }
            column.insert("covering".into(), covering);
            covering_columns.push(box_name);
        }
        // WKB has no statistics of its coordinates, so it gets no `bbox`.
        // Its covering's statistics can't stand in for them:
        // a box that crosses the antimeridian has `xmin > xmax`,
        // and the smallest `xmin` and largest `xmax` would leave it out.
        let is_native = encoding.list_depth.is_some();
        // A spherical edge can bulge beyond its vertices.
        let spherical = ext_metadata.get("edges").is_some_and(|e| e == "spherical");
        let axes = &["x", "y", "z"][..2 + has_z as usize];
        if let Some(statistics) = statistics
            && is_native
            && !spherical
            && let Some(bbox) = bbox_from_leaf_statistics(statistics, name, axes)
        {
            column.insert("bbox".into(), bbox.into());
        }

        primary_column.get_or_insert(name.as_str());
        columns.insert(name.to_string(), Value::Object(column));
    }

    let Some(primary_column) = primary_column else {
        return Ok(None);
    };
    let geo = json!({
        "version": version,
        "primary_column": primary_column,
        "columns": columns,
    });
    Ok(Some(KeyValue {
        key: GEO_KEY.to_string(),
        value: Some(geo.to_string()),
    }))
}

fn extension_metadata(name: &str, ext: &ExtensionType) -> PolarsResult<Map<String, Value>> {
    match ext.metadata.as_deref() {
        None => Ok(Map::new()),
        Some(md) => match serde_json::from_str(md) {
            Ok(Value::Object(md)) => Ok(md),
            _ => polars_bail!(
                InvalidOperation: "cannot write column '{}' as GeoParquet: invalid extension metadata", name
            ),
        },
    }
}

/// The geoarrow-rs naming scheme for the bbox column of a geometry column.
fn covering_name(geometry: &str) -> String {
    match geometry {
        "geometry" | "geography" => "bbox".into(),
        _ => format!("{geometry}_bbox"),
    }
}

/// Returns the name of the box column that covers `geometry`, and its `covering` metadata.
/// Returns `None` if there is no `geoarrow.box` column named after [`covering_name`].
fn bbox_covering(
    schema: &ArrowSchema,
    geometry: &str,
    geometry_crs: Option<&Value>,
) -> PolarsResult<Option<(String, Value)>> {
    let box_name = covering_name(geometry);
    let Some(ArrowDataType::Extension(ext)) = schema.get(box_name.as_str()).map(|f| &f.dtype)
    else {
        return Ok(None);
    };
    if ext.name != BOX_EXTENSION_NAME {
        return Ok(None);
    }

    let layout = box_layout(&ext.inner).ok_or_else(|| {
        polars_err!(
            InvalidOperation: "cannot write column '{}' as GeoParquet covering of '{}': \
            storage type {:?} must have Float64 fields xmin, ymin, [zmin,] xmax, ymax, [zmax] in that order",
            box_name, geometry, ext.inner
        )
    })?;

    let box_crs = extension_metadata(&box_name, ext)?.remove("crs");
    if let Some(crs) = box_crs.filter(|crs| !crs.is_null())
        && Some(&crs) != geometry_crs
    {
        polars_bail!(
            InvalidOperation: "cannot write column '{}' as GeoParquet covering of '{}': its CRS {} differs from the geometry's",
            box_name, geometry, crs
        );
    }

    let bbox: Map<String, Value> = layout
        .iter()
        .map(|field| (field.to_string(), json!([box_name, field])))
        .collect();
    Ok(Some((box_name, json!({ "bbox": bbox }))))
}

/// Returns `[mins..., maxs...]`, or `None` if the statistics can't tell.
fn bbox_from_leaf_statistics(
    statistics: &FileStatistics,
    column: &str,
    axes: &[&str],
) -> Option<Vec<f64>> {
    let mut mins = Vec::with_capacity(axes.len());
    let mut maxs = Vec::with_capacity(axes.len());
    for axis in axes {
        let (min, max) = leaf_min_max_over_row_groups(statistics, column, axis)?;
        mins.push(min);
        maxs.push(max);
    }
    mins.extend(maxs);
    // JSON can't hold infinities.
    mins.iter().all(|v| v.is_finite()).then_some(mins)
}

/// Returns `None` if the statistics are missing or encrypted, or the leaf only holds nulls and NaNs.
fn leaf_min_max_over_row_groups(
    statistics: &FileStatistics,
    column: &str,
    leaf: &str,
) -> Option<(f64, f64)> {
    let leaf_index = statistics.columns.iter().position(|c| {
        let path = &c.path_in_schema;
        path.first().is_some_and(|p| p.as_str() == column)
            && path.last().is_some_and(|p| p.as_str() == leaf)
    })?;
    let mut bounds: Option<(f64, f64)> = None;
    for row_group in statistics.row_groups {
        let chunk = &row_group.columns[leaf_index];
        // Hidden statistics must not leak through the `bbox`.
        if chunk.crypto_metadata.is_some() {
            return None;
        }
        let stats = chunk.meta_data.as_ref()?.statistics.as_ref()?;
        // Row groups of only nulls and NaNs have no minimum and maximum.
        let (Some(min), Some(max)) = (&stats.min_value, &stats.max_value) else {
            continue;
        };
        let (min, max) = (decode::<f64>(min), decode::<f64>(max));
        bounds = Some(match bounds {
            None => (min, max),
            Some((lo, hi)) => (lo.min(min), hi.max(max)),
        });
    }
    bounds
}

/// Whether a PROJJSON CRS is OGC:CRS84, by its `id` as the GeoParquet spec suggests.
fn is_crs84(projjson: &Map<String, Value>) -> bool {
    projjson
        .get("id")
        .is_some_and(|id| id["authority"] == "OGC" && id["code"] == "CRS84")
}
