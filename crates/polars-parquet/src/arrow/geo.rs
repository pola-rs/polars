//! Conversion between GeoParquet `geo` file metadata and GeoArrow extension types.
//!
//! # Bounding box coverings
//!
//! GeoParquet 1.1 lets a geometry column have a bounding box per-row in another column.
//! This column is stored simply as a struct. We are kind enough to write the `geoarrow.box` datatype,
//! but that is not required for the GeoParquet spec of course.
//! Which bbox links to which geometry column lives only in the metadata (they can have any name).
//! Upon reading, that column becomes a `geoarrow.box` with the extension metadata of its geometry,
//! if its fields match one of the [`BOX_LAYOUTS`].
//!
//! On the GeoArrow side though, there is no 'relating'.
//! We still have to somehow indicate an intention of this.
//! In GeoPandas and geoarrow-rs, this is done by naming.
//! We match per-column bounding boxes based on the names, which is the same way GeoPandas/geoarrow-rs does it.
//! E.g. a column called `"foo"` will have a bounding box in `"foo_bbox"`.
//! GeoPandas and `geoarrow-rs` also allow a simplified form if you only have a single column called `"geometry"` or `"geography"`.
//! For those, you can use `"bbox"` instead of a full `"geometry_bbox"`.
//! When both `geometry` and `geography` exist and are geometry columns, `bbox` covers neither.
//! Writing any other box is an error, as a plain struct would silently drop what it means.
use polars_arrow::datatypes::{ArrowDataType, ArrowSchema, ExtensionType, Field, Metadata};
use polars_error::{PolarsResult, polars_bail, polars_err, polars_warn};
use polars_parquet_format::RowGroup;
use polars_utils::pl_str::PlSmallStr;
use serde_json::{Map, Value, json};

use crate::parquet::metadata::{ColumnDescriptor, KeyValue};
use crate::parquet::types::decode;

pub const GEO_KEY: &str = "geo";

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct GeoParquetVersion {
    major: u32,
    minor: u32,
}

impl GeoParquetVersion {
    const V1_0: Self = Self { major: 1, minor: 0 };
    const V1_1: Self = Self { major: 1, minor: 1 };

    /// Parses `major.minor[.patch]`.
    fn parse(version: &str) -> Option<Self> {
        let parts: Vec<u32> = version
            .split('.')
            .map(|part| part.parse().ok())
            .collect::<Option<_>>()?;
        match parts[..] {
            [major, minor] | [major, minor, _] => Some(Self { major, minor }),
            _ => None,
        }
    }

    fn has_native_encodings(self) -> bool {
        self >= Self::V1_1
    }

    fn has_coverings(self) -> bool {
        self >= Self::V1_1
    }
}

impl std::fmt::Display for GeoParquetVersion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}.{}.0", self.major, self.minor)
    }
}

/// The geoparquet spec states that something that has no `crs` key should be considered "OGC: CRS84".
const DEFAULT_CRS: &str = "OGC:CRS84";

pub(crate) const BOX_EXTENSION_NAME: &str = "geoarrow.box";

// TODO: accept Float32 bbox coverings too (requires float32 -> Float64 widening).

/// The canonical field layouts of a `geoarrow.box`.
/// GeoParquet also allows a covering's fields as FLOAT, which cannot be read as a box yet.
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

/// Returns the fields of the coordinate struct inside `dtype`, `None` for WKB.
fn coordinate_fields<'a>(encoding: &Encoding, mut dtype: &'a ArrowDataType) -> Option<&'a [Field]> {
    for _ in 0..encoding.list_depth? {
        dtype = match dtype {
            ArrowDataType::List(f) | ArrowDataType::LargeList(f) => &f.dtype,
            _ => return None,
        };
    }
    match dtype {
        ArrowDataType::Struct(fields) => Some(fields),
        _ => None,
    }
}

/// Returns `Some(has_z)` if `dtype` is a valid storage type for `encoding`.
fn check_storage(encoding: &Encoding, dtype: &ArrowDataType) -> Option<bool> {
    if encoding.list_depth.is_none() {
        return matches!(
            dtype,
            ArrowDataType::Binary | ArrowDataType::LargeBinary | ArrowDataType::BinaryView
        )
        .then_some(false);
    }
    let fields = coordinate_fields(encoding, dtype)?;
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
    apply_geo_metadata_inner(metadata, schema);
}

fn apply_geo_metadata_inner(metadata: &Metadata, schema: &mut ArrowSchema) {
    let Some(geo) = metadata.get(GEO_KEY) else {
        return;
    };
    let Ok(geo) = serde_json::from_str::<Value>(geo) else {
        polars_warn!("ignoring invalid GeoParquet metadata");
        return;
    };
    let version = geo["version"].as_str().unwrap_or_default();
    if GeoParquetVersion::parse(version).is_none_or(|v| v.major != 1) {
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
    let Some(canonical_storage) = canonical_box_storage(storage, layout) else {
        polars_warn!(
            "ignoring GeoParquet covering of '{}': type {:?} of column '{}' is not a struct of Float64 fields xmin, ymin, [zmin,] xmax, ymax, [zmax]",
            geometry,
            storage,
            box_name
        );
        return;
    };

    field.dtype = ArrowDataType::Extension(Box::new(ExtensionType {
        name: BOX_EXTENSION_NAME.into(),
        inner: canonical_storage,
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

/// Returns `dtype` with its fields reordered into canonical `layout` order if it is a
/// `Struct` of `Float64` fields matching `layout` in any order.
fn canonical_box_storage(dtype: &ArrowDataType, layout: &[&str]) -> Option<ArrowDataType> {
    let ArrowDataType::Struct(fields) = dtype else {
        return None;
    };
    if fields.len() != layout.len() {
        return None;
    }
    let reordered = layout
        .iter()
        .map(|name| {
            fields
                .iter()
                .find(|f| f.name == *name && f.dtype == ArrowDataType::Float64)
                .cloned()
        })
        .collect::<Option<Vec<_>>>()?;
    Some(ArrowDataType::Struct(reordered))
}

/// Returns the layout of `dtype` if it is a valid `geoarrow.box` storage type.
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
pub(crate) struct FileStatistics<'a> {
    pub(crate) columns: &'a [ColumnDescriptor],
    pub(crate) row_groups: &'a [RowGroup],
}

/// Checks that `schema` can be written as GeoParquet, so that a write fails before any file is opened.
pub fn validate_geo_schema(schema: &ArrowSchema) -> PolarsResult<()> {
    geo_key_value(schema, None).map(|_| ())
}

/// Builds the GeoParquet `geo` metadata for the GeoArrow columns in `schema` if
/// `POLARS_GEOPARQUET_WRITE` is set to `1.0` or `1.1`.
/// The `bbox` of each geometry column is only filled in if `statistics` are given.
/// A `geoarrow.box` column that covers no geometry column is an error.
pub(crate) fn geo_key_value(
    schema: &ArrowSchema,
    statistics: Option<&FileStatistics<'_>>,
) -> PolarsResult<Option<KeyValue>> {
    let version = match std::env::var("POLARS_GEOPARQUET_WRITE").as_deref() {
        Err(_) | Ok("") => return Ok(None),
        Ok("1.0") => GeoParquetVersion::V1_0,
        Ok("1.1") => GeoParquetVersion::V1_1,
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
        if !is_geometry_extension(ext) {
            continue;
        }
        let name = &field.name;
        let encoding = ENCODINGS
            .iter()
            .find(|e| {
                ext.name == e.geoarrow && (e.list_depth.is_none() || version.has_native_encodings())
            })
            .ok_or_else(|| {
                polars_err!(
                    InvalidOperation: "cannot write column '{}' of type '{}' as GeoParquet {}",
                    name, ext.name, version
                )
            })?;
        let has_z = check_storage(encoding, &ext.inner).ok_or_else(|| {
            let has_m = coordinate_fields(encoding, &ext.inner)
                .is_some_and(|fields| fields.iter().any(|f| f.name == "m"));
            if has_m {
                return polars_err!(
                    InvalidOperation: "cannot write column '{}' as GeoParquet {}: it cannot store M values",
                    name, version
                );
            }
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
        let crs = geoparquet_crs(name, &ext_metadata)?;
        if let Some(crs) = &crs {
            column.insert("crs".into(), crs.clone());
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

        if version.has_coverings()
            && let Some((box_name, covering)) =
                bbox_covering_col_of_geometry(schema, name, crs.as_ref())?
        {
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
    check_boxes_are_coverings(schema, version, &covering_columns)?;

    Ok(primary_column.map(|primary_column| {
        let geo = json!({
            "version": version.to_string(),
            "primary_column": primary_column,
            "columns": columns,
        });
        KeyValue {
            key: GEO_KEY.to_string(),
            value: Some(geo.to_string()),
        }
    }))
}

fn check_boxes_are_coverings(
    schema: &ArrowSchema,
    version: GeoParquetVersion,
    covering_columns: &[String],
) -> PolarsResult<()> {
    for field in schema.iter_values() {
        let ArrowDataType::Extension(ext) = &field.dtype else {
            continue;
        };
        if ext.name != BOX_EXTENSION_NAME
            || covering_columns.iter().any(|c| c == field.name.as_str())
        {
            continue;
        }
        // A box that can't be a covering under any name should not be told to rename.
        let reason = if !version.has_coverings() {
            "GeoParquet 1.0 has no coverings; write GeoParquet 1.1 instead"
        } else if box_layout(&ext.inner).is_none() {
            "a covering's fields must be Float64 xmin, ymin, [zmin,] xmax, ymax, [zmax] in that order"
        } else if field.name == "bbox" && bbox_is_ambiguous(schema) {
            "it could cover both 'geometry' and 'geography'; rename one of them"
        } else {
            "no geometry column has it as its covering name; \
            name it '<geometry>_bbox', or 'bbox' for a column named 'geometry' or 'geography'"
        };
        // Writing it as a plain struct would silently drop what it means, so the user has to opt in.
        polars_bail!(
            InvalidOperation: "cannot write box column '{}' as GeoParquet: {}; \
            to write it as a plain struct, convert it with `.ext.storage()`",
            field.name, reason
        );
    }
    Ok(())
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

/// Returns the GeoParquet `crs` of column `name`.
/// None -> the key is left out meaning OGC:CRS84.
fn geoparquet_crs(name: &str, ext_metadata: &Map<String, Value>) -> PolarsResult<Option<Value>> {
    let crs_type = ext_metadata.get("crs_type").and_then(Value::as_str);
    match ext_metadata.get("crs") {
        // Geoarrow designates an 'unknown' CRS by leaving `crs` out,
        // while GeoParquet explicitly writes it as `Null`.
        // Leaving the key out would claim the data is in OGC:CRS84.
        None | Some(Value::Null) => Ok(Some(Value::Null)),
        // The default is left out, as it is the only string CRS we can write without PROJ.
        // An SRID that reads "OGC:CRS84" means whatever its producer says, so it is refused.
        Some(Value::String(crs))
            if crs == DEFAULT_CRS && matches!(crs_type, None | Some("authority_code")) =>
        {
            Ok(None)
        },
        Some(Value::Object(projjson)) if is_crs84(projjson) => Ok(None),
        Some(crs @ Value::Object(_)) => Ok(Some(crs.clone())),
        Some(crs) => {
            let of_type = crs_type
                .map(|t| format!(" of type '{t}'"))
                .unwrap_or_default();
            polars_bail!(
                InvalidOperation: "cannot write column '{}' as GeoParquet: CRS {}{} must be a PROJJSON object, null, or the authority code '{}'; \
                convert it to PROJJSON, or pass your own '{}' key in the file metadata",
                name, crs, of_type, DEFAULT_CRS, GEO_KEY
            )
        },
    }
}

fn is_geometry_extension(ext: &ExtensionType) -> bool {
    // GeoParquet has no box encoding, so a box is written as a plain struct column.
    ext.name.starts_with("geoarrow.") && ext.name != BOX_EXTENSION_NAME
}

fn is_geometry_column(schema: &ArrowSchema, name: &str) -> bool {
    matches!(
        schema.get(name).map(|f| &f.dtype),
        Some(ArrowDataType::Extension(ext)) if is_geometry_extension(ext)
    )
}

fn bbox_is_ambiguous(schema: &ArrowSchema) -> bool {
    is_geometry_column(schema, "geometry") && is_geometry_column(schema, "geography")
}

fn covering_name(geometry: &str) -> String {
    match geometry {
        "geometry" | "geography" => "bbox".into(),
        _ => format!("{geometry}_bbox"),
    }
}

/// Returns the name of the box column that covers `geometry`, and its `covering` metadata.
/// Returns `None` if there is no `geoarrow.box` column named after [`covering_name`],
/// or if its fields can't form a covering.
fn bbox_covering_col_of_geometry(
    schema: &ArrowSchema,
    geometry: &str,
    geometry_crs: Option<&Value>,
) -> PolarsResult<Option<(String, Value)>> {
    let box_name = covering_name(geometry);
    // [`check_boxes_are_coverings`] errors on a box that covers nothing.
    if box_name == "bbox" && bbox_is_ambiguous(schema) {
        return Ok(None);
    }
    let Some(ArrowDataType::Extension(ext)) = schema.get(box_name.as_str()).map(|f| &f.dtype)
    else {
        return Ok(None);
    };
    if ext.name != BOX_EXTENSION_NAME {
        return Ok(None);
    }

    let Some(layout) = box_layout(&ext.inner) else {
        return Ok(None);
    };

    // A box without a CRS takes its geometry's
    let box_metadata = extension_metadata(&box_name, ext)?;
    if let Some(crs) = box_metadata.get("crs").filter(|crs| !crs.is_null())
        && geoparquet_crs(&box_name, &box_metadata)?.as_ref() != geometry_crs
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

#[cfg(test)]
mod tests {
    use std::io::Cursor;
    use std::sync::Arc;

    use polars_arrow::array::{BinaryViewArray, Float64Array, StructArray};
    use polars_arrow::bitmap::Bitmap;
    use polars_arrow::record_batch::RecordBatch;
    use polars_buffer::Buffer;

    use super::*;
    use crate::arrow::read::schema::{
        SchemaInferenceOptions, parquet_to_arrow_schema_with_options,
    };
    use crate::arrow::read::{BasicDecompressor, PageReader, column_iter_to_arrays, read_metadata};
    use crate::arrow::write::{
        CompressionOptions, Encoding as ParquetEncoding, FileWriter, StatisticsOptions, Version,
        WriteOptions, row_group_iter,
    };

    fn make_geo_metadata(covering: Value) -> Metadata {
        let geo = json!({
            "version": "1.1.0",
            "primary_column": "geometry",
            "columns": {
                "geometry": {
                    "encoding": "WKB",
                    "geometry_types": [],
                    "covering": covering,
                }
            }
        });
        Metadata::from([(PlSmallStr::from_static(GEO_KEY), geo.to_string().into())])
    }

    fn base_schema(bbox_fields: Vec<Field>) -> ArrowSchema {
        ArrowSchema::from_iter([
            Field::new("geometry".into(), ArrowDataType::BinaryView, true),
            Field::new("bbox".into(), ArrowDataType::Struct(bbox_fields), true),
        ])
    }

    #[test]
    fn test_out_of_order_bbox_schema_canonicalized() {
        for (order, expected) in [
            (
                vec!["xmax", "xmin", "ymax", "ymin"],
                vec!["xmin", "ymin", "xmax", "ymax"],
            ),
            (
                vec!["zmax", "ymax", "xmax", "zmin", "ymin", "xmin"],
                vec!["xmin", "ymin", "zmin", "xmax", "ymax", "zmax"],
            ),
        ] {
            let fields = order
                .iter()
                .map(|n| Field::new((*n).into(), ArrowDataType::Float64, false))
                .collect();
            let covering = json!({
                "bbox": order.iter().map(|n| ((*n).to_string(), json!(["bbox", n]))).collect::<Map<_, _>>()
            });
            let metadata = make_geo_metadata(covering);
            let mut schema = base_schema(fields);
            apply_geo_metadata_inner(&metadata, &mut schema);

            let ArrowDataType::Extension(ext) = &schema.get("bbox").unwrap().dtype else {
                panic!("expected bbox to become geoarrow.box extension");
            };
            assert_eq!(ext.name, BOX_EXTENSION_NAME);
            let ArrowDataType::Struct(out_fields) = &ext.inner else {
                panic!("expected Struct inner type");
            };
            let names: Vec<_> = out_fields.iter().map(|f| f.name.as_str()).collect();
            assert_eq!(names, expected);
        }
    }

    #[test]
    fn test_invalid_bbox_schemas_ignored() {
        // Float32 field should not be promoted.
        let mut schema = base_schema(vec![
            Field::new("xmax".into(), ArrowDataType::Float32, false),
            Field::new("xmin".into(), ArrowDataType::Float64, false),
            Field::new("ymax".into(), ArrowDataType::Float64, false),
            Field::new("ymin".into(), ArrowDataType::Float64, false),
        ]);
        let covering = json!({
            "bbox": {
                "xmin": ["bbox", "xmin"],
                "ymin": ["bbox", "ymin"],
                "xmax": ["bbox", "xmax"],
                "ymax": ["bbox", "ymax"],
            }
        });
        apply_geo_metadata_inner(&make_geo_metadata(covering.clone()), &mut schema);
        assert!(matches!(
            schema.get("bbox").unwrap().dtype,
            ArrowDataType::Struct(_)
        ));

        // Extra field should not be promoted.
        let mut schema_extra = base_schema(vec![
            Field::new("xmax".into(), ArrowDataType::Float64, false),
            Field::new("xmin".into(), ArrowDataType::Float64, false),
            Field::new("ymax".into(), ArrowDataType::Float64, false),
            Field::new("ymin".into(), ArrowDataType::Float64, false),
            Field::new("extra".into(), ArrowDataType::Float64, false),
        ]);
        apply_geo_metadata_inner(&make_geo_metadata(covering.clone()), &mut schema_extra);
        assert!(matches!(
            schema_extra.get("bbox").unwrap().dtype,
            ArrowDataType::Struct(_)
        ));

        // Duplicate field name should not be promoted.
        let mut schema_dup = base_schema(vec![
            Field::new("xmax".into(), ArrowDataType::Float64, false),
            Field::new("xmax".into(), ArrowDataType::Float64, false),
            Field::new("ymax".into(), ArrowDataType::Float64, false),
            Field::new("ymin".into(), ArrowDataType::Float64, false),
        ]);
        apply_geo_metadata_inner(&make_geo_metadata(covering), &mut schema_dup);
        assert!(matches!(
            schema_dup.get("bbox").unwrap().dtype,
            ArrowDataType::Struct(_)
        ));
    }

    #[test]
    fn test_read_out_of_order_bbox_roundtrip_decode() {
        // Store fields in GeoParquet `example.parquet` order: xmax, xmin, ymax, ymin.
        let disk_order = ["xmax", "xmin", "ymax", "ymin"];
        let bbox_fields: Vec<Field> = disk_order
            .iter()
            .map(|n| Field::new((*n).into(), ArrowDataType::Float64, true))
            .collect();
        let write_schema = base_schema(bbox_fields.clone());

        let make_batch = |offset: f64, with_null: bool| {
            let geom = BinaryViewArray::from_slice([Some(b"pt1".as_slice()), Some(b"pt2")]);
            let xmax = Float64Array::from_slice([10.0 + offset, 20.0 + offset]).boxed();
            let xmin = Float64Array::from_slice([1.0 + offset, 2.0 + offset]).boxed();
            let ymax = Float64Array::from_slice([30.0 + offset, 40.0 + offset]).boxed();
            let ymin = Float64Array::from_slice([3.0 + offset, 4.0 + offset]).boxed();
            let validity = with_null.then(|| Bitmap::from([true, false]));
            let bbox = StructArray::new(
                ArrowDataType::Struct(bbox_fields.clone()),
                2,
                vec![xmax, xmin, ymax, ymin],
                validity,
            );
            RecordBatch::try_new(
                2,
                Arc::new(write_schema.clone()),
                vec![geom.boxed(), bbox.boxed()],
            )
            .unwrap()
        };

        let batches = [make_batch(0.0, true), make_batch(100.0, false)];

        let options = WriteOptions {
            statistics: StatisticsOptions::default(),
            version: Version::V2,
            compression: CompressionOptions::Uncompressed,
            data_page_size: None,
        };
        let encodings: Buffer<Vec<ParquetEncoding>> = write_schema
            .iter_values()
            .map(|f| match f.dtype() {
                ArrowDataType::Struct(children) => {
                    vec![ParquetEncoding::Plain; children.len()]
                },
                _ => vec![ParquetEncoding::Plain],
            })
            .collect();

        let mut buf = Vec::new();
        let mut writer =
            FileWriter::try_new(&mut buf, write_schema.clone(), options, None).unwrap();
        let fields = writer.parquet_schema().fields().to_vec();
        for batch in batches {
            let num_rows = batch.height() as u64;
            let row_group = row_group_iter(batch, encodings.clone(), fields.clone(), options);
            writer.write(num_rows, row_group).unwrap();
        }
        let covering = json!({
            "bbox": {
                "xmax": ["bbox", "xmax"],
                "xmin": ["bbox", "xmin"],
                "ymax": ["bbox", "ymax"],
                "ymin": ["bbox", "ymin"],
            }
        });
        let geo_kv = KeyValue {
            key: GEO_KEY.to_string(),
            value: Some(
                json!({
                    "version": "1.1.0",
                    "primary_column": "geometry",
                    "columns": {
                        "geometry": {
                            "encoding": "WKB",
                            "geometry_types": [],
                            "covering": covering,
                        }
                    }
                })
                .to_string(),
            ),
        };
        writer.end(Some(vec![geo_kv])).unwrap();

        let buf = Buffer::from_vec(buf);
        let mut reader = Cursor::new(buf.as_slice());
        let file_md = read_metadata(&mut reader).unwrap();
        let kv_md: Metadata = file_md
            .key_value_metadata()
            .iter()
            .flatten()
            .filter_map(|kv| {
                kv.value
                    .as_ref()
                    .map(|v| (kv.key.as_str().into(), v.as_str().into()))
            })
            .collect();
        let mut arrow_schema = parquet_to_arrow_schema_with_options(
            file_md.schema().fields(),
            &SchemaInferenceOptions::default(),
            None,
        )
        .unwrap();
        apply_geo_metadata_inner(&kv_md, &mut arrow_schema);

        let bbox_field = arrow_schema.get("bbox").unwrap().clone();
        let ArrowDataType::Extension(ext) = bbox_field.dtype() else {
            panic!("expected Extension");
        };
        assert_eq!(ext.name, BOX_EXTENSION_NAME);
        let ArrowDataType::Struct(canonical_fields) = &ext.inner else {
            panic!("expected Struct");
        };
        let canonical_names: Vec<_> = canonical_fields.iter().map(|f| f.name.as_str()).collect();
        assert_eq!(canonical_names, vec!["xmin", "ymin", "xmax", "ymax"]);

        // Decode both row groups using the canonical schema field.
        for (rg_idx, expected_offset) in [(0, 0.0), (1, 100.0)] {
            let rg = &file_md.row_groups[rg_idx];
            let leaf_types: Vec<_> = rg
                .columns_under_root_iter("bbox")
                .unwrap()
                .map(|c| &c.descriptor().descriptor.primitive_type)
                .collect();
            let decompressors: Vec<_> = rg
                .columns_under_root_iter("bbox")
                .unwrap()
                .map(|col_md| {
                    let range = col_md.byte_range().unwrap();
                    let chunk = buf.clone().sliced(range.start as usize..range.end as usize);
                    let pages =
                        PageReader::new(Cursor::new(chunk), col_md, vec![], usize::MAX).unwrap();
                    BasicDecompressor::new(pages, vec![])
                })
                .collect();
            let (arrays, _) =
                column_iter_to_arrays(decompressors, leaf_types, bbox_field.clone(), None).unwrap();
            assert_eq!(arrays.len(), 1);
            assert_eq!(arrays[0].dtype(), bbox_field.dtype());
            let struct_arr = arrays[0].as_any().downcast_ref::<StructArray>().unwrap();
            assert_eq!(struct_arr.len(), 2);
            if rg_idx == 0 {
                assert_eq!(struct_arr.validity(), Some(&Bitmap::from([true, false])));
            } else {
                assert_eq!(struct_arr.validity(), None);
            }
            let vals = |i: usize| {
                let arr = struct_arr.values()[i]
                    .as_any()
                    .downcast_ref::<Float64Array>()
                    .unwrap();
                if rg_idx == 0 {
                    // Row 1 has null struct validity; row 0 holds the valid bbox.
                    vec![arr.value(0)]
                } else {
                    arr.values().as_slice().to_vec()
                }
            };
            // Canonical order: [xmin, ymin, xmax, ymax]
            if rg_idx == 0 {
                assert_eq!(vals(0), vec![1.0]);
                assert_eq!(vals(1), vec![3.0]);
                assert_eq!(vals(2), vec![10.0]);
                assert_eq!(vals(3), vec![30.0]);
            } else {
                assert_eq!(vals(0), vec![1.0 + expected_offset, 2.0 + expected_offset]);
                assert_eq!(vals(1), vec![3.0 + expected_offset, 4.0 + expected_offset]);
                assert_eq!(
                    vals(2),
                    vec![10.0 + expected_offset, 20.0 + expected_offset]
                );
                assert_eq!(
                    vals(3),
                    vec![30.0 + expected_offset, 40.0 + expected_offset]
                );
            }
        }
    }
}
