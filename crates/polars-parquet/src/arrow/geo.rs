//! Conversion between GeoParquet `geo` file metadata and GeoArrow extension types.
use polars_arrow::datatypes::{ArrowDataType, ArrowSchema, ExtensionType, Field, Metadata};
use polars_error::{PolarsResult, polars_bail, polars_err, polars_warn};
use serde_json::{Map, Value, json};

use crate::parquet::metadata::KeyValue;

pub const GEO_KEY: &str = "geo";

/// The geoparquet spec states that something that has no `crs` key should be considered "OGC: CRS84".
const DEFAULT_CRS: &str = "OGC:CRS84";

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

        field.dtype = ArrowDataType::Extension(Box::new(ExtensionType {
            name: encoding.geoarrow.into(),
            inner: storage.clone(),
            metadata: (!ext_metadata.is_empty())
                .then(|| Value::Object(ext_metadata).to_string().into()),
        }));
    }
}

/// Called once per write, before any file is opened.
pub fn validate_geo_schema(schema: &ArrowSchema) -> PolarsResult<()> {
    geo_key_value(schema)?;
    Ok(())
}

/// Builds the GeoParquet `geo` metadata for the GeoArrow columns in `schema` if
/// `POLARS_GEOPARQUET_WRITE` is set to `1.0` or `1.1`.
pub(crate) fn geo_key_value(schema: &ArrowSchema) -> PolarsResult<Option<KeyValue>> {
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
    for field in schema.iter_values() {
        let ArrowDataType::Extension(ext) = &field.dtype else {
            continue;
        };
        if !ext.name.starts_with("geoarrow.") {
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
        if let Some(crs) = geoparquet_crs(name, &ext_metadata)? {
            column.insert("crs".into(), crs);
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

/// Whether a PROJJSON CRS is OGC:CRS84, by its `id` as the GeoParquet spec suggests.
fn is_crs84(projjson: &Map<String, Value>) -> bool {
    projjson
        .get("id")
        .is_some_and(|id| id["authority"] == "OGC" && id["code"] == "CRS84")
}
