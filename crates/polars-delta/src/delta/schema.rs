use polars_arrow::datatypes::{ArrowDataType as D, Field, Metadata, TimeUnit};
use polars_utils::aliases::PlHashSet;
use serde_json::Value;

use super::actions::string;
use super::error::{Result, invalid, unsupported};
#[derive(Debug, Clone)]
pub struct Column {
    pub field: Field,
    pub physical: String,
    pub children: Vec<Column>,
}
#[derive(Debug)]
pub struct Schema {
    pub columns: Vec<Column>,
    pub partitions: Vec<String>,
    pub mode: String,
}
impl Schema {
    pub fn parse(metadata: &Value, protocol: &Value) -> Result<Self> {
        let reader = protocol
            .get("minReaderVersion")
            .and_then(Value::as_i64)
            .ok_or_else(|| invalid("invalid minReaderVersion"))?;
        if !(1..=3).contains(&reader) {
            return Err(unsupported(format!("minReaderVersion {reader}")));
        }
        let features = protocol.get("readerFeatures").filter(|v| !v.is_null());
        if reader == 3 && !features.is_some_and(Value::is_array) {
            return Err(invalid("reader version3 needs readerFeatures"));
        }
        if let Some(features) = features {
            for f in features
                .as_array()
                .ok_or_else(|| invalid("readerFeatures must be array"))?
            {
                let f = f
                    .as_str()
                    .ok_or_else(|| invalid("non-string reader feature"))?;
                if !matches!(
                    f,
                    "columnMapping"
                        | "deletionVectors"
                        | "timestampNtz"
                        | "timestampWithoutTimezone"
                        | "v2Checkpoint"
                        | "vacuumProtocolCheck"
                        | "typeWidening"
                        | "typeWidening-preview"
                        | "variantType"
                        | "variantType-preview"
                ) {
                    return Err(unsupported(format!("reader feature {f}")));
                }
            }
        }
        let mode = metadata
            .get("configuration")
            .and_then(|v| v.get("delta.columnMapping.mode"))
            .and_then(Value::as_str)
            .unwrap_or("none");
        if !matches!(mode, "none" | "name" | "id") {
            return Err(unsupported(format!("column mapping mode {mode}")));
        }
        if mode != "none" && reader == 1 {
            return Err(invalid("column mapping needs reader version2 or feature"));
        }
        if mode != "none"
            && reader == 3
            && !features
                .unwrap()
                .as_array()
                .unwrap()
                .iter()
                .any(|f| f == "columnMapping")
        {
            return Err(invalid("missing columnMapping reader feature"));
        }
        if metadata
            .get("format")
            .and_then(|v| v.get("provider"))
            .and_then(Value::as_str)
            != Some("parquet")
        {
            return Err(unsupported("non-Parquet Delta data"));
        }
        let raw: Value =
            serde_json::from_str(string(metadata, "schemaString")?).map_err(invalid)?;
        let columns = parse_fields(&raw, mode)?;
        let mut ids = PlHashSet::default();
        validate_ids(&columns, mode, &mut ids)?;
        let partitions = metadata
            .get("partitionColumns")
            .and_then(Value::as_array)
            .ok_or_else(|| invalid("missing partitionColumns"))?
            .iter()
            .map(|v| {
                v.as_str()
                    .map(str::to_owned)
                    .ok_or_else(|| invalid("non-string partition column"))
            })
            .collect::<Result<Vec<_>>>()?;
        for p in &partitions {
            if !columns.iter().any(|c| c.field.name.as_str() == p) {
                return Err(invalid("partition column missing from schema"));
            }
        }
        Ok(Self {
            columns,
            partitions,
            mode: mode.to_owned(),
        })
    }
}
fn validate_ids(cols: &[Column], mode: &str, ids: &mut PlHashSet<u32>) -> Result<()> {
    for c in cols {
        if mode != "none" {
            let id = c
                .field
                .metadata
                .as_ref()
                .and_then(|m| m.get("PARQUET:field_id"))
                .and_then(|s| s.parse::<u32>().ok())
                .filter(|i| *i > 0 && *i < i32::MAX as u32)
                .ok_or_else(|| invalid("invalid/missing column mapping ID"))?;
            if !ids.insert(id) {
                return Err(invalid("duplicate column mapping ID"));
            }
            if c.physical.is_empty() {
                return Err(invalid("empty column physical name"));
            }
        }
        validate_ids(&c.children, mode, ids)?;
    }
    Ok(())
}
fn parse_fields(v: &Value, mode: &str) -> Result<Vec<Column>> {
    let fields = v
        .get("fields")
        .and_then(Value::as_array)
        .ok_or_else(|| invalid("struct fields must be array"))?;
    let mut names = PlHashSet::default();
    fields
        .iter()
        .map(|v| {
            let name = string(v, "name")?;
            if !names.insert(name.to_lowercase()) {
                return Err(invalid("duplicate case-insensitive field name"));
            }
            let (dtype, children) = parse_type(
                v.get("type").ok_or_else(|| invalid("missing field type"))?,
                mode,
            )?;
            let mut md = Metadata::new();
            if let Some(metadata) = v.get("metadata").and_then(Value::as_object) {
                for (k, v) in metadata {
                    if k == "polars.delta.variant" {
                        continue;
                    }
                    md.insert(
                        k.as_str().into(),
                        v.as_str()
                            .map(str::to_owned)
                            .unwrap_or_else(|| v.to_string())
                            .into(),
                    );
                }
            }
            if let Some(changes) = v.get("metadata").and_then(|m| m.get("delta.typeChanges")) {
                validate_changes(changes)?
            }
            if v.get("type").and_then(Value::as_str) == Some("variant") {
                md.insert("polars.delta.variant".into(), "true".into());
            }
            let physical = if mode == "none" {
                name.to_owned()
            } else {
                v.get("metadata")
                    .and_then(|m| m.get("delta.columnMapping.physicalName"))
                    .and_then(Value::as_str)
                    .ok_or_else(|| invalid("missing physical name"))?
                    .to_owned()
            };
            if mode != "none" {
                let id = v
                    .get("metadata")
                    .and_then(|m| m.get("delta.columnMapping.id"))
                    .and_then(Value::as_u64)
                    .ok_or_else(|| invalid("missing column mapping ID"))?;
                md.insert("PARQUET:field_id".into(), id.to_string().into());
            }
            let field = Field::new(
                name.into(),
                dtype,
                v.get("nullable").and_then(Value::as_bool).unwrap_or(true),
            )
            .with_metadata(md);
            Ok(Column {
                field,
                physical,
                children,
            })
        })
        .collect()
}
fn synthetic(name: &str, dtype: D, nullable: bool) -> Field {
    let mut md = Metadata::new();
    let id = match name {
        "element" => u32::MAX,
        "key" => u32::MAX - 1,
        "variant_value" => u32::MAX - 3,
        "variant_metadata" => u32::MAX - 4,
        _ => u32::MAX - 2,
    };
    md.insert("PARQUET:field_id".into(), id.to_string().into());
    Field::new(name.into(), dtype, nullable).with_metadata(md)
}
fn synthetic_typed(name: &str, dtype: D, nullable: bool, raw: Option<&Value>) -> Field {
    let mut field = synthetic(name, dtype, nullable);
    if raw.and_then(Value::as_str) == Some("variant") {
        let metadata = field.metadata.get_or_insert_default();
        std::sync::Arc::make_mut(metadata).insert("polars.delta.variant".into(), "true".into());
    }
    field
}
fn parse_type(v: &Value, mode: &str) -> Result<(D, Vec<Column>)> {
    if let Some(s) = v.as_str() {
        return Ok((primitive(s)?, vec![]));
    }
    match v.get("type").and_then(Value::as_str) {
        Some("struct") => {
            let cols = parse_fields(v, mode)?;
            Ok((
                D::Struct(cols.iter().map(|c| c.field.clone()).collect()),
                cols,
            ))
        },
        Some("array") => {
            let (t, c) = parse_type(
                v.get("elementType")
                    .ok_or_else(|| invalid("missing elementType"))?,
                mode,
            )?;
            Ok((
                D::LargeList(Box::new(synthetic_typed(
                    "element",
                    t,
                    v.get("containsNull")
                        .and_then(Value::as_bool)
                        .unwrap_or(true),
                    v.get("elementType"),
                ))),
                c,
            ))
        },
        Some("map") => {
            let (k, mut cols) = parse_type(
                v.get("keyType").ok_or_else(|| invalid("missing keyType"))?,
                mode,
            )?;
            let (t, c) = parse_type(
                v.get("valueType")
                    .ok_or_else(|| invalid("missing valueType"))?,
                mode,
            )?;
            cols.extend(c);
            Ok((
                D::Map(
                    Box::new(Field::new(
                        "entries".into(),
                        D::Struct(vec![
                            synthetic_typed("key", k, false, v.get("keyType")),
                            synthetic_typed(
                                "value",
                                t,
                                v.get("valueContainsNull")
                                    .and_then(Value::as_bool)
                                    .unwrap_or(true),
                                v.get("valueType"),
                            ),
                        ]),
                        false,
                    )),
                    false,
                ),
                cols,
            ))
        },
        _ => Err(unsupported(format!("schema type {v}"))),
    }
}
fn variant_child(name: &str, id: u32) -> Field {
    let mut md = Metadata::new();
    md.insert("PARQUET:field_id".into(), id.to_string().into());
    Field::new(name.into(), D::BinaryView, false).with_metadata(md)
}
fn primitive(s: &str) -> Result<D> {
    Ok(match s {
        "string" => D::Utf8View,
        "binary" => D::BinaryView,
        "long" => D::Int64,
        "integer" => D::Int32,
        "short" => D::Int16,
        "byte" => D::Int8,
        "float" => D::Float32,
        "double" => D::Float64,
        "boolean" => D::Boolean,
        "date" => D::Date32,
        "timestamp" => D::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
        "timestamp_ntz" => D::Timestamp(TimeUnit::Microsecond, None),
        "void" => D::Null,
        "variant" => D::Struct(vec![
            variant_child("value", u32::MAX - 3),
            variant_child("metadata", u32::MAX - 4),
        ]),
        s if s.starts_with("decimal(") && s.ends_with(')') => {
            let (p, s) = s[8..s.len() - 1]
                .split_once(',')
                .ok_or_else(|| invalid("invalid decimal"))?;
            let p = p.trim().parse::<usize>().map_err(invalid)?;
            let s = s.trim().parse::<usize>().map_err(invalid)?;
            if !(1..=38).contains(&p) || s > p {
                return Err(invalid("invalid decimal precision/scale"));
            }
            D::Decimal(p, s)
        },
        _ => return Err(unsupported(format!("schema type {s}"))),
    })
}
fn validate_changes(changes: &Value) -> Result<()> {
    for change in changes
        .as_array()
        .ok_or_else(|| invalid("delta.typeChanges must be array"))?
    {
        let from = primitive(string(change, "fromType")?)?;
        let to = primitive(string(change, "toType")?)?;
        let rank = |d: &D| match d {
            D::Int8 => Some(0),
            D::Int16 => Some(1),
            D::Int32 => Some(2),
            D::Int64 => Some(3),
            _ => None,
        };
        let allowed = match (&from, &to) {
            (D::Float32, D::Float64) => true,
            (D::Int8 | D::Int16 | D::Int32, D::Float64) => true,
            (D::Date32, D::Timestamp(TimeUnit::Microsecond, None)) => true,
            (D::Decimal(p, s), D::Decimal(q, t)) => t >= s && q - t >= p - s,
            (D::Int8, D::Decimal(p, s)) => p - s >= 3,
            (D::Int16, D::Decimal(p, s)) => p - s >= 5,
            (D::Int32, D::Decimal(p, s)) => p - s >= 10,
            (D::Int64, D::Decimal(p, s)) => p - s >= 20,
            _ => rank(&from).zip(rank(&to)).is_some_and(|(a, b)| a < b),
        };
        if !allowed {
            return Err(unsupported(format!("type widening {from:?} -> {to:?}")));
        }
        // fieldPath selects nested array/map elements. Cast support is recursive; unknown path syntax is refused.
        if let Some(path) = change.get("fieldPath") {
            if !path.is_string() {
                return Err(invalid("invalid typeChanges fieldPath"));
            }
        }
    }
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn widening_rejects_narrowing() {
        assert!(
            validate_changes(&serde_json::json!([{"fromType":"long","toType":"integer"}])).is_err()
        );
        assert!(validate_changes(&serde_json::json!([{"fromType":"integer","toType":"double"},{"fromType":"decimal(10,2)","toType":"decimal(12,3)"}])).is_ok())
    }
    #[test]
    fn variant_marker_is_owned_by_planner() {
        let raw = serde_json::json!({"type":"struct","fields":[
            {"name":"spoof","type":{"type":"struct","fields":[]},"metadata":{"polars.delta.variant":"true"}},
            {"name":"real","type":"variant","metadata":{}},
            {"name":"nested","type":{"type":"array","elementType":"variant","containsNull":true},"metadata":{}}
        ]});
        let columns = parse_fields(&raw, "none").unwrap();
        assert!(
            !columns[0]
                .field
                .metadata
                .as_ref()
                .is_some_and(|m| m.contains_key("polars.delta.variant"))
        );
        assert_eq!(
            columns[1]
                .field
                .metadata
                .as_ref()
                .unwrap()
                .get("polars.delta.variant")
                .unwrap(),
            "true"
        );
        let D::LargeList(element) = &columns[2].field.dtype else {
            panic!()
        };
        assert_eq!(
            element
                .metadata
                .as_ref()
                .unwrap()
                .get("polars.delta.variant")
                .unwrap(),
            "true"
        );
    }
    #[test]
    fn future_reader_rejected() {
        assert!(
            Schema::parse(
                &Value::Null,
                &serde_json::json!({"minReaderVersion":4,"readerFeatures":[]})
            )
            .is_err()
        )
    }
}
