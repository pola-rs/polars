//! Iceberg table metadata (`metadata.json`): the subset needed for scan planning.
use std::sync::Arc;

use polars_utils::aliases::PlHashMap;
use serde::Deserialize;
use serde_json::Value as JsonValue;

use crate::iceberg::error::{
    IcebergResult, err_invalid_data, err_invalid_input, err_not_implemented,
};

#[derive(Debug, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub struct TableMetadata {
    pub format_version: u8,
    #[serde(default)]
    pub current_schema_id: Option<i32>,
    #[serde(default)]
    schemas: Vec<SchemaJson>,
    /// Format v1 only (superseded by `schemas`).
    #[serde(default)]
    schema: Option<SchemaJson>,
    #[serde(default)]
    partition_specs: Vec<PartitionSpecJson>,
    /// Format v1 only (superseded by `partition-specs`).
    #[serde(default)]
    partition_spec: Option<Vec<PartitionFieldJson>>,
    #[serde(default)]
    pub current_snapshot_id: Option<i64>,
    #[serde(default)]
    pub snapshots: Vec<Snapshot>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub struct Snapshot {
    pub snapshot_id: i64,
    #[serde(default)]
    pub parent_snapshot_id: Option<i64>,
    /// Absent for format v1 snapshots that list their manifests inline (not supported).
    #[serde(default)]
    pub manifest_list: Option<String>,
    #[serde(default)]
    pub summary: Option<PlHashMap<String, JsonValue>>,
    #[serde(default)]
    pub schema_id: Option<i32>,
}

impl Snapshot {
    pub fn operation(&self) -> Option<&str> {
        self.summary.as_ref()?.get("operation")?.as_str()
    }
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "kebab-case")]
struct SchemaJson {
    #[serde(default)]
    schema_id: Option<i32>,
    fields: Vec<JsonValue>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "kebab-case")]
struct PartitionSpecJson {
    spec_id: i32,
    fields: Vec<PartitionFieldJson>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "kebab-case")]
struct PartitionFieldJson {
    #[serde(default)]
    source_id: Option<i32>,
    #[serde(default)]
    source_ids: Option<Vec<i32>>,
    #[serde(default)]
    field_id: Option<i32>,
    transform: String,
}

/// Parsed table metadata with resolved schemas and partition specs.
pub struct Table {
    pub metadata: TableMetadata,
    pub schemas: PlHashMap<i32, Arc<Schema>>,
    pub specs: PlHashMap<i32, Arc<PartitionSpec>>,
}

impl Table {
    pub fn parse(bytes: &[u8]) -> IcebergResult<Self> {
        let mut metadata: TableMetadata = serde_json::from_slice(bytes)
            .map_err(|e| err_invalid_data(format!("invalid table metadata JSON: {e}")))?;

        if metadata.format_version > 3 {
            return Err(err_not_implemented(format!(
                "table format version {}",
                metadata.format_version
            )));
        }

        let mut schema_jsons = std::mem::take(&mut metadata.schemas);
        if schema_jsons.is_empty() {
            schema_jsons.extend(metadata.schema.take());
        }

        let schemas = schema_jsons
            .into_iter()
            .map(|s| {
                let id = s.schema_id.unwrap_or(0);
                let fields = s
                    .fields
                    .iter()
                    .map(parse_nested_field)
                    .collect::<IcebergResult<_>>()?;
                Ok((id, Arc::new(Schema::new(id, fields))))
            })
            .collect::<IcebergResult<PlHashMap<_, _>>>()?;

        let mut spec_jsons = std::mem::take(&mut metadata.partition_specs);
        if spec_jsons.is_empty() {
            if let Some(fields) = metadata.partition_spec.take() {
                spec_jsons.push(PartitionSpecJson { spec_id: 0, fields });
            }
        }

        let specs = spec_jsons
            .into_iter()
            .map(|s| {
                let fields = s
                    .fields
                    .into_iter()
                    .enumerate()
                    .map(|(i, f)| {
                        let source_id = match (f.source_id, f.source_ids.as_deref()) {
                            (Some(id), _) => id,
                            (None, Some([id])) => *id,
                            (None, Some(_)) => {
                                return Err(err_not_implemented(format!(
                                    "multi-argument partition transform '{}'",
                                    f.transform
                                )));
                            },
                            (None, None) => {
                                return Err(err_invalid_data("partition field without source-id"));
                            },
                        };
                        Ok(PartitionField {
                            source_id,
                            field_id: f.field_id.unwrap_or(1000 + i as i32),
                            transform: Transform::parse(&f.transform),
                        })
                    })
                    .collect::<IcebergResult<_>>()?;
                Ok((s.spec_id, Arc::new(PartitionSpec { fields })))
            })
            .collect::<IcebergResult<PlHashMap<_, _>>>()?;

        Ok(Self {
            metadata,
            schemas,
            specs,
        })
    }

    pub fn current_schema(&self) -> IcebergResult<&Arc<Schema>> {
        let id = self.metadata.current_schema_id.unwrap_or(0);
        self.schemas
            .get(&id)
            .ok_or_else(|| err_invalid_data(format!("current schema {id} not found")))
    }

    pub fn schema_by_id(&self, id: i32) -> IcebergResult<&Arc<Schema>> {
        self.schemas
            .get(&id)
            .ok_or_else(|| err_invalid_data(format!("schema {id} not found")))
    }

    pub fn snapshot_by_id(&self, id: i64) -> Option<&Snapshot> {
        self.metadata.snapshots.iter().find(|s| s.snapshot_id == id)
    }

    pub fn current_snapshot(&self) -> Option<&Snapshot> {
        let id = self.metadata.current_snapshot_id?;
        // `-1` means no current snapshot in older writers.
        self.snapshot_by_id(id)
    }

    /// Ancestors of `snapshot` (including itself), newest first.
    pub fn ancestors<'a>(&'a self, snapshot: &'a Snapshot) -> impl Iterator<Item = &'a Snapshot> {
        let mut next = Some(snapshot);
        std::iter::from_fn(move || {
            let current = next?;
            next = current
                .parent_snapshot_id
                .and_then(|id| self.snapshot_by_id(id));
            Some(current)
        })
    }

    /// Snapshot selected by a user-supplied ID, as `IcebergScanResolver` does.
    pub fn snapshot_for_scan(&self, snapshot_id: i64) -> IcebergResult<(&Snapshot, i32)> {
        let snapshot = self.snapshot_by_id(snapshot_id).ok_or_else(|| {
            err_invalid_input(format!("iceberg snapshot ID not found: {snapshot_id}"))
        })?;
        let schema_id = snapshot.schema_id.ok_or_else(|| {
            err_invalid_input(format!(
                "iceberg: requested snapshot {snapshot_id} did not contain a schema ID"
            ))
        })?;
        Ok((snapshot, schema_id))
    }
}

pub struct PartitionSpec {
    pub fields: Vec<PartitionField>,
}

pub struct PartitionField {
    pub source_id: i32,
    pub field_id: i32,
    pub transform: Transform,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Transform {
    Identity,
    Bucket(u32),
    Truncate(u32),
    Year,
    Month,
    Day,
    Hour,
    Void,
    Other(String),
}

impl Transform {
    fn parse(s: &str) -> Self {
        let arg = |prefix: &str| {
            s.strip_prefix(prefix)?
                .strip_prefix('[')?
                .strip_suffix(']')?
                .trim()
                .parse()
                .ok()
        };
        match s {
            "identity" => Self::Identity,
            "year" => Self::Year,
            "month" => Self::Month,
            "day" => Self::Day,
            "hour" => Self::Hour,
            "void" => Self::Void,
            _ => {
                if let Some(n) = arg("bucket") {
                    Self::Bucket(n)
                } else if let Some(n) = arg("truncate") {
                    Self::Truncate(n)
                } else {
                    Self::Other(s.to_owned())
                }
            },
        }
    }
}

// Schema and types.

#[derive(Debug, Clone, PartialEq)]
pub enum PrimitiveType {
    Boolean,
    Int,
    Long,
    Float,
    Double,
    Decimal { precision: u32, scale: u32 },
    Date,
    Time,
    Timestamp,
    Timestamptz,
    TimestampNs,
    TimestamptzNs,
    String,
    Uuid,
    Fixed(u64),
    Binary,
    Unknown,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Type {
    Primitive(PrimitiveType),
    Struct(Vec<NestedField>),
    List {
        element_id: i32,
        element_required: bool,
        element: Box<Type>,
    },
    Map {
        key_id: i32,
        key: Box<Type>,
        value_id: i32,
        value_required: bool,
        value: Box<Type>,
    },
}

impl Type {
    pub fn is_primitive(&self) -> bool {
        matches!(self, Type::Primitive(_))
    }

    pub fn as_primitive(&self) -> Option<&PrimitiveType> {
        match self {
            Type::Primitive(p) => Some(p),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct NestedField {
    pub id: i32,
    pub name: String,
    pub required: bool,
    pub field_type: Type,
    pub initial_default: Option<JsonValue>,
}

pub struct Schema {
    pub schema_id: i32,
    pub fields: Vec<NestedField>,
    /// Field ID → field, for all fields including nested ones (list elements and map keys /
    /// values are represented as fields named `element`, `key` and `value`).
    by_id: PlHashMap<i32, NestedField>,
}

impl Schema {
    pub fn new(schema_id: i32, fields: Vec<NestedField>) -> Self {
        let mut by_id = PlHashMap::default();
        fn index(field: &NestedField, by_id: &mut PlHashMap<i32, NestedField>) {
            by_id.insert(field.id, field.clone());
            index_type(&field.field_type, by_id);
        }
        fn index_type(ty: &Type, by_id: &mut PlHashMap<i32, NestedField>) {
            match ty {
                Type::Primitive(_) => {},
                Type::Struct(fields) => fields.iter().for_each(|f| index(f, by_id)),
                Type::List {
                    element_id,
                    element_required,
                    element,
                } => index(
                    &NestedField {
                        id: *element_id,
                        name: "element".into(),
                        required: *element_required,
                        field_type: (**element).clone(),
                        initial_default: None,
                    },
                    by_id,
                ),
                Type::Map {
                    key_id,
                    key,
                    value_id,
                    value_required,
                    value,
                } => {
                    index(
                        &NestedField {
                            id: *key_id,
                            name: "key".into(),
                            required: true,
                            field_type: (**key).clone(),
                            initial_default: None,
                        },
                        by_id,
                    );
                    index(
                        &NestedField {
                            id: *value_id,
                            name: "value".into(),
                            required: *value_required,
                            field_type: (**value).clone(),
                            initial_default: None,
                        },
                        by_id,
                    );
                },
            }
        }
        fields.iter().for_each(|f| index(f, &mut by_id));

        Self {
            schema_id,
            fields,
            by_id,
        }
    }

    pub fn field_by_id(&self, id: i32) -> Option<&NestedField> {
        self.by_id.get(&id)
    }

    /// All field IDs, including nested ones.
    pub fn field_ids(&self) -> impl Iterator<Item = i32> + '_ {
        self.by_id.keys().copied()
    }

    /// Top-level fields selected by name, in schema order. Unknown names are ignored.
    pub fn select(&self, names: &[String]) -> Vec<&NestedField> {
        self.fields
            .iter()
            .filter(|f| names.iter().any(|n| n == &f.name))
            .collect()
    }
}

fn parse_nested_field(json: &JsonValue) -> IcebergResult<NestedField> {
    let obj = json
        .as_object()
        .ok_or_else(|| err_invalid_data("schema field is not an object"))?;
    let id = obj
        .get("id")
        .and_then(|v| v.as_i64())
        .ok_or_else(|| err_invalid_data("schema field without id"))? as i32;
    let name = obj
        .get("name")
        .and_then(|v| v.as_str())
        .ok_or_else(|| err_invalid_data("schema field without name"))?
        .to_owned();
    let required = obj
        .get("required")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let field_type = parse_type(
        obj.get("type")
            .ok_or_else(|| err_invalid_data("schema field without type"))?,
    )?;
    let initial_default = obj.get("initial-default").filter(|v| !v.is_null()).cloned();

    Ok(NestedField {
        id,
        name,
        required,
        field_type,
        initial_default,
    })
}

fn parse_type(json: &JsonValue) -> IcebergResult<Type> {
    match json {
        JsonValue::String(s) => parse_primitive(s).map(Type::Primitive),
        JsonValue::Object(obj) => {
            let get_i32 = |k: &str| {
                obj.get(k)
                    .and_then(|v| v.as_i64())
                    .map(|v| v as i32)
                    .ok_or_else(|| err_invalid_data(format!("type without '{k}'")))
            };
            let get_type = |k: &str| {
                parse_type(
                    obj.get(k)
                        .ok_or_else(|| err_invalid_data(format!("type without '{k}'")))?,
                )
                .map(Box::new)
            };
            match obj.get("type").and_then(|v| v.as_str()) {
                Some("struct") => Ok(Type::Struct(
                    obj.get("fields")
                        .and_then(|v| v.as_array())
                        .ok_or_else(|| err_invalid_data("struct type without fields"))?
                        .iter()
                        .map(parse_nested_field)
                        .collect::<IcebergResult<_>>()?,
                )),
                Some("list") => Ok(Type::List {
                    element_id: get_i32("element-id")?,
                    element_required: obj
                        .get("element-required")
                        .and_then(|v| v.as_bool())
                        .unwrap_or(false),
                    element: get_type("element")?,
                }),
                Some("map") => Ok(Type::Map {
                    key_id: get_i32("key-id")?,
                    key: get_type("key")?,
                    value_id: get_i32("value-id")?,
                    value_required: obj
                        .get("value-required")
                        .and_then(|v| v.as_bool())
                        .unwrap_or(false),
                    value: get_type("value")?,
                }),
                other => Err(err_invalid_data(format!("unknown nested type {other:?}"))),
            }
        },
        _ => Err(err_invalid_data(format!("invalid type: {json}"))),
    }
}

fn parse_primitive(s: &str) -> IcebergResult<PrimitiveType> {
    use PrimitiveType as P;
    Ok(match s {
        "boolean" => P::Boolean,
        "int" => P::Int,
        "long" => P::Long,
        "float" => P::Float,
        "double" => P::Double,
        "date" => P::Date,
        "time" => P::Time,
        "timestamp" => P::Timestamp,
        "timestamptz" => P::Timestamptz,
        "timestamp_ns" => P::TimestampNs,
        "timestamptz_ns" => P::TimestamptzNs,
        "string" => P::String,
        "uuid" => P::Uuid,
        "binary" => P::Binary,
        "unknown" => P::Unknown,
        s => {
            if let Some(args) = s.strip_prefix("decimal(").and_then(|r| r.strip_suffix(')')) {
                let (p, sc) = args
                    .split_once(',')
                    .ok_or_else(|| err_invalid_data(format!("invalid type '{s}'")))?;
                P::Decimal {
                    precision: p
                        .trim()
                        .parse()
                        .map_err(|_| err_invalid_data(format!("invalid type '{s}'")))?,
                    scale: sc
                        .trim()
                        .parse()
                        .map_err(|_| err_invalid_data(format!("invalid type '{s}'")))?,
                }
            } else if let Some(n) = s.strip_prefix("fixed[").and_then(|r| r.strip_suffix(']')) {
                P::Fixed(
                    n.trim()
                        .parse()
                        .map_err(|_| err_invalid_data(format!("invalid type '{s}'")))?,
                )
            } else {
                return Err(err_not_implemented(format!("type '{s}'")));
            }
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_types() {
        assert_eq!(
            parse_primitive("decimal(10, 2)").unwrap(),
            PrimitiveType::Decimal {
                precision: 10,
                scale: 2
            }
        );
        assert_eq!(
            parse_primitive("fixed[3]").unwrap(),
            PrimitiveType::Fixed(3)
        );
        assert_eq!(Transform::parse("bucket[16]"), Transform::Bucket(16));
        assert_eq!(Transform::parse("truncate[4]"), Transform::Truncate(4));
        assert_eq!(Transform::parse("identity"), Transform::Identity);
    }
}
