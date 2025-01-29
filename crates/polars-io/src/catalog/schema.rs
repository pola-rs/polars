use polars_core::prelude::{DataType, Field};
use polars_core::schema::{Schema, SchemaRef};
use polars_error::{polars_bail, polars_err, to_compute_err, PolarsResult};
use polars_utils::error::TruncateErrorDetail;
use polars_utils::format_pl_smallstr;
use polars_utils::pl_str::PlSmallStr;

use super::unity::models::{ColumnInfo, ColumnTypeJson, TableInfo};
use crate::catalog::unity::models::ColumnTypeJsonType;
use crate::utils::decode_json_response;

/// Returns `(schema, hive_schema)`
pub fn table_info_to_schemas(
    table_info: &TableInfo,
) -> PolarsResult<(Option<SchemaRef>, Option<SchemaRef>)> {
    let Some(columns) = table_info.columns.as_deref() else {
        return Ok((None, None));
    };

    let mut schema = Schema::default();
    let mut hive_schema = Schema::default();

    for (i, col) in columns.iter().enumerate() {
        if let Some(position) = col.position {
            if usize::try_from(position).unwrap() != i {
                polars_bail!(
                    ComputeError:
                    "not yet supported: position was not ordered"
                )
            }
        }

        let field = column_info_to_field(col)?;

        if let Some(i) = col.partition_index {
            if usize::try_from(i).unwrap() != hive_schema.len() {
                polars_bail!(
                    ComputeError:
                    "not yet supported: partition_index was not ordered"
                )
            }

            hive_schema.extend([field]);
        } else {
            schema.extend([field])
        }
    }

    Ok((
        Some(schema.into()),
        Some(hive_schema)
            .filter(|x| !x.is_empty())
            .map(|x| x.into()),
    ))
}

pub fn column_info_to_field(column_info: &ColumnInfo) -> PolarsResult<Field> {
    Ok(Field::new(
        column_info.name.clone(),
        parse_type_json_str(&column_info.type_json)?,
    ))
}

/// e.g.
/// ```json
/// {"name":"Int64","type":"long","nullable":true}
/// {"name":"List","type":{"type":"array","elementType":"long","containsNull":true},"nullable":true}
/// ```
pub fn parse_type_json_str(type_json: &str) -> PolarsResult<DataType> {
    let decoded: ColumnTypeJson = decode_json_response(type_json.as_bytes())?;

    parse_type_json(&decoded).map_err(|e| {
        e.wrap_msg(|e| {
            format!(
                "error parsing type response: {}, type_json: {}",
                e,
                TruncateErrorDetail(type_json)
            )
        })
    })
}

/// We prefer this as `type_text` cannot be trusted for consistency (e.g. we may expect `decimal(int,int)`
/// but instead get `decimal`, or `struct<...>` and instead get `struct`).
pub fn parse_type_json(type_json: &ColumnTypeJson) -> PolarsResult<DataType> {
    use ColumnTypeJsonType::*;

    let out = match &type_json.type_ {
        TypeName(name) => match name.as_str() {
            "array" => {
                let inner_json: &ColumnTypeJsonType =
                    type_json.element_type.as_ref().ok_or_else(|| {
                        polars_err!(
                            ComputeError:
                            "missing elementType in response for array type"
                        )
                    })?;

                let inner_dtype = parse_type_json_type(inner_json)?;

                DataType::List(Box::new(inner_dtype))
            },

            "struct" => {
                let fields_json: &[ColumnTypeJson] =
                    type_json.fields.as_deref().ok_or_else(|| {
                        polars_err!(
                            ComputeError:
                            "missing elementType in response for array type"
                        )
                    })?;

                let fields = fields_json
                    .iter()
                    .map(|x| {
                        let name = x.name.clone().ok_or_else(|| {
                            polars_err!(
                                ComputeError:
                                "missing name in fields response for struct type"
                            )
                        })?;
                        let dtype = parse_type_json(x)?;

                        Ok(Field::new(name, dtype))
                    })
                    .collect::<PolarsResult<Vec<_>>>()?;

                DataType::Struct(fields)
            },

            "map" => {
                let key_type = type_json.key_type.as_ref().ok_or_else(|| {
                    polars_err!(
                        ComputeError:
                        "missing keyType in response for map type"
                    )
                })?;

                let value_type = type_json.value_type.as_ref().ok_or_else(|| {
                    polars_err!(
                        ComputeError:
                        "missing valueType in response for map type"
                    )
                })?;

                DataType::List(Box::new(DataType::Struct(vec![
                    Field::new(
                        PlSmallStr::from_static("key"),
                        parse_type_json_type(key_type)?,
                    ),
                    Field::new(
                        PlSmallStr::from_static("value"),
                        parse_type_json_type(value_type)?,
                    ),
                ])))
            },

            name => parse_type_text(name)?,
        },

        TypeJson(type_json) => parse_type_json(type_json.as_ref())?,
    };

    Ok(out)
}

fn parse_type_json_type(type_json_type: &ColumnTypeJsonType) -> PolarsResult<DataType> {
    use ColumnTypeJsonType::*;

    match type_json_type {
        TypeName(name) => parse_type_text(name),
        TypeJson(type_json) => parse_type_json(type_json.as_ref()),
    }
}

/// Parses the string variant of the `type` field within a `type_json`. This can be understood as
/// the leaf / non-nested datatypes of the field.
///
/// References:
/// * https://spark.apache.org/docs/latest/sql-ref-datatypes.html
/// * https://docs.databricks.com/api/workspace/tables/get
/// * https://docs.databricks.com/en/sql/language-manual/sql-ref-datatypes.html
///
/// Notes:
/// * `type_precision` and `type_scale` in the API response are defined as supplementary data to
///   the `type_text`, but from testing they aren't actually used - e.g. a decimal type would have a
///   `type_text` of `decimal(18, 2)`
fn parse_type_text(type_text: &str) -> PolarsResult<DataType> {
    use polars_core::prelude::TimeUnit;
    use DataType::*;

    let dtype = match type_text {
        "boolean" => Boolean,

        "tinyint" | "byte" => Int8,
        "smallint" | "short" => Int16,
        "int" | "integer" => Int32,
        "bigint" | "long" => Int64,

        "float" | "real" => Float32,
        "double" => Float64,

        "date" => Date,
        "timestamp" | "timestamp_ntz" | "timestamp_ltz" => Datetime(TimeUnit::Nanoseconds, None),

        "string" => String,
        "binary" => Binary,

        "null" | "void" => Null,

        v => {
            if v.starts_with("decimal") {
                // e.g. decimal(38,18)
                (|| {
                    let (precision, scale) = v
                        .get(7..)?
                        .strip_prefix('(')?
                        .strip_suffix(')')?
                        .split_once(',')?;
                    let precision: usize = precision.parse().ok()?;
                    let scale: usize = scale.parse().ok()?;

                    Some(DataType::Decimal(Some(precision), Some(scale)))
                })()
                .ok_or_else(|| {
                    polars_err!(
                        ComputeError:
                        "type format did not match decimal(int,int): {}",
                        v
                    )
                })?
            } else {
                polars_bail!(
                    ComputeError:
                    "parse_type_text unknown type name: {}",
                    v
                )
            }
        },
    };

    Ok(dtype)
}

// Conversion functions to API format. Mainly used for constructing the request to create tables.

pub fn schema_to_column_info_list(schema: &Schema) -> PolarsResult<Vec<ColumnInfo>> {
    schema
        .iter()
        .enumerate()
        .map(|(i, (name, dtype))| {
            let name = name.clone();
            let type_text = dtype_to_type_text(dtype)?;
            let type_name = dtype_to_type_name(dtype)?;
            let type_json = serde_json::to_string(&field_to_type_json(name.clone(), dtype)?)
                .map_err(to_compute_err)?;

            Ok(ColumnInfo {
                name,
                type_name,
                type_text,
                type_json,
                position: Some(i.try_into().unwrap()),
                comment: None,
                partition_index: None,
            })
        })
        .collect::<PolarsResult<_>>()
}

/// Creates the `type_text` field of the API. Opposite of [`parse_type_text`]
fn dtype_to_type_text(dtype: &DataType) -> PolarsResult<PlSmallStr> {
    use polars_core::prelude::TimeUnit;
    use DataType::*;

    macro_rules! S {
        ($e:expr) => {
            PlSmallStr::from_static($e)
        };
    }

    let out = match dtype {
        Boolean => S!("boolean"),

        Int8 => S!("tinyint"),
        Int16 => S!("smallint"),
        Int32 => S!("int"),
        Int64 => S!("bigint"),

        Float32 => S!("float"),
        Float64 => S!("double"),

        Date => S!("date"),
        Datetime(TimeUnit::Nanoseconds, None) => S!("timestamp_ntz"),

        String => S!("string"),
        Binary => S!("binary"),

        Null => S!("null"),

        Decimal(precision, scale) => {
            let precision = precision.unwrap_or(38);
            let scale = scale.unwrap_or(0);

            format_pl_smallstr!("decimal({},{})", precision, scale)
        },

        List(inner) => {
            if let Some((key_type, value_type)) = get_list_map_type(inner) {
                format_pl_smallstr!(
                    "map<{},{}>",
                    dtype_to_type_text(key_type)?,
                    dtype_to_type_text(value_type)?
                )
            } else {
                format_pl_smallstr!("array<{}>", dtype_to_type_text(inner)?)
            }
        },

        Struct(fields) => {
            // Yes, it's possible to construct column names containing the brackets. This won't
            // affect us as we parse using `type_json` rather than this field.
            let mut out = std::string::String::from("struct<");

            for Field { name, dtype } in fields {
                out.push_str(name);
                out.push(':');
                out.push_str(&dtype_to_type_text(dtype)?);
                out.push(',');
            }

            if out.ends_with(',') {
                out.truncate(out.len() - 1);
            }

            out.push('>');

            out.into()
        },

        v => polars_bail!(
            ComputeError:
            "dtype_to_type_text unsupported type: {}",
            v
        ),
    };

    Ok(out)
}

/// Creates the `type_name` field, from testing this wasn't exactly the same as the `type_text` field.
fn dtype_to_type_name(dtype: &DataType) -> PolarsResult<PlSmallStr> {
    use polars_core::prelude::TimeUnit;
    use DataType::*;

    macro_rules! S {
        ($e:expr) => {
            PlSmallStr::from_static($e)
        };
    }

    let out = match dtype {
        Boolean => S!("BOOLEAN"),

        Int8 => S!("BYTE"),
        Int16 => S!("SHORT"),
        Int32 => S!("INT"),
        Int64 => S!("LONG"),

        Float32 => S!("FLOAT"),
        Float64 => S!("DOUBLE"),

        Date => S!("DATE"),
        Datetime(TimeUnit::Nanoseconds, None) => S!("TIMESTAMP_NTZ"),
        String => S!("STRING"),
        Binary => S!("BINARY"),

        Null => S!("NULL"),

        Decimal(..) => S!("DECIMAL"),

        List(inner) => {
            if get_list_map_type(inner).is_some() {
                S!("MAP")
            } else {
                S!("ARRAY")
            }
        },

        Struct(..) => S!("STRUCT"),

        v => polars_bail!(
            ComputeError:
            "dtype_to_type_text unsupported type: {}",
            v
        ),
    };

    Ok(out)
}

/// Creates the `type_json` field.
fn field_to_type_json(name: PlSmallStr, dtype: &DataType) -> PolarsResult<ColumnTypeJson> {
    Ok(ColumnTypeJson {
        name: Some(name),
        type_: dtype_to_type_json(dtype)?,
        nullable: Some(true),
        // We set this to Some(_) so that the output matches the one generated by Databricks.
        metadata: Some(Default::default()),

        ..Default::default()
    })
}

fn dtype_to_type_json(dtype: &DataType) -> PolarsResult<ColumnTypeJsonType> {
    use polars_core::prelude::TimeUnit;
    use DataType::*;

    macro_rules! S {
        ($e:expr) => {
            ColumnTypeJsonType::from_static_type_name($e)
        };
    }

    let out = match dtype {
        Boolean => S!("boolean"),

        Int8 => S!("byte"),
        Int16 => S!("short"),
        Int32 => S!("integer"),
        Int64 => S!("long"),

        Float32 => S!("float"),
        Float64 => S!("double"),

        Date => S!("date"),
        Datetime(TimeUnit::Nanoseconds, None) => S!("timestamp_ntz"),

        String => S!("string"),
        Binary => S!("binary"),

        Null => S!("null"),

        Decimal(..) => ColumnTypeJsonType::TypeName(dtype_to_type_text(dtype)?),

        List(inner) => {
            let out = if let Some((key_type, value_type)) = get_list_map_type(inner) {
                ColumnTypeJson {
                    type_: ColumnTypeJsonType::from_static_type_name("map"),
                    key_type: Some(dtype_to_type_json(key_type)?),
                    value_type: Some(dtype_to_type_json(value_type)?),
                    value_contains_null: Some(true),

                    ..Default::default()
                }
            } else {
                ColumnTypeJson {
                    type_: ColumnTypeJsonType::from_static_type_name("array"),
                    element_type: Some(dtype_to_type_json(inner)?),
                    contains_null: Some(true),

                    ..Default::default()
                }
            };

            ColumnTypeJsonType::TypeJson(Box::new(out))
        },

        Struct(fields) => {
            let out = ColumnTypeJson {
                type_: ColumnTypeJsonType::from_static_type_name("struct"),
                fields: Some(
                    fields
                        .iter()
                        .map(|Field { name, dtype }| field_to_type_json(name.clone(), dtype))
                        .collect::<PolarsResult<_>>()?,
                ),

                ..Default::default()
            };

            ColumnTypeJsonType::TypeJson(Box::new(out))
        },

        v => polars_bail!(
            ComputeError:
            "dtype_to_type_text unsupported type: {}",
            v
        ),
    };

    Ok(out)
}

/// Tries to interpret the List type as a `map` field, which is essentially
/// List(Struct(("key", <dtype>), ("value", <dtyoe>))).
///
/// Returns `Option<(key_type, value_type)>`
fn get_list_map_type(list_inner_dtype: &DataType) -> Option<(&DataType, &DataType)> {
    let DataType::Struct(fields) = list_inner_dtype else {
        return None;
    };

    let [fld1, fld2] = fields.as_slice() else {
        return None;
    };

    if !(fld1.name == "key" && fld2.name == "value") {
        return None;
    }

    Some((fld1.dtype(), fld2.dtype()))
}
