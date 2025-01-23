use polars_core::prelude::{DataType, Field};
use polars_core::schema::{Schema, SchemaRef};
use polars_error::{polars_bail, polars_err, PolarsResult};
use polars_utils::pl_str::PlSmallStr;

use super::unity::models::TableInfo;

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
        let dtype = parse_type_str(&col.type_text)?;

        if let Some(position) = col.position {
            if usize::try_from(position).unwrap() != i {
                polars_bail!(
                    ComputeError:
                    "not yet supported: position was not ordered"
                )
            }
        }

        if let Some(i) = col.partition_index {
            if usize::try_from(i).unwrap() != hive_schema.len() {
                polars_bail!(
                    ComputeError:
                    "not yet supported: partition_index was not ordered"
                )
            }

            hive_schema.extend([Field::new(col.name.as_str().into(), dtype)]);
        } else {
            schema.extend([Field::new(col.name.as_str().into(), dtype)])
        }
    }

    Ok((
        Some(schema.into()),
        Some(hive_schema)
            .filter(|x| !x.is_empty())
            .map(|x| x.into()),
    ))
}

/// Parse a type string from a catalog API response.
///
/// References:
/// * https://spark.apache.org/docs/latest/sql-ref-datatypes.html
/// * https://docs.databricks.com/api/workspace/tables/get
/// * https://docs.databricks.com/en/sql/language-manual/sql-ref-datatypes.html
///
/// Note: `type_precision` and `type_scale` in the API response are defined as supplementary data to
/// the `type_text`, but from testing they aren't actually used - e.g. a decimal type would have a
/// `type_text` of `decimal(18, 2)`
fn parse_type_str(type_text: &str) -> PolarsResult<DataType> {
    use DataType::*;

    let dtype = match type_text {
        "boolean" => Boolean,

        "byte" | "tinyint" => Int8,
        "short" | "smallint" => Int16,
        "int" | "integer" => Int32,
        "long" | "bigint" => Int64,

        "float" | "real" => Float32,
        "double" => Float64,

        "date" => Date,
        "timestamp" | "timestamp_ltz" | "timestamp_ntz" => {
            Datetime(polars_core::prelude::TimeUnit::Nanoseconds, None)
        },

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
            } else if v.starts_with("array") {
                // e.g. array<int>
                DataType::List(Box::new(parse_type_str(extract_angle_brackets_inner(
                    v, "array",
                )?)?))
            } else if v.starts_with("struct") {
                parse_struct_type_str(v)?
            } else if v.starts_with("map") {
                // e.g. map<int,string>
                let inner = extract_angle_brackets_inner(v, "map")?;
                let (key_type_str, value_type_str) = split_comma_nesting_aware(inner);
                DataType::List(Box::new(DataType::Struct(vec![
                    Field::new(
                        PlSmallStr::from_static("key"),
                        parse_type_str(key_type_str)?,
                    ),
                    Field::new(
                        PlSmallStr::from_static("value"),
                        parse_type_str(value_type_str)?,
                    ),
                ])))
            } else {
                polars_bail!(
                    ComputeError:
                    "parse_type_str unknown type name: {}",
                    v
                )
            }
        },
    };

    Ok(dtype)
}

/// `array<inner> -> inner`
fn extract_angle_brackets_inner<'a>(value: &'a str, name: &'static str) -> PolarsResult<&'a str> {
    let i = value.find('<');
    let j = value.rfind('>');

    if i.is_none() || j.is_none() || i.unwrap().saturating_add(1) >= j.unwrap() {
        polars_bail!(
            ComputeError:
            "type format did not match {}<...>: {}",
            name, value
        )
    }

    let i = i.unwrap();
    let j = j.unwrap();

    let inner = value[i + 1..j].trim();

    Ok(inner)
}

/// `struct<default:decimal(38,18),promotional:struct<default:decimal(38,18)>,effective_list:struct<default:decimal(38,18)>>`
fn parse_struct_type_str(value: &str) -> PolarsResult<DataType> {
    let orig_value = value;
    let mut value = extract_angle_brackets_inner(value, "struct")?;

    let mut fields = vec![];

    while !value.is_empty() {
        let (field_str, new_value) = split_comma_nesting_aware(value);
        value = new_value;

        let (name, dtype_str) = field_str.split_once(':').ok_or_else(|| {
            polars_err!(
                ComputeError:
                "type format did not match struct<name:type,..>: {}",
                orig_value
            )
        })?;

        let dtype = parse_type_str(dtype_str)?;

        fields.push(Field::new(name.into(), dtype));
    }

    Ok(DataType::Struct(fields))
}

/// `default:decimal(38,18),promotional:struct<default:decimal(38,18)>` ->
/// * 1: `default:decimal(38,18)`
/// * 2: `struct<default:decimal(38,18)>`
///
/// If there are no splits, returns the full string and an empty string.
fn split_comma_nesting_aware(value: &str) -> (&str, &str) {
    let mut bracket_level = 0usize;
    let mut angle_bracket_level = 0usize;

    for (i, b) in value.as_bytes().iter().enumerate() {
        match b {
            b'(' => bracket_level += 1,
            b')' => bracket_level = bracket_level.saturating_sub(1),
            b'<' => angle_bracket_level += 1,
            b'>' => angle_bracket_level = angle_bracket_level.saturating_sub(1),
            b',' if bracket_level == 0 && angle_bracket_level == 0 => {
                return (&value[..i], &value[1 + i..])
            },
            _ => {},
        }
    }

    (value, &value[value.len()..])
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_parse_type_str_nested_struct() {
        use super::{parse_type_str, DataType, Field};

        let type_str = "struct<default:decimal(38,18),promotional:struct<default:decimal(38,18)>,effective_list:struct<default:decimal(38,18)>>";
        let dtype = parse_type_str(type_str).unwrap();

        use DataType::*;

        assert_eq!(
            dtype,
            Struct(vec![
                Field::new("default".into(), Decimal(Some(38), Some(18))),
                Field::new(
                    "promotional".into(),
                    Struct(vec![Field::new(
                        "default".into(),
                        Decimal(Some(38), Some(18))
                    )])
                ),
                Field::new(
                    "effective_list".into(),
                    Struct(vec![Field::new(
                        "default".into(),
                        Decimal(Some(38), Some(18))
                    )])
                )
            ])
        );
    }

    #[test]
    fn test_parse_type_str_map() {
        use super::{parse_type_str, DataType, Field};

        let type_str = "map<array<int>,array<decimal(18,2)>>";
        let dtype = parse_type_str(type_str).unwrap();

        use DataType::*;

        assert_eq!(
            dtype,
            List(Box::new(Struct(vec![
                Field::new("key".into(), List(Box::new(Int32))),
                Field::new("value".into(), List(Box::new(Decimal(Some(18), Some(2)))))
            ])))
        );
    }
}
