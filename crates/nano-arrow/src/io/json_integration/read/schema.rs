use serde_derive::Deserialize;
use serde_json::Value;

use crate::{
    error::{Error, Result},
    io::ipc::IpcField,
};

use crate::datatypes::{
    get_extension, DataType, Field, IntegerType, IntervalUnit, Metadata, Schema, TimeUnit,
    UnionMode,
};

fn to_time_unit(item: Option<&Value>) -> Result<TimeUnit> {
    match item {
        Some(p) if p == "SECOND" => Ok(TimeUnit::Second),
        Some(p) if p == "MILLISECOND" => Ok(TimeUnit::Millisecond),
        Some(p) if p == "MICROSECOND" => Ok(TimeUnit::Microsecond),
        Some(p) if p == "NANOSECOND" => Ok(TimeUnit::Nanosecond),
        _ => Err(Error::OutOfSpec("time unit missing or invalid".to_string())),
    }
}

fn to_int(item: &Value) -> Result<IntegerType> {
    Ok(match item.get("isSigned") {
        Some(&Value::Bool(true)) => match item.get("bitWidth") {
            Some(Value::Number(n)) => match n.as_u64() {
                Some(8) => IntegerType::Int8,
                Some(16) => IntegerType::Int16,
                Some(32) => IntegerType::Int32,
                Some(64) => IntegerType::Int64,
                _ => {
                    return Err(Error::OutOfSpec(
                        "int bitWidth missing or invalid".to_string(),
                    ))
                }
            },
            _ => {
                return Err(Error::OutOfSpec(
                    "int bitWidth missing or invalid".to_string(),
                ))
            }
        },
        Some(&Value::Bool(false)) => match item.get("bitWidth") {
            Some(Value::Number(n)) => match n.as_u64() {
                Some(8) => IntegerType::UInt8,
                Some(16) => IntegerType::UInt16,
                Some(32) => IntegerType::UInt32,
                Some(64) => IntegerType::UInt64,
                _ => {
                    return Err(Error::OutOfSpec(
                        "int bitWidth missing or invalid".to_string(),
                    ))
                }
            },
            _ => {
                return Err(Error::OutOfSpec(
                    "int bitWidth missing or invalid".to_string(),
                ))
            }
        },
        _ => {
            return Err(Error::OutOfSpec(
                "int signed missing or invalid".to_string(),
            ))
        }
    })
}

fn deserialize_fields(children: Option<&Value>) -> Result<Vec<Field>> {
    children
        .map(|x| {
            if let Value::Array(values) = x {
                values
                    .iter()
                    .map(deserialize_field)
                    .collect::<Result<Vec<_>>>()
            } else {
                Err(Error::OutOfSpec("children must be an array".to_string()))
            }
        })
        .unwrap_or_else(|| Ok(vec![]))
}

fn read_metadata(metadata: &Value) -> Result<Metadata> {
    match metadata {
        Value::Array(ref values) => {
            let mut res = Metadata::new();
            for value in values {
                match value.as_object() {
                    Some(map) => {
                        if map.len() != 2 {
                            return Err(Error::OutOfSpec(
                                "Field 'metadata' must have exact two entries for each key-value map".to_string(),
                            ));
                        }
                        if let (Some(k), Some(v)) = (map.get("key"), map.get("value")) {
                            if let (Some(k_str), Some(v_str)) = (k.as_str(), v.as_str()) {
                                res.insert(k_str.to_string().clone(), v_str.to_string().clone());
                            } else {
                                return Err(Error::OutOfSpec(
                                    "Field 'metadata' must have map value of string type"
                                        .to_string(),
                                ));
                            }
                        } else {
                            return Err(Error::OutOfSpec(
                                "Field 'metadata' lacks map keys named \"key\" or \"value\""
                                    .to_string(),
                            ));
                        }
                    }
                    _ => {
                        return Err(Error::OutOfSpec(
                            "Field 'metadata' contains non-object key-value pair".to_string(),
                        ));
                    }
                }
            }
            Ok(res)
        }
        Value::Object(ref values) => {
            let mut res = Metadata::new();
            for (k, v) in values {
                if let Some(str_value) = v.as_str() {
                    res.insert(k.clone(), str_value.to_string().clone());
                } else {
                    return Err(Error::OutOfSpec(format!(
                        "Field 'metadata' contains non-string value for key {k}"
                    )));
                }
            }
            Ok(res)
        }
        _ => Err(Error::OutOfSpec(
            "Invalid json value type for field".to_string(),
        )),
    }
}

fn to_data_type(item: &Value, mut children: Vec<Field>) -> Result<DataType> {
    let type_ = item
        .get("name")
        .ok_or_else(|| Error::OutOfSpec("type missing".to_string()))?;

    let type_ = if let Value::String(name) = type_ {
        name.as_str()
    } else {
        return Err(Error::OutOfSpec("type is not a string".to_string()));
    };

    use DataType::*;
    Ok(match type_ {
        "null" => Null,
        "bool" => Boolean,
        "binary" => Binary,
        "largebinary" => LargeBinary,
        "fixedsizebinary" => {
            // return a list with any type as its child isn't defined in the map
            if let Some(Value::Number(size)) = item.get("byteWidth") {
                DataType::FixedSizeBinary(size.as_i64().unwrap() as usize)
            } else {
                return Err(Error::OutOfSpec(
                    "Expecting a byteWidth for fixedsizebinary".to_string(),
                ));
            }
        }
        "utf8" => Utf8,
        "largeutf8" => LargeUtf8,
        "decimal" => {
            // return a list with any type as its child isn't defined in the map
            let precision = item
                .get("precision")
                .ok_or_else(|| Error::OutOfSpec("Expecting a precision for decimal".to_string()))?
                .as_u64()
                .unwrap() as usize;

            let scale = item
                .get("scale")
                .ok_or_else(|| Error::OutOfSpec("Expecting a scale for decimal".to_string()))?
                .as_u64()
                .unwrap() as usize;

            let bit_width = match item.get("bitWidth") {
                Some(s) => s.as_u64().unwrap() as usize,
                None => 128,
            };

            match bit_width {
                128 => DataType::Decimal(precision, scale),
                256 => DataType::Decimal256(precision, scale),
                _ => todo!(),
            }
        }
        "floatingpoint" => match item.get("precision") {
            Some(p) if p == "HALF" => DataType::Float16,
            Some(p) if p == "SINGLE" => DataType::Float32,
            Some(p) if p == "DOUBLE" => DataType::Float64,
            _ => {
                return Err(Error::OutOfSpec(
                    "floatingpoint precision missing or invalid".to_string(),
                ))
            }
        },
        "timestamp" => {
            let unit = to_time_unit(item.get("unit"))?;
            let tz = match item.get("timezone") {
                None => Ok(None),
                Some(Value::String(tz)) => Ok(Some(tz.clone())),
                _ => Err(Error::OutOfSpec("timezone must be a string".to_string())),
            }?;
            DataType::Timestamp(unit, tz)
        }
        "date" => match item.get("unit") {
            Some(p) if p == "DAY" => DataType::Date32,
            Some(p) if p == "MILLISECOND" => DataType::Date64,
            _ => return Err(Error::OutOfSpec("date unit missing or invalid".to_string())),
        },
        "time" => {
            let unit = to_time_unit(item.get("unit"))?;
            match item.get("bitWidth") {
                Some(p) if p == 32 => DataType::Time32(unit),
                Some(p) if p == 64 => DataType::Time64(unit),
                _ => {
                    return Err(Error::OutOfSpec(
                        "time bitWidth missing or invalid".to_string(),
                    ))
                }
            }
        }
        "duration" => {
            let unit = to_time_unit(item.get("unit"))?;
            DataType::Duration(unit)
        }
        "interval" => match item.get("unit") {
            Some(p) if p == "DAY_TIME" => DataType::Interval(IntervalUnit::DayTime),
            Some(p) if p == "YEAR_MONTH" => DataType::Interval(IntervalUnit::YearMonth),
            Some(p) if p == "MONTH_DAY_NANO" => DataType::Interval(IntervalUnit::MonthDayNano),
            _ => {
                return Err(Error::OutOfSpec(
                    "interval unit missing or invalid".to_string(),
                ))
            }
        },
        "int" => to_int(item).map(|x| x.into())?,
        "list" => DataType::List(Box::new(children.pop().unwrap())),
        "largelist" => DataType::LargeList(Box::new(children.pop().unwrap())),
        "fixedsizelist" => {
            if let Some(Value::Number(size)) = item.get("listSize") {
                DataType::FixedSizeList(
                    Box::new(children.pop().unwrap()),
                    size.as_i64().unwrap() as usize,
                )
            } else {
                return Err(Error::OutOfSpec(
                    "Expecting a listSize for fixedsizelist".to_string(),
                ));
            }
        }
        "struct" => DataType::Struct(children),
        "union" => {
            let mode = if let Some(Value::String(mode)) = item.get("mode") {
                UnionMode::sparse(mode == "SPARSE")
            } else {
                return Err(Error::OutOfSpec("union requires mode".to_string()));
            };
            let ids = if let Some(Value::Array(ids)) = item.get("typeIds") {
                Some(ids.iter().map(|x| x.as_i64().unwrap() as i32).collect())
            } else {
                return Err(Error::OutOfSpec("union requires ids".to_string()));
            };
            DataType::Union(children, ids, mode)
        }
        "map" => {
            let sorted_keys = if let Some(Value::Bool(sorted_keys)) = item.get("keysSorted") {
                *sorted_keys
            } else {
                return Err(Error::OutOfSpec("sorted keys not defined".to_string()));
            };
            DataType::Map(Box::new(children.pop().unwrap()), sorted_keys)
        }
        other => {
            return Err(Error::NotYetImplemented(format!(
                "invalid json value type \"{other}\""
            )))
        }
    })
}

fn deserialize_ipc_field(value: &Value) -> Result<IpcField> {
    let map = if let Value::Object(map) = value {
        map
    } else {
        return Err(Error::OutOfSpec(
            "Invalid json value type for field".to_string(),
        ));
    };

    let fields = map
        .get("children")
        .map(|x| {
            if let Value::Array(values) = x {
                values
                    .iter()
                    .map(deserialize_ipc_field)
                    .collect::<Result<Vec<_>>>()
            } else {
                Err(Error::OutOfSpec("children must be an array".to_string()))
            }
        })
        .unwrap_or_else(|| Ok(vec![]))?;

    let dictionary_id = if let Some(dictionary) = map.get("dictionary") {
        match dictionary.get("id") {
            Some(Value::Number(n)) => Some(n.as_i64().unwrap()),
            _ => {
                return Err(Error::OutOfSpec("Field missing 'id' attribute".to_string()));
            }
        }
    } else {
        None
    };
    Ok(IpcField {
        fields,
        dictionary_id,
    })
}

fn deserialize_field(value: &Value) -> Result<Field> {
    let map = if let Value::Object(map) = value {
        map
    } else {
        return Err(Error::OutOfSpec(
            "Invalid json value type for field".to_string(),
        ));
    };

    let name = match map.get("name") {
        Some(Value::String(name)) => name.to_string(),
        _ => {
            return Err(Error::OutOfSpec(
                "Field missing 'name' attribute".to_string(),
            ));
        }
    };
    let is_nullable = match map.get("nullable") {
        Some(&Value::Bool(b)) => b,
        _ => {
            return Err(Error::OutOfSpec(
                "Field missing 'nullable' attribute".to_string(),
            ));
        }
    };

    let metadata = if let Some(metadata) = map.get("metadata") {
        read_metadata(metadata)?
    } else {
        Metadata::default()
    };

    let extension = get_extension(&metadata);

    let type_ = map
        .get("type")
        .ok_or_else(|| Error::OutOfSpec("type missing".to_string()))?;

    let children = deserialize_fields(map.get("children"))?;
    let data_type = to_data_type(type_, children)?;

    let data_type = if let Some((name, metadata)) = extension {
        DataType::Extension(name, Box::new(data_type), metadata)
    } else {
        data_type
    };

    let data_type = if let Some(dictionary) = map.get("dictionary") {
        let index_type = match dictionary.get("indexType") {
            Some(t) => to_int(t)?,
            _ => {
                return Err(Error::OutOfSpec(
                    "Field missing 'indexType' attribute".to_string(),
                ));
            }
        };
        let is_ordered = match dictionary.get("isOrdered") {
            Some(&Value::Bool(n)) => n,
            _ => {
                return Err(Error::OutOfSpec(
                    "Field missing 'isOrdered' attribute".to_string(),
                ));
            }
        };
        DataType::Dictionary(index_type, Box::new(data_type), is_ordered)
    } else {
        data_type
    };

    Ok(Field {
        name,
        data_type,
        is_nullable,
        metadata,
    })
}

#[derive(Deserialize)]
struct MetadataKeyValue {
    key: String,
    value: String,
}

/// Parse a `metadata` definition from a JSON representation.
/// The JSON can either be an Object or an Array of Objects.
fn from_metadata(json: &Value) -> Result<Metadata> {
    match json {
        Value::Array(_) => {
            let values: Vec<MetadataKeyValue> = serde_json::from_value(json.clone())?;
            Ok(values
                .into_iter()
                .map(|key_value| (key_value.key, key_value.value))
                .collect())
        }
        Value::Object(md) => md
            .iter()
            .map(|(k, v)| {
                if let Value::String(v) = v {
                    Ok((k.to_string(), v.to_string()))
                } else {
                    Err(Error::OutOfSpec(
                        "metadata `value` field must be a string".to_string(),
                    ))
                }
            })
            .collect::<Result<_>>(),
        _ => Err(Error::OutOfSpec(
            "`metadata` field must be an object".to_string(),
        )),
    }
}

/// Deserializes a [`Value`]
pub fn deserialize_schema(value: &Value) -> Result<(Schema, Vec<IpcField>)> {
    let schema = if let Value::Object(schema) = value {
        schema
    } else {
        return Err(Error::OutOfSpec(
            "Invalid json value type for schema".to_string(),
        ));
    };

    let fields = if let Some(Value::Array(fields)) = schema.get("fields") {
        fields
            .iter()
            .map(deserialize_field)
            .collect::<Result<_>>()?
    } else {
        return Err(Error::OutOfSpec(
            "Schema fields should be an array".to_string(),
        ));
    };

    let ipc_fields = if let Some(Value::Array(fields)) = schema.get("fields") {
        fields
            .iter()
            .map(deserialize_ipc_field)
            .collect::<Result<_>>()?
    } else {
        return Err(Error::OutOfSpec(
            "Schema fields should be an array".to_string(),
        ));
    };

    let metadata = if let Some(value) = schema.get("metadata") {
        from_metadata(value)?
    } else {
        Metadata::default()
    };

    Ok((Schema { fields, metadata }, ipc_fields))
}
