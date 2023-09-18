use std::borrow::Borrow;

use indexmap::map::IndexMap as HashMap;
use indexmap::set::IndexSet as HashSet;
use json_deserializer::{Number, Value};

use crate::datatypes::*;
use crate::error::{Error, Result};

const ITEM_NAME: &str = "item";

/// Infers [`DataType`] from [`Value`].
pub fn infer(json: &Value) -> Result<DataType> {
    Ok(match json {
        Value::Bool(_) => DataType::Boolean,
        Value::Array(array) => infer_array(array)?,
        Value::Null => DataType::Null,
        Value::Number(number) => infer_number(number),
        Value::String(_) => DataType::Utf8,
        Value::Object(inner) => infer_object(inner)?,
    })
}

/// Infers [`Schema`] from JSON [`Value`] in (pandas-compatible) records format.
pub fn infer_records_schema(json: &Value) -> Result<Schema> {
    let outer_array = match json {
        Value::Array(array) => Ok(array),
        _ => Err(Error::ExternalFormat(
            "outer type is not an array".to_string(),
        )),
    }?;

    let fields = match outer_array.iter().next() {
        Some(Value::Object(record)) => record
            .iter()
            .map(|(name, json)| {
                let data_type = infer(json)?;

                Ok(Field {
                    name: name.clone(),
                    data_type,
                    is_nullable: true,
                    metadata: Metadata::default(),
                })
            })
            .collect::<Result<Vec<_>>>(),
        None => Ok(vec![]),
        _ => Err(Error::ExternalFormat(
            "first element in array is not a record".to_string(),
        )),
    }?;

    Ok(Schema {
        fields,
        metadata: Metadata::default(),
    })
}

fn filter_map_nulls(dt: DataType) -> Option<DataType> {
    if dt == DataType::Null {
        None
    } else {
        Some(dt)
    }
}

fn infer_object(inner: &HashMap<String, Value>) -> Result<DataType> {
    let fields = inner
        .iter()
        .filter_map(|(key, value)| {
            infer(value)
                .map(|dt| filter_map_nulls(dt).map(|dt| (key, dt)))
                .transpose()
        })
        .map(|maybe_dt| {
            let (key, dt) = maybe_dt?;
            Ok(Field::new(key, dt, true))
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(DataType::Struct(fields))
}

fn infer_array(values: &[Value]) -> Result<DataType> {
    let types = values
        .iter()
        .map(infer)
        .filter_map(|x| x.map(filter_map_nulls).transpose())
        // deduplicate entries
        .collect::<Result<HashSet<_>>>()?;

    let dt = if !types.is_empty() {
        let types = types.into_iter().collect::<Vec<_>>();
        coerce_data_type(&types)
    } else {
        DataType::Null
    };

    // if a record contains only nulls, it is not
    // added to values
    Ok(if dt == DataType::Null {
        dt
    } else {
        DataType::List(Box::new(Field::new(ITEM_NAME, dt, true)))
    })
}

fn infer_number(n: &Number) -> DataType {
    match n {
        Number::Float(..) => DataType::Float64,
        Number::Integer(..) => DataType::Int64,
    }
}

/// Coerce an heterogeneous set of [`DataType`] into a single one. Rules:
/// * The empty set is coerced to `Null`
/// * `Int64` and `Float64` are `Float64`
/// * Lists and scalars are coerced to a list of a compatible scalar
/// * Structs contain the union of all fields
/// * All other types are coerced to `Utf8`
pub(crate) fn coerce_data_type<A: Borrow<DataType>>(datatypes: &[A]) -> DataType {
    use DataType::*;

    if datatypes.is_empty() {
        return DataType::Null;
    }

    let are_all_equal = datatypes.windows(2).all(|w| w[0].borrow() == w[1].borrow());

    if are_all_equal {
        return datatypes[0].borrow().clone();
    }

    let are_all_structs = datatypes.iter().all(|x| matches!(x.borrow(), Struct(_)));

    if are_all_structs {
        // all are structs => union of all fields (that may have equal names)
        let fields = datatypes.iter().fold(vec![], |mut acc, dt| {
            if let Struct(new_fields) = dt.borrow() {
                acc.extend(new_fields);
            };
            acc
        });
        // group fields by unique
        let fields = fields.iter().fold(
            HashMap::<&String, HashSet<&DataType>>::new(),
            |mut acc, field| {
                match acc.entry(&field.name) {
                    indexmap::map::Entry::Occupied(mut v) => {
                        v.get_mut().insert(&field.data_type);
                    }
                    indexmap::map::Entry::Vacant(v) => {
                        let mut a = HashSet::new();
                        a.insert(&field.data_type);
                        v.insert(a);
                    }
                }
                acc
            },
        );
        // and finally, coerce each of the fields within the same name
        let fields = fields
            .into_iter()
            .map(|(name, dts)| {
                let dts = dts.into_iter().collect::<Vec<_>>();
                Field::new(name, coerce_data_type(&dts), true)
            })
            .collect();
        return Struct(fields);
    } else if datatypes.len() > 2 {
        return Utf8;
    }
    let (lhs, rhs) = (datatypes[0].borrow(), datatypes[1].borrow());

    return match (lhs, rhs) {
        (lhs, rhs) if lhs == rhs => lhs.clone(),
        (List(lhs), List(rhs)) => {
            let inner = coerce_data_type(&[lhs.data_type(), rhs.data_type()]);
            List(Box::new(Field::new(ITEM_NAME, inner, true)))
        }
        (scalar, List(list)) => {
            let inner = coerce_data_type(&[scalar, list.data_type()]);
            List(Box::new(Field::new(ITEM_NAME, inner, true)))
        }
        (List(list), scalar) => {
            let inner = coerce_data_type(&[scalar, list.data_type()]);
            List(Box::new(Field::new(ITEM_NAME, inner, true)))
        }
        (Float64, Int64) => Float64,
        (Int64, Float64) => Float64,
        (Int64, Boolean) => Int64,
        (Boolean, Int64) => Int64,
        (_, _) => Utf8,
    };
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_coersion_scalar_and_list() {
        use crate::datatypes::DataType::*;

        assert_eq!(
            coerce_data_type(&[
                Float64,
                List(Box::new(Field::new(ITEM_NAME, Float64, true)))
            ]),
            List(Box::new(Field::new(ITEM_NAME, Float64, true))),
        );
        assert_eq!(
            coerce_data_type(&[Float64, List(Box::new(Field::new(ITEM_NAME, Int64, true)))]),
            List(Box::new(Field::new(ITEM_NAME, Float64, true))),
        );
        assert_eq!(
            coerce_data_type(&[Int64, List(Box::new(Field::new(ITEM_NAME, Int64, true)))]),
            List(Box::new(Field::new(ITEM_NAME, Int64, true))),
        );
        // boolean and number are incompatible, return utf8
        assert_eq!(
            coerce_data_type(&[
                Boolean,
                List(Box::new(Field::new(ITEM_NAME, Float64, true)))
            ]),
            List(Box::new(Field::new(ITEM_NAME, Utf8, true))),
        );
    }

    #[test]
    fn test_coersion_of_nulls() {
        assert_eq!(coerce_data_type(&[DataType::Null]), DataType::Null);
        assert_eq!(
            coerce_data_type(&[DataType::Null, DataType::Boolean]),
            DataType::Utf8
        );
        let vec: Vec<DataType> = vec![];
        assert_eq!(coerce_data_type(vec.as_slice()), DataType::Null);
    }
}
