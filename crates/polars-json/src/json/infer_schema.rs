use std::borrow::Borrow;

use arrow::datatypes::{DataType, Field, Schema};
use indexmap::map::Entry;
use indexmap::IndexMap;
use simd_json::borrowed::Object;
use simd_json::{BorrowedValue, StaticNode};

use super::*;

const ITEM_NAME: &str = "item";

/// Infers [`DataType`] from [`Value`][Value].
///
/// [Value]: simd_json::value::Value
pub fn infer(json: &BorrowedValue) -> PolarsResult<DataType> {
    Ok(match json {
        BorrowedValue::Static(StaticNode::Bool(_)) => DataType::Boolean,
        BorrowedValue::Static(StaticNode::U64(_) | StaticNode::I64(_)) => DataType::Int64,
        BorrowedValue::Static(StaticNode::F64(_)) => DataType::Float64,
        BorrowedValue::Static(StaticNode::Null) => DataType::Null,
        BorrowedValue::Array(array) => infer_array(array)?,
        BorrowedValue::String(_) => DataType::LargeUtf8,
        BorrowedValue::Object(inner) => infer_object(inner)?,
    })
}

/// Infers [`Schema`] from JSON [`Value`][Value] in (pandas-compatible) records format.
///
/// [Value]: simd_json::value::Value
pub fn infer_records_schema(json: &BorrowedValue) -> PolarsResult<Schema> {
    let outer_array = match json {
        BorrowedValue::Array(array) => Ok(array),
        _ => Err(PolarsError::ComputeError(
            "outer type is not an array".into(),
        )),
    }?;

    let fields = match outer_array.iter().next() {
        Some(BorrowedValue::Object(record)) => record
            .iter()
            .map(|(name, json)| {
                let data_type = infer(json)?;

                Ok(Field {
                    name: name.to_string(),
                    data_type: DataType::LargeList(Box::new(Field {
                        name: format!("{name}-records"),
                        data_type,
                        is_nullable: true,
                        metadata: Default::default(),
                    })),
                    is_nullable: true,
                    metadata: Default::default(),
                })
            })
            .collect::<PolarsResult<Vec<_>>>(),
        None => Ok(vec![]),
        _ => Err(PolarsError::ComputeError(
            "first element in array is not a record".into(),
        )),
    }?;

    Ok(Schema {
        fields,
        metadata: Default::default(),
    })
}

fn filter_map_nulls(dt: DataType) -> Option<DataType> {
    if dt == DataType::Null {
        None
    } else {
        Some(dt)
    }
}

fn infer_object(inner: &Object) -> PolarsResult<DataType> {
    let fields = inner
        .iter()
        .filter_map(|(key, value)| {
            infer(value)
                .map(|dt| filter_map_nulls(dt).map(|dt| (key, dt)))
                .transpose()
        })
        .map(|maybe_dt| {
            let (key, dt) = maybe_dt?;
            Ok(Field::new(key.as_ref(), dt, true))
        })
        .collect::<PolarsResult<Vec<_>>>()?;
    Ok(DataType::Struct(fields))
}

fn infer_array(values: &[BorrowedValue]) -> PolarsResult<DataType> {
    let types = values
        .iter()
        .map(infer)
        .filter_map(|x| x.map(filter_map_nulls).transpose())
        // deduplicate entries
        .collect::<PolarsResult<PlHashSet<_>>>()?;

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
        DataType::LargeList(Box::new(Field::new(ITEM_NAME, dt, true)))
    })
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
        return Null;
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
            IndexMap::<&str, PlHashSet<&DataType>, ahash::RandomState>::default(),
            |mut acc, field| {
                match acc.entry(field.name.as_str()) {
                    Entry::Occupied(mut v) => {
                        v.get_mut().insert(&field.data_type);
                    },
                    Entry::Vacant(v) => {
                        let mut a = PlHashSet::default();
                        a.insert(&field.data_type);
                        v.insert(a);
                    },
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
        return LargeUtf8;
    }
    let (lhs, rhs) = (datatypes[0].borrow(), datatypes[1].borrow());

    return match (lhs, rhs) {
        (lhs, rhs) if lhs == rhs => lhs.clone(),
        (LargeList(lhs), LargeList(rhs)) => {
            let inner = coerce_data_type(&[lhs.data_type(), rhs.data_type()]);
            LargeList(Box::new(Field::new(ITEM_NAME, inner, true)))
        },
        (scalar, List(list)) => {
            let inner = coerce_data_type(&[scalar, list.data_type()]);
            LargeList(Box::new(Field::new(ITEM_NAME, inner, true)))
        },
        (LargeList(list), scalar) => {
            let inner = coerce_data_type(&[scalar, list.data_type()]);
            LargeList(Box::new(Field::new(ITEM_NAME, inner, true)))
        },
        (Float64, Int64) => Float64,
        (Int64, Float64) => Float64,
        (Int64, Boolean) => Int64,
        (Boolean, Int64) => Int64,
        (_, _) => LargeUtf8,
    };
}
