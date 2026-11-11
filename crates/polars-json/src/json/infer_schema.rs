use std::borrow::Borrow;

use indexmap::map::Entry;
use polars_arrow::datatypes::{ArrowDataType, Field};
use polars_utils::pl_str::PlSmallStr;
use simd_json::borrowed::Object;
use simd_json::{BorrowedValue, StaticNode};

use super::*;

const ITEM_NAME: &str = "item";

/// Infers [`ArrowDataType`] from [`Value`][Value].
///
/// [Value]: simd_json::value::Value
pub fn infer(json: &BorrowedValue) -> PolarsResult<ArrowDataType> {
    Ok(match json {
        BorrowedValue::Static(StaticNode::Bool(_)) => ArrowDataType::Boolean,
        BorrowedValue::Static(StaticNode::I64(_)) => ArrowDataType::Int64,
        BorrowedValue::Static(StaticNode::U64(x)) if *x <= i64::MAX as u64 => ArrowDataType::Int64,
        BorrowedValue::Static(StaticNode::U64(_) | StaticNode::U128(_) | StaticNode::I128(_)) => {
            ArrowDataType::Int128
        },
        BorrowedValue::Static(StaticNode::F64(_)) => ArrowDataType::Float64,
        BorrowedValue::Static(StaticNode::Null) => ArrowDataType::Null,
        BorrowedValue::Array(array) => infer_array(array)?,
        BorrowedValue::String(_) => ArrowDataType::LargeUtf8,
        BorrowedValue::Object(inner) => infer_object(inner)?,
    })
}

fn infer_object(inner: &Object) -> PolarsResult<ArrowDataType> {
    // Objects parsed from a tape keep repeated keys; each key is one field.
    let mut fields = PlIndexMap::<&str, ArrowDataType>::with_capacity(inner.len());
    for (key, value) in inner.iter() {
        let dt = infer(value)?;
        match fields.entry(key.as_ref()) {
            Entry::Occupied(mut prev) => {
                let coerced = coerce_dtype(&[prev.get().clone(), dt]);
                prev.insert(coerced);
            },
            Entry::Vacant(v) => {
                v.insert(dt);
            },
        }
    }
    Ok(ArrowDataType::Struct(
        fields
            .into_iter()
            .map(|(key, dt)| Field::new(key.into(), dt, true))
            .collect(),
    ))
}

fn infer_array(values: &[BorrowedValue]) -> PolarsResult<ArrowDataType> {
    let types = values
        .iter()
        .map(infer)
        // deduplicate entries
        .collect::<PolarsResult<PlIndexSet<_>>>()?;

    let dt = if !types.is_empty() {
        let types = types.into_iter().collect::<Vec<_>>();
        coerce_dtype(&types)
    } else {
        ArrowDataType::Null
    };

    Ok(ArrowDataType::LargeList(Box::new(Field::new(
        PlSmallStr::from_static(ITEM_NAME),
        dt,
        true,
    ))))
}

/// Coerce an heterogeneous set of [`ArrowDataType`] into a single one. Rules:
/// * The empty set is coerced to `Null`
/// * `Int64` and `Float64` are `Float64`
/// * Lists and scalars are coerced to a list of a compatible scalar
/// * Structs contain the union of all fields
/// * All other types are coerced to `Utf8`
pub(crate) fn coerce_dtype<A: Borrow<ArrowDataType>>(datatypes: &[A]) -> ArrowDataType {
    use ArrowDataType::*;

    if datatypes.is_empty() {
        return Null;
    }

    let are_all_equal = datatypes.windows(2).all(|w| w[0].borrow() == w[1].borrow());

    if are_all_equal {
        return datatypes[0].borrow().clone();
    }
    let mut are_all_structs = true;
    let mut are_all_lists = true;
    for dt in datatypes {
        are_all_structs &= matches!(dt.borrow(), Struct(_));
        are_all_lists &= matches!(dt.borrow(), LargeList(_));
    }

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
            PlIndexMap::<&str, PlHashSet<&ArrowDataType>>::default(),
            |mut acc, field| {
                match acc.entry(field.name.as_str()) {
                    Entry::Occupied(mut v) => {
                        v.get_mut().insert(&field.dtype);
                    },
                    Entry::Vacant(v) => {
                        let mut a = PlHashSet::default();
                        a.insert(&field.dtype);
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
                Field::new(name.into(), coerce_dtype(&dts), true)
            })
            .collect();
        return Struct(fields);
    } else if are_all_lists {
        let inner_types: Vec<&ArrowDataType> = datatypes
            .iter()
            .map(|dt| {
                if let LargeList(inner) = dt.borrow() {
                    inner.dtype()
                } else {
                    unreachable!();
                }
            })
            .collect();
        return LargeList(Box::new(Field::new(
            PlSmallStr::from_static(ITEM_NAME),
            coerce_dtype(inner_types.as_slice()),
            true,
        )));
    } else if datatypes.len() > 2 {
        return datatypes
            .iter()
            .map(|t| t.borrow().clone())
            .reduce(|a, b| coerce_dtype(&[a, b]))
            .expect("not empty");
    }
    let (lhs, rhs) = (datatypes[0].borrow(), datatypes[1].borrow());

    match (lhs, rhs) {
        (lhs, rhs) if lhs == rhs => lhs.clone(),
        (LargeList(lhs), LargeList(rhs)) => {
            let inner = coerce_dtype(&[lhs.dtype(), rhs.dtype()]);
            LargeList(Box::new(Field::new(
                PlSmallStr::from_static(ITEM_NAME),
                inner,
                true,
            )))
        },
        (scalar, LargeList(list)) => {
            let inner = coerce_dtype(&[scalar, list.dtype()]);
            LargeList(Box::new(Field::new(
                PlSmallStr::from_static(ITEM_NAME),
                inner,
                true,
            )))
        },
        (LargeList(list), scalar) => {
            let inner = coerce_dtype(&[scalar, list.dtype()]);
            LargeList(Box::new(Field::new(
                PlSmallStr::from_static(ITEM_NAME),
                inner,
                true,
            )))
        },
        (Float64, Int64) => Float64,
        (Int64, Float64) => Float64,
        (Int64, Boolean) => Int64,
        (Boolean, Int64) => Int64,
        (Null, rhs) => rhs.clone(),
        (lhs, Null) => lhs.clone(),
        (_, _) => LargeUtf8,
    }
}
