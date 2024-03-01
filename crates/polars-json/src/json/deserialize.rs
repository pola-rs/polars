use std::borrow::Borrow;
use std::fmt::Write;

use arrow::array::*;
use arrow::bitmap::MutableBitmap;
use arrow::datatypes::{ArrowDataType, IntervalUnit};
use arrow::offset::{Offset, Offsets};
use arrow::temporal_conversions;
use arrow::types::NativeType;
use num_traits::NumCast;
use simd_json::{BorrowedValue, StaticNode};

use super::*;

const JSON_NULL_VALUE: BorrowedValue = BorrowedValue::Static(StaticNode::Null);

fn deserialize_boolean_into<'a, A: Borrow<BorrowedValue<'a>>>(
    target: &mut MutableBooleanArray,
    rows: &[A],
) {
    let iter = rows.iter().map(|row| match row.borrow() {
        BorrowedValue::Static(StaticNode::Bool(v)) => Some(v),
        _ => None,
    });
    target.extend_trusted_len(iter);
}

fn deserialize_primitive_into<'a, T: NativeType + NumCast, A: Borrow<BorrowedValue<'a>>>(
    target: &mut MutablePrimitiveArray<T>,
    rows: &[A],
) {
    let iter = rows.iter().map(|row| match row.borrow() {
        BorrowedValue::Static(StaticNode::I64(v)) => T::from(*v),
        BorrowedValue::Static(StaticNode::U64(v)) => T::from(*v),
        BorrowedValue::Static(StaticNode::F64(v)) => T::from(*v),
        BorrowedValue::Static(StaticNode::Bool(v)) => T::from(*v as u8),
        _ => None,
    });
    target.extend_trusted_len(iter);
}

fn deserialize_binary<'a, A: Borrow<BorrowedValue<'a>>>(rows: &[A]) -> BinaryArray<i64> {
    let iter = rows.iter().map(|row| match row.borrow() {
        BorrowedValue::String(v) => Some(v.as_bytes()),
        _ => None,
    });
    BinaryArray::from_trusted_len_iter(iter)
}

fn deserialize_utf8_into<'a, O: Offset, A: Borrow<BorrowedValue<'a>>>(
    target: &mut MutableUtf8Array<O>,
    rows: &[A],
) {
    let mut scratch = String::new();
    for row in rows {
        match row.borrow() {
            BorrowedValue::String(v) => target.push(Some(v.as_ref())),
            BorrowedValue::Static(StaticNode::Bool(v)) => {
                target.push(Some(if *v { "true" } else { "false" }))
            },
            BorrowedValue::Static(node) if !matches!(node, StaticNode::Null) => {
                write!(scratch, "{node}").unwrap();
                target.push(Some(scratch.as_str()));
                scratch.clear();
            },
            _ => target.push_null(),
        }
    }
}

fn deserialize_utf8view_into<'a, A: Borrow<BorrowedValue<'a>>>(
    target: &mut MutableBinaryViewArray<str>,
    rows: &[A],
) {
    let mut scratch = String::new();
    for row in rows {
        match row.borrow() {
            BorrowedValue::String(v) => target.push_value(v.as_ref()),
            BorrowedValue::Static(StaticNode::Bool(v)) => {
                target.push_value(if *v { "true" } else { "false" })
            },
            BorrowedValue::Static(node) if !matches!(node, StaticNode::Null) => {
                write!(scratch, "{node}").unwrap();
                target.push_value(scratch.as_str());
                scratch.clear();
            },
            _ => target.push_null(),
        }
    }
}

fn deserialize_list<'a, A: Borrow<BorrowedValue<'a>>>(
    rows: &[A],
    data_type: ArrowDataType,
) -> ListArray<i64> {
    let child = ListArray::<i64>::get_child_type(&data_type);

    let mut validity = MutableBitmap::with_capacity(rows.len());
    let mut offsets = Offsets::<i64>::with_capacity(rows.len());
    let mut inner = vec![];
    rows.iter().for_each(|row| match row.borrow() {
        BorrowedValue::Array(value) => {
            inner.extend(value.iter());
            validity.push(true);
            offsets
                .try_push(value.len())
                .expect("List offset is too large :/");
        },
        BorrowedValue::Static(StaticNode::Null) => {
            validity.push(false);
            offsets.extend_constant(1)
        },
        value @ (BorrowedValue::Static(_) | BorrowedValue::String(_)) => {
            inner.push(value);
            validity.push(true);
            offsets.try_push(1).expect("List offset is too large :/");
        },
        _ => {
            validity.push(false);
            offsets.extend_constant(1);
        },
    });

    let values = _deserialize(&inner, child.clone());

    ListArray::<i64>::new(data_type, offsets.into(), values, validity.into())
}

fn deserialize_struct<'a, A: Borrow<BorrowedValue<'a>>>(
    rows: &[A],
    data_type: ArrowDataType,
) -> StructArray {
    let fields = StructArray::get_fields(&data_type);

    let mut values = fields
        .iter()
        .map(|f| (f.name.as_str(), (f.data_type(), vec![])))
        .collect::<PlHashMap<_, _>>();

    let mut validity = MutableBitmap::with_capacity(rows.len());

    rows.iter().for_each(|row| {
        match row.borrow() {
            BorrowedValue::Object(value) => {
                values.iter_mut().for_each(|(s, (_, inner))| {
                    inner.push(value.get(*s).unwrap_or(&JSON_NULL_VALUE))
                });
                validity.push(true);
            },
            _ => {
                values
                    .iter_mut()
                    .for_each(|(_, (_, inner))| inner.push(&JSON_NULL_VALUE));
                validity.push(false);
            },
        };
    });

    // ensure we collect in the proper order
    let values = fields
        .iter()
        .map(|fld| {
            let (data_type, vals) = values.get(fld.name.as_str()).unwrap();
            _deserialize(vals, (*data_type).clone())
        })
        .collect::<Vec<_>>();

    StructArray::new(data_type.clone(), values, validity.into())
}

fn fill_array_from<B, T, A>(
    f: fn(&mut MutablePrimitiveArray<T>, &[B]),
    data_type: ArrowDataType,
    rows: &[B],
) -> Box<dyn Array>
where
    T: NativeType,
    A: From<MutablePrimitiveArray<T>> + Array,
{
    let mut array = MutablePrimitiveArray::<T>::with_capacity(rows.len()).to(data_type);
    f(&mut array, rows);
    Box::new(A::from(array))
}

/// A trait describing an array with a backing store that can be preallocated to
/// a given size.
pub(crate) trait Container {
    /// Create this array with a given capacity.
    fn with_capacity(capacity: usize) -> Self
    where
        Self: Sized;
}

impl<O: Offset> Container for MutableBinaryArray<O> {
    fn with_capacity(capacity: usize) -> Self {
        MutableBinaryArray::with_capacity(capacity)
    }
}

impl Container for MutableBooleanArray {
    fn with_capacity(capacity: usize) -> Self {
        MutableBooleanArray::with_capacity(capacity)
    }
}

impl Container for MutableFixedSizeBinaryArray {
    fn with_capacity(capacity: usize) -> Self {
        MutableFixedSizeBinaryArray::with_capacity(capacity, 0)
    }
}

impl Container for MutableBinaryViewArray<str> {
    fn with_capacity(capacity: usize) -> Self
    where
        Self: Sized,
    {
        MutableBinaryViewArray::with_capacity(capacity)
    }
}

impl<O: Offset, M: MutableArray + Default + 'static> Container for MutableListArray<O, M> {
    fn with_capacity(capacity: usize) -> Self {
        MutableListArray::with_capacity(capacity)
    }
}

impl<T: NativeType> Container for MutablePrimitiveArray<T> {
    fn with_capacity(capacity: usize) -> Self {
        MutablePrimitiveArray::with_capacity(capacity)
    }
}

impl<O: Offset> Container for MutableUtf8Array<O> {
    fn with_capacity(capacity: usize) -> Self {
        MutableUtf8Array::with_capacity(capacity)
    }
}

fn fill_generic_array_from<B, M, A>(f: fn(&mut M, &[B]), rows: &[B]) -> Box<dyn Array>
where
    M: Container,
    A: From<M> + Array,
{
    let mut array = M::with_capacity(rows.len());
    f(&mut array, rows);
    Box::new(A::from(array))
}

pub(crate) fn _deserialize<'a, A: Borrow<BorrowedValue<'a>>>(
    rows: &[A],
    data_type: ArrowDataType,
) -> Box<dyn Array> {
    match &data_type {
        ArrowDataType::Null => Box::new(NullArray::new(data_type, rows.len())),
        ArrowDataType::Boolean => {
            fill_generic_array_from::<_, _, BooleanArray>(deserialize_boolean_into, rows)
        },
        ArrowDataType::Int8 => {
            fill_array_from::<_, _, PrimitiveArray<i8>>(deserialize_primitive_into, data_type, rows)
        },
        ArrowDataType::Int16 => fill_array_from::<_, _, PrimitiveArray<i16>>(
            deserialize_primitive_into,
            data_type,
            rows,
        ),
        ArrowDataType::Int32
        | ArrowDataType::Date32
        | ArrowDataType::Time32(_)
        | ArrowDataType::Interval(IntervalUnit::YearMonth) => {
            fill_array_from::<_, _, PrimitiveArray<i32>>(
                deserialize_primitive_into,
                data_type,
                rows,
            )
        },
        ArrowDataType::Interval(IntervalUnit::DayTime) => {
            unimplemented!("There is no natural representation of DayTime in JSON.")
        },
        ArrowDataType::Int64
        | ArrowDataType::Date64
        | ArrowDataType::Time64(_)
        | ArrowDataType::Duration(_) => fill_array_from::<_, _, PrimitiveArray<i64>>(
            deserialize_primitive_into,
            data_type,
            rows,
        ),
        ArrowDataType::Timestamp(tu, tz) => {
            let iter = rows.iter().map(|row| match row.borrow() {
                BorrowedValue::Static(StaticNode::I64(v)) => Some(*v),
                BorrowedValue::String(v) => match (tu, tz) {
                    (_, None) => temporal_conversions::utf8_to_naive_timestamp_scalar(v, "%+", tu),
                    (_, Some(ref tz)) => {
                        let tz = temporal_conversions::parse_offset(tz).unwrap();
                        temporal_conversions::utf8_to_timestamp_scalar(v, "%+", &tz, tu)
                    },
                },
                _ => None,
            });
            Box::new(Int64Array::from_iter(iter).to(data_type))
        },
        ArrowDataType::UInt8 => {
            fill_array_from::<_, _, PrimitiveArray<u8>>(deserialize_primitive_into, data_type, rows)
        },
        ArrowDataType::UInt16 => fill_array_from::<_, _, PrimitiveArray<u16>>(
            deserialize_primitive_into,
            data_type,
            rows,
        ),
        ArrowDataType::UInt32 => fill_array_from::<_, _, PrimitiveArray<u32>>(
            deserialize_primitive_into,
            data_type,
            rows,
        ),
        ArrowDataType::UInt64 => fill_array_from::<_, _, PrimitiveArray<u64>>(
            deserialize_primitive_into,
            data_type,
            rows,
        ),
        ArrowDataType::Float16 => unreachable!(),
        ArrowDataType::Float32 => fill_array_from::<_, _, PrimitiveArray<f32>>(
            deserialize_primitive_into,
            data_type,
            rows,
        ),
        ArrowDataType::Float64 => fill_array_from::<_, _, PrimitiveArray<f64>>(
            deserialize_primitive_into,
            data_type,
            rows,
        ),
        ArrowDataType::LargeUtf8 => {
            fill_generic_array_from::<_, _, Utf8Array<i64>>(deserialize_utf8_into, rows)
        },
        ArrowDataType::Utf8View => {
            fill_generic_array_from::<_, _, Utf8ViewArray>(deserialize_utf8view_into, rows)
        },
        ArrowDataType::LargeList(_) => Box::new(deserialize_list(rows, data_type)),
        ArrowDataType::LargeBinary => Box::new(deserialize_binary(rows)),
        ArrowDataType::Struct(_) => Box::new(deserialize_struct(rows, data_type)),
        _ => todo!(),
    }
}

pub fn deserialize(json: &BorrowedValue, data_type: ArrowDataType) -> PolarsResult<Box<dyn Array>> {
    match json {
        BorrowedValue::Array(rows) => match data_type {
            ArrowDataType::LargeList(inner) => Ok(_deserialize(rows, inner.data_type)),
            _ => todo!("read an Array from a non-Array data type"),
        },
        _ => Ok(_deserialize(&[json], data_type)),
    }
}
