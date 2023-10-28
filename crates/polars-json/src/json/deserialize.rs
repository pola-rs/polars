use std::borrow::Borrow;
use std::fmt::Write;

use arrow::array::*;
use arrow::bitmap::MutableBitmap;
use arrow::chunk::Chunk;
use arrow::datatypes::{ArrowSchema, DataType, Field, IntervalUnit};
use arrow::legacy::prelude::*;
use arrow::offset::{Offset, Offsets};
use arrow::temporal_conversions;
use arrow::types::{f16, NativeType};
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

fn deserialize_list<'a, A: Borrow<BorrowedValue<'a>>>(
    rows: &[A],
    data_type: DataType,
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

// TODO: due to nesting, deduplicating this from the above is trickier than
// other `deserialize_xxx_into` functions. Punting on that for now.
fn deserialize_list_into<'a, A: Borrow<BorrowedValue<'a>>>(
    target: &mut MutableListArray<i64, Box<dyn MutableArray>>,
    rows: &[A],
) {
    let empty = vec![];
    let inner: Vec<_> = rows
        .iter()
        .flat_map(|row| match row.borrow() {
            BorrowedValue::Array(value) => value.iter(),
            _ => empty.iter(),
        })
        .collect();

    deserialize_into(target.mut_values(), &inner);

    let lengths = rows.iter().map(|row| match row.borrow() {
        BorrowedValue::Array(value) => Some(value.len()),
        _ => None,
    });

    target
        .try_extend_from_lengths(lengths)
        .expect("Offsets overflow");
}

fn primitive_dispatch<'a, A: Borrow<BorrowedValue<'a>>, T: NativeType>(
    target: &mut Box<dyn MutableArray>,
    rows: &[A],
    deserialize_into: fn(&mut MutablePrimitiveArray<T>, &[A]) -> (),
) {
    generic_deserialize_into(target, rows, deserialize_into)
}

fn generic_deserialize_into<'a, A: Borrow<BorrowedValue<'a>>, M: 'static>(
    target: &mut Box<dyn MutableArray>,
    rows: &[A],
    deserialize_into: fn(&mut M, &[A]) -> (),
) {
    deserialize_into(target.as_mut_any().downcast_mut::<M>().unwrap(), rows);
}

/// Deserialize `rows` by extending them into the given `target`
fn deserialize_into<'a, A: Borrow<BorrowedValue<'a>>>(
    target: &mut Box<dyn MutableArray>,
    rows: &[A],
) {
    match target.data_type() {
        DataType::Boolean => generic_deserialize_into(target, rows, deserialize_boolean_into),
        DataType::Float32 => primitive_dispatch::<_, f32>(target, rows, deserialize_primitive_into),
        DataType::Float64 => primitive_dispatch::<_, f64>(target, rows, deserialize_primitive_into),
        DataType::Int8 => primitive_dispatch::<_, i8>(target, rows, deserialize_primitive_into),
        DataType::Int16 => primitive_dispatch::<_, i16>(target, rows, deserialize_primitive_into),
        DataType::Int32 => primitive_dispatch::<_, i32>(target, rows, deserialize_primitive_into),
        DataType::Int64 => primitive_dispatch::<_, i64>(target, rows, deserialize_primitive_into),
        DataType::UInt8 => primitive_dispatch::<_, u8>(target, rows, deserialize_primitive_into),
        DataType::UInt16 => primitive_dispatch::<_, u16>(target, rows, deserialize_primitive_into),
        DataType::UInt32 => primitive_dispatch::<_, u32>(target, rows, deserialize_primitive_into),
        DataType::UInt64 => primitive_dispatch::<_, u64>(target, rows, deserialize_primitive_into),
        DataType::LargeUtf8 => generic_deserialize_into::<_, MutableUtf8Array<i64>>(
            target,
            rows,
            deserialize_utf8_into,
        ),
        DataType::LargeList(_) => deserialize_list_into(
            target
                .as_mut_any()
                .downcast_mut::<MutableListArray<i64, Box<dyn MutableArray>>>()
                .unwrap(),
            rows,
        ),
        _ => {
            todo!()
        },
    }
}

fn deserialize_struct<'a, A: Borrow<BorrowedValue<'a>>>(
    rows: &[A],
    data_type: DataType,
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
    data_type: DataType,
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
    data_type: DataType,
) -> Box<dyn Array> {
    match &data_type {
        DataType::Null => Box::new(NullArray::new(data_type, rows.len())),
        DataType::Boolean => {
            fill_generic_array_from::<_, _, BooleanArray>(deserialize_boolean_into, rows)
        },
        DataType::Int8 => {
            fill_array_from::<_, _, PrimitiveArray<i8>>(deserialize_primitive_into, data_type, rows)
        },
        DataType::Int16 => fill_array_from::<_, _, PrimitiveArray<i16>>(
            deserialize_primitive_into,
            data_type,
            rows,
        ),
        DataType::Int32
        | DataType::Date32
        | DataType::Time32(_)
        | DataType::Interval(IntervalUnit::YearMonth) => {
            fill_array_from::<_, _, PrimitiveArray<i32>>(
                deserialize_primitive_into,
                data_type,
                rows,
            )
        },
        DataType::Interval(IntervalUnit::DayTime) => {
            unimplemented!("There is no natural representation of DayTime in JSON.")
        },
        DataType::Int64 | DataType::Date64 | DataType::Time64(_) | DataType::Duration(_) => {
            fill_array_from::<_, _, PrimitiveArray<i64>>(
                deserialize_primitive_into,
                data_type,
                rows,
            )
        },
        DataType::Timestamp(tu, tz) => {
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
        DataType::UInt8 => {
            fill_array_from::<_, _, PrimitiveArray<u8>>(deserialize_primitive_into, data_type, rows)
        },
        DataType::UInt16 => fill_array_from::<_, _, PrimitiveArray<u16>>(
            deserialize_primitive_into,
            data_type,
            rows,
        ),
        DataType::UInt32 => fill_array_from::<_, _, PrimitiveArray<u32>>(
            deserialize_primitive_into,
            data_type,
            rows,
        ),
        DataType::UInt64 => fill_array_from::<_, _, PrimitiveArray<u64>>(
            deserialize_primitive_into,
            data_type,
            rows,
        ),
        DataType::Float16 => unreachable!(),
        DataType::Float32 => fill_array_from::<_, _, PrimitiveArray<f32>>(
            deserialize_primitive_into,
            data_type,
            rows,
        ),
        DataType::Float64 => fill_array_from::<_, _, PrimitiveArray<f64>>(
            deserialize_primitive_into,
            data_type,
            rows,
        ),
        DataType::LargeUtf8 => {
            fill_generic_array_from::<_, _, Utf8Array<i64>>(deserialize_utf8_into, rows)
        },
        DataType::LargeList(_) => Box::new(deserialize_list(rows, data_type)),
        DataType::LargeBinary => Box::new(deserialize_binary(rows)),
        DataType::Struct(_) => Box::new(deserialize_struct(rows, data_type)),
        _ => todo!(),
    }
}

pub fn deserialize(json: &BorrowedValue, data_type: DataType) -> PolarsResult<Box<dyn Array>> {
    match json {
        BorrowedValue::Array(rows) => match data_type {
            DataType::LargeList(inner) => Ok(_deserialize(rows, inner.data_type)),
            _ => todo!("read an Array from a non-Array data type"),
        },
        _ => Ok(_deserialize(&[json], data_type)),
    }
}

fn allocate_array(f: &Field) -> Box<dyn MutableArray> {
    match f.data_type() {
        DataType::Int8 => Box::new(MutablePrimitiveArray::<i8>::new()),
        DataType::Int16 => Box::new(MutablePrimitiveArray::<i16>::new()),
        DataType::Int32 => Box::new(MutablePrimitiveArray::<i32>::new()),
        DataType::Int64 => Box::new(MutablePrimitiveArray::<i64>::new()),
        DataType::UInt8 => Box::new(MutablePrimitiveArray::<u8>::new()),
        DataType::UInt16 => Box::new(MutablePrimitiveArray::<u16>::new()),
        DataType::UInt32 => Box::new(MutablePrimitiveArray::<u32>::new()),
        DataType::UInt64 => Box::new(MutablePrimitiveArray::<u64>::new()),
        DataType::Float16 => Box::new(MutablePrimitiveArray::<f16>::new()),
        DataType::Float32 => Box::new(MutablePrimitiveArray::<f32>::new()),
        DataType::Float64 => Box::new(MutablePrimitiveArray::<f64>::new()),
        DataType::LargeList(inner) => match inner.data_type() {
            DataType::LargeList(_) => Box::new(MutableListArray::<i64, _>::new_from(
                allocate_array(inner),
                inner.data_type().clone(),
                0,
            )),
            _ => allocate_array(inner),
        },
        _ => todo!(),
    }
}

/// Deserializes a `json` [`simd_json::value::Value`] serialized in Pandas record format into
/// a [`Chunk`].
///
/// Uses the `Schema` provided, which can be inferred from arbitrary JSON with
/// [`infer_records_schema`].
///
/// This is CPU-bounded.
///
/// # Errors
///
/// This function errors iff either:
///
/// * `json` is not an [`Array`]
/// * `data_type` contains any incompatible types:
///   * [`DataType::Struct`]
///   * [`DataType::Dictionary`]
///   * [`DataType::LargeList`]
pub fn deserialize_records(
    json: &BorrowedValue,
    schema: &ArrowSchema,
) -> PolarsResult<Chunk<ArrayRef>> {
    let mut results = schema
        .fields
        .iter()
        .map(|f| (f.name.as_str(), allocate_array(f)))
        .collect::<PlHashMap<_, _>>();

    match json {
        BorrowedValue::Array(rows) => {
            for row in rows.iter() {
                match row {
                    BorrowedValue::Object(record) => {
                        for (key, value) in record.iter() {
                            let arr = results.get_mut(key.as_ref()).ok_or_else(|| {
                                PolarsError::ComputeError(format!("unexpected key: '{key}'").into())
                            })?;
                            deserialize_into(arr, &[value]);
                        }
                    },
                    _ => {
                        return Err(PolarsError::ComputeError(
                            "each row must be an Object".into(),
                        ))
                    },
                }
            }
        },
        _ => {
            return Err(PolarsError::ComputeError(
                "outer type must be an Array".into(),
            ))
        },
    }

    Ok(Chunk::new(
        results.into_values().map(|mut ma| ma.as_box()).collect(),
    ))
}
