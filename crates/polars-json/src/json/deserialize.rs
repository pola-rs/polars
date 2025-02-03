use std::borrow::Borrow;
use std::fmt::Write;

use arrow::array::*;
use arrow::bitmap::BitmapBuilder;
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
) -> PolarsResult<()> {
    let mut err_idx = rows.len();
    let iter = rows.iter().enumerate().map(|(i, row)| match row.borrow() {
        BorrowedValue::Static(StaticNode::Bool(v)) => Some(v),
        BorrowedValue::Static(StaticNode::Null) => None,
        _ => {
            err_idx = if err_idx == rows.len() { i } else { err_idx };
            None
        },
    });
    target.extend_trusted_len(iter);
    check_err_idx(rows, err_idx, "boolean")
}

fn deserialize_primitive_into<'a, T: NativeType + NumCast, A: Borrow<BorrowedValue<'a>>>(
    target: &mut MutablePrimitiveArray<T>,
    rows: &[A],
) -> PolarsResult<()> {
    let mut err_idx = rows.len();
    let iter = rows.iter().enumerate().map(|(i, row)| match row.borrow() {
        BorrowedValue::Static(StaticNode::I64(v)) => T::from(*v),
        BorrowedValue::Static(StaticNode::U64(v)) => T::from(*v),
        BorrowedValue::Static(StaticNode::F64(v)) => T::from(*v),
        BorrowedValue::Static(StaticNode::Bool(v)) => T::from(*v as u8),
        BorrowedValue::Static(StaticNode::Null) => None,
        _ => {
            err_idx = if err_idx == rows.len() { i } else { err_idx };
            None
        },
    });
    target.extend_trusted_len(iter);
    check_err_idx(rows, err_idx, "numeric")
}

fn deserialize_binary<'a, A: Borrow<BorrowedValue<'a>>>(
    rows: &[A],
) -> PolarsResult<BinaryArray<i64>> {
    let mut err_idx = rows.len();
    let iter = rows.iter().enumerate().map(|(i, row)| match row.borrow() {
        BorrowedValue::String(v) => Some(v.as_bytes()),
        BorrowedValue::Static(StaticNode::Null) => None,
        _ => {
            err_idx = if err_idx == rows.len() { i } else { err_idx };
            None
        },
    });
    let out = BinaryArray::from_trusted_len_iter(iter);
    check_err_idx(rows, err_idx, "binary")?;
    Ok(out)
}

fn deserialize_utf8_into<'a, O: Offset, A: Borrow<BorrowedValue<'a>>>(
    target: &mut MutableUtf8Array<O>,
    rows: &[A],
) -> PolarsResult<()> {
    let mut err_idx = rows.len();
    let mut scratch = String::new();
    for (i, row) in rows.iter().enumerate() {
        match row.borrow() {
            BorrowedValue::String(v) => target.push(Some(v.as_ref())),
            BorrowedValue::Static(StaticNode::Bool(v)) => {
                target.push(Some(if *v { "true" } else { "false" }))
            },
            BorrowedValue::Static(StaticNode::Null) => target.push_null(),
            BorrowedValue::Static(node) => {
                write!(scratch, "{node}").unwrap();
                target.push(Some(scratch.as_str()));
                scratch.clear();
            },
            _ => {
                err_idx = if err_idx == rows.len() { i } else { err_idx };
            },
        }
    }
    check_err_idx(rows, err_idx, "string")
}

fn deserialize_utf8view_into<'a, A: Borrow<BorrowedValue<'a>>>(
    target: &mut MutableBinaryViewArray<str>,
    rows: &[A],
) -> PolarsResult<()> {
    let mut err_idx = rows.len();
    let mut scratch = String::new();
    for (i, row) in rows.iter().enumerate() {
        match row.borrow() {
            BorrowedValue::String(v) => target.push_value(v.as_ref()),
            BorrowedValue::Static(StaticNode::Bool(v)) => {
                target.push_value(if *v { "true" } else { "false" })
            },
            BorrowedValue::Static(StaticNode::Null) => target.push_null(),
            BorrowedValue::Static(node) => {
                write!(scratch, "{node}").unwrap();
                target.push_value(scratch.as_str());
                scratch.clear();
            },
            _ => {
                err_idx = if err_idx == rows.len() { i } else { err_idx };
            },
        }
    }
    check_err_idx(rows, err_idx, "string")
}

fn deserialize_list<'a, A: Borrow<BorrowedValue<'a>>>(
    rows: &[A],
    dtype: ArrowDataType,
    allow_extra_fields_in_struct: bool,
) -> PolarsResult<ListArray<i64>> {
    let mut err_idx = rows.len();
    let child = ListArray::<i64>::get_child_type(&dtype);

    let mut validity = BitmapBuilder::with_capacity(rows.len());
    let mut offsets = Offsets::<i64>::with_capacity(rows.len());
    let mut inner = vec![];
    rows.iter()
        .enumerate()
        .for_each(|(i, row)| match row.borrow() {
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
                err_idx = if err_idx == rows.len() { i } else { err_idx };
            },
        });

    check_err_idx(rows, err_idx, "list")?;

    let values = _deserialize(&inner, child.clone(), allow_extra_fields_in_struct)?;

    Ok(ListArray::<i64>::new(
        dtype,
        offsets.into(),
        values,
        validity.into_opt_validity(),
    ))
}

fn deserialize_struct<'a, A: Borrow<BorrowedValue<'a>>>(
    rows: &[A],
    dtype: ArrowDataType,
    allow_extra_fields_in_struct: bool,
) -> PolarsResult<StructArray> {
    let mut err_idx = rows.len();
    let fields = StructArray::get_fields(&dtype);

    let mut out_values = fields
        .iter()
        .map(|f| (f.name.as_str(), (f.dtype(), vec![])))
        .collect::<PlHashMap<_, _>>();

    let mut validity = BitmapBuilder::with_capacity(rows.len());
    // Custom error tracker
    let mut extra_field = None;

    rows.iter().enumerate().for_each(|(i, row)| {
        match row.borrow() {
            BorrowedValue::Object(values) => {
                let mut n_matched = 0usize;
                for (&key, &mut (_, ref mut inner)) in out_values.iter_mut() {
                    if let Some(v) = values.get(key) {
                        n_matched += 1;
                        inner.push(v)
                    } else {
                        inner.push(&JSON_NULL_VALUE)
                    }
                }

                validity.push(true);

                if n_matched < values.len() && extra_field.is_none() {
                    for k in values.keys() {
                        if !out_values.contains_key(k.as_ref()) {
                            extra_field = Some(k.as_ref())
                        }
                    }
                }
            },
            BorrowedValue::Static(StaticNode::Null) => {
                out_values
                    .iter_mut()
                    .for_each(|(_, (_, inner))| inner.push(&JSON_NULL_VALUE));
                validity.push(false);
            },
            _ => {
                err_idx = if err_idx == rows.len() { i } else { err_idx };
            },
        };
    });

    if let Some(v) = extra_field {
        if !allow_extra_fields_in_struct {
            polars_bail!(
               ComputeError:
               "extra field in struct data: {}, consider increasing infer_schema_length, or \
               manually specifying the full schema to ignore extra fields",
               v
            )
        }
    }

    check_err_idx(rows, err_idx, "struct")?;

    // ensure we collect in the proper order
    let values = fields
        .iter()
        .map(|fld| {
            let (dtype, vals) = out_values.get(fld.name.as_str()).unwrap();
            _deserialize(vals, (*dtype).clone(), allow_extra_fields_in_struct)
        })
        .collect::<PolarsResult<Vec<_>>>()?;

    Ok(StructArray::new(
        dtype.clone(),
        rows.len(),
        values,
        validity.into_opt_validity(),
    ))
}

fn fill_array_from<B, T, A>(
    f: fn(&mut MutablePrimitiveArray<T>, &[B]) -> PolarsResult<()>,
    dtype: ArrowDataType,
    rows: &[B],
) -> PolarsResult<Box<dyn Array>>
where
    T: NativeType,
    A: From<MutablePrimitiveArray<T>> + Array,
{
    let mut array = MutablePrimitiveArray::<T>::with_capacity(rows.len()).to(dtype);
    f(&mut array, rows)?;
    Ok(Box::new(A::from(array)))
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

fn fill_generic_array_from<B, M, A>(
    f: fn(&mut M, &[B]) -> PolarsResult<()>,
    rows: &[B],
) -> PolarsResult<Box<dyn Array>>
where
    M: Container,
    A: From<M> + Array,
{
    let mut array = M::with_capacity(rows.len());
    f(&mut array, rows)?;
    Ok(Box::new(A::from(array)))
}

pub(crate) fn _deserialize<'a, A: Borrow<BorrowedValue<'a>>>(
    rows: &[A],
    dtype: ArrowDataType,
    allow_extra_fields_in_struct: bool,
) -> PolarsResult<Box<dyn Array>> {
    match &dtype {
        ArrowDataType::Null => {
            if let Some(err_idx) = (0..rows.len())
                .find(|i| !matches!(rows[*i].borrow(), BorrowedValue::Static(StaticNode::Null)))
            {
                check_err_idx(rows, err_idx, "null")?;
            }

            Ok(Box::new(NullArray::new(dtype, rows.len())))
        },
        ArrowDataType::Boolean => {
            fill_generic_array_from::<_, _, BooleanArray>(deserialize_boolean_into, rows)
        },
        ArrowDataType::Int8 => {
            fill_array_from::<_, _, PrimitiveArray<i8>>(deserialize_primitive_into, dtype, rows)
        },
        ArrowDataType::Int16 => {
            fill_array_from::<_, _, PrimitiveArray<i16>>(deserialize_primitive_into, dtype, rows)
        },
        ArrowDataType::Int32
        | ArrowDataType::Date32
        | ArrowDataType::Time32(_)
        | ArrowDataType::Interval(IntervalUnit::YearMonth) => {
            fill_array_from::<_, _, PrimitiveArray<i32>>(deserialize_primitive_into, dtype, rows)
        },
        ArrowDataType::Interval(IntervalUnit::DayTime) => {
            unimplemented!("There is no natural representation of DayTime in JSON.")
        },
        ArrowDataType::Int64
        | ArrowDataType::Date64
        | ArrowDataType::Time64(_)
        | ArrowDataType::Duration(_) => {
            fill_array_from::<_, _, PrimitiveArray<i64>>(deserialize_primitive_into, dtype, rows)
        },
        ArrowDataType::Timestamp(tu, tz) => {
            let mut err_idx = rows.len();
            let iter = rows.iter().enumerate().map(|(i, row)| match row.borrow() {
                BorrowedValue::Static(StaticNode::I64(v)) => Some(*v),
                BorrowedValue::String(v) => match (tu, tz) {
                    (_, None) => {
                        polars_compute::cast::temporal::utf8_to_naive_timestamp_scalar(v, "%+", tu)
                    },
                    (_, Some(ref tz)) => {
                        let tz = temporal_conversions::parse_offset(tz.as_str()).unwrap();
                        temporal_conversions::utf8_to_timestamp_scalar(v, "%+", &tz, tu)
                    },
                },
                BorrowedValue::Static(StaticNode::Null) => None,
                _ => {
                    err_idx = if err_idx == rows.len() { i } else { err_idx };
                    None
                },
            });
            let out = Box::new(Int64Array::from_iter(iter).to(dtype));
            check_err_idx(rows, err_idx, "timestamp")?;
            Ok(out)
        },
        ArrowDataType::UInt8 => {
            fill_array_from::<_, _, PrimitiveArray<u8>>(deserialize_primitive_into, dtype, rows)
        },
        ArrowDataType::UInt16 => {
            fill_array_from::<_, _, PrimitiveArray<u16>>(deserialize_primitive_into, dtype, rows)
        },
        ArrowDataType::UInt32 => {
            fill_array_from::<_, _, PrimitiveArray<u32>>(deserialize_primitive_into, dtype, rows)
        },
        ArrowDataType::UInt64 => {
            fill_array_from::<_, _, PrimitiveArray<u64>>(deserialize_primitive_into, dtype, rows)
        },
        ArrowDataType::Float16 => unreachable!(),
        ArrowDataType::Float32 => {
            fill_array_from::<_, _, PrimitiveArray<f32>>(deserialize_primitive_into, dtype, rows)
        },
        ArrowDataType::Float64 => {
            fill_array_from::<_, _, PrimitiveArray<f64>>(deserialize_primitive_into, dtype, rows)
        },
        ArrowDataType::LargeUtf8 => {
            fill_generic_array_from::<_, _, Utf8Array<i64>>(deserialize_utf8_into, rows)
        },
        ArrowDataType::Utf8View => {
            fill_generic_array_from::<_, _, Utf8ViewArray>(deserialize_utf8view_into, rows)
        },
        ArrowDataType::LargeList(_) => Ok(Box::new(deserialize_list(
            rows,
            dtype,
            allow_extra_fields_in_struct,
        )?)),
        ArrowDataType::LargeBinary => Ok(Box::new(deserialize_binary(rows)?)),
        ArrowDataType::Struct(_) => Ok(Box::new(deserialize_struct(
            rows,
            dtype,
            allow_extra_fields_in_struct,
        )?)),
        _ => todo!(),
    }
}

pub fn deserialize(
    json: &BorrowedValue,
    dtype: ArrowDataType,
    allow_extra_fields_in_struct: bool,
) -> PolarsResult<Box<dyn Array>> {
    match json {
        BorrowedValue::Array(rows) => match dtype {
            ArrowDataType::LargeList(inner) => {
                _deserialize(rows, inner.dtype, allow_extra_fields_in_struct)
            },
            _ => todo!("read an Array from a non-Array data type"),
        },
        _ => _deserialize(&[json], dtype, allow_extra_fields_in_struct),
    }
}

fn check_err_idx<'a>(
    rows: &[impl Borrow<BorrowedValue<'a>>],
    err_idx: usize,
    type_name: &'static str,
) -> PolarsResult<()> {
    if err_idx != rows.len() {
        polars_bail!(
            ComputeError:
            r#"error deserializing value "{:?}" as {}. \
            Try increasing `infer_schema_length` or specifying a schema.
            "#,
            rows[err_idx].borrow(), type_name,
        )
    }

    Ok(())
}
