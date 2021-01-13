use crate::prelude::*;
use arrow::compute::kernels::arithmetic::{divide, multiply};
use arrow::datatypes::ToByteSlice;
use arrow::error::{ArrowError, Result};
use arrow::{array::*, buffer::Buffer};

/// Cast `array` to the provided data type and return a new Array with
/// type `to_type`, if possible.
///
/// Behavior:
/// * Boolean to Utf8: `true` => '1', `false` => `0`
/// * Utf8 to numeric: strings that can't be parsed to numbers return null, float strings
///   in integer casts return null
/// * Numeric to boolean: 0 returns `false`, any other value returns `true`
/// * List to List: the underlying data type is cast
/// * Primitive to List: a list array with 1 value per slot is created
/// * Date32 and Date64: precision lost when going to higher interval
/// * Time32 and Time64: precision lost when going to higher interval
/// * Timestamp and Date{32|64}: precision lost when going to higher interval
/// * Temporal to/from backing primitive: zero-copy with data type change
///
/// Unsupported Casts
/// * To or from `StructArray`
/// * List to primitive
/// * Utf8 to boolean
/// * Interval and duration
pub fn cast(array: &ArrayRef, to_type: &ArrowDataType) -> Result<ArrayRef> {
    use ArrowDataType::*;
    let from_type = array.data_type();

    // clone array if types are the same
    if from_type == to_type {
        return Ok(array.clone());
    }
    match (from_type, to_type) {
        (LargeList(_), LargeList(ref to)) => {
            let data = array.data_ref();
            let underlying_array = make_array(data.child_data()[0].clone());
            let cast_array = cast(&underlying_array, &to)?;
            let array_data = ArrayData::new(
                *to.clone(),
                array.len(),
                Some(cast_array.null_count()),
                cast_array
                    .data()
                    .null_bitmap()
                    .clone()
                    .map(|bitmap| bitmap.into_buffer()),
                array.offset(),
                // reuse offset buffer
                data.buffers().to_vec(),
                vec![cast_array.data()],
            );
            let list = ListArray::from(Arc::new(array_data));
            Ok(Arc::new(list) as ArrayRef)
        }
        (List(_), LargeList(_)) => {
            // TODO! test this
            let data = array.data_ref();
            let underlying_array = data.child_data()[0].clone();

            let offset_ptr = data.buffers()[0].raw_data() as *const i32;
            // offsets in the list array. These indicate where a new list starts
            let offsets = unsafe { std::slice::from_raw_parts(offset_ptr, array.len()) };
            let mut offset_builder = Int64BufferBuilder::new(offsets.len());

            offsets
                .iter()
                .for_each(|v| offset_builder.append(*v as i64).unwrap());
            let offset_buffer = offset_builder.finish();
            offset_builder.append(0).unwrap();

            let mut builder = ArrayData::builder(ArrowDataType::LargeList(Box::new(
                underlying_array.data_type().clone(),
            )))
            .len(array.len())
            .null_count(array.null_count())
            .add_buffer(offset_buffer)
            .add_child_data(underlying_array);

            if let Some(buf) = data.null_buffer() {
                builder = builder.null_bit_buffer(buf.clone())
            }
            let data = builder.build();
            Ok(Arc::new(LargeListArray::from(data)))
        }
        (Utf8, LargeUtf8) => {
            let list_data = array.data();
            let str_values_buf = list_data.child_data()[0].buffers()[0].clone();
            // We get the offsets of the strings in the original array
            let offset_ptr = list_data.buffers()[0].raw_data() as *const i32;
            // offsets in the list array. These indicate where a new list starts
            let offsets = unsafe { std::slice::from_raw_parts(offset_ptr, array.len()) };

            let mut offset_builder = Int64BufferBuilder::new(offsets.len());

            offsets
                .iter()
                .for_each(|v| offset_builder.append(*v as i64).unwrap());
            let offset_buffer = offset_builder.finish();
            offset_builder.append(0).unwrap();

            let mut builder = ArrayData::builder(ArrowDataType::LargeUtf8)
                .len(array.len())
                .add_buffer(offset_buffer)
                .add_buffer(str_values_buf);

            if let Some(buf) = list_data.null_buffer() {
                builder = builder.null_bit_buffer(buf.clone())
            }
            let data = builder.build();
            Ok(Arc::new(LargeStringArray::from(data)))
        }
        (LargeList(_), _) => Err(ArrowError::ComputeError(
            "Cannot cast list to non-list data types".to_string(),
        )),
        (_, LargeList(ref to)) => {
            // cast primitive to list's primitive
            let cast_array = cast(array, &to)?;
            // create offsets, where if array.len() = 2, we have [0,1,2]

            // Todo! use alignedvec
            let offsets: Vec<i64> = (0..=array.len() as i64).collect();
            let value_offsets = Buffer::from(offsets[..].to_byte_slice());
            let list_data = ArrayData::new(
                *to.clone(),
                array.len(),
                Some(cast_array.null_count()),
                cast_array
                    .data()
                    .null_bitmap()
                    .clone()
                    .map(|bitmap| bitmap.into_buffer()),
                0,
                vec![value_offsets],
                vec![cast_array.data()],
            );
            let list_array = Arc::new(LargeListArray::from(Arc::new(list_data))) as ArrayRef;

            Ok(list_array)
        }
        (_, Boolean) => match from_type {
            UInt8 => cast_numeric_to_bool::<UInt8Type>(array),
            UInt16 => cast_numeric_to_bool::<UInt16Type>(array),
            UInt32 => cast_numeric_to_bool::<UInt32Type>(array),
            UInt64 => cast_numeric_to_bool::<UInt64Type>(array),
            Int8 => cast_numeric_to_bool::<Int8Type>(array),
            Int16 => cast_numeric_to_bool::<Int16Type>(array),
            Int32 => cast_numeric_to_bool::<Int32Type>(array),
            Int64 => cast_numeric_to_bool::<Int64Type>(array),
            Float32 => cast_numeric_to_bool::<Float32Type>(array),
            Float64 => cast_numeric_to_bool::<Float64Type>(array),
            Utf8 => Err(ArrowError::ComputeError(format!(
                "Casting from {:?} to {:?} not supported",
                from_type, to_type,
            ))),
            _ => Err(ArrowError::ComputeError(format!(
                "Casting from {:?} to {:?} not supported",
                from_type, to_type,
            ))),
        },
        (Boolean, _) => match to_type {
            UInt8 => cast_bool_to_numeric::<UInt8Type>(array),
            UInt16 => cast_bool_to_numeric::<UInt16Type>(array),
            UInt32 => cast_bool_to_numeric::<UInt32Type>(array),
            UInt64 => cast_bool_to_numeric::<UInt64Type>(array),
            Int8 => cast_bool_to_numeric::<Int8Type>(array),
            Int16 => cast_bool_to_numeric::<Int16Type>(array),
            Int32 => cast_bool_to_numeric::<Int32Type>(array),
            Int64 => cast_bool_to_numeric::<Int64Type>(array),
            Float32 => cast_bool_to_numeric::<Float32Type>(array),
            Float64 => cast_bool_to_numeric::<Float64Type>(array),
            LargeUtf8 => {
                let from = array.as_any().downcast_ref::<BooleanArray>().unwrap();
                let mut b = LargeStringBuilder::new(array.len());
                for i in 0..array.len() {
                    if array.is_null(i) {
                        b.append(false)?;
                    } else {
                        b.append_value(if from.value(i) { "1" } else { "0" })?;
                    }
                }

                Ok(Arc::new(b.finish()) as ArrayRef)
            }
            _ => Err(ArrowError::ComputeError(format!(
                "Casting from {:?} to {:?} not supported",
                from_type, to_type,
            ))),
        },
        (LargeUtf8, _) => match to_type {
            UInt8 => cast_string_to_numeric::<UInt8Type>(array),
            UInt16 => cast_string_to_numeric::<UInt16Type>(array),
            UInt32 => cast_string_to_numeric::<UInt32Type>(array),
            UInt64 => cast_string_to_numeric::<UInt64Type>(array),
            Int8 => cast_string_to_numeric::<Int8Type>(array),
            Int16 => cast_string_to_numeric::<Int16Type>(array),
            Int32 => cast_string_to_numeric::<Int32Type>(array),
            Int64 => cast_string_to_numeric::<Int64Type>(array),
            Float32 => cast_string_to_numeric::<Float32Type>(array),
            Float64 => cast_string_to_numeric::<Float64Type>(array),
            _ => Err(ArrowError::ComputeError(format!(
                "Casting from {:?} to {:?} not supported",
                from_type, to_type,
            ))),
        },
        (_, LargeUtf8) => match from_type {
            UInt8 => cast_numeric_to_string::<UInt8Type>(array),
            UInt16 => cast_numeric_to_string::<UInt16Type>(array),
            UInt32 => cast_numeric_to_string::<UInt32Type>(array),
            UInt64 => cast_numeric_to_string::<UInt64Type>(array),
            Int8 => cast_numeric_to_string::<Int8Type>(array),
            Int16 => cast_numeric_to_string::<Int16Type>(array),
            Int32 => cast_numeric_to_string::<Int32Type>(array),
            Int64 => cast_numeric_to_string::<Int64Type>(array),
            Float32 => cast_numeric_to_string::<Float32Type>(array),
            Float64 => cast_numeric_to_string::<Float64Type>(array),
            _ => Err(ArrowError::ComputeError(format!(
                "Casting from {:?} to {:?} not supported",
                from_type, to_type,
            ))),
        },

        // start numeric casts
        (UInt8, UInt16) => cast_numeric_arrays::<UInt8Type, UInt16Type>(array),
        (UInt8, UInt32) => cast_numeric_arrays::<UInt8Type, UInt32Type>(array),
        (UInt8, UInt64) => cast_numeric_arrays::<UInt8Type, UInt64Type>(array),
        (UInt8, Int8) => cast_numeric_arrays::<UInt8Type, Int8Type>(array),
        (UInt8, Int16) => cast_numeric_arrays::<UInt8Type, Int16Type>(array),
        (UInt8, Int32) => cast_numeric_arrays::<UInt8Type, Int32Type>(array),
        (UInt8, Int64) => cast_numeric_arrays::<UInt8Type, Int64Type>(array),
        (UInt8, Float32) => cast_numeric_arrays::<UInt8Type, Float32Type>(array),
        (UInt8, Float64) => cast_numeric_arrays::<UInt8Type, Float64Type>(array),

        (UInt16, UInt8) => cast_numeric_arrays::<UInt16Type, UInt8Type>(array),
        (UInt16, UInt32) => cast_numeric_arrays::<UInt16Type, UInt32Type>(array),
        (UInt16, UInt64) => cast_numeric_arrays::<UInt16Type, UInt64Type>(array),
        (UInt16, Int8) => cast_numeric_arrays::<UInt16Type, Int8Type>(array),
        (UInt16, Int16) => cast_numeric_arrays::<UInt16Type, Int16Type>(array),
        (UInt16, Int32) => cast_numeric_arrays::<UInt16Type, Int32Type>(array),
        (UInt16, Int64) => cast_numeric_arrays::<UInt16Type, Int64Type>(array),
        (UInt16, Float32) => cast_numeric_arrays::<UInt16Type, Float32Type>(array),
        (UInt16, Float64) => cast_numeric_arrays::<UInt16Type, Float64Type>(array),

        (UInt32, UInt8) => cast_numeric_arrays::<UInt32Type, UInt8Type>(array),
        (UInt32, UInt16) => cast_numeric_arrays::<UInt32Type, UInt16Type>(array),
        (UInt32, UInt64) => cast_numeric_arrays::<UInt32Type, UInt64Type>(array),
        (UInt32, Int8) => cast_numeric_arrays::<UInt32Type, Int8Type>(array),
        (UInt32, Int16) => cast_numeric_arrays::<UInt32Type, Int16Type>(array),
        (UInt32, Int32) => cast_numeric_arrays::<UInt32Type, Int32Type>(array),
        (UInt32, Int64) => cast_numeric_arrays::<UInt32Type, Int64Type>(array),
        (UInt32, Float32) => cast_numeric_arrays::<UInt32Type, Float32Type>(array),
        (UInt32, Float64) => cast_numeric_arrays::<UInt32Type, Float64Type>(array),

        (UInt64, UInt8) => cast_numeric_arrays::<UInt64Type, UInt8Type>(array),
        (UInt64, UInt16) => cast_numeric_arrays::<UInt64Type, UInt16Type>(array),
        (UInt64, UInt32) => cast_numeric_arrays::<UInt64Type, UInt32Type>(array),
        (UInt64, Int8) => cast_numeric_arrays::<UInt64Type, Int8Type>(array),
        (UInt64, Int16) => cast_numeric_arrays::<UInt64Type, Int16Type>(array),
        (UInt64, Int32) => cast_numeric_arrays::<UInt64Type, Int32Type>(array),
        (UInt64, Int64) => cast_numeric_arrays::<UInt64Type, Int64Type>(array),
        (UInt64, Float32) => cast_numeric_arrays::<UInt64Type, Float32Type>(array),
        (UInt64, Float64) => cast_numeric_arrays::<UInt64Type, Float64Type>(array),

        (Int8, UInt8) => cast_numeric_arrays::<Int8Type, UInt8Type>(array),
        (Int8, UInt16) => cast_numeric_arrays::<Int8Type, UInt16Type>(array),
        (Int8, UInt32) => cast_numeric_arrays::<Int8Type, UInt32Type>(array),
        (Int8, UInt64) => cast_numeric_arrays::<Int8Type, UInt64Type>(array),
        (Int8, Int16) => cast_numeric_arrays::<Int8Type, Int16Type>(array),
        (Int8, Int32) => cast_numeric_arrays::<Int8Type, Int32Type>(array),
        (Int8, Int64) => cast_numeric_arrays::<Int8Type, Int64Type>(array),
        (Int8, Float32) => cast_numeric_arrays::<Int8Type, Float32Type>(array),
        (Int8, Float64) => cast_numeric_arrays::<Int8Type, Float64Type>(array),

        (Int16, UInt8) => cast_numeric_arrays::<Int16Type, UInt8Type>(array),
        (Int16, UInt16) => cast_numeric_arrays::<Int16Type, UInt16Type>(array),
        (Int16, UInt32) => cast_numeric_arrays::<Int16Type, UInt32Type>(array),
        (Int16, UInt64) => cast_numeric_arrays::<Int16Type, UInt64Type>(array),
        (Int16, Int8) => cast_numeric_arrays::<Int16Type, Int8Type>(array),
        (Int16, Int32) => cast_numeric_arrays::<Int16Type, Int32Type>(array),
        (Int16, Int64) => cast_numeric_arrays::<Int16Type, Int64Type>(array),
        (Int16, Float32) => cast_numeric_arrays::<Int16Type, Float32Type>(array),
        (Int16, Float64) => cast_numeric_arrays::<Int16Type, Float64Type>(array),

        (Int32, UInt8) => cast_numeric_arrays::<Int32Type, UInt8Type>(array),
        (Int32, UInt16) => cast_numeric_arrays::<Int32Type, UInt16Type>(array),
        (Int32, UInt32) => cast_numeric_arrays::<Int32Type, UInt32Type>(array),
        (Int32, UInt64) => cast_numeric_arrays::<Int32Type, UInt64Type>(array),
        (Int32, Int8) => cast_numeric_arrays::<Int32Type, Int8Type>(array),
        (Int32, Int16) => cast_numeric_arrays::<Int32Type, Int16Type>(array),
        (Int32, Int64) => cast_numeric_arrays::<Int32Type, Int64Type>(array),
        (Int32, Float32) => cast_numeric_arrays::<Int32Type, Float32Type>(array),
        (Int32, Float64) => cast_numeric_arrays::<Int32Type, Float64Type>(array),

        (Int64, UInt8) => cast_numeric_arrays::<Int64Type, UInt8Type>(array),
        (Int64, UInt16) => cast_numeric_arrays::<Int64Type, UInt16Type>(array),
        (Int64, UInt32) => cast_numeric_arrays::<Int64Type, UInt32Type>(array),
        (Int64, UInt64) => cast_numeric_arrays::<Int64Type, UInt64Type>(array),
        (Int64, Int8) => cast_numeric_arrays::<Int64Type, Int8Type>(array),
        (Int64, Int16) => cast_numeric_arrays::<Int64Type, Int16Type>(array),
        (Int64, Int32) => cast_numeric_arrays::<Int64Type, Int32Type>(array),
        (Int64, Float32) => cast_numeric_arrays::<Int64Type, Float32Type>(array),
        (Int64, Float64) => cast_numeric_arrays::<Int64Type, Float64Type>(array),

        (Float32, UInt8) => cast_numeric_arrays::<Float32Type, UInt8Type>(array),
        (Float32, UInt16) => cast_numeric_arrays::<Float32Type, UInt16Type>(array),
        (Float32, UInt32) => cast_numeric_arrays::<Float32Type, UInt32Type>(array),
        (Float32, UInt64) => cast_numeric_arrays::<Float32Type, UInt64Type>(array),
        (Float32, Int8) => cast_numeric_arrays::<Float32Type, Int8Type>(array),
        (Float32, Int16) => cast_numeric_arrays::<Float32Type, Int16Type>(array),
        (Float32, Int32) => cast_numeric_arrays::<Float32Type, Int32Type>(array),
        (Float32, Int64) => cast_numeric_arrays::<Float32Type, Int64Type>(array),
        (Float32, Float64) => cast_numeric_arrays::<Float32Type, Float64Type>(array),

        (Float64, UInt8) => cast_numeric_arrays::<Float64Type, UInt8Type>(array),
        (Float64, UInt16) => cast_numeric_arrays::<Float64Type, UInt16Type>(array),
        (Float64, UInt32) => cast_numeric_arrays::<Float64Type, UInt32Type>(array),
        (Float64, UInt64) => cast_numeric_arrays::<Float64Type, UInt64Type>(array),
        (Float64, Int8) => cast_numeric_arrays::<Float64Type, Int8Type>(array),
        (Float64, Int16) => cast_numeric_arrays::<Float64Type, Int16Type>(array),
        (Float64, Int32) => cast_numeric_arrays::<Float64Type, Int32Type>(array),
        (Float64, Int64) => cast_numeric_arrays::<Float64Type, Int64Type>(array),
        (Float64, Float32) => cast_numeric_arrays::<Float64Type, Float32Type>(array),
        // end numeric casts

        // temporal casts
        (Int32, Date32(_)) => cast_array_data::<Date32Type>(array, to_type.clone()),
        (Int32, Time32(_)) => cast_array_data::<Date32Type>(array, to_type.clone()),
        (Date32(_), Int32) => cast_array_data::<Int32Type>(array, to_type.clone()),
        (Time32(_), Int32) => cast_array_data::<Int32Type>(array, to_type.clone()),
        (Int64, Date64(_)) => cast_array_data::<Date64Type>(array, to_type.clone()),
        (Int64, Time64(_)) => cast_array_data::<Date64Type>(array, to_type.clone()),
        (Date64(_), Int64) => cast_array_data::<Int64Type>(array, to_type.clone()),
        (Time64(_), Int64) => cast_array_data::<Int64Type>(array, to_type.clone()),
        (Date32(DateUnit::Day), Date64(DateUnit::Millisecond)) => {
            let date_array = array.as_any().downcast_ref::<Date32Array>().unwrap();
            let mut b = Date64Builder::new(array.len());
            for i in 0..array.len() {
                if array.is_null(i) {
                    b.append_null()?;
                } else {
                    b.append_value(date_array.value(i) as i64 * MILLISECONDS_IN_DAY)?;
                }
            }

            Ok(Arc::new(b.finish()) as ArrayRef)
        }
        (Date64(DateUnit::Millisecond), Date32(DateUnit::Day)) => {
            let date_array = array.as_any().downcast_ref::<Date64Array>().unwrap();
            let mut b = Date32Builder::new(array.len());
            for i in 0..array.len() {
                if array.is_null(i) {
                    b.append_null()?;
                } else {
                    b.append_value((date_array.value(i) / MILLISECONDS_IN_DAY) as i32)?;
                }
            }

            Ok(Arc::new(b.finish()) as ArrayRef)
        }
        (Time32(TimeUnit::Second), Time32(TimeUnit::Millisecond)) => {
            let time_array = Time32MillisecondArray::from(array.data());
            let mult = Time32MillisecondArray::from(vec![MILLISECONDS as i32; array.len()]);
            let time32_ms = multiply(&time_array, &mult)?;

            Ok(Arc::new(time32_ms) as ArrayRef)
        }
        (Time32(TimeUnit::Millisecond), Time32(TimeUnit::Second)) => {
            let time_array = Time32SecondArray::from(array.data());
            let divisor = Time32SecondArray::from(vec![MILLISECONDS as i32; array.len()]);
            let time32_s = divide(&time_array, &divisor)?;

            Ok(Arc::new(time32_s) as ArrayRef)
        }
        (Time32(from_unit), Time64(to_unit)) => {
            let time_array = Int32Array::from(array.data());
            // note: (numeric_cast + SIMD multiply) is faster than (cast & multiply)
            let c: Int64Array = numeric_cast(&time_array);
            let from_size = time_unit_multiple(&from_unit);
            let to_size = time_unit_multiple(&to_unit);
            // from is only smaller than to if 64milli/64second don't exist
            let mult = Int64Array::from(vec![to_size / from_size; array.len()]);
            let converted = multiply(&c, &mult)?;
            let array_ref = Arc::new(converted) as ArrayRef;
            use TimeUnit::*;
            match to_unit {
                Microsecond => {
                    cast_array_data::<TimestampMicrosecondType>(&array_ref, to_type.clone())
                }
                Nanosecond => {
                    cast_array_data::<TimestampNanosecondType>(&array_ref, to_type.clone())
                }
                _ => unreachable!("array type not supported"),
            }
        }
        (Time64(TimeUnit::Microsecond), Time64(TimeUnit::Nanosecond)) => {
            let time_array = Time64NanosecondArray::from(array.data());
            let mult = Time64NanosecondArray::from(vec![MILLISECONDS; array.len()]);
            let time64_ns = multiply(&time_array, &mult)?;

            Ok(Arc::new(time64_ns) as ArrayRef)
        }
        (Time64(TimeUnit::Nanosecond), Time64(TimeUnit::Microsecond)) => {
            let time_array = Time64MicrosecondArray::from(array.data());
            let divisor = Time64MicrosecondArray::from(vec![MILLISECONDS; array.len()]);
            let time64_us = divide(&time_array, &divisor)?;

            Ok(Arc::new(time64_us) as ArrayRef)
        }
        (Time64(from_unit), Time32(to_unit)) => {
            let time_array = Int64Array::from(array.data());
            let from_size = time_unit_multiple(&from_unit);
            let to_size = time_unit_multiple(&to_unit);
            let divisor = from_size / to_size;
            match to_unit {
                TimeUnit::Second => {
                    let mut b = Time32SecondBuilder::new(array.len());
                    for i in 0..array.len() {
                        if array.is_null(i) {
                            b.append_null()?;
                        } else {
                            b.append_value((time_array.value(i) as i64 / divisor) as i32)?;
                        }
                    }

                    Ok(Arc::new(b.finish()) as ArrayRef)
                }
                TimeUnit::Millisecond => {
                    // currently can't dedup this builder [ARROW-4164]
                    let mut b = Time32MillisecondBuilder::new(array.len());
                    for i in 0..array.len() {
                        if array.is_null(i) {
                            b.append_null()?;
                        } else {
                            b.append_value((time_array.value(i) as i64 / divisor) as i32)?;
                        }
                    }

                    Ok(Arc::new(b.finish()) as ArrayRef)
                }
                _ => unreachable!("array type not supported"),
            }
        }
        (Timestamp(_, _), Int64) => cast_array_data::<Int64Type>(array, to_type.clone()),
        (Int64, Timestamp(to_unit, _)) => {
            use TimeUnit::*;
            match to_unit {
                Second => cast_array_data::<TimestampSecondType>(array, to_type.clone()),
                Millisecond => cast_array_data::<TimestampMillisecondType>(array, to_type.clone()),
                Microsecond => cast_array_data::<TimestampMicrosecondType>(array, to_type.clone()),
                Nanosecond => cast_array_data::<TimestampNanosecondType>(array, to_type.clone()),
            }
        }
        (Timestamp(from_unit, _), Timestamp(to_unit, _)) => {
            let time_array = Int64Array::from(array.data());
            let from_size = time_unit_multiple(&from_unit);
            let to_size = time_unit_multiple(&to_unit);
            // we either divide or multiply, depending on size of each unit
            // units are never the same when the types are the same
            let converted = if from_size >= to_size {
                divide(
                    &time_array,
                    &Int64Array::from(vec![from_size / to_size; array.len()]),
                )?
            } else {
                multiply(
                    &time_array,
                    &Int64Array::from(vec![to_size / from_size; array.len()]),
                )?
            };
            let array_ref = Arc::new(converted) as ArrayRef;
            use TimeUnit::*;
            match to_unit {
                Second => cast_array_data::<TimestampSecondType>(&array_ref, to_type.clone()),
                Millisecond => {
                    cast_array_data::<TimestampMillisecondType>(&array_ref, to_type.clone())
                }
                Microsecond => {
                    cast_array_data::<TimestampMicrosecondType>(&array_ref, to_type.clone())
                }
                Nanosecond => {
                    cast_array_data::<TimestampNanosecondType>(&array_ref, to_type.clone())
                }
            }
        }
        (Timestamp(from_unit, _), Date32(_)) => {
            let time_array = Int64Array::from(array.data());
            let from_size = time_unit_multiple(&from_unit) * SECONDS_IN_DAY;
            let mut b = Date32Builder::new(array.len());
            for i in 0..array.len() {
                if array.is_null(i) {
                    b.append_null()?;
                } else {
                    b.append_value((time_array.value(i) / from_size) as i32)?;
                }
            }

            Ok(Arc::new(b.finish()) as ArrayRef)
        }
        (Timestamp(from_unit, _), Date64(_)) => {
            let from_size = time_unit_multiple(&from_unit);
            let to_size = MILLISECONDS;
            if from_size != to_size {
                let time_array = Date64Array::from(array.data());
                Ok(Arc::new(divide(
                    &time_array,
                    &Date64Array::from(vec![from_size / to_size; array.len()]),
                )?) as ArrayRef)
            } else {
                cast_array_data::<Date64Type>(array, to_type.clone())
            }
        }
        // date64 to timestamp might not make sense,

        // end temporal casts
        (_, _) => Err(ArrowError::ComputeError(format!(
            "Casting from {:?} to {:?} not supported",
            from_type, to_type,
        ))),
    }
}

/// Get the time unit as a multiple of a second
fn time_unit_multiple(unit: &TimeUnit) -> i64 {
    match unit {
        TimeUnit::Second => 1,
        TimeUnit::Millisecond => MILLISECONDS,
        TimeUnit::Microsecond => MICROSECONDS,
        TimeUnit::Nanosecond => NANOSECONDS,
    }
}

/// Number of seconds in a day
const SECONDS_IN_DAY: i64 = 86_400;
/// Number of milliseconds in a second
const MILLISECONDS: i64 = 1_000;
/// Number of microseconds in a second
const MICROSECONDS: i64 = 1_000_000;
/// Number of nanoseconds in a second
const NANOSECONDS: i64 = 1_000_000_000;
/// Number of milliseconds in a day
const MILLISECONDS_IN_DAY: i64 = SECONDS_IN_DAY * MILLISECONDS;

/// Cast an array by changing its array_data type to the desired type
///
/// Arrays should have the same primitive data type, otherwise this should fail.
/// We do not perform this check on primitive data types as we only use this
/// function internally, where it is guaranteed to be infallible.
#[allow(clippy::unnecessary_wraps)]
fn cast_array_data<TO>(array: &ArrayRef, to_type: ArrowDataType) -> Result<ArrayRef>
where
    TO: ArrowNumericType,
{
    let data = Arc::new(ArrayData::new(
        to_type,
        array.len(),
        Some(array.null_count()),
        array
            .data()
            .null_bitmap()
            .clone()
            .map(|bitmap| bitmap.into_buffer()),
        array.data().offset(),
        array.data().buffers().to_vec(),
        vec![],
    ));
    Ok(Arc::new(PrimitiveArray::<TO>::from(data)) as ArrayRef)
}

/// Convert Array into a PrimitiveArray of type, and apply numeric cast
#[allow(clippy::unnecessary_wraps)]
fn cast_numeric_arrays<FROM, TO>(from: &ArrayRef) -> Result<ArrayRef>
where
    FROM: ArrowNumericType,
    TO: ArrowNumericType,
    FROM::Native: num::NumCast,
    TO::Native: num::NumCast,
{
    Ok(Arc::new(numeric_cast::<FROM, TO>(
        from.as_any()
            .downcast_ref::<PrimitiveArray<FROM>>()
            .unwrap(),
    )))
}

/// Natural cast between numeric types
#[allow(clippy::unnecessary_wraps)]
fn numeric_cast<T, R>(from: &PrimitiveArray<T>) -> PrimitiveArray<R>
where
    T: ArrowNumericType,
    R: ArrowNumericType,
    T::Native: num::NumCast,
    R::Native: num::NumCast,
{
    from.iter()
        .map(|v| match v {
            Some(v) => num::cast::cast::<T::Native, R::Native>(v),
            None => None,
        })
        .collect()
}

/// Cast numeric types to Utf8
#[allow(clippy::unnecessary_wraps)]
fn cast_numeric_to_string<FROM>(array: &ArrayRef) -> Result<ArrayRef>
where
    FROM: ArrowNumericType,
    FROM::Native: std::string::ToString,
{
    numeric_to_string_cast::<FROM>(
        array
            .as_any()
            .downcast_ref::<PrimitiveArray<FROM>>()
            .unwrap(),
    )
    .map(|to| Arc::new(to) as ArrayRef)
}

#[allow(clippy::unnecessary_wraps)]
fn numeric_to_string_cast<T>(from: &PrimitiveArray<T>) -> Result<LargeStringArray>
where
    T: ArrowPrimitiveType + ArrowNumericType,
    T::Native: std::string::ToString,
{
    let mut b = LargeStringBuilder::new(from.len());

    for i in 0..from.len() {
        if from.is_null(i) {
            b.append(false)?;
        } else {
            b.append_value(&from.value(i).to_string())?;
        }
    }

    Ok(b.finish())
}

/// Cast numeric types to Utf8
#[allow(clippy::unnecessary_wraps)]
fn cast_string_to_numeric<TO>(from: &ArrayRef) -> Result<ArrayRef>
where
    TO: ArrowNumericType,
{
    Ok(Arc::new(string_to_numeric_cast::<TO>(
        from.as_any().downcast_ref::<LargeStringArray>().unwrap(),
    )))
}

fn string_to_numeric_cast<T>(from: &LargeStringArray) -> PrimitiveArray<T>
where
    T: ArrowNumericType,
{
    (0..from.len())
        .map(|i| {
            if from.is_null(i) {
                None
            } else {
                match from.value(i).parse::<T::Native>() {
                    Ok(v) => Some(v),
                    Err(_) => None,
                }
            }
        })
        .collect()
}

/// Cast numeric types to Boolean
///
/// Any zero value returns `false` while non-zero returns `true`
#[allow(clippy::unnecessary_wraps)]
fn cast_numeric_to_bool<FROM>(from: &ArrayRef) -> Result<ArrayRef>
where
    FROM: ArrowNumericType,
{
    numeric_to_bool_cast::<FROM>(
        from.as_any()
            .downcast_ref::<PrimitiveArray<FROM>>()
            .unwrap(),
    )
    .map(|to| Arc::new(to) as ArrayRef)
}

#[allow(clippy::unnecessary_wraps)]
fn numeric_to_bool_cast<T>(from: &PrimitiveArray<T>) -> Result<BooleanArray>
where
    T: ArrowPrimitiveType + ArrowNumericType,
{
    let mut b = BooleanBuilder::new(from.len());

    for i in 0..from.len() {
        if from.is_null(i) {
            b.append_null()?;
        } else if from.value(i) != T::default_value() {
            b.append_value(true)?;
        } else {
            b.append_value(false)?;
        }
    }

    Ok(b.finish())
}

/// Cast Boolean types to numeric
///
/// `false` returns 0 while `true` returns 1
#[allow(clippy::unnecessary_wraps)]
fn cast_bool_to_numeric<TO>(from: &ArrayRef) -> Result<ArrayRef>
where
    TO: ArrowNumericType,
    TO::Native: num::cast::NumCast,
{
    Ok(Arc::new(bool_to_numeric_cast::<TO>(
        from.as_any().downcast_ref::<BooleanArray>().unwrap(),
    )))
}

fn bool_to_numeric_cast<T>(from: &BooleanArray) -> PrimitiveArray<T>
where
    T: ArrowNumericType,
    T::Native: num::NumCast,
{
    (0..from.len())
        .map(|i| {
            if from.is_null(i) {
                None
            } else if from.value(i) {
                // a workaround to cast a primitive to T::Native, infallible
                num::cast::cast(1)
            } else {
                Some(T::default_value())
            }
        })
        .collect()
}
