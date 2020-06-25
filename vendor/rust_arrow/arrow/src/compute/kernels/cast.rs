// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! Defines cast kernels for `ArrayRef`, allowing casting arrays between supported
//! datatypes.
//!
//! Example:
//!
//! ```
//! use some::array::*;
//! use some::compute::cast;
//! use some::datatypes::DataType;
//! use std::sync::Arc;
//!
//! let a = Int32Array::from(vec![5, 6, 7]);
//! let array = Arc::new(a) as ArrayRef;
//! let b = cast(&array, &DataType::Float64).unwrap();
//! let c = b.as_any().downcast_ref::<Float64Array>().unwrap();
//! assert_eq!(5.0, c.value(0));
//! assert_eq!(6.0, c.value(1));
//! assert_eq!(7.0, c.value(2));
//! ```

use std::str;
use std::sync::Arc;

use crate::array::*;
use crate::buffer::Buffer;
use crate::compute::kernels::arithmetic::{divide, multiply};
use crate::datatypes::*;
use crate::error::{ArrowError, Result};

/// Cast array to provided data type
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
pub fn cast(array: &ArrayRef, to_type: &DataType) -> Result<ArrayRef> {
    use DataType::*;
    let from_type = array.data_type();

    // clone array if types are the same
    if from_type == to_type {
        return Ok(array.clone());
    }
    match (from_type, to_type) {
        (Struct(_), _) => Err(ArrowError::ComputeError(
            "Cannot cast from struct to other types".to_string(),
        )),
        (_, Struct(_)) => Err(ArrowError::ComputeError(
            "Cannot cast to struct from other types".to_string(),
        )),
        (List(_), List(ref to)) => {
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
                    .map(|bitmap| bitmap.bits),
                array.offset(),
                // reuse offset buffer
                data.buffers().to_vec(),
                vec![cast_array.data()],
            );
            let list = ListArray::from(Arc::new(array_data));
            Ok(Arc::new(list) as ArrayRef)
        }
        (List(_), _) => Err(ArrowError::ComputeError(
            "Cannot cast list to non-list data types".to_string(),
        )),
        (_, List(ref to)) => {
            // cast primitive to list's primitive
            let cast_array = cast(array, &to)?;
            // create offsets, where if array.len() = 2, we have [0,1,2]
            let offsets: Vec<i32> = (0..=array.len() as i32).collect();
            let value_offsets = Buffer::from(offsets[..].to_byte_slice());
            let list_data = ArrayData::new(
                *to.clone(),
                array.len(),
                Some(cast_array.null_count()),
                cast_array
                    .data()
                    .null_bitmap()
                    .clone()
                    .map(|bitmap| bitmap.bits),
                0,
                vec![value_offsets],
                vec![cast_array.data()],
            );
            let list_array = Arc::new(ListArray::from(Arc::new(list_data))) as ArrayRef;

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
            Utf8 => {
                let from = array.as_any().downcast_ref::<BooleanArray>().unwrap();
                let mut b = StringBuilder::new(array.len());
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
        (Utf8, _) => match to_type {
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
        (_, Utf8) => match from_type {
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
            Binary => {
                let from = array.as_any().downcast_ref::<BinaryArray>().unwrap();
                let mut b = StringBuilder::new(array.len());
                for i in 0..array.len() {
                    if array.is_null(i) {
                        b.append_null()?;
                    } else {
                        match str::from_utf8(from.value(i)) {
                            Ok(s) => b.append_value(s)?,
                            Err(_) => b.append_null()?, // not valid UTF8
                        }
                    }
                }

                Ok(Arc::new(b.finish()) as ArrayRef)
            }
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
            let mult =
                Time32MillisecondArray::from(vec![MILLISECONDS as i32; array.len()]);
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
            let c: Int64Array = numeric_cast(&time_array)?;
            let from_size = time_unit_multiple(&from_unit);
            let to_size = time_unit_multiple(&to_unit);
            // from is only smaller than to if 64milli/64second don't exist
            let mult = Int64Array::from(vec![to_size / from_size; array.len()]);
            let converted = multiply(&c, &mult)?;
            let array_ref = Arc::new(converted) as ArrayRef;
            use TimeUnit::*;
            match to_unit {
                Microsecond => cast_array_data::<TimestampMicrosecondType>(
                    &array_ref,
                    to_type.clone(),
                ),
                Nanosecond => cast_array_data::<TimestampNanosecondType>(
                    &array_ref,
                    to_type.clone(),
                ),
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
                            b.append_value(
                                (time_array.value(i) as i64 / divisor) as i32,
                            )?;
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
                            b.append_value(
                                (time_array.value(i) as i64 / divisor) as i32,
                            )?;
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
                Millisecond => {
                    cast_array_data::<TimestampMillisecondType>(array, to_type.clone())
                }
                Microsecond => {
                    cast_array_data::<TimestampMicrosecondType>(array, to_type.clone())
                }
                Nanosecond => {
                    cast_array_data::<TimestampNanosecondType>(array, to_type.clone())
                }
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
                Second => {
                    cast_array_data::<TimestampSecondType>(&array_ref, to_type.clone())
                }
                Millisecond => cast_array_data::<TimestampMillisecondType>(
                    &array_ref,
                    to_type.clone(),
                ),
                Microsecond => cast_array_data::<TimestampMicrosecondType>(
                    &array_ref,
                    to_type.clone(),
                ),
                Nanosecond => cast_array_data::<TimestampNanosecondType>(
                    &array_ref,
                    to_type.clone(),
                ),
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
fn cast_array_data<TO>(array: &ArrayRef, to_type: DataType) -> Result<ArrayRef>
where
    TO: ArrowNumericType,
{
    let data = Arc::new(ArrayData::new(
        to_type,
        array.len(),
        Some(array.null_count()),
        array.data().null_bitmap().clone().map(|bitmap| bitmap.bits),
        array.data().offset(),
        array.data().buffers().to_vec(),
        vec![],
    ));
    Ok(Arc::new(PrimitiveArray::<TO>::from(data)) as ArrayRef)
}

/// Convert Array into a PrimitiveArray of type, and apply numeric cast
fn cast_numeric_arrays<FROM, TO>(from: &ArrayRef) -> Result<ArrayRef>
where
    FROM: ArrowNumericType,
    TO: ArrowNumericType,
    FROM::Native: num::NumCast,
    TO::Native: num::NumCast,
{
    numeric_cast::<FROM, TO>(
        from.as_any()
            .downcast_ref::<PrimitiveArray<FROM>>()
            .unwrap(),
    )
    .map(|to| Arc::new(to) as ArrayRef)
}

/// Natural cast between numeric types
fn numeric_cast<T, R>(from: &PrimitiveArray<T>) -> Result<PrimitiveArray<R>>
where
    T: ArrowNumericType,
    R: ArrowNumericType,
    T::Native: num::NumCast,
    R::Native: num::NumCast,
{
    let mut b = PrimitiveBuilder::<R>::new(from.len());

    for i in 0..from.len() {
        if from.is_null(i) {
            b.append_null()?;
        } else {
            // some casts return None, such as a negative value to u{8|16|32|64}
            match num::cast::cast(from.value(i)) {
                Some(v) => b.append_value(v)?,
                None => b.append_null()?,
            };
        }
    }

    Ok(b.finish())
}

/// Cast numeric types to Utf8
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

fn numeric_to_string_cast<T>(from: &PrimitiveArray<T>) -> Result<StringArray>
where
    T: ArrowPrimitiveType + ArrowNumericType,
    T::Native: std::string::ToString,
{
    let mut b = StringBuilder::new(from.len());

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
fn cast_string_to_numeric<TO>(from: &ArrayRef) -> Result<ArrayRef>
where
    TO: ArrowNumericType,
{
    string_to_numeric_cast::<TO>(from.as_any().downcast_ref::<StringArray>().unwrap())
        .map(|to| Arc::new(to) as ArrayRef)
}

fn string_to_numeric_cast<T>(from: &StringArray) -> Result<PrimitiveArray<T>>
where
    T: ArrowNumericType,
{
    let mut b = PrimitiveBuilder::<T>::new(from.len());

    for i in 0..from.len() {
        if from.is_null(i) {
            b.append_null()?;
        } else {
            match from.value(i).parse::<T::Native>() {
                Ok(v) => b.append_value(v)?,
                _ => b.append_null()?,
            };
        }
    }

    Ok(b.finish())
}

/// Cast numeric types to Boolean
///
/// Any zero value returns `false` while non-zero returns `true`
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
fn cast_bool_to_numeric<TO>(from: &ArrayRef) -> Result<ArrayRef>
where
    TO: ArrowNumericType,
    TO::Native: num::cast::NumCast,
{
    bool_to_numeric_cast::<TO>(from.as_any().downcast_ref::<BooleanArray>().unwrap())
        .map(|to| Arc::new(to) as ArrayRef)
}

fn bool_to_numeric_cast<T>(from: &BooleanArray) -> Result<PrimitiveArray<T>>
where
    T: ArrowNumericType,
    T::Native: num::NumCast,
{
    let mut b = PrimitiveBuilder::<T>::new(from.len());

    for i in 0..from.len() {
        if from.is_null(i) {
            b.append_null()?;
        } else if from.value(i) {
            // a workaround to cast a primitive to T::Native, infallible
            match num::cast::cast(1) {
                Some(v) => b.append_value(v)?,
                None => b.append_null()?,
            };
        } else {
            b.append_value(T::default_value())?;
        }
    }

    Ok(b.finish())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::Buffer;

    #[test]
    fn test_cast_i32_to_f64() {
        let a = Int32Array::from(vec![5, 6, 7, 8, 9]);
        let array = Arc::new(a) as ArrayRef;
        let b = cast(&array, &DataType::Float64).unwrap();
        let c = b.as_any().downcast_ref::<Float64Array>().unwrap();
        assert_eq!(5.0, c.value(0));
        assert_eq!(6.0, c.value(1));
        assert_eq!(7.0, c.value(2));
        assert_eq!(8.0, c.value(3));
        assert_eq!(9.0, c.value(4));
    }

    #[test]
    fn test_cast_i32_to_u8() {
        let a = Int32Array::from(vec![-5, 6, -7, 8, 100000000]);
        let array = Arc::new(a) as ArrayRef;
        let b = cast(&array, &DataType::UInt8).unwrap();
        let c = b.as_any().downcast_ref::<UInt8Array>().unwrap();
        assert_eq!(false, c.is_valid(0));
        assert_eq!(6, c.value(1));
        assert_eq!(false, c.is_valid(2));
        assert_eq!(8, c.value(3));
        // overflows return None
        assert_eq!(false, c.is_valid(4));
    }

    #[test]
    fn test_cast_i32_to_u8_sliced() {
        let a = Int32Array::from(vec![-5, 6, -7, 8, 100000000]);
        let array = Arc::new(a) as ArrayRef;
        assert_eq!(0, array.offset());
        let array = array.slice(2, 3);
        assert_eq!(2, array.offset());
        let b = cast(&array, &DataType::UInt8).unwrap();
        assert_eq!(3, b.len());
        assert_eq!(0, b.offset());
        let c = b.as_any().downcast_ref::<UInt8Array>().unwrap();
        assert_eq!(false, c.is_valid(0));
        assert_eq!(8, c.value(1));
        // overflows return None
        assert_eq!(false, c.is_valid(2));
    }

    #[test]
    fn test_cast_i32_to_i32() {
        let a = Int32Array::from(vec![5, 6, 7, 8, 9]);
        let array = Arc::new(a) as ArrayRef;
        let b = cast(&array, &DataType::Int32).unwrap();
        let c = b.as_any().downcast_ref::<Int32Array>().unwrap();
        assert_eq!(5, c.value(0));
        assert_eq!(6, c.value(1));
        assert_eq!(7, c.value(2));
        assert_eq!(8, c.value(3));
        assert_eq!(9, c.value(4));
    }

    #[test]
    fn test_cast_i32_to_list_i32() {
        let a = Int32Array::from(vec![5, 6, 7, 8, 9]);
        let array = Arc::new(a) as ArrayRef;
        let b = cast(&array, &DataType::List(Box::new(DataType::Int32))).unwrap();
        assert_eq!(5, b.len());
        let arr = b.as_any().downcast_ref::<ListArray>().unwrap();
        assert_eq!(0, arr.value_offset(0));
        assert_eq!(1, arr.value_offset(1));
        assert_eq!(2, arr.value_offset(2));
        assert_eq!(3, arr.value_offset(3));
        assert_eq!(4, arr.value_offset(4));
        assert_eq!(1, arr.value_length(0));
        assert_eq!(1, arr.value_length(1));
        assert_eq!(1, arr.value_length(2));
        assert_eq!(1, arr.value_length(3));
        assert_eq!(1, arr.value_length(4));
        let values = arr.values();
        let c = values.as_any().downcast_ref::<Int32Array>().unwrap();
        assert_eq!(5, c.value(0));
        assert_eq!(6, c.value(1));
        assert_eq!(7, c.value(2));
        assert_eq!(8, c.value(3));
        assert_eq!(9, c.value(4));
    }

    #[test]
    fn test_cast_i32_to_list_i32_nullable() {
        let a = Int32Array::from(vec![Some(5), None, Some(7), Some(8), Some(9)]);
        let array = Arc::new(a) as ArrayRef;
        let b = cast(&array, &DataType::List(Box::new(DataType::Int32))).unwrap();
        assert_eq!(5, b.len());
        assert_eq!(1, b.null_count());
        let arr = b.as_any().downcast_ref::<ListArray>().unwrap();
        assert_eq!(0, arr.value_offset(0));
        assert_eq!(1, arr.value_offset(1));
        assert_eq!(2, arr.value_offset(2));
        assert_eq!(3, arr.value_offset(3));
        assert_eq!(4, arr.value_offset(4));
        assert_eq!(1, arr.value_length(0));
        assert_eq!(1, arr.value_length(1));
        assert_eq!(1, arr.value_length(2));
        assert_eq!(1, arr.value_length(3));
        assert_eq!(1, arr.value_length(4));
        let values = arr.values();
        let c = values.as_any().downcast_ref::<Int32Array>().unwrap();
        assert_eq!(1, c.null_count());
        assert_eq!(5, c.value(0));
        assert_eq!(false, c.is_valid(1));
        assert_eq!(7, c.value(2));
        assert_eq!(8, c.value(3));
        assert_eq!(9, c.value(4));
    }

    #[test]
    fn test_cast_i32_to_list_f64_nullable_sliced() {
        let a = Int32Array::from(vec![Some(5), None, Some(7), Some(8), None, Some(10)]);
        let array = Arc::new(a) as ArrayRef;
        let array = array.slice(2, 4);
        let b = cast(&array, &DataType::List(Box::new(DataType::Float64))).unwrap();
        assert_eq!(4, b.len());
        assert_eq!(1, b.null_count());
        let arr = b.as_any().downcast_ref::<ListArray>().unwrap();
        assert_eq!(0, arr.value_offset(0));
        assert_eq!(1, arr.value_offset(1));
        assert_eq!(2, arr.value_offset(2));
        assert_eq!(3, arr.value_offset(3));
        assert_eq!(1, arr.value_length(0));
        assert_eq!(1, arr.value_length(1));
        assert_eq!(1, arr.value_length(2));
        assert_eq!(1, arr.value_length(3));
        let values = arr.values();
        let c = values.as_any().downcast_ref::<Float64Array>().unwrap();
        assert_eq!(1, c.null_count());
        assert_eq!(7.0, c.value(0));
        assert_eq!(8.0, c.value(1));
        assert_eq!(false, c.is_valid(2));
        assert_eq!(10.0, c.value(3));
    }

    #[test]
    fn test_cast_utf8_to_i32() {
        let a = StringArray::from(vec!["5", "6", "seven", "8", "9.1"]);
        let array = Arc::new(a) as ArrayRef;
        let b = cast(&array, &DataType::Int32).unwrap();
        let c = b.as_any().downcast_ref::<Int32Array>().unwrap();
        assert_eq!(5, c.value(0));
        assert_eq!(6, c.value(1));
        assert_eq!(false, c.is_valid(2));
        assert_eq!(8, c.value(3));
        assert_eq!(false, c.is_valid(2));
    }

    #[test]
    fn test_cast_bool_to_i32() {
        let a = BooleanArray::from(vec![Some(true), Some(false), None]);
        let array = Arc::new(a) as ArrayRef;
        let b = cast(&array, &DataType::Int32).unwrap();
        let c = b.as_any().downcast_ref::<Int32Array>().unwrap();
        assert_eq!(1, c.value(0));
        assert_eq!(0, c.value(1));
        assert_eq!(false, c.is_valid(2));
    }

    #[test]
    fn test_cast_bool_to_f64() {
        let a = BooleanArray::from(vec![Some(true), Some(false), None]);
        let array = Arc::new(a) as ArrayRef;
        let b = cast(&array, &DataType::Float64).unwrap();
        let c = b.as_any().downcast_ref::<Float64Array>().unwrap();
        assert_eq!(1.0, c.value(0));
        assert_eq!(0.0, c.value(1));
        assert_eq!(false, c.is_valid(2));
    }

    #[test]
    #[should_panic(
        expected = "Casting from Int32 to Timestamp(Microsecond, None) not supported"
    )]
    fn test_cast_int32_to_timestamp() {
        let a = Int32Array::from(vec![Some(2), Some(10), None]);
        let array = Arc::new(a) as ArrayRef;
        cast(&array, &DataType::Timestamp(TimeUnit::Microsecond, None)).unwrap();
    }

    #[test]
    fn test_cast_list_i32_to_list_u16() {
        // Construct a value array
        let value_data = Int32Array::from(vec![0, 0, 0, -1, -2, -1, 2, 100000000]).data();

        let value_offsets = Buffer::from(&[0, 3, 6, 8].to_byte_slice());

        // Construct a list array from the above two
        let list_data_type = DataType::List(Box::new(DataType::Int32));
        let list_data = ArrayData::builder(list_data_type.clone())
            .len(3)
            .add_buffer(value_offsets.clone())
            .add_child_data(value_data.clone())
            .build();
        let list_array = Arc::new(ListArray::from(list_data)) as ArrayRef;

        let cast_array =
            cast(&list_array, &DataType::List(Box::new(DataType::UInt16))).unwrap();
        // 3 negative values should get lost when casting to unsigned,
        // 1 value should overflow
        assert_eq!(4, cast_array.null_count());
        // offsets should be the same
        assert_eq!(
            list_array.data().buffers().to_vec(),
            cast_array.data().buffers().to_vec()
        );
        let array = cast_array
            .as_ref()
            .as_any()
            .downcast_ref::<ListArray>()
            .unwrap();
        assert_eq!(DataType::UInt16, array.value_type());
        assert_eq!(4, array.values().null_count());
        assert_eq!(3, array.value_length(0));
        assert_eq!(3, array.value_length(1));
        assert_eq!(2, array.value_length(2));
        let values = array.values();
        let u16arr = values.as_any().downcast_ref::<UInt16Array>().unwrap();
        assert_eq!(8, u16arr.len());
        assert_eq!(4, u16arr.null_count());

        assert_eq!(0, u16arr.value(0));
        assert_eq!(0, u16arr.value(1));
        assert_eq!(0, u16arr.value(2));
        assert_eq!(false, u16arr.is_valid(3));
        assert_eq!(false, u16arr.is_valid(4));
        assert_eq!(false, u16arr.is_valid(5));
        assert_eq!(2, u16arr.value(6));
        assert_eq!(false, u16arr.is_valid(7));
    }

    #[test]
    #[should_panic(
        expected = "Casting from Int32 to Timestamp(Microsecond, None) not supported"
    )]
    fn test_cast_list_i32_to_list_timestamp() {
        // Construct a value array
        let value_data =
            Int32Array::from(vec![0, 0, 0, -1, -2, -1, 2, 8, 100000000]).data();

        let value_offsets = Buffer::from(&[0, 3, 6, 9].to_byte_slice());

        // Construct a list array from the above two
        let list_data_type = DataType::List(Box::new(DataType::Int32));
        let list_data = ArrayData::builder(list_data_type.clone())
            .len(3)
            .add_buffer(value_offsets.clone())
            .add_child_data(value_data.clone())
            .build();
        let list_array = Arc::new(ListArray::from(list_data)) as ArrayRef;

        cast(
            &list_array,
            &DataType::List(Box::new(DataType::Timestamp(TimeUnit::Microsecond, None))),
        )
        .unwrap();
    }

    #[test]
    fn test_cast_date32_to_date64() {
        let a = Date32Array::from(vec![10000, 17890]);
        let array = Arc::new(a) as ArrayRef;
        let b = cast(&array, &DataType::Date64(DateUnit::Millisecond)).unwrap();
        let c = b.as_any().downcast_ref::<Date64Array>().unwrap();
        assert_eq!(864000000000, c.value(0));
        assert_eq!(1545696000000, c.value(1));
    }

    #[test]
    fn test_cast_date64_to_date32() {
        let a = Date64Array::from(vec![Some(864000000005), Some(1545696000001), None]);
        let array = Arc::new(a) as ArrayRef;
        let b = cast(&array, &DataType::Date32(DateUnit::Day)).unwrap();
        let c = b.as_any().downcast_ref::<Date32Array>().unwrap();
        assert_eq!(10000, c.value(0));
        assert_eq!(17890, c.value(1));
        assert!(c.is_null(2));
    }

    #[test]
    fn test_cast_date32_to_int32() {
        let a = Date32Array::from(vec![10000, 17890]);
        let array = Arc::new(a) as ArrayRef;
        let b = cast(&array, &DataType::Int32).unwrap();
        let c = b.as_any().downcast_ref::<Int32Array>().unwrap();
        assert_eq!(10000, c.value(0));
        assert_eq!(17890, c.value(1));
    }

    #[test]
    fn test_cast_int32_to_date32() {
        let a = Int32Array::from(vec![10000, 17890]);
        let array = Arc::new(a) as ArrayRef;
        let b = cast(&array, &DataType::Date32(DateUnit::Day)).unwrap();
        let c = b.as_any().downcast_ref::<Date32Array>().unwrap();
        assert_eq!(10000, c.value(0));
        assert_eq!(17890, c.value(1));
    }

    #[test]
    fn test_cast_timestamp_to_date32() {
        let a = TimestampMillisecondArray::from_opt_vec(
            vec![Some(864000000005), Some(1545696000001), None],
            Some(Arc::new(String::from("UTC"))),
        );
        let array = Arc::new(a) as ArrayRef;
        let b = cast(&array, &DataType::Date32(DateUnit::Day)).unwrap();
        let c = b.as_any().downcast_ref::<Date32Array>().unwrap();
        assert_eq!(10000, c.value(0));
        assert_eq!(17890, c.value(1));
        assert!(c.is_null(2));
    }

    #[test]
    fn test_cast_timestamp_to_date64() {
        let a = TimestampMillisecondArray::from_opt_vec(
            vec![Some(864000000005), Some(1545696000001), None],
            None,
        );
        let array = Arc::new(a) as ArrayRef;
        let b = cast(&array, &DataType::Date64(DateUnit::Millisecond)).unwrap();
        let c = b.as_any().downcast_ref::<Date64Array>().unwrap();
        assert_eq!(864000000005, c.value(0));
        assert_eq!(1545696000001, c.value(1));
        assert!(c.is_null(2));
    }

    #[test]
    fn test_cast_timestamp_to_i64() {
        let a = TimestampMillisecondArray::from_opt_vec(
            vec![Some(864000000005), Some(1545696000001), None],
            Some(Arc::new("UTC".to_string())),
        );
        let array = Arc::new(a) as ArrayRef;
        let b = cast(&array, &DataType::Int64).unwrap();
        let c = b.as_any().downcast_ref::<Int64Array>().unwrap();
        assert_eq!(&DataType::Int64, c.data_type());
        assert_eq!(864000000005, c.value(0));
        assert_eq!(1545696000001, c.value(1));
        assert!(c.is_null(2));
    }

    #[test]
    fn test_cast_between_timestamps() {
        let a = TimestampMillisecondArray::from_opt_vec(
            vec![Some(864000003005), Some(1545696002001), None],
            None,
        );
        let array = Arc::new(a) as ArrayRef;
        let b = cast(&array, &DataType::Timestamp(TimeUnit::Second, None)).unwrap();
        let c = b.as_any().downcast_ref::<TimestampSecondArray>().unwrap();
        assert_eq!(864000003, c.value(0));
        assert_eq!(1545696002, c.value(1));
        assert!(c.is_null(2));
    }

    #[test]
    fn test_cast_from_f64() {
        let f64_values: Vec<f64> = vec![
            std::i64::MIN as f64,
            std::i32::MIN as f64,
            std::i16::MIN as f64,
            std::i8::MIN as f64,
            0_f64,
            std::u8::MAX as f64,
            std::u16::MAX as f64,
            std::u32::MAX as f64,
            std::u64::MAX as f64,
        ];
        let f64_array: ArrayRef = Arc::new(Float64Array::from(f64_values.clone()));

        let f64_expected = vec![
            "-9223372036854776000.0",
            "-2147483648.0",
            "-32768.0",
            "-128.0",
            "0.0",
            "255.0",
            "65535.0",
            "4294967295.0",
            "18446744073709552000.0",
        ];
        assert_eq!(
            f64_expected,
            get_cast_values::<Float64Type>(&f64_array, &DataType::Float64)
        );

        let f32_expected = vec![
            "-9223372000000000000.0",
            "-2147483600.0",
            "-32768.0",
            "-128.0",
            "0.0",
            "255.0",
            "65535.0",
            "4294967300.0",
            "18446744000000000000.0",
        ];
        assert_eq!(
            f32_expected,
            get_cast_values::<Float32Type>(&f64_array, &DataType::Float32)
        );

        let i64_expected = vec![
            "-9223372036854775808",
            "-2147483648",
            "-32768",
            "-128",
            "0",
            "255",
            "65535",
            "4294967295",
            "null",
        ];
        assert_eq!(
            i64_expected,
            get_cast_values::<Int64Type>(&f64_array, &DataType::Int64)
        );

        let i32_expected = vec![
            "null",
            "-2147483648",
            "-32768",
            "-128",
            "0",
            "255",
            "65535",
            "null",
            "null",
        ];
        assert_eq!(
            i32_expected,
            get_cast_values::<Int32Type>(&f64_array, &DataType::Int32)
        );

        let i16_expected = vec![
            "null", "null", "-32768", "-128", "0", "255", "null", "null", "null",
        ];
        assert_eq!(
            i16_expected,
            get_cast_values::<Int16Type>(&f64_array, &DataType::Int16)
        );

        let i8_expected = vec![
            "null", "null", "null", "-128", "0", "null", "null", "null", "null",
        ];
        assert_eq!(
            i8_expected,
            get_cast_values::<Int8Type>(&f64_array, &DataType::Int8)
        );

        let u64_expected = vec![
            "null",
            "null",
            "null",
            "null",
            "0",
            "255",
            "65535",
            "4294967295",
            "null",
        ];
        assert_eq!(
            u64_expected,
            get_cast_values::<UInt64Type>(&f64_array, &DataType::UInt64)
        );

        let u32_expected = vec![
            "null",
            "null",
            "null",
            "null",
            "0",
            "255",
            "65535",
            "4294967295",
            "null",
        ];
        assert_eq!(
            u32_expected,
            get_cast_values::<UInt32Type>(&f64_array, &DataType::UInt32)
        );

        let u16_expected = vec![
            "null", "null", "null", "null", "0", "255", "65535", "null", "null",
        ];
        assert_eq!(
            u16_expected,
            get_cast_values::<UInt16Type>(&f64_array, &DataType::UInt16)
        );

        let u8_expected = vec![
            "null", "null", "null", "null", "0", "255", "null", "null", "null",
        ];
        assert_eq!(
            u8_expected,
            get_cast_values::<UInt8Type>(&f64_array, &DataType::UInt8)
        );
    }

    #[test]
    fn test_cast_from_f32() {
        let f32_values: Vec<f32> = vec![
            std::i32::MIN as f32,
            std::i32::MIN as f32,
            std::i16::MIN as f32,
            std::i8::MIN as f32,
            0_f32,
            std::u8::MAX as f32,
            std::u16::MAX as f32,
            std::u32::MAX as f32,
            std::u32::MAX as f32,
        ];
        let f32_array: ArrayRef = Arc::new(Float32Array::from(f32_values.clone()));

        let f64_expected = vec![
            "-2147483648.0",
            "-2147483648.0",
            "-32768.0",
            "-128.0",
            "0.0",
            "255.0",
            "65535.0",
            "4294967296.0",
            "4294967296.0",
        ];
        assert_eq!(
            f64_expected,
            get_cast_values::<Float64Type>(&f32_array, &DataType::Float64)
        );

        let f32_expected = vec![
            "-2147483600.0",
            "-2147483600.0",
            "-32768.0",
            "-128.0",
            "0.0",
            "255.0",
            "65535.0",
            "4294967300.0",
            "4294967300.0",
        ];
        assert_eq!(
            f32_expected,
            get_cast_values::<Float32Type>(&f32_array, &DataType::Float32)
        );

        let i64_expected = vec![
            "-2147483648",
            "-2147483648",
            "-32768",
            "-128",
            "0",
            "255",
            "65535",
            "4294967296",
            "4294967296",
        ];
        assert_eq!(
            i64_expected,
            get_cast_values::<Int64Type>(&f32_array, &DataType::Int64)
        );

        let i32_expected = vec![
            "-2147483648",
            "-2147483648",
            "-32768",
            "-128",
            "0",
            "255",
            "65535",
            "null",
            "null",
        ];
        assert_eq!(
            i32_expected,
            get_cast_values::<Int32Type>(&f32_array, &DataType::Int32)
        );

        let i16_expected = vec![
            "null", "null", "-32768", "-128", "0", "255", "null", "null", "null",
        ];
        assert_eq!(
            i16_expected,
            get_cast_values::<Int16Type>(&f32_array, &DataType::Int16)
        );

        let i8_expected = vec![
            "null", "null", "null", "-128", "0", "null", "null", "null", "null",
        ];
        assert_eq!(
            i8_expected,
            get_cast_values::<Int8Type>(&f32_array, &DataType::Int8)
        );

        let u64_expected = vec![
            "null",
            "null",
            "null",
            "null",
            "0",
            "255",
            "65535",
            "4294967296",
            "4294967296",
        ];
        assert_eq!(
            u64_expected,
            get_cast_values::<UInt64Type>(&f32_array, &DataType::UInt64)
        );

        let u32_expected = vec![
            "null", "null", "null", "null", "0", "255", "65535", "null", "null",
        ];
        assert_eq!(
            u32_expected,
            get_cast_values::<UInt32Type>(&f32_array, &DataType::UInt32)
        );

        let u16_expected = vec![
            "null", "null", "null", "null", "0", "255", "65535", "null", "null",
        ];
        assert_eq!(
            u16_expected,
            get_cast_values::<UInt16Type>(&f32_array, &DataType::UInt16)
        );

        let u8_expected = vec![
            "null", "null", "null", "null", "0", "255", "null", "null", "null",
        ];
        assert_eq!(
            u8_expected,
            get_cast_values::<UInt8Type>(&f32_array, &DataType::UInt8)
        );
    }

    #[test]
    fn test_cast_from_uint64() {
        let u64_values: Vec<u64> = vec![
            0,
            std::u8::MAX as u64,
            std::u16::MAX as u64,
            std::u32::MAX as u64,
            std::u64::MAX,
        ];
        let u64_array: ArrayRef = Arc::new(UInt64Array::from(u64_values.clone()));

        let f64_expected = vec![
            "0.0",
            "255.0",
            "65535.0",
            "4294967295.0",
            "18446744073709552000.0",
        ];
        assert_eq!(
            f64_expected,
            get_cast_values::<Float64Type>(&u64_array, &DataType::Float64)
        );

        let f32_expected = vec![
            "0.0",
            "255.0",
            "65535.0",
            "4294967300.0",
            "18446744000000000000.0",
        ];
        assert_eq!(
            f32_expected,
            get_cast_values::<Float32Type>(&u64_array, &DataType::Float32)
        );

        let i64_expected = vec!["0", "255", "65535", "4294967295", "null"];
        assert_eq!(
            i64_expected,
            get_cast_values::<Int64Type>(&u64_array, &DataType::Int64)
        );

        let i32_expected = vec!["0", "255", "65535", "null", "null"];
        assert_eq!(
            i32_expected,
            get_cast_values::<Int32Type>(&u64_array, &DataType::Int32)
        );

        let i16_expected = vec!["0", "255", "null", "null", "null"];
        assert_eq!(
            i16_expected,
            get_cast_values::<Int16Type>(&u64_array, &DataType::Int16)
        );

        let i8_expected = vec!["0", "null", "null", "null", "null"];
        assert_eq!(
            i8_expected,
            get_cast_values::<Int8Type>(&u64_array, &DataType::Int8)
        );

        let u64_expected =
            vec!["0", "255", "65535", "4294967295", "18446744073709551615"];
        assert_eq!(
            u64_expected,
            get_cast_values::<UInt64Type>(&u64_array, &DataType::UInt64)
        );

        let u32_expected = vec!["0", "255", "65535", "4294967295", "null"];
        assert_eq!(
            u32_expected,
            get_cast_values::<UInt32Type>(&u64_array, &DataType::UInt32)
        );

        let u16_expected = vec!["0", "255", "65535", "null", "null"];
        assert_eq!(
            u16_expected,
            get_cast_values::<UInt16Type>(&u64_array, &DataType::UInt16)
        );

        let u8_expected = vec!["0", "255", "null", "null", "null"];
        assert_eq!(
            u8_expected,
            get_cast_values::<UInt8Type>(&u64_array, &DataType::UInt8)
        );
    }

    #[test]
    fn test_cast_from_uint32() {
        let u32_values: Vec<u32> = vec![
            0,
            std::u8::MAX as u32,
            std::u16::MAX as u32,
            std::u32::MAX as u32,
        ];
        let u32_array: ArrayRef = Arc::new(UInt32Array::from(u32_values.clone()));

        let f64_expected = vec!["0.0", "255.0", "65535.0", "4294967295.0"];
        assert_eq!(
            f64_expected,
            get_cast_values::<Float64Type>(&u32_array, &DataType::Float64)
        );

        let f32_expected = vec!["0.0", "255.0", "65535.0", "4294967300.0"];
        assert_eq!(
            f32_expected,
            get_cast_values::<Float32Type>(&u32_array, &DataType::Float32)
        );

        let i64_expected = vec!["0", "255", "65535", "4294967295"];
        assert_eq!(
            i64_expected,
            get_cast_values::<Int64Type>(&u32_array, &DataType::Int64)
        );

        let i32_expected = vec!["0", "255", "65535", "null"];
        assert_eq!(
            i32_expected,
            get_cast_values::<Int32Type>(&u32_array, &DataType::Int32)
        );

        let i16_expected = vec!["0", "255", "null", "null"];
        assert_eq!(
            i16_expected,
            get_cast_values::<Int16Type>(&u32_array, &DataType::Int16)
        );

        let i8_expected = vec!["0", "null", "null", "null"];
        assert_eq!(
            i8_expected,
            get_cast_values::<Int8Type>(&u32_array, &DataType::Int8)
        );

        let u64_expected = vec!["0", "255", "65535", "4294967295"];
        assert_eq!(
            u64_expected,
            get_cast_values::<UInt64Type>(&u32_array, &DataType::UInt64)
        );

        let u32_expected = vec!["0", "255", "65535", "4294967295"];
        assert_eq!(
            u32_expected,
            get_cast_values::<UInt32Type>(&u32_array, &DataType::UInt32)
        );

        let u16_expected = vec!["0", "255", "65535", "null"];
        assert_eq!(
            u16_expected,
            get_cast_values::<UInt16Type>(&u32_array, &DataType::UInt16)
        );

        let u8_expected = vec!["0", "255", "null", "null"];
        assert_eq!(
            u8_expected,
            get_cast_values::<UInt8Type>(&u32_array, &DataType::UInt8)
        );
    }

    #[test]
    fn test_cast_from_uint16() {
        let u16_values: Vec<u16> = vec![0, std::u8::MAX as u16, std::u16::MAX as u16];
        let u16_array: ArrayRef = Arc::new(UInt16Array::from(u16_values.clone()));

        let f64_expected = vec!["0.0", "255.0", "65535.0"];
        assert_eq!(
            f64_expected,
            get_cast_values::<Float64Type>(&u16_array, &DataType::Float64)
        );

        let f32_expected = vec!["0.0", "255.0", "65535.0"];
        assert_eq!(
            f32_expected,
            get_cast_values::<Float32Type>(&u16_array, &DataType::Float32)
        );

        let i64_expected = vec!["0", "255", "65535"];
        assert_eq!(
            i64_expected,
            get_cast_values::<Int64Type>(&u16_array, &DataType::Int64)
        );

        let i32_expected = vec!["0", "255", "65535"];
        assert_eq!(
            i32_expected,
            get_cast_values::<Int32Type>(&u16_array, &DataType::Int32)
        );

        let i16_expected = vec!["0", "255", "null"];
        assert_eq!(
            i16_expected,
            get_cast_values::<Int16Type>(&u16_array, &DataType::Int16)
        );

        let i8_expected = vec!["0", "null", "null"];
        assert_eq!(
            i8_expected,
            get_cast_values::<Int8Type>(&u16_array, &DataType::Int8)
        );

        let u64_expected = vec!["0", "255", "65535"];
        assert_eq!(
            u64_expected,
            get_cast_values::<UInt64Type>(&u16_array, &DataType::UInt64)
        );

        let u32_expected = vec!["0", "255", "65535"];
        assert_eq!(
            u32_expected,
            get_cast_values::<UInt32Type>(&u16_array, &DataType::UInt32)
        );

        let u16_expected = vec!["0", "255", "65535"];
        assert_eq!(
            u16_expected,
            get_cast_values::<UInt16Type>(&u16_array, &DataType::UInt16)
        );

        let u8_expected = vec!["0", "255", "null"];
        assert_eq!(
            u8_expected,
            get_cast_values::<UInt8Type>(&u16_array, &DataType::UInt8)
        );
    }

    #[test]
    fn test_cast_from_uint8() {
        let u8_values: Vec<u8> = vec![0, std::u8::MAX];
        let u8_array: ArrayRef = Arc::new(UInt8Array::from(u8_values.clone()));

        let f64_expected = vec!["0.0", "255.0"];
        assert_eq!(
            f64_expected,
            get_cast_values::<Float64Type>(&u8_array, &DataType::Float64)
        );

        let f32_expected = vec!["0.0", "255.0"];
        assert_eq!(
            f32_expected,
            get_cast_values::<Float32Type>(&u8_array, &DataType::Float32)
        );

        let i64_expected = vec!["0", "255"];
        assert_eq!(
            i64_expected,
            get_cast_values::<Int64Type>(&u8_array, &DataType::Int64)
        );

        let i32_expected = vec!["0", "255"];
        assert_eq!(
            i32_expected,
            get_cast_values::<Int32Type>(&u8_array, &DataType::Int32)
        );

        let i16_expected = vec!["0", "255"];
        assert_eq!(
            i16_expected,
            get_cast_values::<Int16Type>(&u8_array, &DataType::Int16)
        );

        let i8_expected = vec!["0", "null"];
        assert_eq!(
            i8_expected,
            get_cast_values::<Int8Type>(&u8_array, &DataType::Int8)
        );

        let u64_expected = vec!["0", "255"];
        assert_eq!(
            u64_expected,
            get_cast_values::<UInt64Type>(&u8_array, &DataType::UInt64)
        );

        let u32_expected = vec!["0", "255"];
        assert_eq!(
            u32_expected,
            get_cast_values::<UInt32Type>(&u8_array, &DataType::UInt32)
        );

        let u16_expected = vec!["0", "255"];
        assert_eq!(
            u16_expected,
            get_cast_values::<UInt16Type>(&u8_array, &DataType::UInt16)
        );

        let u8_expected = vec!["0", "255"];
        assert_eq!(
            u8_expected,
            get_cast_values::<UInt8Type>(&u8_array, &DataType::UInt8)
        );
    }

    #[test]
    fn test_cast_from_int64() {
        let i64_values: Vec<i64> = vec![
            std::i64::MIN,
            std::i32::MIN as i64,
            std::i16::MIN as i64,
            std::i8::MIN as i64,
            0,
            std::i8::MAX as i64,
            std::i16::MAX as i64,
            std::i32::MAX as i64,
            std::i64::MAX,
        ];
        let i64_array: ArrayRef = Arc::new(Int64Array::from(i64_values.clone()));

        let f64_expected = vec![
            "-9223372036854776000.0",
            "-2147483648.0",
            "-32768.0",
            "-128.0",
            "0.0",
            "127.0",
            "32767.0",
            "2147483647.0",
            "9223372036854776000.0",
        ];
        assert_eq!(
            f64_expected,
            get_cast_values::<Float64Type>(&i64_array, &DataType::Float64)
        );

        let f32_expected = vec![
            "-9223372000000000000.0",
            "-2147483600.0",
            "-32768.0",
            "-128.0",
            "0.0",
            "127.0",
            "32767.0",
            "2147483600.0",
            "9223372000000000000.0",
        ];
        assert_eq!(
            f32_expected,
            get_cast_values::<Float32Type>(&i64_array, &DataType::Float32)
        );

        let i64_expected = vec![
            "-9223372036854775808",
            "-2147483648",
            "-32768",
            "-128",
            "0",
            "127",
            "32767",
            "2147483647",
            "9223372036854775807",
        ];
        assert_eq!(
            i64_expected,
            get_cast_values::<Int64Type>(&i64_array, &DataType::Int64)
        );

        let i32_expected = vec![
            "null",
            "-2147483648",
            "-32768",
            "-128",
            "0",
            "127",
            "32767",
            "2147483647",
            "null",
        ];
        assert_eq!(
            i32_expected,
            get_cast_values::<Int32Type>(&i64_array, &DataType::Int32)
        );

        let i16_expected = vec![
            "null", "null", "-32768", "-128", "0", "127", "32767", "null", "null",
        ];
        assert_eq!(
            i16_expected,
            get_cast_values::<Int16Type>(&i64_array, &DataType::Int16)
        );

        let i8_expected = vec![
            "null", "null", "null", "-128", "0", "127", "null", "null", "null",
        ];
        assert_eq!(
            i8_expected,
            get_cast_values::<Int8Type>(&i64_array, &DataType::Int8)
        );

        let u64_expected = vec![
            "null",
            "null",
            "null",
            "null",
            "0",
            "127",
            "32767",
            "2147483647",
            "9223372036854775807",
        ];
        assert_eq!(
            u64_expected,
            get_cast_values::<UInt64Type>(&i64_array, &DataType::UInt64)
        );

        let u32_expected = vec![
            "null",
            "null",
            "null",
            "null",
            "0",
            "127",
            "32767",
            "2147483647",
            "null",
        ];
        assert_eq!(
            u32_expected,
            get_cast_values::<UInt32Type>(&i64_array, &DataType::UInt32)
        );

        let u16_expected = vec![
            "null", "null", "null", "null", "0", "127", "32767", "null", "null",
        ];
        assert_eq!(
            u16_expected,
            get_cast_values::<UInt16Type>(&i64_array, &DataType::UInt16)
        );

        let u8_expected = vec![
            "null", "null", "null", "null", "0", "127", "null", "null", "null",
        ];
        assert_eq!(
            u8_expected,
            get_cast_values::<UInt8Type>(&i64_array, &DataType::UInt8)
        );
    }

    #[test]
    fn test_cast_from_int32() {
        let i32_values: Vec<i32> = vec![
            std::i32::MIN as i32,
            std::i16::MIN as i32,
            std::i8::MIN as i32,
            0,
            std::i8::MAX as i32,
            std::i16::MAX as i32,
            std::i32::MAX as i32,
        ];
        let i32_array: ArrayRef = Arc::new(Int32Array::from(i32_values.clone()));

        let f64_expected = vec![
            "-2147483648.0",
            "-32768.0",
            "-128.0",
            "0.0",
            "127.0",
            "32767.0",
            "2147483647.0",
        ];
        assert_eq!(
            f64_expected,
            get_cast_values::<Float64Type>(&i32_array, &DataType::Float64)
        );

        let f32_expected = vec![
            "-2147483600.0",
            "-32768.0",
            "-128.0",
            "0.0",
            "127.0",
            "32767.0",
            "2147483600.0",
        ];
        assert_eq!(
            f32_expected,
            get_cast_values::<Float32Type>(&i32_array, &DataType::Float32)
        );

        let i16_expected = vec!["null", "-32768", "-128", "0", "127", "32767", "null"];
        assert_eq!(
            i16_expected,
            get_cast_values::<Int16Type>(&i32_array, &DataType::Int16)
        );

        let i8_expected = vec!["null", "null", "-128", "0", "127", "null", "null"];
        assert_eq!(
            i8_expected,
            get_cast_values::<Int8Type>(&i32_array, &DataType::Int8)
        );

        let u64_expected =
            vec!["null", "null", "null", "0", "127", "32767", "2147483647"];
        assert_eq!(
            u64_expected,
            get_cast_values::<UInt64Type>(&i32_array, &DataType::UInt64)
        );

        let u32_expected =
            vec!["null", "null", "null", "0", "127", "32767", "2147483647"];
        assert_eq!(
            u32_expected,
            get_cast_values::<UInt32Type>(&i32_array, &DataType::UInt32)
        );

        let u16_expected = vec!["null", "null", "null", "0", "127", "32767", "null"];
        assert_eq!(
            u16_expected,
            get_cast_values::<UInt16Type>(&i32_array, &DataType::UInt16)
        );

        let u8_expected = vec!["null", "null", "null", "0", "127", "null", "null"];
        assert_eq!(
            u8_expected,
            get_cast_values::<UInt8Type>(&i32_array, &DataType::UInt8)
        );
    }

    #[test]
    fn test_cast_from_int16() {
        let i16_values: Vec<i16> = vec![
            std::i16::MIN,
            std::i8::MIN as i16,
            0,
            std::i8::MAX as i16,
            std::i16::MAX,
        ];
        let i16_array: ArrayRef = Arc::new(Int16Array::from(i16_values.clone()));

        let f64_expected = vec!["-32768.0", "-128.0", "0.0", "127.0", "32767.0"];
        assert_eq!(
            f64_expected,
            get_cast_values::<Float64Type>(&i16_array, &DataType::Float64)
        );

        let f32_expected = vec!["-32768.0", "-128.0", "0.0", "127.0", "32767.0"];
        assert_eq!(
            f32_expected,
            get_cast_values::<Float32Type>(&i16_array, &DataType::Float32)
        );

        let i64_expected = vec!["-32768", "-128", "0", "127", "32767"];
        assert_eq!(
            i64_expected,
            get_cast_values::<Int64Type>(&i16_array, &DataType::Int64)
        );

        let i32_expected = vec!["-32768", "-128", "0", "127", "32767"];
        assert_eq!(
            i32_expected,
            get_cast_values::<Int32Type>(&i16_array, &DataType::Int32)
        );

        let i16_expected = vec!["-32768", "-128", "0", "127", "32767"];
        assert_eq!(
            i16_expected,
            get_cast_values::<Int16Type>(&i16_array, &DataType::Int16)
        );

        let i8_expected = vec!["null", "-128", "0", "127", "null"];
        assert_eq!(
            i8_expected,
            get_cast_values::<Int8Type>(&i16_array, &DataType::Int8)
        );

        let u64_expected = vec!["null", "null", "0", "127", "32767"];
        assert_eq!(
            u64_expected,
            get_cast_values::<UInt64Type>(&i16_array, &DataType::UInt64)
        );

        let u32_expected = vec!["null", "null", "0", "127", "32767"];
        assert_eq!(
            u32_expected,
            get_cast_values::<UInt32Type>(&i16_array, &DataType::UInt32)
        );

        let u16_expected = vec!["null", "null", "0", "127", "32767"];
        assert_eq!(
            u16_expected,
            get_cast_values::<UInt16Type>(&i16_array, &DataType::UInt16)
        );

        let u8_expected = vec!["null", "null", "0", "127", "null"];
        assert_eq!(
            u8_expected,
            get_cast_values::<UInt8Type>(&i16_array, &DataType::UInt8)
        );
    }

    #[test]
    fn test_cast_from_int8() {
        let i8_values: Vec<i8> = vec![std::i8::MIN, 0, std::i8::MAX];
        let i8_array: ArrayRef = Arc::new(Int8Array::from(i8_values.clone()));

        let f64_expected = vec!["-128.0", "0.0", "127.0"];
        assert_eq!(
            f64_expected,
            get_cast_values::<Float64Type>(&i8_array, &DataType::Float64)
        );

        let f32_expected = vec!["-128.0", "0.0", "127.0"];
        assert_eq!(
            f32_expected,
            get_cast_values::<Float32Type>(&i8_array, &DataType::Float32)
        );

        let i64_expected = vec!["-128", "0", "127"];
        assert_eq!(
            i64_expected,
            get_cast_values::<Int64Type>(&i8_array, &DataType::Int64)
        );

        let i32_expected = vec!["-128", "0", "127"];
        assert_eq!(
            i32_expected,
            get_cast_values::<Int32Type>(&i8_array, &DataType::Int32)
        );

        let i16_expected = vec!["-128", "0", "127"];
        assert_eq!(
            i16_expected,
            get_cast_values::<Int16Type>(&i8_array, &DataType::Int16)
        );

        let i8_expected = vec!["-128", "0", "127"];
        assert_eq!(
            i8_expected,
            get_cast_values::<Int8Type>(&i8_array, &DataType::Int8)
        );

        let u64_expected = vec!["null", "0", "127"];
        assert_eq!(
            u64_expected,
            get_cast_values::<UInt64Type>(&i8_array, &DataType::UInt64)
        );

        let u32_expected = vec!["null", "0", "127"];
        assert_eq!(
            u32_expected,
            get_cast_values::<UInt32Type>(&i8_array, &DataType::UInt32)
        );

        let u16_expected = vec!["null", "0", "127"];
        assert_eq!(
            u16_expected,
            get_cast_values::<UInt16Type>(&i8_array, &DataType::UInt16)
        );

        let u8_expected = vec!["null", "0", "127"];
        assert_eq!(
            u8_expected,
            get_cast_values::<UInt8Type>(&i8_array, &DataType::UInt8)
        );
    }

    fn get_cast_values<T>(array: &ArrayRef, dt: &DataType) -> Vec<String>
    where
        T: ArrowNumericType,
    {
        let c = cast(&array, dt).unwrap();
        let a = c.as_any().downcast_ref::<PrimitiveArray<T>>().unwrap();
        let mut v: Vec<String> = vec![];
        for i in 0..array.len() {
            if a.is_null(i) {
                v.push("null".to_string())
            } else {
                v.push(format!("{:?}", a.value(i)));
            }
        }
        v
    }
}
