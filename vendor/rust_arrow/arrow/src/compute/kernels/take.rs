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

//! Defines take kernel for `ArrayRef`

use std::sync::Arc;

use crate::array::*;
use crate::buffer::{Buffer, MutableBuffer};
use crate::compute::util::take_value_indices_from_list;
use crate::datatypes::*;
use crate::error::{ArrowError, Result};
use crate::util::bit_util;

use TimeUnit::*;

/// Take elements from `ArrayRef` by supplying an array of indices.
///
/// Supports:
///  * null indices, returning a null value for the index
///  * checking for overflowing indices
pub fn take(
    values: &ArrayRef,
    indices: &UInt32Array,
    options: Option<TakeOptions>,
) -> Result<ArrayRef> {
    let options = options.unwrap_or_default();
    if options.check_bounds {
        let len = values.len();
        for i in 0..indices.len() {
            if indices.is_valid(i) {
                let ix = indices.value(i) as usize;
                if ix >= len {
                    return Err(ArrowError::ComputeError(
                    format!("Array index out of bounds, cannot get item at index {} from {} entries", ix, len))
                );
                }
            }
        }
    }
    match values.data_type() {
        DataType::Boolean => take_primitive::<BooleanType>(values, indices),
        DataType::Int8 => take_primitive::<Int8Type>(values, indices),
        DataType::Int16 => take_primitive::<Int16Type>(values, indices),
        DataType::Int32 => take_primitive::<Int32Type>(values, indices),
        DataType::Int64 => take_primitive::<Int64Type>(values, indices),
        DataType::UInt8 => take_primitive::<UInt8Type>(values, indices),
        DataType::UInt16 => take_primitive::<UInt16Type>(values, indices),
        DataType::UInt32 => take_primitive::<UInt32Type>(values, indices),
        DataType::UInt64 => take_primitive::<UInt64Type>(values, indices),
        DataType::Float32 => take_primitive::<Float32Type>(values, indices),
        DataType::Float64 => take_primitive::<Float64Type>(values, indices),
        DataType::Date32(_) => take_primitive::<Date32Type>(values, indices),
        DataType::Date64(_) => take_primitive::<Date64Type>(values, indices),
        DataType::Time32(Second) => take_primitive::<Time32SecondType>(values, indices),
        DataType::Time32(Millisecond) => {
            take_primitive::<Time32MillisecondType>(values, indices)
        }
        DataType::Time64(Microsecond) => {
            take_primitive::<Time64MicrosecondType>(values, indices)
        }
        DataType::Time64(Nanosecond) => {
            take_primitive::<Time64NanosecondType>(values, indices)
        }
        DataType::Timestamp(Second, _) => {
            take_primitive::<TimestampSecondType>(values, indices)
        }
        DataType::Timestamp(Millisecond, _) => {
            take_primitive::<TimestampMillisecondType>(values, indices)
        }
        DataType::Timestamp(Microsecond, _) => {
            take_primitive::<TimestampMicrosecondType>(values, indices)
        }
        DataType::Timestamp(Nanosecond, _) => {
            take_primitive::<TimestampNanosecondType>(values, indices)
        }
        DataType::Interval(IntervalUnit::YearMonth) => {
            take_primitive::<IntervalYearMonthType>(values, indices)
        }
        DataType::Interval(IntervalUnit::DayTime) => {
            take_primitive::<IntervalDayTimeType>(values, indices)
        }
        DataType::Duration(TimeUnit::Second) => {
            take_primitive::<DurationSecondType>(values, indices)
        }
        DataType::Duration(TimeUnit::Millisecond) => {
            take_primitive::<DurationMillisecondType>(values, indices)
        }
        DataType::Duration(TimeUnit::Microsecond) => {
            take_primitive::<DurationMicrosecondType>(values, indices)
        }
        DataType::Duration(TimeUnit::Nanosecond) => {
            take_primitive::<DurationNanosecondType>(values, indices)
        }
        DataType::Utf8 => take_string(values, indices),
        DataType::List(_) => take_list(values, indices),
        DataType::Struct(fields) => {
            let struct_: &StructArray =
                values.as_any().downcast_ref::<StructArray>().unwrap();
            let arrays: Result<Vec<ArrayRef>> = struct_
                .columns()
                .iter()
                .map(|a| take(a, indices, Some(options.clone())))
                .collect();
            let arrays = arrays?;
            let pairs: Vec<(Field, ArrayRef)> =
                fields.clone().into_iter().zip(arrays).collect();
            Ok(Arc::new(StructArray::from(pairs)) as ArrayRef)
        }
        t => unimplemented!("Take not supported for data type {:?}", t),
    }
}

/// Options that define how `take` should behave
#[derive(Clone)]
pub struct TakeOptions {
    /// Perform bounds check before taking indices from values.
    /// If enabled, an `ArrowError` is returned if the indices are out of bounds.
    /// If not enabled, and indices exceed bounds, the kernel will panic.
    pub check_bounds: bool,
}

impl Default for TakeOptions {
    fn default() -> Self {
        Self {
            check_bounds: false,
        }
    }
}

/// `take` implementation for primitive arrays
///
/// This checks if an `indices` slot is populated, and gets the value from `values`
///  as the populated index.
/// If the `indices` slot is null, a null value is returned.
/// For example, given:
///     values:  [1, 2, 3, null, 5]
///     indices: [0, null, 4, 3]
/// The result is: [1 (slot 0), null (null slot), 5 (slot 4), null (slot 3)]
fn take_primitive<T>(values: &ArrayRef, indices: &UInt32Array) -> Result<ArrayRef>
where
    T: ArrowPrimitiveType,
{
    let mut builder = PrimitiveBuilder::<T>::new(indices.len());
    let a = values.as_any().downcast_ref::<PrimitiveArray<T>>().unwrap();
    for i in 0..indices.len() {
        if indices.is_null(i) {
            // populate with null if index is null
            builder.append_null()?;
        } else {
            // get index value to use in looking up the value from `values`
            let ix = indices.value(i) as usize;
            if a.is_valid(ix) {
                builder.append_value(a.value(ix))?;
            } else {
                builder.append_null()?;
            }
        }
    }
    Ok(Arc::new(builder.finish()) as ArrayRef)
}

/// `take` implementation for string arrays
fn take_string(values: &ArrayRef, indices: &UInt32Array) -> Result<ArrayRef> {
    let mut builder = StringBuilder::new(indices.len());
    let a = values.as_any().downcast_ref::<StringArray>().unwrap();
    for i in 0..indices.len() {
        if indices.is_null(i) {
            builder.append(false)?;
        } else {
            let ix = indices.value(i) as usize;
            if a.is_null(ix) {
                builder.append(false)?;
            } else {
                builder.append_value(a.value(ix))?;
            }
        }
    }
    Ok(Arc::new(builder.finish()) as ArrayRef)
}

/// `take` implementation for list arrays
///
/// Calculates the index and indexed offset for the inner array,
/// applying `take` on the inner array, then reconstructing a list array
/// with the indexed offsets
fn take_list(values: &ArrayRef, indices: &UInt32Array) -> Result<ArrayRef> {
    // TODO: Some optimizations can be done here such as if it is
    // taking the whole list or a contiguous sublist
    let list: &ListArray = values.as_any().downcast_ref::<ListArray>().unwrap();
    let (list_indices, offsets) = take_value_indices_from_list(values, indices);
    let taken = take(&list.values(), &list_indices, None)?;
    // determine null count and null buffer, which are a function of `values` and `indices`
    let mut null_count = 0;
    let num_bytes = bit_util::ceil(indices.len(), 8);
    let mut null_buf = MutableBuffer::new(num_bytes).with_bitset(num_bytes, false);
    {
        let null_slice = null_buf.data_mut();
        offsets[..]
            .windows(2)
            .enumerate()
            .for_each(|(i, window): (usize, &[i32])| {
                if window[0] != window[1] {
                    // offsets are unequal, slot is not null
                    bit_util::set_bit(null_slice, i);
                } else {
                    null_count += 1;
                }
            });
    }
    let value_offsets = Buffer::from(offsets[..].to_byte_slice());
    // create a new list with taken data and computed null information
    let list_data = ArrayDataBuilder::new(list.data_type().clone())
        .len(indices.len())
        .null_count(null_count)
        .null_bit_buffer(null_buf.freeze())
        .offset(0)
        .add_child_data(taken.data())
        .add_buffer(value_offsets)
        .build();
    let list_array = Arc::new(ListArray::from(list_data)) as ArrayRef;
    Ok(list_array)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_take_primitive_arrays<'a, T>(
        data: Vec<Option<T::Native>>,
        index: &UInt32Array,
        options: Option<TakeOptions>,
        expected_data: Vec<Option<T::Native>>,
    ) where
        T: ArrowPrimitiveType,
        PrimitiveArray<T>: From<Vec<Option<T::Native>>> + ArrayEqual,
    {
        let output = PrimitiveArray::<T>::from(data);
        let expected = PrimitiveArray::<T>::from(expected_data);
        let output = take(&(Arc::new(output) as ArrayRef), index, options).unwrap();
        let output = output.as_any().downcast_ref::<PrimitiveArray<T>>().unwrap();
        assert!(output.equals(&expected))
    }

    // create a simple struct for testing purposes
    fn create_test_struct() -> ArrayRef {
        let boolean_data = BooleanArray::from(vec![true, false, false, true]).data();
        let int_data = Int32Array::from(vec![42, 28, 19, 31]).data();
        let mut field_types = vec![];
        field_types.push(Field::new("a", DataType::Boolean, true));
        field_types.push(Field::new("b", DataType::Int32, true));
        let struct_array_data = ArrayData::builder(DataType::Struct(field_types))
            .len(4)
            .null_count(0)
            .add_child_data(boolean_data)
            .add_child_data(int_data)
            .build();
        let struct_array = StructArray::from(struct_array_data);
        Arc::new(struct_array) as ArrayRef
    }

    #[test]
    fn test_take_primitive() {
        let index = UInt32Array::from(vec![Some(3), None, Some(1), Some(3), Some(2)]);

        // uint8
        test_take_primitive_arrays::<UInt8Type>(
            vec![Some(0), None, Some(2), Some(3), None],
            &index,
            None,
            vec![Some(3), None, None, Some(3), Some(2)],
        );

        // uint16
        test_take_primitive_arrays::<UInt16Type>(
            vec![Some(0), None, Some(2), Some(3), None],
            &index,
            None,
            vec![Some(3), None, None, Some(3), Some(2)],
        );

        // uint32
        test_take_primitive_arrays::<UInt32Type>(
            vec![Some(0), None, Some(2), Some(3), None],
            &index,
            None,
            vec![Some(3), None, None, Some(3), Some(2)],
        );

        // int64
        test_take_primitive_arrays::<Int64Type>(
            vec![Some(0), None, Some(2), Some(-15), None],
            &index,
            None,
            vec![Some(-15), None, None, Some(-15), Some(2)],
        );

        // interval_year_month
        test_take_primitive_arrays::<IntervalYearMonthType>(
            vec![Some(0), None, Some(2), Some(-15), None],
            &index,
            None,
            vec![Some(-15), None, None, Some(-15), Some(2)],
        );

        // interval_day_time
        test_take_primitive_arrays::<IntervalDayTimeType>(
            vec![Some(0), None, Some(2), Some(-15), None],
            &index,
            None,
            vec![Some(-15), None, None, Some(-15), Some(2)],
        );

        // duration_second
        test_take_primitive_arrays::<DurationSecondType>(
            vec![Some(0), None, Some(2), Some(-15), None],
            &index,
            None,
            vec![Some(-15), None, None, Some(-15), Some(2)],
        );

        // duration_millisecond
        test_take_primitive_arrays::<DurationMillisecondType>(
            vec![Some(0), None, Some(2), Some(-15), None],
            &index,
            None,
            vec![Some(-15), None, None, Some(-15), Some(2)],
        );

        // duration_microsecond
        test_take_primitive_arrays::<DurationMicrosecondType>(
            vec![Some(0), None, Some(2), Some(-15), None],
            &index,
            None,
            vec![Some(-15), None, None, Some(-15), Some(2)],
        );

        // duration_nanosecond
        test_take_primitive_arrays::<DurationNanosecondType>(
            vec![Some(0), None, Some(2), Some(-15), None],
            &index,
            None,
            vec![Some(-15), None, None, Some(-15), Some(2)],
        );

        // float32
        test_take_primitive_arrays::<Float32Type>(
            vec![Some(0.0), None, Some(2.21), Some(-3.1), None],
            &index,
            None,
            vec![Some(-3.1), None, None, Some(-3.1), Some(2.21)],
        );

        // float64
        test_take_primitive_arrays::<Float64Type>(
            vec![Some(0.0), None, Some(2.21), Some(-3.1), None],
            &index,
            None,
            vec![Some(-3.1), None, None, Some(-3.1), Some(2.21)],
        );

        // boolean
        test_take_primitive_arrays::<BooleanType>(
            vec![Some(false), None, Some(true), Some(false), None],
            &index,
            None,
            vec![Some(false), None, None, Some(false), Some(true)],
        );
    }

    #[test]
    fn test_take_string() {
        let index = UInt32Array::from(vec![Some(3), None, Some(1), Some(3), Some(4)]);
        let mut builder: StringBuilder = StringBuilder::new(6);
        builder.append_value("one").unwrap();
        builder.append_null().unwrap();
        builder.append_value("three").unwrap();
        builder.append_value("four").unwrap();
        builder.append_value("five").unwrap();
        let array = Arc::new(builder.finish()) as ArrayRef;
        let a = take(&array, &index, None).unwrap();
        assert_eq!(a.len(), index.len());
        builder.append_value("four").unwrap();
        builder.append_null().unwrap();
        builder.append_null().unwrap();
        builder.append_value("four").unwrap();
        builder.append_value("five").unwrap();
        let b = builder.finish();
        assert_eq!(a.data(), b.data());
    }

    #[test]
    fn test_take_list() {
        // Construct a value array, [[0,0,0], [-1,-2,-1], [2,3]]
        let value_data = Int32Array::from(vec![0, 0, 0, -1, -2, -1, 2, 3]).data();
        // Construct offsets
        let value_offsets = Buffer::from(&[0, 3, 6, 8].to_byte_slice());
        // Construct a list array from the above two
        let list_data_type = DataType::List(Box::new(DataType::Int32));
        let list_data = ArrayData::builder(list_data_type.clone())
            .len(3)
            .add_buffer(value_offsets)
            .add_child_data(value_data)
            .build();
        let list_array = Arc::new(ListArray::from(list_data)) as ArrayRef;

        // index returns: [[2,3], null, [-1,-2,-1], [2,3], [0,0,0]]
        let index = UInt32Array::from(vec![Some(2), None, Some(1), Some(2), Some(0)]);

        let a = take(&list_array, &index, None).unwrap();
        let a: &ListArray = a.as_any().downcast_ref::<ListArray>().unwrap();

        // construct a value array with expected results:
        // [[2,3], null, [-1,-2,-1], [2,3], [0,0,0]]
        let expected_data = Int32Array::from(vec![
            Some(2),
            Some(3),
            Some(-1),
            Some(-2),
            Some(-1),
            Some(2),
            Some(3),
            Some(0),
            Some(0),
            Some(0),
        ])
        .data();
        // construct offsets
        let expected_offsets = Buffer::from(&[0, 2, 2, 5, 7, 10].to_byte_slice());
        // construct list array from the two
        let expected_list_data = ArrayData::builder(list_data_type.clone())
            .len(5)
            .null_count(1)
            // null buffer remains the same as only the indices have nulls
            .null_bit_buffer(index.data().null_bitmap().as_ref().unwrap().bits.clone())
            .add_buffer(expected_offsets)
            .add_child_data(expected_data)
            .build();
        let expected_list_array = ListArray::from(expected_list_data);

        assert!(a.equals(&expected_list_array));
    }

    #[test]
    fn test_take_list_with_value_nulls() {
        // Construct a value array, [[0,null,0], [-1,-2,3], [null], [5,null]]
        let value_data = Int32Array::from(vec![
            Some(0),
            None,
            Some(0),
            Some(-1),
            Some(-2),
            Some(3),
            None,
            Some(5),
            None,
        ])
        .data();
        // Construct offsets
        let value_offsets = Buffer::from(&[0, 3, 6, 7, 9].to_byte_slice());
        // Construct a list array from the above two
        let list_data_type = DataType::List(Box::new(DataType::Int32));
        let list_data = ArrayData::builder(list_data_type.clone())
            .len(4)
            .add_buffer(value_offsets)
            .null_count(0)
            .null_bit_buffer(Buffer::from([0b10111101, 0b00000000]))
            .add_child_data(value_data)
            .build();
        let list_array = Arc::new(ListArray::from(list_data)) as ArrayRef;

        // index returns: [[null], null, [-1,-2,3], [2,null], [0,null,0]]
        let index = UInt32Array::from(vec![Some(2), None, Some(1), Some(3), Some(0)]);

        let a = take(&list_array, &index, None).unwrap();
        let a: &ListArray = a.as_any().downcast_ref::<ListArray>().unwrap();

        // construct a value array with expected results:
        // [[null], null, [-1,-2,3], [5,null], [0,null,0]]
        let expected_data = Int32Array::from(vec![
            None,
            Some(-1),
            Some(-2),
            Some(3),
            Some(5),
            None,
            Some(0),
            None,
            Some(0),
        ])
        .data();
        // construct offsets
        let expected_offsets = Buffer::from(&[0, 1, 1, 4, 6, 9].to_byte_slice());
        // construct list array from the two
        let expected_list_data = ArrayData::builder(list_data_type.clone())
            .len(5)
            .null_count(1)
            // null buffer remains the same as only the indices have nulls
            .null_bit_buffer(index.data().null_bitmap().as_ref().unwrap().bits.clone())
            .add_buffer(expected_offsets)
            .add_child_data(expected_data)
            .build();
        let expected_list_array = ListArray::from(expected_list_data);

        assert!(a.equals(&expected_list_array));
    }

    #[test]
    fn test_take_list_with_list_nulls() {
        // Construct a value array, [[0,null,0], [-1,-2,3], null, [5,null]]
        let value_data = Int32Array::from(vec![
            Some(0),
            None,
            Some(0),
            Some(-1),
            Some(-2),
            Some(3),
            Some(5),
            None,
        ])
        .data();
        // Construct offsets
        let value_offsets = Buffer::from(&[0, 3, 6, 6, 8].to_byte_slice());
        // Construct a list array from the above two
        let list_data_type = DataType::List(Box::new(DataType::Int32));
        let list_data = ArrayData::builder(list_data_type.clone())
            .len(4)
            .add_buffer(value_offsets)
            .null_count(1)
            .null_bit_buffer(Buffer::from([0b01111101]))
            .add_child_data(value_data)
            .build();
        let list_array = Arc::new(ListArray::from(list_data)) as ArrayRef;

        // index returns: [null, null, [-1,-2,3], [5,null], [0,null,0]]
        let index = UInt32Array::from(vec![Some(2), None, Some(1), Some(3), Some(0)]);

        let a = take(&list_array, &index, None).unwrap();
        let a: &ListArray = a.as_any().downcast_ref::<ListArray>().unwrap();

        // construct a value array with expected results:
        // [null, null, [-1,-2,3], [5,null], [0,null,0]]
        let expected_data = Int32Array::from(vec![
            Some(-1),
            Some(-2),
            Some(3),
            Some(5),
            None,
            Some(0),
            None,
            Some(0),
        ])
        .data();
        // construct offsets
        let expected_offsets = Buffer::from(&[0, 0, 0, 3, 5, 8].to_byte_slice());
        // construct list array from the two
        let mut null_bits: [u8; 1] = [0; 1];
        bit_util::set_bit(&mut null_bits, 2);
        bit_util::set_bit(&mut null_bits, 3);
        bit_util::set_bit(&mut null_bits, 4);
        let expected_list_data = ArrayData::builder(list_data_type.clone())
            .len(5)
            .null_count(2)
            // null buffer must be recalculated as both values and indices have nulls
            .null_bit_buffer(Buffer::from(null_bits))
            .add_buffer(expected_offsets)
            .add_child_data(expected_data)
            .build();
        let expected_list_array = ListArray::from(expected_list_data);

        assert!(a.equals(&expected_list_array));
    }

    #[test]
    fn test_take_struct() {
        let array = create_test_struct();

        let index = UInt32Array::from(vec![0, 3, 1, 0, 2]);
        let a = take(&array, &index, None).unwrap();
        let a: &StructArray = a.as_any().downcast_ref::<StructArray>().unwrap();
        assert_eq!(index.len(), a.len());
        assert_eq!(0, a.null_count());

        let expected_bool_data =
            BooleanArray::from(vec![true, true, false, true, false]).data();
        let expected_int_data = Int32Array::from(vec![42, 31, 28, 42, 19]).data();
        let mut field_types = vec![];
        field_types.push(Field::new("a", DataType::Boolean, true));
        field_types.push(Field::new("b", DataType::Int32, true));
        let struct_array_data = ArrayData::builder(DataType::Struct(field_types))
            .len(5)
            .null_count(0)
            .add_child_data(expected_bool_data)
            .add_child_data(expected_int_data)
            .build();
        let struct_array = StructArray::from(struct_array_data);
        assert!(a.equals(&struct_array));
    }

    #[test]
    fn test_take_struct_with_nulls() {
        let array = create_test_struct();

        let index = UInt32Array::from(vec![None, Some(3), Some(1), None, Some(0)]);
        let a = take(&array, &index, None).unwrap();
        let a: &StructArray = a.as_any().downcast_ref::<StructArray>().unwrap();
        assert_eq!(index.len(), a.len());
        assert_eq!(0, a.null_count());

        let expected_bool_data =
            BooleanArray::from(vec![None, Some(true), Some(false), None, Some(true)])
                .data();
        let expected_int_data =
            Int32Array::from(vec![None, Some(31), Some(28), None, Some(42)]).data();

        let mut field_types = vec![];
        field_types.push(Field::new("a", DataType::Boolean, true));
        field_types.push(Field::new("b", DataType::Int32, true));
        let struct_array_data = ArrayData::builder(DataType::Struct(field_types))
            .len(5)
            // TODO: see https://issues.apache.org/jira/browse/ARROW-5408 for why count != 2
            .null_count(0)
            .add_child_data(expected_bool_data)
            .add_child_data(expected_int_data)
            .build();
        let struct_array = StructArray::from(struct_array_data);
        assert!(a.equals(&struct_array));
    }

    #[test]
    #[should_panic(
        expected = "Array index out of bounds, cannot get item at index 6 from 5 entries"
    )]
    fn test_take_out_of_bounds() {
        let index = UInt32Array::from(vec![Some(3), None, Some(1), Some(3), Some(6)]);
        let take_opt = TakeOptions { check_bounds: true };

        // int64
        test_take_primitive_arrays::<Int64Type>(
            vec![Some(0), None, Some(2), Some(3), None],
            &index,
            Some(take_opt),
            vec![None],
        );
    }
}
