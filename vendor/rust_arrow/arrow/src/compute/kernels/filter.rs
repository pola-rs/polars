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

//! Defines miscellaneous array kernels.

use std::sync::Arc;

use crate::array::*;
use crate::datatypes::{ArrowNumericType, DataType, TimeUnit};
use crate::error::{ArrowError, Result};

/// Helper function to perform boolean lambda function on values from two arrays.
fn bool_op<T, F>(
    left: &PrimitiveArray<T>,
    right: &PrimitiveArray<T>,
    op: F,
) -> Result<BooleanArray>
where
    T: ArrowNumericType,
    F: Fn(Option<T::Native>, Option<T::Native>) -> bool,
{
    if left.len() != right.len() {
        return Err(ArrowError::ComputeError(
            "Cannot perform math operation on arrays of different length".to_string(),
        ));
    }
    let mut b = BooleanArray::builder(left.len());
    for i in 0..left.len() {
        let index = i;
        let l = if left.is_null(i) {
            None
        } else {
            Some(left.value(index))
        };
        let r = if right.is_null(i) {
            None
        } else {
            Some(right.value(index))
        };
        b.append_value(op(l, r))?;
    }
    Ok(b.finish())
}

macro_rules! filter_array {
    ($array:expr, $filter:expr, $array_type:ident) => {{
        let b = $array.as_any().downcast_ref::<$array_type>().unwrap();
        let mut builder = $array_type::builder(b.len());
        for i in 0..b.len() {
            if $filter.value(i) {
                if b.is_null(i) {
                    builder.append_null()?;
                } else {
                    builder.append_value(b.value(i))?;
                }
            }
        }
        Ok(Arc::new(builder.finish()))
    }};
}

/// Returns the array, taking only the elements matching the filter
pub fn filter(array: &Array, filter: &BooleanArray) -> Result<ArrayRef> {
    match array.data_type() {
        DataType::UInt8 => filter_array!(array, filter, UInt8Array),
        DataType::UInt16 => filter_array!(array, filter, UInt16Array),
        DataType::UInt32 => filter_array!(array, filter, UInt32Array),
        DataType::UInt64 => filter_array!(array, filter, UInt64Array),
        DataType::Int8 => filter_array!(array, filter, Int8Array),
        DataType::Int16 => filter_array!(array, filter, Int16Array),
        DataType::Int32 => filter_array!(array, filter, Int32Array),
        DataType::Int64 => filter_array!(array, filter, Int64Array),
        DataType::Float32 => filter_array!(array, filter, Float32Array),
        DataType::Float64 => filter_array!(array, filter, Float64Array),
        DataType::Boolean => filter_array!(array, filter, BooleanArray),
        DataType::Date32(_) => filter_array!(array, filter, Date32Array),
        DataType::Date64(_) => filter_array!(array, filter, Date64Array),
        DataType::Time32(TimeUnit::Second) => {
            filter_array!(array, filter, Time32SecondArray)
        }
        DataType::Time32(TimeUnit::Millisecond) => {
            filter_array!(array, filter, Time32MillisecondArray)
        }
        DataType::Time64(TimeUnit::Microsecond) => {
            filter_array!(array, filter, Time64MicrosecondArray)
        }
        DataType::Time64(TimeUnit::Nanosecond) => {
            filter_array!(array, filter, Time64NanosecondArray)
        }
        DataType::Duration(TimeUnit::Second) => {
            filter_array!(array, filter, DurationSecondArray)
        }
        DataType::Duration(TimeUnit::Millisecond) => {
            filter_array!(array, filter, DurationMillisecondArray)
        }
        DataType::Duration(TimeUnit::Microsecond) => {
            filter_array!(array, filter, DurationMicrosecondArray)
        }
        DataType::Duration(TimeUnit::Nanosecond) => {
            filter_array!(array, filter, DurationNanosecondArray)
        }
        DataType::Timestamp(TimeUnit::Second, _) => {
            filter_array!(array, filter, TimestampSecondArray)
        }
        DataType::Timestamp(TimeUnit::Millisecond, _) => {
            filter_array!(array, filter, TimestampMillisecondArray)
        }
        DataType::Timestamp(TimeUnit::Microsecond, _) => {
            filter_array!(array, filter, TimestampMicrosecondArray)
        }
        DataType::Timestamp(TimeUnit::Nanosecond, _) => {
            filter_array!(array, filter, TimestampNanosecondArray)
        }
        DataType::Binary => {
            let b = array.as_any().downcast_ref::<BinaryArray>().unwrap();
            let mut values: Vec<&[u8]> = Vec::with_capacity(b.len());
            for i in 0..b.len() {
                if filter.value(i) {
                    values.push(b.value(i));
                }
            }
            Ok(Arc::new(BinaryArray::from(values)))
        }
        DataType::Utf8 => {
            let b = array.as_any().downcast_ref::<StringArray>().unwrap();
            let mut values: Vec<&str> = Vec::with_capacity(b.len());
            for i in 0..b.len() {
                if filter.value(i) {
                    values.push(b.value(i));
                }
            }
            Ok(Arc::new(StringArray::from(values)))
        }
        other => Err(ArrowError::ComputeError(format!(
            "filter not supported for {:?}",
            other
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    macro_rules! def_temporal_test {
        ($test:ident, $array_type: ident, $data: expr) => {
            #[test]
            fn $test() {
                let a = $data;
                let b = BooleanArray::from(vec![true, false, true, false]);
                let c = filter(&a, &b).unwrap();
                let d = c.as_ref().as_any().downcast_ref::<$array_type>().unwrap();
                assert_eq!(2, d.len());
                assert_eq!(1, d.value(0));
                assert_eq!(3, d.value(1));
            }
        };
    }

    def_temporal_test!(
        test_filter_date32,
        Date32Array,
        Date32Array::from(vec![1, 2, 3, 4])
    );
    def_temporal_test!(
        test_filter_date64,
        Date64Array,
        Date64Array::from(vec![1, 2, 3, 4])
    );
    def_temporal_test!(
        test_filter_time32_second,
        Time32SecondArray,
        Time32SecondArray::from(vec![1, 2, 3, 4])
    );
    def_temporal_test!(
        test_filter_time32_millisecond,
        Time32MillisecondArray,
        Time32MillisecondArray::from(vec![1, 2, 3, 4])
    );
    def_temporal_test!(
        test_filter_time64_microsecond,
        Time64MicrosecondArray,
        Time64MicrosecondArray::from(vec![1, 2, 3, 4])
    );
    def_temporal_test!(
        test_filter_time64_nanosecond,
        Time64NanosecondArray,
        Time64NanosecondArray::from(vec![1, 2, 3, 4])
    );
    def_temporal_test!(
        test_filter_duration_second,
        DurationSecondArray,
        DurationSecondArray::from(vec![1, 2, 3, 4])
    );
    def_temporal_test!(
        test_filter_duration_millisecond,
        DurationMillisecondArray,
        DurationMillisecondArray::from(vec![1, 2, 3, 4])
    );
    def_temporal_test!(
        test_filter_duration_microsecond,
        DurationMicrosecondArray,
        DurationMicrosecondArray::from(vec![1, 2, 3, 4])
    );
    def_temporal_test!(
        test_filter_duration_nanosecond,
        DurationNanosecondArray,
        DurationNanosecondArray::from(vec![1, 2, 3, 4])
    );
    def_temporal_test!(
        test_filter_timestamp_second,
        TimestampSecondArray,
        TimestampSecondArray::from_vec(vec![1, 2, 3, 4], None)
    );
    def_temporal_test!(
        test_filter_timestamp_millisecond,
        TimestampMillisecondArray,
        TimestampMillisecondArray::from_vec(vec![1, 2, 3, 4], None)
    );
    def_temporal_test!(
        test_filter_timestamp_microsecond,
        TimestampMicrosecondArray,
        TimestampMicrosecondArray::from_vec(vec![1, 2, 3, 4], None)
    );
    def_temporal_test!(
        test_filter_timestamp_nanosecond,
        TimestampNanosecondArray,
        TimestampNanosecondArray::from_vec(vec![1, 2, 3, 4], None)
    );

    #[test]
    fn test_filter_array() {
        let a = Int32Array::from(vec![5, 6, 7, 8, 9]);
        let b = BooleanArray::from(vec![true, false, false, true, false]);
        let c = filter(&a, &b).unwrap();
        let d = c.as_ref().as_any().downcast_ref::<Int32Array>().unwrap();
        assert_eq!(2, d.len());
        assert_eq!(5, d.value(0));
        assert_eq!(8, d.value(1));
    }

    #[test]
    fn test_filter_string_array() {
        let a = StringArray::from(vec!["hello", " ", "world", "!"]);
        let b = BooleanArray::from(vec![true, false, true, false]);
        let c = filter(&a, &b).unwrap();
        let d = c.as_ref().as_any().downcast_ref::<StringArray>().unwrap();
        assert_eq!(2, d.len());
        assert_eq!("hello", d.value(0));
        assert_eq!("world", d.value(1));
    }

    #[test]
    fn test_filter_array_with_null() {
        let a = Int32Array::from(vec![Some(5), None]);
        let b = BooleanArray::from(vec![false, true]);
        let c = filter(&a, &b).unwrap();
        let d = c.as_ref().as_any().downcast_ref::<Int32Array>().unwrap();
        assert_eq!(1, d.len());
        assert_eq!(true, d.is_null(0));
    }
}
