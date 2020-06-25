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

//! Defines trait for array element comparison

use std::cmp::Ordering;

use crate::array::*;
use crate::datatypes::*;
use crate::error::{ArrowError, Result};

use TimeUnit::*;

/// Trait for Arrays that can be sorted
///
/// Example:
/// ```
/// use std::cmp::Ordering;
/// use some::array::*;
/// use some::datatypes::*;
///
/// let arr: Box<dyn OrdArray> = Box::new(PrimitiveArray::<Int64Type>::from(vec![
///     Some(-2),
///     Some(89),
///     Some(-64),
///     Some(101),
/// ]));
///
/// assert_eq!(arr.cmp_value(1, 2), Ordering::Greater);
/// ```
pub trait OrdArray: Array {
    /// Return ordering between array element at index i and j
    fn cmp_value(&self, i: usize, j: usize) -> Ordering;
}

impl<T: ArrowPrimitiveType> OrdArray for PrimitiveArray<T>
where
    T::Native: std::cmp::Ord,
{
    fn cmp_value(&self, i: usize, j: usize) -> Ordering {
        self.value(i).cmp(&self.value(j))
    }
}

impl OrdArray for StringArray {
    fn cmp_value(&self, i: usize, j: usize) -> Ordering {
        self.value(i).cmp(self.value(j))
    }
}

impl OrdArray for NullArray {
    fn cmp_value(&self, _i: usize, _j: usize) -> Ordering {
        Ordering::Equal
    }
}

/// Convert ArrayRef to OrdArray trait object
pub fn as_ordarray(values: &ArrayRef) -> Result<Box<&OrdArray>> {
    match values.data_type() {
        DataType::Boolean => Ok(Box::new(as_boolean_array(&values))),
        DataType::Utf8 => Ok(Box::new(as_string_array(&values))),
        DataType::Null => Ok(Box::new(as_null_array(&values))),
        DataType::Int8 => Ok(Box::new(as_primitive_array::<Int8Type>(&values))),
        DataType::Int16 => Ok(Box::new(as_primitive_array::<Int16Type>(&values))),
        DataType::Int32 => Ok(Box::new(as_primitive_array::<Int32Type>(&values))),
        DataType::Int64 => Ok(Box::new(as_primitive_array::<Int64Type>(&values))),
        DataType::UInt8 => Ok(Box::new(as_primitive_array::<UInt8Type>(&values))),
        DataType::UInt16 => Ok(Box::new(as_primitive_array::<UInt16Type>(&values))),
        DataType::UInt32 => Ok(Box::new(as_primitive_array::<UInt32Type>(&values))),
        DataType::UInt64 => Ok(Box::new(as_primitive_array::<UInt64Type>(&values))),
        DataType::Date32(_) => Ok(Box::new(as_primitive_array::<Date32Type>(&values))),
        DataType::Date64(_) => Ok(Box::new(as_primitive_array::<Date64Type>(&values))),
        DataType::Time32(Second) => {
            Ok(Box::new(as_primitive_array::<Time32SecondType>(&values)))
        }
        DataType::Time32(Millisecond) => Ok(Box::new(as_primitive_array::<
            Time32MillisecondType,
        >(&values))),
        DataType::Time64(Microsecond) => Ok(Box::new(as_primitive_array::<
            Time64MicrosecondType,
        >(&values))),
        DataType::Time64(Nanosecond) => Ok(Box::new(as_primitive_array::<
            Time64NanosecondType,
        >(&values))),
        DataType::Timestamp(Second, _) => {
            Ok(Box::new(as_primitive_array::<TimestampSecondType>(&values)))
        }
        DataType::Timestamp(Millisecond, _) => Ok(Box::new(as_primitive_array::<
            TimestampMillisecondType,
        >(&values))),
        DataType::Timestamp(Microsecond, _) => Ok(Box::new(as_primitive_array::<
            TimestampMicrosecondType,
        >(&values))),
        DataType::Timestamp(Nanosecond, _) => Ok(Box::new(as_primitive_array::<
            TimestampNanosecondType,
        >(&values))),
        DataType::Interval(IntervalUnit::YearMonth) => Ok(Box::new(
            as_primitive_array::<IntervalYearMonthType>(&values),
        )),
        DataType::Interval(IntervalUnit::DayTime) => {
            Ok(Box::new(as_primitive_array::<IntervalDayTimeType>(&values)))
        }
        DataType::Duration(TimeUnit::Second) => {
            Ok(Box::new(as_primitive_array::<DurationSecondType>(&values)))
        }
        DataType::Duration(TimeUnit::Millisecond) => Ok(Box::new(as_primitive_array::<
            DurationMillisecondType,
        >(&values))),
        DataType::Duration(TimeUnit::Microsecond) => Ok(Box::new(as_primitive_array::<
            DurationMicrosecondType,
        >(&values))),
        DataType::Duration(TimeUnit::Nanosecond) => Ok(Box::new(as_primitive_array::<
            DurationNanosecondType,
        >(&values))),
        t => Err(ArrowError::ComputeError(format!(
            "Lexical Sort not supported for data type {:?}",
            t
        ))),
    }
}
