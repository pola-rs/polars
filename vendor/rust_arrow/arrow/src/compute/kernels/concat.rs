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

//! Defines concat kernel for `ArrayRef`
//!
//! Example:
//!
//! ```
//! use std::sync::Arc;
//! use some::array::{ArrayRef, StringArray};
//! use some::compute::concat;
//!
//! let arr = concat(&vec![
//!     Arc::new(StringArray::from(vec!["hello", "world"])) as ArrayRef,
//!     Arc::new(StringArray::from(vec!["!"])) as ArrayRef,
//! ]).unwrap();
//! assert_eq!(arr.len(), 3);
//! ```

use crate::array::*;
use crate::datatypes::*;
use crate::error::{ArrowError, Result};

use TimeUnit::*;

/// Concatenate multiple `ArrayRef` with the same type.
///
/// Returns a new ArrayRef.
pub fn concat(array_list: &[ArrayRef]) -> Result<ArrayRef> {
    if array_list.is_empty() {
        return Err(ArrowError::ComputeError(
            "concat requires input of at least one array".to_string(),
        ));
    }
    let array_data_list = &array_list
        .iter()
        .map(|a| a.data_ref().clone())
        .collect::<Vec<ArrayDataRef>>();

    match array_data_list[0].data_type() {
        DataType::Utf8 => {
            let mut builder = StringArray::builder(0);
            builder.append_data(array_data_list)?;
            Ok(ArrayBuilder::finish(&mut builder))
        }
        DataType::Boolean => {
            let mut builder = PrimitiveArray::<BooleanType>::builder(0);
            builder.append_data(array_data_list)?;
            Ok(ArrayBuilder::finish(&mut builder))
        }
        DataType::Int8 => concat_primitive::<Int8Type>(array_data_list),
        DataType::Int16 => concat_primitive::<Int16Type>(array_data_list),
        DataType::Int32 => concat_primitive::<Int32Type>(array_data_list),
        DataType::Int64 => concat_primitive::<Int64Type>(array_data_list),
        DataType::UInt8 => concat_primitive::<UInt8Type>(array_data_list),
        DataType::UInt16 => concat_primitive::<UInt16Type>(array_data_list),
        DataType::UInt32 => concat_primitive::<UInt32Type>(array_data_list),
        DataType::UInt64 => concat_primitive::<UInt64Type>(array_data_list),
        DataType::Float32 => concat_primitive::<Float32Type>(array_data_list),
        DataType::Float64 => concat_primitive::<Float64Type>(array_data_list),
        DataType::Date32(_) => concat_primitive::<Date32Type>(array_data_list),
        DataType::Date64(_) => concat_primitive::<Date64Type>(array_data_list),
        DataType::Time32(Second) => concat_primitive::<Time32SecondType>(array_data_list),
        DataType::Time32(Millisecond) => {
            concat_primitive::<Time32MillisecondType>(array_data_list)
        }
        DataType::Time64(Microsecond) => {
            concat_primitive::<Time64MicrosecondType>(array_data_list)
        }
        DataType::Time64(Nanosecond) => {
            concat_primitive::<Time64NanosecondType>(array_data_list)
        }
        DataType::Timestamp(Second, _) => {
            concat_primitive::<TimestampSecondType>(array_data_list)
        }
        DataType::Timestamp(Millisecond, _) => {
            concat_primitive::<TimestampMillisecondType>(array_data_list)
        }
        DataType::Timestamp(Microsecond, _) => {
            concat_primitive::<TimestampMicrosecondType>(array_data_list)
        }
        DataType::Timestamp(Nanosecond, _) => {
            concat_primitive::<TimestampNanosecondType>(array_data_list)
        }
        DataType::Interval(IntervalUnit::YearMonth) => {
            concat_primitive::<IntervalYearMonthType>(array_data_list)
        }
        DataType::Interval(IntervalUnit::DayTime) => {
            concat_primitive::<IntervalDayTimeType>(array_data_list)
        }
        DataType::Duration(TimeUnit::Second) => {
            concat_primitive::<DurationSecondType>(array_data_list)
        }
        DataType::Duration(TimeUnit::Millisecond) => {
            concat_primitive::<DurationMillisecondType>(array_data_list)
        }
        DataType::Duration(TimeUnit::Microsecond) => {
            concat_primitive::<DurationMicrosecondType>(array_data_list)
        }
        DataType::Duration(TimeUnit::Nanosecond) => {
            concat_primitive::<DurationNanosecondType>(array_data_list)
        }
        t => Err(ArrowError::ComputeError(format!(
            "Concat not supported for data type {:?}",
            t
        ))),
    }
}

#[inline]
fn concat_primitive<T>(array_data_list: &[ArrayDataRef]) -> Result<ArrayRef>
where
    T: ArrowNumericType,
{
    let mut builder = PrimitiveArray::<T>::builder(0);
    builder.append_data(array_data_list)?;
    Ok(ArrayBuilder::finish(&mut builder))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::convert::TryFrom;
    use std::sync::Arc;

    #[test]
    fn test_concat_empty_vec() -> Result<()> {
        let re = concat(&vec![]);
        assert!(re.is_err());
        Ok(())
    }

    #[test]
    fn test_concat_incompatible_datatypes() -> Result<()> {
        let re = concat(&vec![
            Arc::new(PrimitiveArray::<Int64Type>::from(vec![
                Some(-1),
                Some(2),
                None,
            ])) as ArrayRef,
            Arc::new(
                StringArray::try_from(vec![Some("hello"), Some("bar"), Some("world")])
                    .expect("Unable to create string array"),
            ) as ArrayRef,
        ]);
        assert!(re.is_err());
        Ok(())
    }

    #[test]
    fn test_concat_string_arrays() -> Result<()> {
        let arr = concat(&vec![
            Arc::new(
                StringArray::try_from(vec![Some("hello"), Some("world")])
                    .expect("Unable to create string array"),
            ) as ArrayRef,
            Arc::new(StringArray::from(vec!["1", "2", "3", "4", "6"])).slice(1, 3),
            Arc::new(
                StringArray::try_from(vec![Some("foo"), Some("bar"), None, Some("baz")])
                    .expect("Unable to create string array"),
            ) as ArrayRef,
        ])?;

        let expected_output = Arc::new(
            StringArray::try_from(vec![
                Some("hello"),
                Some("world"),
                Some("2"),
                Some("3"),
                Some("4"),
                Some("foo"),
                Some("bar"),
                None,
                Some("baz"),
            ])
            .expect("Unable to create string array"),
        ) as ArrayRef;

        assert!(
            arr.equals(&(*expected_output)),
            "expect {:#?} to be: {:#?}",
            arr,
            &expected_output
        );

        Ok(())
    }

    #[test]
    fn test_concat_primitive_arrays() -> Result<()> {
        let arr = concat(&vec![
            Arc::new(PrimitiveArray::<Int64Type>::from(vec![
                Some(-1),
                Some(-1),
                Some(2),
                None,
                None,
            ])) as ArrayRef,
            Arc::new(PrimitiveArray::<Int64Type>::from(vec![
                Some(101),
                Some(102),
                Some(103),
                None,
            ])) as ArrayRef,
            Arc::new(PrimitiveArray::<Int64Type>::from(vec![
                Some(256),
                Some(512),
                Some(1024),
            ])) as ArrayRef,
        ])?;

        let expected_output = Arc::new(PrimitiveArray::<Int64Type>::from(vec![
            Some(-1),
            Some(-1),
            Some(2),
            None,
            None,
            Some(101),
            Some(102),
            Some(103),
            None,
            Some(256),
            Some(512),
            Some(1024),
        ])) as ArrayRef;

        assert!(
            arr.equals(&(*expected_output)),
            "expect {:#?} to be: {:#?}",
            arr,
            &expected_output
        );

        Ok(())
    }

    #[test]
    fn test_concat_boolean_primitive_arrays() -> Result<()> {
        let arr = concat(&vec![
            Arc::new(PrimitiveArray::<BooleanType>::from(vec![
                Some(true),
                Some(true),
                Some(false),
                None,
                None,
                Some(false),
            ])) as ArrayRef,
            Arc::new(PrimitiveArray::<BooleanType>::from(vec![
                None,
                Some(false),
                Some(true),
                Some(false),
            ])) as ArrayRef,
        ])?;

        let expected_output = Arc::new(PrimitiveArray::<BooleanType>::from(vec![
            Some(true),
            Some(true),
            Some(false),
            None,
            None,
            Some(false),
            None,
            Some(false),
            Some(true),
            Some(false),
        ])) as ArrayRef;

        assert!(
            arr.equals(&(*expected_output)),
            "expect {:#?} to be: {:#?}",
            arr,
            &expected_output
        );

        Ok(())
    }
}
