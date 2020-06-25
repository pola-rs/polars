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

//! Defines temporal kernels for time and date related functions.

use chrono::Timelike;

use crate::array::*;
use crate::datatypes::*;
use crate::error::Result;

/// Extracts the hours of a given temporal array as an array of integers
pub fn hour<T>(array: &PrimitiveArray<T>) -> Result<Int32Array>
where
    T: ArrowTemporalType + ArrowNumericType,
    i64: std::convert::From<T::Native>,
{
    let mut b = Int32Builder::new(array.len());
    for i in 0..array.len() {
        if array.is_null(i) {
            b.append_null()?;
        } else {
            match array.data_type() {
                &DataType::Time32(_) | &DataType::Time64(_) => {
                    match array.value_as_time(i) {
                        Some(time) => b.append_value(time.hour() as i32)?,
                        None => b.append_null()?,
                    }
                }
                _ => match array.value_as_datetime(i) {
                    Some(dt) => b.append_value(dt.hour() as i32)?,
                    None => b.append_null()?,
                },
            }
        }
    }

    Ok(b.finish())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_temporal_array_date64_hour() {
        let a: PrimitiveArray<Date64Type> =
            vec![Some(1514764800000), None, Some(1550636625000)].into();

        // get hour from temporal
        let b = hour(&a).unwrap();
        assert_eq!(0, b.value(0));
        assert_eq!(false, b.is_valid(1));
        assert_eq!(4, b.value(2));
    }

    #[test]
    fn test_temporal_array_time32_second_hour() {
        let a: PrimitiveArray<Time32SecondType> = vec![37800, 86339].into();

        // get hour from temporal
        let b = hour(&a).unwrap();
        assert_eq!(10, b.value(0));
        assert_eq!(23, b.value(1));
    }
}
