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

//! Defines kernel to extract a substring of a \[Large\]StringArray

use crate::{
    array::*,
    datatypes::DataType,
    error::{Error, Result},
    offset::{Offset, Offsets},
};

/// Returns a Utf8Array<O> with a substring starting from `start` and with optional length `length` of each of the elements in `array`.
/// `start` can be negative, in which case the start counts from the end of the string.
pub fn utf8_substring<O: Offset>(
    array: &Utf8Array<O>,
    start: O,
    length: &Option<O>,
) -> Utf8Array<O> {
    let length = length.map(|v| v.to_usize());

    let iter = array.values_iter().map(|str_val| {
        // compute where we should start slicing this entry.
        let start = if start >= O::zero() {
            start.to_usize()
        } else {
            let start = (O::zero() - start).to_usize();
            str_val
                .char_indices()
                .rev()
                .nth(start)
                .map(|(idx, _)| idx + 1)
                .unwrap_or(0)
        };

        let mut iter_chars = str_val.char_indices();
        if let Some((start_idx, _)) = iter_chars.nth(start) {
            // length of the str
            let len_end = str_val.len() - start_idx;

            // length to slice
            let length = length.unwrap_or(len_end);

            if length == 0 {
                return "";
            }
            // compute
            let end_idx = iter_chars
                .nth(length.saturating_sub(1))
                .map(|(idx, _)| idx)
                .unwrap_or(str_val.len());

            &str_val[start_idx..end_idx]
        } else {
            ""
        }
    });

    let new = Utf8Array::<O>::from_trusted_len_values_iter(iter);
    new.with_validity(array.validity().cloned())
}

/// Returns a BinaryArray<O> with a substring starting from `start` and with optional length `length` of each of the elements in `array`.
/// `start` can be negative, in which case the start counts from the end of the string.
pub fn binary_substring<O: Offset>(
    array: &BinaryArray<O>,
    start: O,
    length: &Option<O>,
) -> BinaryArray<O> {
    let validity = array.validity();
    let offsets = array.offsets();
    let values = array.values();

    let mut new_offsets = Offsets::<O>::with_capacity(array.len());
    let mut new_values = Vec::<u8>::new(); // we have no way to estimate how much this will be.

    offsets.buffer().windows(2).for_each(|windows| {
        let length_i: O = windows[1] - windows[0];

        // compute where we should start slicing this entry
        let start = windows[0]
            + if start >= O::zero() {
                start
            } else {
                length_i + start
            };
        let start = start.clamp(windows[0], windows[1]);

        let length: O = length
            .unwrap_or(length_i)
            // .max(0) is not needed as it is guaranteed
            .min(windows[1] - start); // so we do not go beyond this entry
        new_offsets
            .try_push(length)
            .expect("Substring is always smaller than original - overflow never happens");

        // we need usize for ranges
        let start = start.to_usize();
        let length = length.to_usize();

        new_values.extend_from_slice(&values[start..start + length]);
    });

    BinaryArray::<O>::new(
        array.data_type().clone(),
        new_offsets.into(),
        new_values.into(),
        validity.cloned(),
    )
}

/// Returns an ArrayRef with a substring starting from `start` and with optional length `length` of each of the elements in `array`.
/// `start` can be negative, in which case the start counts from the end of the string.
/// this function errors when the passed array is not a \[Large\]String array.
pub fn substring(array: &dyn Array, start: i64, length: &Option<u64>) -> Result<Box<dyn Array>> {
    match array.data_type() {
        DataType::Binary => Ok(Box::new(binary_substring(
            array
                .as_any()
                .downcast_ref::<BinaryArray<i32>>()
                .expect("A binary is expected"),
            start as i32,
            &length.map(|e| e as i32),
        ))),
        DataType::LargeBinary => Ok(Box::new(binary_substring(
            array
                .as_any()
                .downcast_ref::<BinaryArray<i64>>()
                .expect("A large binary is expected"),
            start,
            &length.map(|e| e as i64),
        ))),
        DataType::LargeUtf8 => Ok(Box::new(utf8_substring(
            array
                .as_any()
                .downcast_ref::<Utf8Array<i64>>()
                .expect("A large string is expected"),
            start,
            &length.map(|e| e as i64),
        ))),
        DataType::Utf8 => Ok(Box::new(utf8_substring(
            array
                .as_any()
                .downcast_ref::<Utf8Array<i32>>()
                .expect("A string is expected"),
            start as i32,
            &length.map(|e| e as i32),
        ))),
        _ => Err(Error::InvalidArgumentError(format!(
            "substring does not support type {:?}",
            array.data_type()
        ))),
    }
}

/// Checks if an array of type `datatype` can perform substring operation
///
/// # Examples
/// ```
/// use arrow2::compute::substring::can_substring;
/// use arrow2::datatypes::{DataType};
///
/// let data_type = DataType::Utf8;
/// assert_eq!(can_substring(&data_type), true);

/// let data_type = DataType::Null;
/// assert_eq!(can_substring(&data_type), false);
/// ```
pub fn can_substring(data_type: &DataType) -> bool {
    matches!(
        data_type,
        DataType::LargeUtf8 | DataType::Utf8 | DataType::LargeBinary | DataType::Binary
    )
}
