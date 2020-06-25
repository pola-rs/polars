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

//! Common utilities for computation kernels.

use crate::array::*;
use crate::bitmap::Bitmap;
use crate::buffer::Buffer;
#[cfg(feature = "simd")]
use crate::datatypes::*;
use crate::error::Result;
#[cfg(feature = "simd")]
use num::One;
#[cfg(feature = "simd")]
use std::cmp::min;

/// Applies a given binary operation, `op`, to two references to `Option<Bitmap>`'s.
///
/// This function is useful when implementing operations on higher level arrays.
pub(super) fn apply_bin_op_to_option_bitmap<F>(
    left: &Option<Bitmap>,
    right: &Option<Bitmap>,
    op: F,
) -> Result<Option<Buffer>>
where
    F: Fn(&Buffer, &Buffer) -> Result<Buffer>,
{
    match *left {
        None => match *right {
            None => Ok(None),
            Some(ref r) => Ok(Some(r.bits.clone())),
        },
        Some(ref l) => match *right {
            None => Ok(Some(l.bits.clone())),
            Some(ref r) => Ok(Some(op(&l.bits, &r.bits)?)),
        },
    }
}

/// Takes/filters a list array's inner data using the offsets of the list array.
///
/// Where a list array has indices `[0,2,5,10]`, taking indices of `[2,0]` returns
/// an array of the indices `[5..10, 0..2]` and offsets `[0,5,7]` (5 elements and 2
/// elements)
pub(super) fn take_value_indices_from_list(
    values: &ArrayRef,
    indices: &UInt32Array,
) -> (UInt32Array, Vec<i32>) {
    // TODO: benchmark this function, there might be a faster unsafe alternative
    // get list array's offsets
    let list: &ListArray = values.as_any().downcast_ref::<ListArray>().unwrap();
    let offsets: Vec<u32> = (0..=list.len())
        .map(|i| list.value_offset(i) as u32)
        .collect();
    let mut new_offsets = Vec::with_capacity(indices.len());
    let mut values = Vec::new();
    let mut current_offset = 0;
    // add first offset
    new_offsets.push(0);
    // compute the value indices, and set offsets accordingly
    for i in 0..indices.len() {
        if indices.is_valid(i) {
            let ix = indices.value(i) as usize;
            let start = offsets[ix];
            let end = offsets[ix + 1];
            current_offset += (end - start) as i32;
            new_offsets.push(current_offset);
            // if start == end, this slot is empty
            if start != end {
                // type annotation needed to guide compiler a bit
                let mut offsets: Vec<Option<u32>> =
                    (start..end).map(Some).collect::<Vec<Option<u32>>>();
                values.append(&mut offsets);
            }
        } else {
            new_offsets.push(current_offset);
        }
    }
    (UInt32Array::from(values), new_offsets)
}

/// Creates a new SIMD mask, i.e. `packed_simd::m32x16` or similar. that indicates if the
/// corresponding array slots represented by the mask are 'valid'.  
///
/// Lanes of the SIMD mask can be set to 'valid' (`true`) if the corresponding array slot is not
/// `NULL`, as indicated by it's `Bitmap`, and is within the length of the array.  Lanes outside the
/// length represent padding and are set to 'invalid' (`false`).
#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "simd"))]
unsafe fn is_valid<T>(
    bitmap: &Option<Bitmap>,
    i: usize,
    simd_width: usize,
    array_len: usize,
) -> T::SimdMask
where
    T: ArrowNumericType,
{
    let simd_upper_bound = i + simd_width;
    let mut validity = T::mask_init(true);

    // Validity based on `Bitmap`
    if let Some(b) = bitmap {
        for j in i..min(array_len, simd_upper_bound) {
            if !b.is_set(j) {
                validity = T::mask_set(validity, j - i, false);
            }
        }
    }

    // Validity based on the length of the Array
    for j in array_len..simd_upper_bound {
        validity = T::mask_set(validity, j - i, false);
    }

    validity
}

/// Performs a SIMD load but sets all 'invalid' lanes to a constant value.
///
/// 'invalid' lanes are lanes where the corresponding array slots are either `NULL` or between the
/// length and capacity of the array, i.e. in the padded region.
///
/// Note that `array` below has it's own `Bitmap` separate from the `bitmap` argument.  This
/// function is used to prepare `array`'s for binary operations.  The `bitmap` argument is the
/// `Bitmap` after the binary operation.
#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "simd"))]
pub(super) unsafe fn simd_load_set_invalid<T>(
    array: &PrimitiveArray<T>,
    bitmap: &Option<Bitmap>,
    i: usize,
    simd_width: usize,
    fill_value: T::Native,
) -> T::Simd
where
    T: ArrowNumericType,
    T::Native: One,
{
    let simd_with_zeros = T::load(array.value_slice(i, simd_width));
    T::mask_select(
        is_valid::<T>(bitmap, i, simd_width, array.len()),
        simd_with_zeros,
        T::init(fill_value),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::sync::Arc;

    use crate::array::ArrayData;
    use crate::datatypes::{DataType, ToByteSlice};

    #[test]
    fn test_apply_bin_op_to_option_bitmap() {
        assert_eq!(
            Ok(None),
            apply_bin_op_to_option_bitmap(&None, &None, |a, b| a & b)
        );
        assert_eq!(
            Ok(Some(Buffer::from([0b01101010]))),
            apply_bin_op_to_option_bitmap(
                &Some(Bitmap::from(Buffer::from([0b01101010]))),
                &None,
                |a, b| a & b,
            )
        );
        assert_eq!(
            Ok(Some(Buffer::from([0b01001110]))),
            apply_bin_op_to_option_bitmap(
                &None,
                &Some(Bitmap::from(Buffer::from([0b01001110]))),
                |a, b| a & b,
            )
        );
        assert_eq!(
            Ok(Some(Buffer::from([0b01001010]))),
            apply_bin_op_to_option_bitmap(
                &Some(Bitmap::from(Buffer::from([0b01101010]))),
                &Some(Bitmap::from(Buffer::from([0b01001110]))),
                |a, b| a & b,
            )
        );
    }

    #[test]
    fn test_take_value_index_from_list() {
        let value_data = Int32Array::from((0..10).collect::<Vec<i32>>()).data();
        let value_offsets = Buffer::from(&[0, 2, 5, 10].to_byte_slice());
        let list_data_type = DataType::List(Box::new(DataType::Int32));
        let list_data = ArrayData::builder(list_data_type.clone())
            .len(3)
            .add_buffer(value_offsets.clone())
            .add_child_data(value_data.clone())
            .build();
        let array = Arc::new(ListArray::from(list_data)) as ArrayRef;
        let index = UInt32Array::from(vec![2, 0]);
        let (indexed, offsets) = take_value_indices_from_list(&array, &index);
        assert_eq!(vec![0, 5, 7], offsets);
        let data = UInt32Array::from(vec![
            Some(5),
            Some(6),
            Some(7),
            Some(8),
            Some(9),
            Some(0),
            Some(1),
        ])
        .data();
        assert_eq!(data, indexed.data());
    }

    #[test]
    #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "simd"))]
    fn test_is_valid() {
        let a = Int32Array::from(vec![
            Some(15),
            None,
            None,
            Some(1),
            None,
            None,
            Some(5),
            None,
            None,
            Some(4),
        ]);
        let simd_lanes = 16;
        let data = a.data();
        let bitmap = data.null_bitmap();
        let result = unsafe { is_valid::<Int32Type>(&bitmap, 0, simd_lanes, a.len()) };
        for i in 0..simd_lanes {
            if i % 3 != 0 || i > 9 {
                assert_eq!(false, result.extract(i));
            } else {
                assert_eq!(true, result.extract(i));
            }
        }
    }

    #[test]
    #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "simd"))]
    fn test_simd_load_set_invalid() {
        let a = Int64Array::from(vec![None, Some(15), Some(5), Some(0)]);
        let new_bitmap = &Some(Bitmap::from(Buffer::from([0b00001010])));
        let simd_lanes = 8;
        let result = unsafe {
            simd_load_set_invalid::<Int64Type>(&a, &new_bitmap, 0, simd_lanes, 1)
        };
        for i in 0..simd_lanes {
            if i == 1 {
                assert_eq!(15_i64, result.extract(i));
            } else if i == 3 {
                assert_eq!(0_i64, result.extract(i));
            } else {
                assert_eq!(1_i64, result.extract(i));
            }
        }
    }
}
