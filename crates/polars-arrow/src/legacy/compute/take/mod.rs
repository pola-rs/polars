pub mod bitmap;
mod boolean;
#[cfg(feature = "dtype-array")]
mod fixed_size_list;

use polars_utils::slice::GetSaferUnchecked;
use crate::array::*;
use crate::bitmap::MutableBitmap;
use crate::buffer::Buffer;
use crate::datatypes::{ArrowDataType, PhysicalType};
use crate::legacy::bit_util::unset_bit_raw;
use crate::legacy::prelude::*;
use crate::legacy::trusted_len::{TrustedLenPush};
use crate::legacy::utils::CustomIterTools;
use crate::offset::Offsets;
use crate::trusted_len::TrustedLen;
use crate::types::NativeType;
use crate::with_match_primitive_type;

/// # Safety
/// Does not do bounds checks
pub unsafe fn take_unchecked(arr: &dyn Array, idx: &IdxArr) -> ArrayRef {
    if idx.null_count() == idx.len() {
        return new_null_array(arr.data_type().clone(), idx.len());
    }
    use PhysicalType::*;
    match arr.data_type().to_physical_type() {
        Primitive(primitive) => with_match_primitive_type!(primitive, |$T| {
            let arr: &PrimitiveArray<$T> = arr.as_any().downcast_ref().unwrap();
            take_primitive_unchecked::<$T>(arr, idx)
        }),
        Boolean => {
            let arr = arr.as_any().downcast_ref().unwrap();
            Box::new(boolean::take_unchecked(arr, idx))
        },
        #[cfg(feature = "dtype-array")]
        FixedSizeList => {
            let arr = arr.as_any().downcast_ref().unwrap();
            Box::new(fixed_size_list::take_unchecked(arr, idx))
        },
        // TODO! implement proper unchecked version
        #[cfg(feature = "compute")]
        _ => {
            use crate::compute::take::take;
            take(arr, idx).unwrap()
        },
        #[cfg(not(feature = "compute"))]
        _ => {
            panic!("activate compute feature")
        },
    }
}

/// Take kernel for single chunk with nulls and arrow array as index that may have nulls.
/// # Safety
/// caller must ensure indices are in bounds
pub unsafe fn take_primitive_unchecked<T: NativeType>(
    arr: &PrimitiveArray<T>,
    indices: &IdxArr,
) -> Box<PrimitiveArray<T>> {
    let array_values = arr.values().as_slice();
    let index_values = indices.values().as_slice();
    let validity_values = arr.validity().expect("should have nulls");

    // first take the values, these are always needed
    let values: Vec<T> = index_values
        .iter()
        .map(|idx| {
            *array_values.get_unchecked_release(*idx as usize)
        })
        .collect_trusted();

    let arr = if arr.null_count() > 0 {
        // the validity buffer we will fill with all valid. And we unset the ones that are null
        // in later checks
        // this is in the assumption that most values will be valid.
        // Maybe we could add another branch based on the null count
        let mut validity = MutableBitmap::with_capacity(indices.len());
        validity.extend_constant(indices.len(), true);
        let validity_ptr = validity.as_slice().as_ptr() as *mut u8;

        if let Some(validity_indices) = indices.validity().as_ref() {
            index_values.iter().enumerate().for_each(|(i, idx)| {
                // i is iteration count
                // idx is the index that we take from the values array.
                let idx = *idx as usize;
                if !validity_indices.get_bit_unchecked(i) || !validity_values.get_bit_unchecked(idx) {
                    unset_bit_raw(validity_ptr, i);
                }
            });
        } else {
            index_values.iter().enumerate().for_each(|(i, idx)| {
                let idx = *idx as usize;
                if !validity_values.get_bit_unchecked(idx) {
                    unset_bit_raw(validity_ptr, i);
                }
            });
        };
        PrimitiveArray::new(T::PRIMITIVE.into(), values.into(), Some(validity.into()))
    } else {
        PrimitiveArray::new(T::PRIMITIVE.into(), values.into(), indices.validity().cloned())
    };

    Box::new(arr)
}

/// Forked and adapted from arrow-rs
/// This is faster because it does no bounds checks and allocates directly into aligned memory
///
/// Takes/filters a list array's inner data using the offsets of the list array.
///
/// Where a list array has indices `[0,2,5,10]`, taking indices of `[2,0]` returns
/// an array of the indices `[5..10, 0..2]` and offsets `[0,5,7]` (5 elements and 2
/// elements)
///
/// # Safety
/// No bounds checks
pub unsafe fn take_value_indices_from_list(
    list: &ListArray<i64>,
    indices: &IdxArr,
) -> (IdxArr, Offsets<i64>) {
    let offsets = list.offsets().as_slice();

    let mut new_offsets = Vec::with_capacity(indices.len());
    // will likely have at least indices.len values
    let mut values = Vec::with_capacity(indices.len());
    let mut current_offset = 0;
    // add first offset
    new_offsets.push(0);
    // compute the value indices, and set offsets accordingly

    let indices_values = indices.values();

    if !indices.has_validity() {
        for i in 0..indices.len() {
            let idx = *indices_values.get_unchecked(i) as usize;
            let start = *offsets.get_unchecked(idx);
            let end = *offsets.get_unchecked(idx + 1);
            current_offset += end - start;
            new_offsets.push(current_offset);

            let mut curr = start;

            // if start == end, this slot is empty
            while curr < end {
                values.push(curr as IdxSize);
                curr += 1;
            }
        }
    } else {
        let validity = indices.validity().expect("should have nulls");

        for i in 0..indices.len() {
            if validity.get_bit_unchecked(i) {
                let idx = *indices_values.get_unchecked(i) as usize;
                let start = *offsets.get_unchecked(idx);
                let end = *offsets.get_unchecked(idx + 1);
                current_offset += end - start;
                new_offsets.push(current_offset);

                let mut curr = start;

                // if start == end, this slot is empty
                while curr < end {
                    values.push(curr as IdxSize);
                    curr += 1;
                }
            } else {
                new_offsets.push(current_offset);
            }
        }
    }

    // Safety:
    // offsets are monotonically increasing.
    unsafe {
        (
            IdxArr::from_data_default(values.into(), None),
            Offsets::new_unchecked(new_offsets),
        )
    }
}
