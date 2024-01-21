pub mod bitmap;
mod boolean;
#[cfg(feature = "dtype-array")]
mod fixed_size_list;

use polars_utils::slice::GetSaferUnchecked;

use crate::array::*;
use crate::bitmap::{Bitmap, MutableBitmap};
use crate::buffer::Buffer;
use crate::datatypes::PhysicalType;
use crate::legacy::bit_util::unset_bit_raw;
use crate::legacy::prelude::*;
use crate::legacy::utils::CustomIterTools;
use crate::offset::Offsets;
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
            take_primitive_unchecked::<$T>(arr, idx).boxed()
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
        BinaryView => take_binview_unchecked(arr.as_any().downcast_ref().unwrap(), idx).boxed(),
        Utf8View => {
            let arr: &Utf8ViewArray = arr.as_any().downcast_ref().unwrap();
            take_binview_unchecked(&arr.to_binview(), idx)
                .to_utf8view_unchecked()
                .boxed()
        },
        Struct => {
            let array = arr.as_any().downcast_ref().unwrap();
            take_struct_unchecked(array, idx).boxed()
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

unsafe fn take_validity_unchecked(validity: Option<&Bitmap>, indices: &IdxArr) -> Option<Bitmap> {
    let indices_validity = indices.validity();
    match (validity, indices_validity) {
        (None, _) => indices_validity.cloned(),
        (Some(validity), None) => {
            let iter = indices
                .values()
                .iter()
                .map(|index| validity.get_bit_unchecked(*index as usize));
            MutableBitmap::from_trusted_len_iter(iter).into()
        },
        (Some(validity), _) => {
            let iter = indices.iter().map(|x| match x {
                Some(index) => validity.get_bit_unchecked(*index as usize),
                None => false,
            });
            MutableBitmap::from_trusted_len_iter(iter).into()
        },
    }
}

/// # Safety
/// No bound checks
pub unsafe fn take_struct_unchecked(array: &StructArray, indices: &IdxArr) -> StructArray {
    let values: Vec<Box<dyn Array>> = array
        .values()
        .iter()
        .map(|a| take_unchecked(a.as_ref(), indices))
        .collect();
    let validity = take_validity_unchecked(array.validity(), indices);
    StructArray::new(array.data_type().clone(), values, validity)
}

/// # Safety
/// No bound checks
unsafe fn take_binview_unchecked(arr: &BinaryViewArray, indices: &IdxArr) -> BinaryViewArray {
    let views = arr.views().clone();
    // PrimitiveArray<u128> is not supported, so we go via i128
    let views = std::mem::transmute::<Buffer<u128>, Buffer<i128>>(views);
    let views = PrimitiveArray::from_data_default(views, arr.validity().cloned());
    let taken_views = take_primitive_unchecked(&views, indices);
    let taken_views_values = taken_views.values().clone();
    let taken_views_values = std::mem::transmute::<Buffer<i128>, Buffer<u128>>(taken_views_values);
    BinaryViewArray::new_unchecked_unknown_md(
        arr.data_type().clone(),
        taken_views_values,
        arr.data_buffers().clone(),
        taken_views.validity().cloned(),
    )
    .maybe_gc()
}

/// Take kernel for single chunk with nulls and arrow array as index that may have nulls.
/// # Safety
/// caller must ensure indices are in bounds
pub unsafe fn take_primitive_unchecked<T: NativeType>(
    arr: &PrimitiveArray<T>,
    indices: &IdxArr,
) -> PrimitiveArray<T> {
    let array_values = arr.values().as_slice();
    let index_values = indices.values().as_slice();

    // first take the values, these are always needed
    let values: Vec<T> = index_values
        .iter()
        .map(|idx| *array_values.get_unchecked_release(*idx as usize))
        .collect_trusted();

    let arr = if arr.null_count() > 0 {
        let validity_values = arr.validity().unwrap();
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
                if !validity_indices.get_bit_unchecked(i) || !validity_values.get_bit_unchecked(idx)
                {
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
        PrimitiveArray::new_unchecked(
            arr.data_type().clone(),
            values.into(),
            Some(validity.into()),
        )
    } else {
        PrimitiveArray::new_unchecked(
            arr.data_type().clone(),
            values.into(),
            indices.validity().cloned(),
        )
    };

    arr
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
