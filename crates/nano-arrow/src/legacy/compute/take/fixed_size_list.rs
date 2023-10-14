use crate::array::growable::{Growable, GrowableFixedSizeList};
use crate::array::{Array, FixedSizeListArray, PrimitiveArray};
use crate::bitmap::{Bitmap, MutableBitmap};
use crate::datatypes::{DataType, PhysicalType};
use crate::legacy::index::{IdxArr, IdxSize};
use crate::legacy::prelude::ArrayRef;
use crate::types::NativeType;
use crate::with_match_primitive_type;

pub unsafe fn take_unchecked(values: &FixedSizeListArray, indices: &IdxArr) -> FixedSizeListArray {
    if let (PhysicalType::Primitive(primitive), 0) = (
        values.values().data_type().to_physical_type(),
        indices.null_count(),
    ) {
        let idx = indices.values().as_slice();
        let child_values = values.values();
        let DataType::FixedSizeList(_, width) = values.data_type() else {
            unreachable!()
        };

        with_match_primitive_type!(primitive, |$T| {
            let arr: &PrimitiveArray<$T> = child_values.as_any().downcast_ref().unwrap();
            return take_unchecked_primitive(values, arr, idx, *width)
        })
    }

    let mut capacity = 0;
    let arrays = indices
        .values()
        .iter()
        .map(|index| {
            let index = *index as usize;
            let slice = values.clone().sliced_unchecked(index, 1);
            capacity += slice.len();
            slice
        })
        .collect::<Vec<FixedSizeListArray>>();

    let arrays = arrays.iter().collect();

    if let Some(validity) = indices.validity() {
        let mut growable: GrowableFixedSizeList =
            GrowableFixedSizeList::new(arrays, true, capacity);

        for index in 0..indices.len() {
            if validity.get_bit(index) {
                growable.extend(index, 0, 1);
            } else {
                growable.extend_validity(1)
            }
        }

        growable.into()
    } else {
        let mut growable: GrowableFixedSizeList =
            GrowableFixedSizeList::new(arrays, false, capacity);
        for index in 0..indices.len() {
            growable.extend(index, 0, 1);
        }

        growable.into()
    }
}

unsafe fn take_bitmap_unchecked(bitmap: &Bitmap, idx: &[IdxSize], width: usize) -> Bitmap {
    let mut out = MutableBitmap::with_capacity(idx.len() * width);
    let (slice, offset, _len) = bitmap.as_slice();

    for &idx in idx {
        out.extend_from_slice_unchecked(slice, offset + idx as usize * width, width)
    }
    out.into()
}

unsafe fn take_unchecked_primitive<T: NativeType>(
    parent: &FixedSizeListArray,
    list_values: &PrimitiveArray<T>,
    idx: &[IdxSize],
    width: usize,
) -> FixedSizeListArray {
    let values = list_values.values().as_slice();
    let mut out = Vec::with_capacity(idx.len() * width);

    for &i in idx {
        let start = i as usize * width;
        let end = start + width;
        out.extend_from_slice(values.get_unchecked(start..end));
    }

    let validity = if list_values.null_count() > 0 {
        let validity = list_values.validity().unwrap();
        Some(take_bitmap_unchecked(validity, idx, width))
    } else {
        None
    };
    let list_values = Box::new(PrimitiveArray::new(
        list_values.data_type().clone(),
        out.into(),
        validity,
    )) as ArrayRef;
    let validity = if parent.null_count() > 0 {
        Some(super::bitmap::take_bitmap_unchecked(
            parent.validity().unwrap(),
            idx,
        ))
    } else {
        None
    };
    FixedSizeListArray::new(parent.data_type().clone(), list_values, validity)
}
