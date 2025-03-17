//! Contains operators to filter arrays such as [`filter`].
mod boolean;
mod primitive;
mod scalar;

#[cfg(all(target_arch = "x86_64", feature = "simd"))]
mod avx512;

use arrow::array::builder::{ArrayBuilder, ShareStrategy, make_builder};
use arrow::array::{
    Array, BinaryViewArray, BooleanArray, PrimitiveArray, Utf8ViewArray, new_empty_array,
};
use arrow::bitmap::Bitmap;
use arrow::bitmap::utils::SlicesIterator;
use arrow::with_match_primitive_type_full;
pub use boolean::filter_boolean_kernel;

pub fn filter(array: &dyn Array, mask: &BooleanArray) -> Box<dyn Array> {
    assert_eq!(array.len(), mask.len());

    // Treat null mask values as false.
    if let Some(validities) = mask.validity() {
        let combined_mask = mask.values() & validities;
        filter_with_bitmap(array, &combined_mask)
    } else {
        filter_with_bitmap(array, mask.values())
    }
}

pub fn filter_with_bitmap(array: &dyn Array, mask: &Bitmap) -> Box<dyn Array> {
    // Many filters involve filtering values in a subsection of the array. When we trim the leading
    // and trailing filtered items, we can close in on those items and not have to perform and
    // thinking about those. The overhead for when there are no leading or trailing filtered values
    // is very minimal: only a clone of the mask and the array.
    //
    // This also allows dispatching to the fast paths way, way, way more often.
    let mut mask = mask.clone();
    let leading_zeros = mask.take_leading_zeros();
    mask.take_trailing_zeros();
    let array = array.sliced(leading_zeros, mask.len());

    let mask = &mask;
    let array = array.as_ref();

    // Fast-path: completely empty or completely full mask.
    let false_count = mask.unset_bits();
    if false_count == mask.len() {
        return new_empty_array(array.dtype().clone());
    }
    if false_count == 0 {
        return array.to_boxed();
    }

    use arrow::datatypes::PhysicalType::*;
    match array.dtype().to_physical_type() {
        Primitive(primitive) => with_match_primitive_type_full!(primitive, |$T| {
            let array: &PrimitiveArray<$T> = array.as_any().downcast_ref().unwrap();
            let (values, validity) = primitive::filter_values_and_validity::<$T>(array.values(), array.validity(), mask);
            Box::new(PrimitiveArray::from_vec(values).with_validity(validity))
        }),
        Boolean => {
            let array = array.as_any().downcast_ref::<BooleanArray>().unwrap();
            let (values, validity) =
                boolean::filter_bitmap_and_validity(array.values(), array.validity(), mask);
            BooleanArray::new(array.dtype().clone(), values, validity).boxed()
        },
        BinaryView => {
            let array = array.as_any().downcast_ref::<BinaryViewArray>().unwrap();
            let views = array.views();
            let validity = array.validity();
            let (views, validity) = primitive::filter_values_and_validity(views, validity, mask);
            unsafe {
                BinaryViewArray::new_unchecked_unknown_md(
                    array.dtype().clone(),
                    views.into(),
                    array.data_buffers().clone(),
                    validity,
                    Some(array.total_buffer_len()),
                )
            }
            .boxed()
        },
        Utf8View => {
            let array = array.as_any().downcast_ref::<Utf8ViewArray>().unwrap();
            let views = array.views();
            let validity = array.validity();
            let (views, validity) = primitive::filter_values_and_validity(views, validity, mask);
            unsafe {
                BinaryViewArray::new_unchecked_unknown_md(
                    arrow::datatypes::ArrowDataType::BinaryView,
                    views.into(),
                    array.data_buffers().clone(),
                    validity,
                    Some(array.total_buffer_len()),
                )
                .to_utf8view_unchecked()
            }
            .boxed()
        },
        _ => {
            let iter = SlicesIterator::new(mask);
            let mut mutable = make_builder(array.dtype());
            mutable.reserve(iter.slots());
            iter.for_each(|(start, len)| {
                mutable.subslice_extend(array, start, len, ShareStrategy::Always)
            });
            mutable.freeze()
        },
    }
}
