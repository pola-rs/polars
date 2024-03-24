//! Contains operators to filter arrays such as [`filter`].
mod boolean;
mod primitive;
mod scalar;

#[cfg(all(target_arch = "x86_64", feature = "simd"))]
mod avx512;

use arrow::array::growable::make_growable;
use arrow::array::{new_empty_array, Array, BinaryViewArray, BooleanArray, PrimitiveArray};
use arrow::bitmap::utils::SlicesIterator;
use arrow::datatypes::ArrowDataType;
use arrow::with_match_primitive_type_full;
use polars_error::PolarsResult;

pub fn filter(array: &dyn Array, mask: &BooleanArray) -> PolarsResult<Box<dyn Array>> {
    assert_eq!(array.len(), mask.len());

    // Treat null mask values as false.
    if let Some(validities) = mask.validity() {
        let values = mask.values();
        let new_values = values & validities;
        let mask = BooleanArray::new(ArrowDataType::Boolean, new_values, None);
        return filter(array, &mask);
    }

    // Fast-path: completely empty or completely full mask.
    let false_count = mask.values().unset_bits();
    if false_count == mask.len() {
        return Ok(new_empty_array(array.data_type().clone()));
    }
    if false_count == 0 {
        return Ok(array.to_boxed());
    }

    use arrow::datatypes::PhysicalType::*;
    match array.data_type().to_physical_type() {
        Primitive(primitive) => with_match_primitive_type_full!(primitive, |$T| {
            let array: &PrimitiveArray<$T> = array.as_any().downcast_ref().unwrap();
            let (values, validity) = primitive::filter_values_and_validity::<$T>(array.values(), array.validity(), mask.values());
            Ok(Box::new(PrimitiveArray::from_vec(values).with_validity(validity)))
        }),
        Boolean => {
            let array = array.as_any().downcast_ref::<BooleanArray>().unwrap();
            let (values, validity) = boolean::filter_bitmap_and_validity(
                array.values(),
                array.validity(),
                mask.values(),
            );
            Ok(BooleanArray::new(array.data_type().clone(), values, validity).boxed())
        },
        BinaryView => {
            let array = array.as_any().downcast_ref::<BinaryViewArray>().unwrap();
            let views = array.views();
            let validity = array.validity();
            let (views, validity) =
                primitive::filter_values_and_validity(views, validity, mask.values());
            Ok(unsafe {
                BinaryViewArray::new_unchecked_unknown_md(
                    array.data_type().clone(),
                    views.into(),
                    array.data_buffers().clone(),
                    validity,
                    Some(array.total_buffer_len()),
                )
            }
            .boxed())
        },
        // Should go via BinaryView
        Utf8View => {
            unreachable!()
        },
        _ => {
            let iter = SlicesIterator::new(mask.values());
            let mut mutable = make_growable(&[array], false, iter.slots());
            // SAFETY:
            // we are in bounds
            iter.for_each(|(start, len)| unsafe { mutable.extend(0, start, len) });
            Ok(mutable.as_box())
        },
    }
}
