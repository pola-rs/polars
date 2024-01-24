//! Contains operators to filter arrays such as [`filter`].
mod boolean;
mod primitive;

use arrow::array::growable::make_growable;
use arrow::array::*;
use arrow::bitmap::utils::{BitChunkIterExact, BitChunksExact, SlicesIterator};
use arrow::bitmap::{Bitmap, MutableBitmap};
use arrow::datatypes::ArrowDataType;
use arrow::types::simd::Simd;
use arrow::types::{BitChunkOnes, NativeType};
use arrow::with_match_primitive_type_full;
use boolean::*;
use polars_error::*;
use primitive::*;

/// Function that can filter arbitrary arrays
pub type Filter<'a> = Box<dyn Fn(&dyn Array) -> Box<dyn Array> + 'a + Send + Sync>;

#[inline]
fn get_leading_ones(chunk: u64) -> u32 {
    if cfg!(target_endian = "little") {
        chunk.trailing_ones()
    } else {
        chunk.leading_ones()
    }
}

pub fn filter(array: &dyn Array, mask: &BooleanArray) -> PolarsResult<Box<dyn Array>> {
    // The validities may be masking out `true` bits, making the filter operation
    // based on the values incorrect
    if let Some(validities) = mask.validity() {
        let values = mask.values();
        let new_values = values & validities;
        let mask = BooleanArray::new(ArrowDataType::Boolean, new_values, None);
        return filter(array, &mask);
    }

    let false_count = mask.values().unset_bits();
    if false_count == mask.len() {
        assert_eq!(array.len(), mask.len());
        return Ok(new_empty_array(array.data_type().clone()));
    }
    if false_count == 0 {
        assert_eq!(array.len(), mask.len());
        return Ok(array.to_boxed());
    }

    use arrow::datatypes::PhysicalType::*;
    match array.data_type().to_physical_type() {
        Primitive(primitive) => with_match_primitive_type_full!(primitive, |$T| {
            let array = array.as_any().downcast_ref().unwrap();
            Ok(Box::new(filter_primitive::<$T>(array, mask.values())))
        }),
        Boolean => {
            let array = array.as_any().downcast_ref::<BooleanArray>().unwrap();
            let (values, validity) =
                filter_bitmap_and_validity(array.values(), array.validity(), mask.values());
            Ok(BooleanArray::new(
                array.data_type().clone(),
                values.freeze(),
                validity.map(|v| v.freeze()),
            )
            .boxed())
        },
        BinaryView => {
            let array = array.as_any().downcast_ref::<BinaryViewArray>().unwrap();
            let views = array.views();
            let validity = array.validity();
            // TODO! we might opt for a filter that maintains the bytes_count
            // currently we don't do that and bytes_len is set to UNKNOWN.
            let (views, validity) = filter_values_and_validity(views, validity, mask.values());
            Ok(unsafe {
                BinaryViewArray::new_unchecked_unknown_md(
                    array.data_type().clone(),
                    views.into(),
                    array.data_buffers().clone(),
                    validity.map(|v| v.freeze()),
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
