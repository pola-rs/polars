//! The kernels that measure what an array holds.

use polars_array::{
    PlArray, PlArrayType, PlBinaryArray, PlBinaryViewArray, PlBitmap, PlBitmapRef, PlBooleanArray,
    PlFixedSizeBinaryArray, PlFixedSizeListArray, PlListArray, PlPrimitiveArray, PlStructArray,
    PlUtf8ViewArray,
};
use polars_arrow::with_match_primitive_type;

/// The length in bytes of every element, read off the views.
pub fn binary_size_bytes(array: &PlBinaryViewArray) -> PlPrimitiveArray<u32> {
    let lengths = match array.scalar_views() {
        Some(view) => PlPrimitiveArray::new_scalar(view.length, array.len()),
        None => PlPrimitiveArray::from_vec(
            array
                .flat_views()
                .unwrap()
                .iter()
                .map(|view| view.length)
                .collect(),
        ),
    };

    lengths.with_validity(array.validity().map(PlBitmap::from))
}

/// The bytes a validity mask takes, which is none where there is no mask.
fn validity_size(validity: Option<PlBitmapRef<'_>>) -> usize {
    validity.map_or(0, |validity| {
        validity.to_flat_or_scalar().as_slice().0.len()
    })
}

/// The number of slots a backing buffer holds: a single one when scalar, one per element flat.
fn buffer_slots(is_scalar: bool, length: usize) -> usize {
    if is_scalar { 1 } else { length }
}

/// Downcasts an array whose [`PlArrayType`] has already been matched on.
#[inline]
fn downcast<A: PlArray>(array: &dyn PlArray) -> &A {
    array
        .as_any()
        .downcast_ref()
        .expect("the array type identifies the concrete array")
}

/// The bytes the views of a view array cover, which is what such an array is measured by.
///
/// The data buffers are shared, so an array is measured by what its views read rather than by what
/// the buffers hold: summing the buffers overestimates and spills to disk when there is no need to.
fn viewed_bytes(array: &PlBinaryViewArray) -> usize {
    match array.scalar_views() {
        // One view is all a scalar array holds, however many elements read it.
        Some(view) => view.length as usize,
        // The array keeps this, so a column measured more than once walks its views only once.
        None => array.total_bytes_len(),
    }
}

/// The bytes the offsets of `array` cover, and the number of offsets it holds.
fn offset_bytes(array: &PlBinaryArray) -> (usize, usize) {
    let slots = if array.offsets_are_scalar() {
        2
    } else {
        array.len() + 1
    };

    let covered = if array.is_empty() {
        0
    } else {
        array.value_range(array.len() - 1).end - array.value_range(0).start
    };

    (covered, slots)
}

/// The bytes the buffers of `array` take, its children included.
pub fn estimated_bytes_size(array: &dyn PlArray) -> usize {
    use PlArrayType as A;

    match array.array_type() {
        A::Null => 0,
        A::Boolean => {
            let array = downcast::<PlBooleanArray>(array);
            array.values().to_flat_or_scalar().as_slice().0.len() + validity_size(array.validity())
        },
        A::Primitive(primitive) => with_match_primitive_type!(primitive, |$T| {
            let array = downcast::<PlPrimitiveArray<$T>>(array);
            buffer_slots(array.values_are_scalar(), array.len()) * size_of::<$T>()
                + validity_size(array.validity())
        }),
        A::Binary => {
            let array = downcast::<PlBinaryArray>(array);
            let (covered, slots) = offset_bytes(array);
            covered + slots * size_of::<u64>() + validity_size(array.validity())
        },
        A::BinaryView => viewed_bytes(downcast::<PlBinaryViewArray>(array)),
        A::Utf8View => viewed_bytes(downcast::<PlUtf8ViewArray>(array).as_binview()),
        A::FixedSizeBinary => {
            let array = downcast::<PlFixedSizeBinaryArray>(array);
            let bytes = match array.scalar_value_ignore_validity() {
                Some(value) => value.len(),
                None => array.flat_values().unwrap().len(),
            };
            bytes + validity_size(array.validity())
        },
        A::Struct => {
            let array = downcast::<PlStructArray>(array);
            array
                .fields()
                .iter()
                .map(|field| estimated_bytes_size(&**field))
                .sum::<usize>()
                + validity_size(array.validity())
        },
        A::List => {
            let array = downcast::<PlListArray>(array);
            let range = if array.is_empty() {
                0..0
            } else {
                let start = array.value_range(0).start;
                start..array.value_range(array.len() - 1).end
            };
            let slots = buffer_slots(array.offsets_are_scalar(), array.len());

            estimated_bytes_size(&*array.values().sliced(range.start, range.len()))
                + slots * size_of::<u64>()
                + validity_size(array.validity())
        },
        A::FixedSizeList => {
            let array = downcast::<PlFixedSizeListArray>(array);
            estimated_bytes_size(array.values()) + validity_size(array.validity())
        },
        A::Object { .. } => 0,
    }
}
