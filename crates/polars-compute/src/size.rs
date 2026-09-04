//! The kernels that measure what an array holds.

use arrow::with_match_primitive_type_full;
use polars_array::{
    ArrayRepr, PlArray, PlArrayType, PlBinaryArray, PlBinaryViewArray, PlBitmap, PlBitmapRef,
    PlBooleanArray, PlFixedSizeBinaryArray, PlFixedSizeListArray, PlListArray, PlPrimitiveArray,
    PlStructArray, PlUtf8ViewArray,
};

/// The length in bytes of every element, read off the views.
pub fn binary_size_bytes(array: &PlBinaryViewArray) -> PlPrimitiveArray<u32> {
    // A scalar views buffer holds the one view every element reads: its length is measured once
    // and repeated in turn, in `O(1)` memory.
    let lengths = match array.views_repr() {
        ArrayRepr::Scalar(view) => PlPrimitiveArray::new_scalar(view.length, array.len()),
        ArrayRepr::Flat(views) => {
            PlPrimitiveArray::from_vec(views.iter().map(|view| view.length).collect())
        },
    };

    lengths.with_validity(array.validity().map(PlBitmap::from))
}

/// The bytes a validity mask takes, which is none where there is no mask.
fn validity_size(validity: Option<PlBitmapRef<'_>>) -> usize {
    validity.map_or(0, |validity| {
        validity.to_flat_or_scalar().as_slice().0.len()
    })
}

/// The number of slots a backing buffer holds: a single one where it stands for a value repeated
/// over every element, and one per element otherwise.
fn buffer_slots(is_scalar: bool, length: usize) -> usize {
    if is_scalar { 1 } else { length }
}

/// Downcasts an array whose [`PlArrayType`] has already been matched on.
///
/// # Panics
/// Panics if `array` is not an `A`, which its array type rules out.
#[inline]
fn downcast<A: PlArray>(array: &dyn PlArray) -> &A {
    array
        .as_any()
        .downcast_ref()
        .expect("the array type identifies the concrete array")
}

/// The bytes the views of a view array cover, which is what such an array is measured by.
///
/// The data buffers behind the views may be shared with another array, so the sum of the buffers
/// would overestimate what this array costs and spill data that did not need spilling. A views
/// buffer of a single slot covers its bytes once, however many elements read it.
fn viewed_bytes(array: &PlBinaryViewArray) -> usize {
    match array.views_repr() {
        ArrayRepr::Scalar(view) => view.length as usize,
        ArrayRepr::Flat(views) => views.iter().map(|view| view.length as usize).sum(),
    }
}

/// The bytes the offsets of `array` cover, and the number of offsets it holds.
fn offset_bytes(array: &PlBinaryArray) -> (usize, usize) {
    // A scalar offsets buffer holds the one range every element covers, so it cuts those bytes
    // out of the values once; a flat one holds the end of every element plus a leading zero.
    let slots = if array.offsets_are_scalar() {
        2
    } else {
        array.len() + 1
    };

    let covered = if array.is_empty() {
        0
    } else {
        // The offsets are what is sliced, not the values, so only the range they cover is held.
        array.value_range(array.len() - 1).end - array.value_range(0).start
    };

    (covered, slots)
}

/// The bytes the buffers of `array` take, its children included.
///
/// A buffer that stands for a value repeated over every element holds a single slot, so this
/// reports what such a chunk costs rather than what it would cost written out.
///
/// # Implementation
/// This is the sum of the sizes of the buffers and masks of `array` and of everything nested under
/// it. Arrays may share buffers and masks, so the size of two of them is not the sum of what this
/// returns for each: a [`PlStructArray`] in particular is an upper bound.
///
/// Slicing an array leaves its allocation as it is, but shrinks what this returns, because what is
/// measured is the part of each buffer the array can see rather than the whole allocation.
pub fn estimated_bytes_size(array: &dyn PlArray) -> usize {
    use PlArrayType as A;

    match array.array_type() {
        // Nulls are stored as nothing but a length.
        A::Null => 0,
        A::Boolean => {
            let array = downcast::<PlBooleanArray>(array);
            array.values().to_flat_or_scalar().as_slice().0.len() + validity_size(array.validity())
        },
        A::Primitive(primitive) => with_match_primitive_type_full!(primitive, |$T| {
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
            let bytes = match array.values_repr() {
                // The bytes of the one element every element reads.
                ArrayRepr::Scalar(value) => value.len(),
                // The bytes of every element, laid end to end.
                ArrayRepr::Flat(values) => values.len(),
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
            // The offsets are what is sliced, so only the values they cover are held.
            let range = if array.is_empty() {
                0..0
            } else {
                let start = array.value_range(0).start;
                start..array.value_range(array.len() - 1).end
            };
            // The offsets are counted one per element rather than one per slot, so that slicing an
            // array in half halves what this returns — the leading offset would tip it over.
            let slots = buffer_slots(array.offsets_are_scalar(), array.len());

            estimated_bytes_size(&*array.values().sliced(range.start, range.len()))
                + slots * size_of::<u64>()
                + validity_size(array.validity())
        },
        A::FixedSizeList => {
            let array = downcast::<PlFixedSizeListArray>(array);
            estimated_bytes_size(array.values()) + validity_size(array.validity())
        },
        // An object array holds its elements behind a trait object, whose size is its own business.
        A::Object { .. } => 0,
    }
}

#[cfg(test)]
mod tests {
    use arrow::bitmap::Bitmap;
    use polars_array::PlNullArray;

    use super::*;

    /// The one view a scalar chunk holds is measured once, and the result repeats the length
    /// rather than holding a slot per element.
    #[test]
    fn a_repeated_value_keeps_its_length_repeated() {
        let scalar = PlBinaryViewArray::new_scalar(b"hello", 100);
        let sizes = binary_size_bytes(&scalar);

        assert!(sizes.values_are_scalar());
        assert_eq!(sizes, PlPrimitiveArray::from_vec(vec![5u32; 100]));
    }

    /// A chunk laid out one view per element is measured one element at a time, and the mask
    /// comes along as it is.
    #[test]
    fn every_element_is_measured() {
        let arr = PlBinaryViewArray::from_iter([Some(&b"a"[..]), None, Some(&b"three"[..])]);
        assert_eq!(
            binary_size_bytes(&arr),
            PlPrimitiveArray::from_iter([Some(1u32), None, Some(5)]),
        );

        // A scalar view under a mask that holds one bit per element stays scalar.
        let masked = PlBinaryViewArray::new_scalar(b"ab", 3).with_validity(Some(
            PlBitmap::from_bitmap(Bitmap::from_iter([true, false, true])),
        ));
        assert_eq!(
            binary_size_bytes(&masked),
            PlPrimitiveArray::from_iter([Some(2u32), None, Some(2)]),
        );
    }

    #[test]
    fn an_empty_chunk_measures_nothing() {
        let empty = PlBinaryViewArray::new_empty();
        assert!(binary_size_bytes(&empty).is_empty());
    }

    /// A buffer that stands for a repeated value holds one slot, and that is what it costs — the
    /// point of the representation, and what measuring the written-out form would miss.
    #[test]
    fn a_repeated_value_costs_one_slot() {
        const LENGTH: usize = 1_000;

        let scalar = PlPrimitiveArray::new_scalar(7i64, LENGTH);
        assert_eq!(estimated_bytes_size(&scalar), size_of::<i64>());

        let flat = PlPrimitiveArray::from_vec(vec![7i64; LENGTH]);
        assert_eq!(estimated_bytes_size(&flat), LENGTH * size_of::<i64>());

        // A mask of a single bit costs the one byte that bit is stored in.
        let masked = PlPrimitiveArray::new_scalar(7i64, LENGTH)
            .with_validity(Some(PlBitmap::new_scalar(true, LENGTH)));
        assert_eq!(estimated_bytes_size(&masked), size_of::<i64>() + 1);

        // Nulls are a length and nothing else.
        assert_eq!(estimated_bytes_size(&PlNullArray::new(LENGTH)), 0);
    }

    /// Every buffer under a nested array is measured, the child's included.
    #[test]
    fn a_nested_chunk_measures_its_child() {
        let values = Box::new(PlPrimitiveArray::from_vec(vec![1i32, 2, 3, 4]));
        let list = PlListArray::from_offsets(values, vec![0u64, 2, 4].into());

        assert_eq!(
            estimated_bytes_size(&list),
            4 * size_of::<i32>() + 2 * size_of::<u64>(),
        );

        let fields: Vec<Box<dyn PlArray>> = vec![
            Box::new(PlPrimitiveArray::from_vec(vec![1i32, 2])),
            Box::new(PlPrimitiveArray::new_scalar(9i64, 2)),
        ];
        assert_eq!(
            estimated_bytes_size(&PlStructArray::new(fields, 2, None)),
            2 * size_of::<i32>() + size_of::<i64>(),
        );
    }

    /// Slicing a list array in half has to halve what it is measured at, which is why its offsets
    /// are counted one per element: the leading offset would leave the halves over half the whole.
    #[test]
    fn slicing_a_list_halves_it() {
        const LENGTH: usize = 10_000;

        let values = Box::new(PlPrimitiveArray::from_vec((0..LENGTH as i64).collect()));
        let list =
            PlListArray::from_offsets(values, (0..=LENGTH as u64).collect::<Vec<_>>().into());

        let whole = estimated_bytes_size(&list);
        let half = estimated_bytes_size(&list.clone().sliced(LENGTH / 2, LENGTH / 2));
        assert!(
            half * 2 <= whole,
            "half of {LENGTH} lists measured {half}, which is over half of {whole}",
        );
    }

    /// Slicing an array shrinks what it can see, and a view array is measured by the bytes its
    /// views cover rather than by the buffers behind them.
    #[test]
    fn slicing_shrinks_what_is_measured() {
        let arr = PlPrimitiveArray::from_vec((0..100i32).collect());
        assert_eq!(
            estimated_bytes_size(&arr.clone().sliced(10, 5)),
            5 * size_of::<i32>(),
        );

        let views = PlBinaryViewArray::from_values_iter([b"aaaa".as_slice(), b"bb"]);
        assert_eq!(estimated_bytes_size(&views), 6);
        // One view stands for every element, so its bytes are counted once.
        assert_eq!(
            estimated_bytes_size(&PlBinaryViewArray::new_scalar(b"aaaa", 100)),
            4
        );
    }
}
