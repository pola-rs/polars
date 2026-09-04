//! The gather kernels over the arrays of `polars-array`.
//!
//! Gathering only ever picks elements out again, so a chunk that repeats one value keeps repeating
//! it however it is gathered from: only the length of the run changes, and the value is never
//! written out. A run of indices that repeats one index reads one element of the values, whose
//! answer stands for the whole gather in turn.

use polars_array::arrow::bridge::{chunk_to_arrow, with_arrow_chunk};
use polars_array::bitmap::combine_validities_and;
use polars_array::{ArrayRepr, PlArray, PlBitmap, PlBitmapRef, PlPrimitiveArray};
use polars_utils::IdxSize;

use super::bitmap::take_bitmap_nulls_unchecked;
use super::take_arrow_unchecked;

/// Returns the elements of `values` at `indices`, one per index, reading a null index as a null
/// element.
///
/// # Safety
/// Every non-null index must be in bounds of `values`.
pub unsafe fn take_unchecked(
    values: &dyn PlArray,
    indices: &PlPrimitiveArray<IdxSize>,
) -> Box<dyn PlArray> {
    if indices.is_empty() {
        return values.sliced(0, 0);
    }

    // An index that is null picks no element at all, so a run of them picks nothing anywhere —
    // which leaves `values` unread, and is the one case in which it may hold no element to read.
    if indices.null_count() == indices.len() {
        return values.full_null_like(indices.len());
    }

    // From here on at least one index is in bounds, so `values` holds at least one element.

    // Indices stored in the scalar representation are one index repeated, and the one element it
    // picks is the answer at every position in turn.
    if let Some(index) = indices.scalar_values() {
        // SAFETY: the index is one of the caller's, and is therefore in bounds.
        let gathered = unsafe { values.new_from_index_unchecked(index as usize, indices.len()) };
        return and_validity(gathered, indices.validity());
    }

    // Values stored in the scalar representation are one value repeated, so whichever elements are
    // picked out of them are that value again: the values stay in `O(1)` memory, and it is the
    // validity mask alone that is gathered.
    //
    // Dropping the validity mask is what leaves the values on their own, and it is `O(1)`: the
    // buffers are handed over as they are.
    let unmasked = values.without_validity();
    if unmasked.is_scalar() {
        // SAFETY: `values` holds at least one element, and the value under a null one is a value
        // like any other here — the mask below is what makes the result null.
        let gathered = unsafe { unmasked.new_from_index_unchecked(0, indices.len()) };

        // SAFETY: the caller's indices are in bounds of the mask, which covers every element.
        let validity = unsafe { gather_validity(values.validity(), indices) };

        return match validity {
            None => gathered,
            Some(validity) => {
                gathered.with_validity_broadcast(Some(validity.into_flat_or_scalar()))
            },
        };
    }

    // Otherwise the chunk holds one slot per element, which is the layout the Arrow kernel reads.
    let indices = chunk_to_arrow(indices);
    with_arrow_chunk(values, |values| unsafe {
        take_arrow_unchecked(values, &indices)
    })
}

/// The validity of a gather from a chunk whose values are stored in the scalar representation:
/// `validity` read at every index, and null wherever the index itself is null.
///
/// # Safety
/// Every non-null index must be in bounds of `validity`.
unsafe fn gather_validity(
    validity: Option<PlBitmapRef<'_>>,
    indices: &PlPrimitiveArray<IdxSize>,
) -> Option<PlBitmap> {
    let gathered = validity.map(|validity| match validity.repr() {
        // One bit says the same of every element, and therefore of every element gathered.
        ArrayRepr::Scalar(bit) => PlBitmap::new_scalar(bit, indices.len()),
        // A null index reads the mask at zero, which the index's own validity masks out below.
        ArrayRepr::Flat(validity) => PlBitmap::from_bitmap(unsafe {
            take_bitmap_nulls_unchecked(validity, &chunk_to_arrow(indices))
        }),
    });

    combine_validities_and(gathered.as_ref().map(PlBitmap::as_ref), indices.validity())
}

/// `array` with `mask` folded into its validity mask.
///
/// # Panics
/// Panics unless `mask` covers exactly the elements of `array`.
fn and_validity(array: Box<dyn PlArray>, mask: Option<PlBitmapRef<'_>>) -> Box<dyn PlArray> {
    let Some(mask) = mask else {
        return array;
    };

    let validity = combine_validities_and(array.validity(), Some(mask));
    array.with_validity_broadcast(validity.map(PlBitmap::into_flat_or_scalar))
}

#[cfg(test)]
mod tests {
    use polars_array::PlBooleanArray;

    use super::*;

    fn indices(idx: impl IntoIterator<Item = Option<IdxSize>>) -> PlPrimitiveArray<IdxSize> {
        PlPrimitiveArray::from_iter(idx)
    }

    fn gather(values: &dyn PlArray, idx: &PlPrimitiveArray<IdxSize>) -> Box<dyn PlArray> {
        unsafe { take_unchecked(values, idx) }
    }

    /// A chunk that repeats one value answers every gather with that value again, however many
    /// elements are asked for and in whatever order.
    #[test]
    fn a_repeated_value_is_gathered_in_place() {
        let scalar = PlPrimitiveArray::new_scalar(7i32, 3);
        let gathered = gather(&scalar, &indices([Some(2), Some(0), Some(0), Some(1)]));

        assert!(gathered.is_scalar());
        assert_eq!(
            gathered.as_any().downcast_ref::<PlPrimitiveArray<i32>>(),
            Some(&PlPrimitiveArray::new_scalar(7i32, 4)),
        );
    }

    /// A chunk that repeats one value under a mask of its own keeps the value repeated; it is the
    /// mask that is read at every index.
    #[test]
    fn a_repeated_value_under_a_mask_has_its_mask_gathered() {
        let scalar = PlPrimitiveArray::new_scalar(7i32, 3)
            .with_validity(Some([true, false, true].into_iter().collect()));
        let gathered = gather(&scalar, &indices([Some(1), Some(2), None]));

        let gathered = gathered
            .as_any()
            .downcast_ref::<PlPrimitiveArray<i32>>()
            .unwrap();
        assert!(gathered.values_are_scalar());
        assert_eq!(
            gathered,
            &PlPrimitiveArray::from_iter([None, Some(7i32), None]),
        );
    }

    /// A run of indices that repeats one index reads one element, which stands for the gather.
    #[test]
    fn a_repeated_index_reads_one_element() {
        let arr = PlPrimitiveArray::from_vec(vec![1i32, 2, 3]);
        let gathered = gather(&arr, &PlPrimitiveArray::new_scalar(1 as IdxSize, 4));

        assert!(gathered.is_scalar());
        assert_eq!(
            gathered.as_any().downcast_ref::<PlPrimitiveArray<i32>>(),
            Some(&PlPrimitiveArray::new_scalar(2i32, 4)),
        );

        // A repeated index that picks a null element picks a null every time.
        let arr = PlPrimitiveArray::from_iter([Some(1i32), None]);
        let gathered = gather(&arr, &PlPrimitiveArray::new_scalar(1 as IdxSize, 3));
        assert_eq!(
            gathered.as_any().downcast_ref::<PlPrimitiveArray<i32>>(),
            Some(&PlPrimitiveArray::new_full_null(3)),
        );
    }

    /// Indices that are all null pick nothing at all, which `values` is never read for.
    #[test]
    fn null_indices_pick_nothing() {
        let empty = PlPrimitiveArray::<i32>::new_empty();
        let gathered = gather(&empty, &indices([None, None]));

        assert_eq!(
            gathered.as_any().downcast_ref::<PlPrimitiveArray<i32>>(),
            Some(&PlPrimitiveArray::new_full_null(2)),
        );
    }

    /// A chunk laid out one element per slot is gathered one element at a time.
    #[test]
    fn every_index_is_read() {
        let arr = PlPrimitiveArray::from_iter([Some(1i32), None, Some(3)]);
        let gathered = gather(&arr, &indices([Some(2), None, Some(1), Some(0)]));

        assert_eq!(
            gathered.as_any().downcast_ref::<PlPrimitiveArray<i32>>(),
            Some(&PlPrimitiveArray::from_iter([
                Some(3i32),
                None,
                None,
                Some(1)
            ])),
        );
    }

    /// Gathering nothing leaves an empty chunk of the same type.
    #[test]
    fn no_indices_gather_nothing() {
        let arr = PlBooleanArray::from_vec(vec![true, false]);
        let gathered = gather(&arr, &indices([]));

        assert!(gathered.is_empty());
        assert_eq!(gathered.array_type(), arr.array_type());
    }
}
