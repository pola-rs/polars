//! The gather kernels over the arrays of `polars-array`.

use polars_array::arrow::bridge::{chunk_to_arrow, with_arrow_chunk};
use polars_array::bitmap::combine_validities_and;
use polars_array::{PlArray, PlBitmap, PlBitmapRef, PlPrimitiveArray};
use polars_utils::IdxSize;

use super::bitmap::take_bitmap_nulls_unchecked;
use super::take_arrow_unchecked;

/// Returns the elements of `values` at `indices`, reading a null index as a null element.
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
    if let Some(index) = indices.scalar_value_ignore_validity() {
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
            Some(validity) => gathered.with_validity(Some(validity)),
        };
    }

    // Otherwise the chunk holds one slot per element, which is the layout the Arrow kernel reads.
    let indices = chunk_to_arrow(indices);
    with_arrow_chunk(values, |values| unsafe {
        take_arrow_unchecked(values, &indices)
    })
}

/// The validity of a gather from a chunk whose values are stored in the scalar representation:
///
/// # Safety
/// Every non-null index must be in bounds of `validity`.
unsafe fn gather_validity(
    validity: Option<PlBitmapRef<'_>>,
    indices: &PlPrimitiveArray<IdxSize>,
) -> Option<PlBitmap> {
    let gathered = validity.map(|validity| match validity.scalar_value() {
        // One bit says the same of every element, and therefore of every element gathered.
        Some(bit) => PlBitmap::new_scalar(bit, indices.len()),
        // A null index reads the mask at zero, which the index's own validity masks out below.
        None => PlBitmap::from_bitmap(unsafe {
            take_bitmap_nulls_unchecked(validity.flat_bitmap().unwrap(), &chunk_to_arrow(indices))
        }),
    });

    combine_validities_and(gathered.as_ref().map(PlBitmap::as_ref), indices.validity())
}

/// `array` with `mask` folded into its validity mask.
fn and_validity(array: Box<dyn PlArray>, mask: Option<PlBitmapRef<'_>>) -> Box<dyn PlArray> {
    let Some(mask) = mask else {
        return array;
    };

    let validity = combine_validities_and(array.validity(), Some(mask));
    array.with_validity(validity)
}
