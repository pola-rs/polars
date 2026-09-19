//! The gather kernels over the arrays of `polars-array`.

use polars_array::arrow::bridge::{chunk_to_arrow, with_arrow_chunk};
use polars_array::bitmap::combine_validities_and;
use polars_array::builder::new_full_null_like;
use polars_array::{PlArray, PlBitmap, PlBitmapRef, PlPrimitiveArray};
use polars_utils::IdxSize;

use super::bitmap::{take_bitmap_nulls_unchecked, take_bitmap_unchecked};
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

    if indices.null_count() == indices.len() {
        return new_full_null_like(values, indices.len());
    }

    if let Some(index) = indices.scalar_value_ignore_validity() {
        // SAFETY: the index is one of the caller's, and is therefore in bounds.
        let gathered = unsafe { values.new_from_index_unchecked(index as usize, indices.len()) };
        return and_validity(gathered, indices.validity());
    }

    // Dropping the mask to ask whether the values all read one slot is a clone of the chunk's
    // buffers; a chunk with no mask is its own values, so it answers the question in place.
    let dropped = values
        .validity()
        .is_some()
        .then(|| values.without_validity());
    let unmasked = dropped.as_deref().unwrap_or(values);
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

    let indices = chunk_to_arrow(indices);
    with_arrow_chunk(values, |values| unsafe {
        take_arrow_unchecked(values, &indices)
    })
}

/// The validity of a gather from a chunk whose values are scalar: the mask alone is gathered.
///
/// # Safety
/// Every non-null index must be in bounds of `validity`.
pub unsafe fn gather_validity(
    validity: Option<PlBitmapRef<'_>>,
    indices: &PlPrimitiveArray<IdxSize>,
) -> Option<PlBitmap> {
    let gathered = validity.map(|validity| match validity.scalar_value() {
        Some(bit) => PlBitmap::new_scalar(bit, indices.len()),
        None => PlBitmap::from_bitmap(unsafe {
            take_bitmap_nulls_unchecked(validity.flat_bitmap().unwrap(), &chunk_to_arrow(indices))
        }),
    });

    combine_validities_and(gathered.as_ref().map(PlBitmap::as_ref), indices.validity())
}

/// As [`gather_validity`], where the indices are a slice and so none of them is null.
///
/// # Safety
/// Every index must be in bounds of `validity`.
pub unsafe fn gather_validity_slice(
    validity: Option<PlBitmapRef<'_>>,
    indices: &[IdxSize],
) -> Option<PlBitmap> {
    validity.map(|validity| match validity.scalar_value() {
        Some(bit) => PlBitmap::new_scalar(bit, indices.len()),
        None => PlBitmap::from_bitmap(unsafe {
            take_bitmap_unchecked(validity.flat_bitmap().unwrap(), indices)
        }),
    })
}

/// `array` with `mask` folded into its validity mask.
fn and_validity(array: Box<dyn PlArray>, mask: Option<PlBitmapRef<'_>>) -> Box<dyn PlArray> {
    let Some(mask) = mask else {
        return array;
    };

    let validity = combine_validities_and(array.validity(), Some(mask));
    array.with_validity(validity)
}
