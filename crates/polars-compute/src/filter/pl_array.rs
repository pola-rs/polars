//! The filter kernels over the arrays of `polars-array`.

use polars_array::arrow::bridge::with_arrow_chunk;
use polars_array::bitmap::combine_validities_and;
use polars_array::{PlArray, PlBitmap, PlBitmapRef, PlBooleanArray};

use super::boolean::filter_boolean_kernel;
use super::filter_arrow_with_bitmap;

/// Keeps the elements of `array` at which `mask` is set, reading a null in `mask` as unset.
pub fn filter(array: &dyn PlArray, mask: &PlBooleanArray) -> Box<dyn PlArray> {
    let mask = combine_validities_and(Some(mask.values()), mask.validity())
        .expect("the values of a mask are a mask of their own");

    filter_with_bitmap(array, mask.as_ref())
}

/// Keeps the elements of `array` at which `mask` is set.
pub fn filter_with_bitmap(array: &dyn PlArray, mask: PlBitmapRef<'_>) -> Box<dyn PlArray> {
    assert_eq!(
        array.len(),
        mask.len(),
        "filter mask covers a different number of elements than the array it filters",
    );

    if let Some(keep) = mask.scalar_value() {
        return if keep {
            array.to_boxed()
        } else {
            array.sliced(0, 0)
        };
    }

    let kept = mask.set_bits();
    if kept == 0 {
        return array.sliced(0, 0);
    }
    if kept == array.len() {
        return array.to_boxed();
    }

    let values = array.without_validity();
    if values.is_scalar() {
        // SAFETY: the mask keeps at least one element, so the array holds at least one.
        let filtered = unsafe { values.new_from_index_unchecked(0, kept) };

        return match array.validity() {
            None => filtered,
            Some(validity) => filtered.with_validity(Some(filter_pl_bitmap(validity, mask, kept))),
        };
    }

    with_arrow_chunk(array, |array| {
        filter_arrow_with_bitmap(array, &mask.to_flat())
    })
}

/// Keeps the bits of `values` at which `mask` is set.
fn filter_pl_bitmap(values: PlBitmapRef<'_>, mask: PlBitmapRef<'_>, kept: usize) -> PlBitmap {
    match values.scalar_value() {
        Some(value) => PlBitmap::new_scalar(value, kept),
        None => PlBitmap::new(
            filter_boolean_kernel(values.flat_bitmap().unwrap(), &mask.to_flat()),
            kept,
        ),
    }
}
