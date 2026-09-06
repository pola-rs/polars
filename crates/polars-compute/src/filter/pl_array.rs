//! The filter kernels over the arrays of `polars-array`.

use polars_array::arrow::bridge::with_arrow_chunk;
use polars_array::bitmap::combine_validities_and;
use polars_array::{PlArray, PlBitmap, PlBitmapRef, PlBooleanArray};

use super::boolean::filter_boolean_kernel;
use super::filter_arrow_with_bitmap;

/// Keeps the elements of `array` at which `mask` is set, reading a null in `mask` as unset.
pub fn filter(array: &dyn PlArray, mask: &PlBooleanArray) -> Box<dyn PlArray> {
    // An element the mask says nothing about is one it does not keep, which is what an unset bit
    // says in turn: the two fold together into the one mask the kernel reads.
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

    // A mask that repeats one bit says the same of every element: it either keeps all of them or
    // none of them, and neither answer has to look at the elements to be given.
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

    // Every element of a chunk whose values are stored in the scalar representation is the one
    // value it repeats, so dropping elements only shortens the run: the values stay in `O(1)`
    // memory, and it is the validity mask alone that is filtered.
    //
    // Dropping the validity mask is what leaves the values on their own, and it is `O(1)`: the
    // buffers are handed over as they are.
    let values = array.without_validity();
    if values.is_scalar() {
        // SAFETY: the mask keeps at least one element, so the array holds at least one.
        let filtered = unsafe { values.new_from_index_unchecked(0, kept) };

        return match array.validity() {
            None => filtered,
            Some(validity) => filtered.with_validity(Some(filter_pl_bitmap(validity, mask, kept))),
        };
    }

    // Otherwise the chunk holds one slot per element, which is the layout the Arrow kernel reads.
    with_arrow_chunk(array, |array| {
        filter_arrow_with_bitmap(array, &mask.to_flat())
    })
}

/// Keeps the bits of `values` at which `mask` is set.
fn filter_pl_bitmap(values: PlBitmapRef<'_>, mask: PlBitmapRef<'_>, kept: usize) -> PlBitmap {
    match values.scalar_value() {
        // One bit says the same of every element, and so of however many of them survive.
        Some(value) => PlBitmap::new_scalar(value, kept),
        None => PlBitmap::new(
            filter_boolean_kernel(values.flat_bitmap().unwrap(), &mask.to_flat()),
            kept,
        ),
    }
}
