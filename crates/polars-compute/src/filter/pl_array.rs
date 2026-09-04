//! The filter kernels over the arrays of `polars-array`.
//!
//! Filtering only ever drops elements, so a chunk that repeats one value keeps repeating it: the
//! run is shortened to the number of elements the mask keeps, in `O(1)`, without the value ever
//! being written out. The same holds of the mask itself — one that is set everywhere keeps the
//! chunk as it is, and one that is set nowhere leaves nothing at all.

use polars_array::arrow::bridge::with_arrow_chunk;
use polars_array::bitmap::combine_validities_and;
use polars_array::{ArrayRepr, PlArray, PlBitmap, PlBitmapRef, PlBooleanArray};

use super::boolean::filter_boolean_kernel;
use super::dyn_array::filter_arrow_with_bitmap;

/// Keeps the elements of `array` at which `mask` is set, reading a null in `mask` as unset.
///
/// # Panics
/// Panics unless `array` and `mask` hold the same number of elements.
pub fn filter(array: &dyn PlArray, mask: &PlBooleanArray) -> Box<dyn PlArray> {
    // An element the mask says nothing about is one it does not keep, which is what an unset bit
    // says in turn: the two fold together into the one mask the kernel reads.
    let mask = combine_validities_and(Some(mask.values()), mask.validity())
        .expect("the values of a mask are a mask of their own");

    filter_with_bitmap(array, mask.as_ref())
}

/// Keeps the elements of `array` at which `mask` is set.
///
/// # Panics
/// Panics unless `array` and `mask` hold the same number of elements.
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
            Some(validity) => filtered.with_validity_broadcast(Some(
                filter_pl_bitmap(validity, mask, kept).into_flat_or_scalar(),
            )),
        };
    }

    // Otherwise the chunk holds one slot per element, which is the layout the Arrow kernel reads.
    with_arrow_chunk(array, |array| {
        filter_arrow_with_bitmap(array, &mask.to_flat())
    })
}

/// Keeps the bits of `values` at which `mask` is set. `kept` is the number of bits `mask` keeps.
fn filter_pl_bitmap(values: PlBitmapRef<'_>, mask: PlBitmapRef<'_>, kept: usize) -> PlBitmap {
    match values.repr() {
        // One bit says the same of every element, and so of however many of them survive.
        ArrayRepr::Scalar(value) => PlBitmap::new_scalar(value, kept),
        ArrayRepr::Flat(values) => {
            PlBitmap::new(filter_boolean_kernel(values, &mask.to_flat()), kept)
        },
    }
}

#[cfg(test)]
mod tests {
    use polars_array::PlPrimitiveArray;

    use super::*;

    fn mask(bits: impl IntoIterator<Item = bool>) -> PlBooleanArray {
        PlBooleanArray::from_vec(bits.into_iter().collect())
    }

    /// A chunk that repeats one value keeps repeating it: only the length of the run changes, and
    /// the values are never written out.
    #[test]
    fn a_repeated_value_stays_repeated() {
        let scalar = PlPrimitiveArray::new_scalar(7i32, 4);
        let filtered = filter(&scalar, &mask([true, false, true, false]));

        assert!(filtered.is_scalar());
        assert_eq!(
            filtered.as_any().downcast_ref::<PlPrimitiveArray<i32>>(),
            Some(&PlPrimitiveArray::new_scalar(7i32, 2)),
        );
    }

    /// A chunk that repeats one value under a mask of its own keeps the value repeated, and it is
    /// the mask that is filtered down to the elements that survive.
    #[test]
    fn a_repeated_value_under_a_mask_keeps_its_mask_filtered() {
        let scalar = PlPrimitiveArray::new_scalar(7i32, 4)
            .with_validity(Some([true, false, true, true].into_iter().collect()));
        let filtered = filter(&scalar, &mask([true, true, false, true]));

        let filtered = filtered
            .as_any()
            .downcast_ref::<PlPrimitiveArray<i32>>()
            .unwrap();
        assert!(filtered.values_are_scalar());
        assert_eq!(
            filtered,
            &PlPrimitiveArray::from_iter([Some(7i32), None, Some(7)]),
        );
    }

    /// A mask that says the same of every element answers without looking at them.
    #[test]
    fn a_repeated_bit_keeps_everything_or_nothing() {
        let arr = PlPrimitiveArray::from_vec(vec![1i32, 2, 3]);

        let kept = filter_with_bitmap(&arr, PlBitmap::new_scalar(true, 3).as_ref());
        assert_eq!(
            kept.as_any().downcast_ref::<PlPrimitiveArray<i32>>(),
            Some(&arr),
        );

        let dropped = filter_with_bitmap(&arr, PlBitmap::new_scalar(false, 3).as_ref());
        assert!(dropped.is_empty());
        assert_eq!(dropped.array_type(), arr.array_type());
    }

    /// A chunk laid out one element per slot is filtered one element at a time.
    #[test]
    fn every_element_is_filtered() {
        let arr = PlPrimitiveArray::from_iter([Some(1i32), None, Some(3), Some(4)]);
        let filtered = filter(&arr, &mask([true, true, false, true]));

        assert_eq!(
            filtered.as_any().downcast_ref::<PlPrimitiveArray<i32>>(),
            Some(&PlPrimitiveArray::from_iter([Some(1i32), None, Some(4)])),
        );
    }

    /// A null in the mask keeps nothing, exactly like an unset bit.
    #[test]
    fn a_null_in_the_mask_keeps_nothing() {
        let arr = PlPrimitiveArray::from_vec(vec![1i32, 2, 3]);
        let mask = PlBooleanArray::from_iter([Some(true), None, Some(true)]);

        assert_eq!(
            filter(&arr, &mask)
                .as_any()
                .downcast_ref::<PlPrimitiveArray<i32>>(),
            Some(&PlPrimitiveArray::from_vec(vec![1i32, 3])),
        );
    }
}
