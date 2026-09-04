#![allow(unsafe_op_in_unsafe_fn)]
//! The kernels that overwrite the elements a mask picks out.
//!
//! These read the arrays of `polars-array`, whose buffers may stand for a value repeated over
//! every element rather than holding one slot per element. That answers the whole of some of these
//! kernels in `O(1)`: a mask of a single set bit overwrites every element, and one of a single
//! unset bit overwrites none — see [`set_with_mask`] and [`set_at_nulls`].
//!
//! Where the elements do have to be walked, they are walked one run of the mask at a time rather
//! than one element at a time, which is what [`runs`] is for.

use std::ops::BitOr;

use arrow::bitmap::Bitmap;
use arrow::bitmap::utils::SlicesIterator;
use arrow::types::NativeType;
use polars_array::{ArrayRepr, PlBooleanArray, PlPrimitiveArray};
use polars_error::{PolarsResult, polars_err};
use polars_utils::IdxSize;

/// The runs of `mask` as `(start, end, is_set)`, covering every bit of it in order.
///
/// [`SlicesIterator`] yields only the runs of set bits, so the gaps between them — and any run
/// before the first or after the last — are the unset ones.
fn runs(mask: &Bitmap) -> impl Iterator<Item = (usize, usize, bool)> + '_ {
    let length = mask.len();
    let mut set_runs = SlicesIterator::new(mask);
    // A set run held back while the gap before it is yielded first.
    let mut held: Option<(usize, usize)> = None;
    let mut next = 0;

    std::iter::from_fn(move || {
        if let Some((start, end)) = held.take() {
            next = end;
            return Some((start, end, true));
        }

        match set_runs.next() {
            Some((start, len)) => {
                let end = start + len;
                if next < start {
                    held = Some((start, end));
                    let gap = (next, start, false);
                    next = start;
                    Some(gap)
                } else {
                    next = end;
                    Some((start, end, true))
                }
            },
            // Everything after the last set run is unset.
            None => (next < length).then(|| {
                let run = (next, length, false);
                next = length;
                run
            }),
        }
    })
}

/// The values buffer of `arr` as one slot per element, writing out a buffer of a single slot.
fn values_written_out<T: NativeType>(arr: &PlPrimitiveArray<T>) -> Vec<T> {
    match arr.values_repr() {
        ArrayRepr::Scalar(value) => vec![value; arr.len()],
        ArrayRepr::Flat(values) => values.as_slice().to_vec(),
    }
}

/// Sets the elements of `array` that are null to `value`.
///
/// This is faster than inverting the mask and combining it, because the runs of the validity mask
/// say directly which elements to copy and which to overwrite.
pub fn set_at_nulls<T: NativeType>(array: &PlPrimitiveArray<T>, value: T) -> PlPrimitiveArray<T> {
    if array.null_count() == 0 {
        return array.clone();
    }

    let validity = array
        .validity()
        .expect("a chunk with nulls in it holds a validity mask");

    let Some(validity) = validity.flat_bitmap() else {
        // The mask holds a single bit and there is a null under it, so it is unset and every
        // element is null: every one of them is overwritten, in `O(1)`.
        return PlPrimitiveArray::new_scalar(value, array.len());
    };

    // A values buffer of a single slot still has to be written out, because the result holds
    // `value` wherever the mask is unset and that one value everywhere else. Which buffer the
    // runs are read out of is settled once, ahead of the loop over them.
    let mut av = Vec::with_capacity(array.len());
    match array.values_repr() {
        ArrayRepr::Scalar(repeated) => {
            for (lower, upper, truthy) in runs(validity) {
                let fill = if truthy { repeated } else { value };
                av.extend(std::iter::repeat_n(fill, upper - lower));
            }
        },
        ArrayRepr::Flat(values) => {
            for (lower, upper, truthy) in runs(validity) {
                if truthy {
                    av.extend_from_slice(&values[lower..upper]);
                } else {
                    av.extend(std::iter::repeat_n(value, upper - lower));
                }
            }
        },
    }

    PlPrimitiveArray::from_vec(av)
}

/// Sets the elements of `array` that `mask` picks out to `value`.
///
/// The validity of the result is that of `array` with everything `mask` picks out marked as being
/// there, since those elements now hold `value`. The validity of `mask` itself is not read.
///
/// This is fast when large runs of bits are set or unset.
pub fn set_with_mask<T: NativeType>(
    array: &PlPrimitiveArray<T>,
    mask: &PlBooleanArray,
    value: T,
) -> PlPrimitiveArray<T> {
    assert_eq!(array.len(), mask.len(), "the mask must cover every element");

    let mask_values = match mask.values().repr() {
        // Every element is picked out, so every one of them holds `value` and none is null.
        ArrayRepr::Scalar(true) => return PlPrimitiveArray::new_scalar(value, array.len()),
        // No element is picked out, so nothing changes.
        ArrayRepr::Scalar(false) => return array.clone(),
        ArrayRepr::Flat(mask_values) => mask_values,
    };

    let mut buf = Vec::with_capacity(array.len());
    match array.values_repr() {
        ArrayRepr::Scalar(repeated) => {
            for (lower, upper, truthy) in runs(mask_values) {
                let fill = if truthy { value } else { repeated };
                buf.extend(std::iter::repeat_n(fill, upper - lower));
            }
        },
        ArrayRepr::Flat(values) => {
            for (lower, upper, truthy) in runs(mask_values) {
                if truthy {
                    buf.extend(std::iter::repeat_n(value, upper - lower));
                } else {
                    buf.extend_from_slice(&values[lower..upper]);
                }
            }
        },
    }

    // Wherever the mask is set the element now holds `value`, so it is no longer null.
    let validity = array
        .validity()
        .map(|validity| validity.to_flat().bitor(mask_values));

    PlPrimitiveArray::new(buf.into(), array.len(), validity)
}

/// Sets the elements of `array` at `idx` to `value`, leaving its validity as it is.
///
/// The values are copied in one go and then overwritten in place.
///
/// # Errors
/// This function errors if any index is out of bounds of `array`.
pub fn scatter_single_non_null<T, I>(
    array: &PlPrimitiveArray<T>,
    idx: I,
    value: T,
) -> PolarsResult<PlPrimitiveArray<T>>
where
    T: NativeType,
    I: IntoIterator<Item = IdxSize>,
{
    // Distinct indices hold distinct values after the scatter, so the values are written out even
    // where they were a single slot standing for all of them.
    let mut buf = values_written_out(array);
    let mut_slice = buf.as_mut_slice();

    idx.into_iter().try_for_each::<_, PolarsResult<_>>(|idx| {
        let val = mut_slice
            .get_mut(idx as usize)
            .ok_or_else(|| polars_err!(ComputeError: "index is out of bounds"))?;
        *val = value;
        Ok(())
    })?;

    let validity = array
        .validity()
        .map(|validity| validity.to_flat().into_owned());

    Ok(PlPrimitiveArray::new(buf.into(), array.len(), validity))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The runs of a mask have to cover it exactly once, in order.
    #[test]
    fn runs_cover_the_whole_mask() {
        let masks: [Vec<bool>; 7] = [
            vec![],
            vec![true],
            vec![false],
            vec![true, true, true],
            vec![false, false, false],
            vec![false, true, true, false, false, true],
            vec![true, false, false, true, true, false],
        ];

        for mask in masks {
            let bitmap: Bitmap = mask.iter().copied().collect();

            let mut covered = Vec::new();
            let mut next = 0;
            for (lower, upper, truthy) in runs(&bitmap) {
                assert_eq!(lower, next, "the runs must be in order and leave no gap");
                assert!(lower < upper, "a run must hold at least one bit");
                covered.extend(std::iter::repeat_n(truthy, upper - lower));
                next = upper;
            }
            assert_eq!(next, mask.len());
            assert_eq!(covered, mask);
        }
    }

    #[test]
    fn setting_at_nulls() {
        let arr: PlPrimitiveArray<i32> = [Some(1), None, Some(3), None, None].into_iter().collect();
        let out = set_at_nulls(&arr, 9);
        assert_eq!(out.iter().collect::<Vec<_>>(), [1, 9, 3, 9, 9].map(Some));
        assert_eq!(out.null_count(), 0);

        // A chunk with no nulls is handed straight back.
        let arr = PlPrimitiveArray::from_vec(vec![1i32, 2, 3]);
        assert_eq!(
            set_at_nulls(&arr, 9).iter().collect::<Vec<_>>(),
            [1, 2, 3].map(Some)
        );
    }

    /// A mask of one bit and a values buffer of one slot answer without being written out.
    #[test]
    fn setting_at_nulls_reads_the_repeated_forms() {
        const LENGTH: usize = 5;

        // Every element is null, so every one is overwritten and the result is itself scalar.
        let all_null = PlPrimitiveArray::<i32>::new_full_null(LENGTH);
        let out = set_at_nulls(&all_null, 9);
        assert!(out.is_scalar(), "the result must stay in the scalar form");
        assert_eq!(out.iter().collect::<Vec<_>>(), [Some(9); LENGTH]);

        // A repeated value under a flat mask is written out, and agrees with the flat form.
        let mask = [true, false, true, false, true];
        let scalar = PlPrimitiveArray::new_scalar(7i32, LENGTH)
            .with_validity(Some(mask.into_iter().collect()));
        let flat = PlPrimitiveArray::from_vec(vec![7i32; LENGTH])
            .with_validity(Some(mask.into_iter().collect()));
        assert_eq!(
            set_at_nulls(&scalar, 9).iter().collect::<Vec<_>>(),
            set_at_nulls(&flat, 9).iter().collect::<Vec<_>>(),
        );
        assert_eq!(
            set_at_nulls(&scalar, 9).iter().collect::<Vec<_>>(),
            [7, 9, 7, 9, 7].map(Some),
        );
    }

    #[test]
    fn setting_with_a_mask() {
        let arr: PlPrimitiveArray<u32> = (0..10u32).collect();
        let mask: PlBooleanArray = [
            false, true, false, true, false, true, false, true, false, false,
        ]
        .into_iter()
        .collect();
        let out = set_with_mask(&arr, &mask, 100);
        assert_eq!(
            out.iter().flatten().collect::<Vec<_>>(),
            [0, 100, 2, 100, 4, 100, 6, 100, 8, 9],
        );

        // Where the mask is set the element is no longer null.
        let arr: PlPrimitiveArray<u32> = [None, None, None].into_iter().collect();
        let mask: PlBooleanArray = [true, true, false].into_iter().collect();
        let out = set_with_mask(&arr, &mask, 1);
        assert_eq!(out.iter().collect::<Vec<_>>(), [Some(1), Some(1), None]);
    }

    /// A mask of a single bit answers in `O(1)`, either way it is set.
    #[test]
    fn setting_with_a_repeated_mask() {
        const LENGTH: usize = 5;
        let arr: PlPrimitiveArray<u32> = [Some(0), None, Some(2), None, Some(4)]
            .into_iter()
            .collect();

        let all_set = PlBooleanArray::new_scalar(true, LENGTH);
        let out = set_with_mask(&arr, &all_set, 100);
        assert!(out.is_scalar(), "the result must stay in the scalar form");
        assert_eq!(out.iter().collect::<Vec<_>>(), [Some(100); LENGTH]);
        // It agrees with the same mask written out.
        let flat_set = PlBooleanArray::from_values(std::iter::repeat_n(true, LENGTH).collect());
        assert_eq!(
            out.iter().collect::<Vec<_>>(),
            set_with_mask(&arr, &flat_set, 100)
                .iter()
                .collect::<Vec<_>>(),
        );

        let none_set = PlBooleanArray::new_scalar(false, LENGTH);
        let out = set_with_mask(&arr, &none_set, 100);
        assert_eq!(
            out.iter().collect::<Vec<_>>(),
            arr.iter().collect::<Vec<_>>()
        );
        let flat_unset = PlBooleanArray::from_values(std::iter::repeat_n(false, LENGTH).collect());
        assert_eq!(
            out.iter().collect::<Vec<_>>(),
            set_with_mask(&arr, &flat_unset, 100)
                .iter()
                .collect::<Vec<_>>(),
        );
    }

    #[test]
    fn scattering_a_single_value() {
        let arr = PlPrimitiveArray::from_vec(vec![1u32, 2, 3]);
        let out = scatter_single_non_null(&arr, std::iter::once(1), 100).unwrap();
        assert_eq!(out.iter().flatten().collect::<Vec<_>>(), [1, 100, 3]);

        assert!(scatter_single_non_null(&arr, std::iter::once(100), 100).is_err());

        // A values buffer of a single slot is written out, since the scatter makes the elements
        // differ.
        let scalar = PlPrimitiveArray::new_scalar(7u32, 3);
        let out = scatter_single_non_null(&scalar, std::iter::once(1), 100).unwrap();
        assert_eq!(out.iter().flatten().collect::<Vec<_>>(), [7, 100, 7]);
    }
}
