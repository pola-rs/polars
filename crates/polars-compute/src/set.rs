#![allow(unsafe_op_in_unsafe_fn)]
//! The kernels that overwrite the elements a mask picks out.

use std::ops::BitOr;

use arrow::bitmap::Bitmap;
use arrow::bitmap::utils::SlicesIterator;
use arrow::types::NativeType;
use polars_array::{PlBitmap, PlBooleanArray, PlPrimitiveArray};
use polars_error::{PolarsResult, polars_err};
use polars_utils::IdxSize;

/// The runs of `mask` as `(start, end, is_set)`, covering every bit of it in order.
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
    match arr.scalar_value_ignore_validity() {
        Some(value) => vec![value; arr.len()],
        None => arr.flat_values().unwrap().as_slice().to_vec(),
    }
}

/// Sets the elements of `array` that are null to `value`.
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
    if let Some(repeated) = array.scalar_value_ignore_validity() {
        for (lower, upper, truthy) in runs(validity) {
            let fill = if truthy { repeated } else { value };
            av.extend(std::iter::repeat_n(fill, upper - lower));
        }
    } else {
        let values = array.flat_values().unwrap();
        for (lower, upper, truthy) in runs(validity) {
            if truthy {
                av.extend_from_slice(&values[lower..upper]);
            } else {
                av.extend(std::iter::repeat_n(value, upper - lower));
            }
        }
    }

    PlPrimitiveArray::from_vec(av)
}

/// Sets the elements of `array` that `mask` picks out to `value`.
pub fn set_with_mask<T: NativeType>(
    array: &PlPrimitiveArray<T>,
    mask: &PlBooleanArray,
    value: T,
) -> PlPrimitiveArray<T> {
    assert_eq!(array.len(), mask.len(), "the mask must cover every element");

    match mask.scalar_value_ignore_validity() {
        // Every element is picked out, so every one of them holds `value` and none is null.
        Some(true) => return PlPrimitiveArray::new_scalar(value, array.len()),
        // No element is picked out, so nothing changes.
        Some(false) => return array.clone(),
        None => {},
    }
    let mask_values = mask.flat_values().unwrap();

    let mut buf = Vec::with_capacity(array.len());
    if let Some(repeated) = array.scalar_value_ignore_validity() {
        for (lower, upper, truthy) in runs(mask_values) {
            let fill = if truthy { value } else { repeated };
            buf.extend(std::iter::repeat_n(fill, upper - lower));
        }
    } else {
        let values = array.flat_values().unwrap();
        for (lower, upper, truthy) in runs(mask_values) {
            if truthy {
                buf.extend(std::iter::repeat_n(value, upper - lower));
            } else {
                buf.extend_from_slice(&values[lower..upper]);
            }
        }
    }

    // Wherever the mask is set the element now holds `value`, so it is no longer null.
    let validity = array
        .validity()
        .map(|validity| validity.to_flat().bitor(mask_values));

    PlPrimitiveArray::new(buf.into(), array.len(), validity.map(PlBitmap::from_bitmap))
}

/// Sets the elements of `array` at `idx` to `value`, leaving its validity as it is.
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

    Ok(PlPrimitiveArray::new(
        buf.into(),
        array.len(),
        validity.map(PlBitmap::from_bitmap),
    ))
}
