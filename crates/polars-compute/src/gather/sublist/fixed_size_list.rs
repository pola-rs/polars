//! The `arr.get` kernels over the arrays of `polars-array`.

use arrow::legacy::index::IndexToUsize;
use polars_array::bitmap::combine_validities_and;
use polars_array::builder::new_full_null_like;
use polars_array::{PlArray, PlBitmapRef, PlFixedSizeListArray, PlPrimitiveArray};
use polars_error::{PolarsResult, polars_bail};
use polars_utils::IdxSize;

use crate::gather::take_unchecked;

/// The position `index` picks out of a list of `width` values, or `None` if it falls outside it.
#[inline]
fn position_in(index: i64, width: usize) -> Option<usize> {
    index.negative_to_usize(width)
}

/// `out` masked off wherever `mask` leaves an element null, on top of the mask it carries itself.
///
/// An element that is null holds no values, whatever the values under it read as, so it holds no
/// value at any index either.
fn masked_off(out: Box<dyn PlArray>, mask: Option<PlBitmapRef<'_>>) -> Box<dyn PlArray> {
    if mask.is_none() {
        return out;
    }

    let validity = combine_validities_and(out.validity(), mask);
    out.with_validity(validity)
}

/// Returns the value at `index` within every element of `arr`.
pub fn sub_fixed_size_list_get_literal(
    arr: &PlFixedSizeListArray,
    index: i64,
    null_on_oob: bool,
) -> PolarsResult<Box<dyn PlArray>> {
    if arr.is_empty() {
        return Ok(arr.values().sliced(0, 0));
    }

    // Every element is `width` values wide, so the index falls either within all of them or within
    // none: it is resolved once, and an out of bounds one is answered without a value being read.
    let Some(offset) = position_in(index, arr.width()) else {
        if !null_on_oob {
            polars_bail!(ComputeError: "get index is out of bounds");
        }
        return Ok(new_full_null_like(arr.values(), arr.len()));
    };

    // Values that hold the single element every element of `arr` repeats are indexed in place: the
    // value at `offset` within that one element is the answer at every element in turn, in `O(1)`.
    if let Some(values) = arr.scalar_value_ignore_validity() {
        // SAFETY: `offset` is within the width, which is how many values the one element holds.
        let out = unsafe { values.new_from_index_unchecked(offset, arr.len()) };
        return Ok(masked_off(out, arr.validity()));
    }

    let indices = (0..arr.len())
        // SAFETY: `i` is an element of `arr`.
        .map(|i| (unsafe { arr.value_range_unchecked(i) }.start + offset) as IdxSize)
        .collect::<Vec<_>>();

    // SAFETY: every index lands within the element it is read for.
    let out = unsafe { take_unchecked(arr.values(), &PlPrimitiveArray::from_vec(indices)) };
    Ok(masked_off(out, arr.validity()))
}

/// Returns the value at the index `index` holds for it within every element of `arr`.
pub fn sub_fixed_size_list_get(
    arr: &PlFixedSizeListArray,
    index: &PlPrimitiveArray<i64>,
    null_on_oob: bool,
) -> PolarsResult<Box<dyn PlArray>> {
    assert_eq!(
        arr.len(),
        index.len(),
        "`arr.get` reads one index per element of the array it indexes",
    );

    if arr.is_empty() {
        return Ok(arr.values().sliced(0, 0));
    }

    // Indices stored in the scalar representation are one index shared by every element, which
    // lands at the same position within all of them: it is resolved once, like a literal one.
    if let Some(value) = index.scalar_value_ignore_validity() {
        let out = sub_fixed_size_list_get_literal(arr, value, null_on_oob)?;

        // An index that is null picks out no value at all, which is the null an out of bounds one
        // reads as in turn.
        let Some(validity) = index.validity() else {
            return Ok(out);
        };
        if !null_on_oob && validity.unset_bits() > 0 {
            polars_bail!(ComputeError: "get index is out of bounds");
        }

        let validity = combine_validities_and(out.validity(), Some(validity));
        return Ok(out.with_validity(validity));
    }

    let width = arr.width();
    let mut out_of_bounds = false;
    let indices = index
        .iter()
        .enumerate()
        .map(|(i, index)| {
            let position = index.and_then(|index| position_in(index, width));
            out_of_bounds |= position.is_none();

            position.map(|position| {
                // SAFETY: `i` is an element of `arr`, which holds as many as `index`.
                (unsafe { arr.value_range_unchecked(i) }.start + position) as IdxSize
            })
        })
        .collect::<PlPrimitiveArray<IdxSize>>();

    if !null_on_oob && out_of_bounds {
        polars_bail!(ComputeError: "get index is out of bounds");
    }

    // SAFETY: every index lands within the element it is read for.
    let out = unsafe { take_unchecked(arr.values(), &indices) };
    Ok(masked_off(out, arr.validity()))
}
