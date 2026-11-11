use polars_arrow::array::builder::{ArrayBuilder, ShareStrategy, make_builder};
use polars_arrow::array::{
    Array, ChildPositions, LargeUnionArray, new_null_array,
};
use polars_arrow::bitmap::{Bitmap, BitmapBuilder};
use polars_arrow::compute::utils::combine_validities_and;
use polars_arrow::datatypes::{ArrowDataType, IdxArr};
use polars_utils::IdxSize;

use super::bitmap::take_bitmap_nulls_unchecked;
use super::has_native_take;

pub(super) unsafe fn take_unchecked(array: &LargeUnionArray, indices: &IdxArr) -> LargeUnionArray {
    let validity = array
        .validity()
        .map(|b| take_bitmap_nulls_unchecked(b, indices));
    let validity = combine_validities_and(validity.as_ref(), indices.validity());
    gather_union_slots(
        array.dtype(),
        array.types(),
        array.offsets(),
        array.fields(),
        validity,
        indices.values().iter().map(|i| *i as usize),
    )
}

/// Gathers the queued `positions` of `child` into a dense array, with nulls for `None`.
///
/// # Safety
/// `positions` must have been created for `child.len()`, and every queued position must be in
/// bounds of `child`.
pub unsafe fn gather_child(child: &dyn Array, mut positions: ChildPositions) -> Box<dyn Array> {
    if has_native_take(child.dtype()) {
        if let ChildPositions::Idxs { idxs, has_none } = &mut positions {
            let mut idxs = std::mem::take(idxs);
            let validity = has_none.then(|| {
                // The native take reads the value under a null index, so it must be in bounds.
                let mut validity = BitmapBuilder::with_capacity(idxs.len());
                for i in idxs.iter_mut() {
                    validity.push(*i != IdxSize::MAX);
                    if *i == IdxSize::MAX {
                        *i = 0;
                    }
                }
                validity.freeze()
            });
            let idxs = IdxArr::from_vec(idxs).with_validity(validity);
            if *has_none && child.is_empty() {
                let placeholder = new_null_array(child.dtype().clone(), 1);
                return unsafe { super::take_unchecked(placeholder.as_ref(), &idxs) };
            }
            return unsafe { super::take_unchecked(child, &idxs) };
        }
    }
    let mut builder = make_builder(child.dtype());
    unsafe { positions.flush(&mut *builder, child, ShareStrategy::Always) };
    builder.freeze()
}

/// Gathers the union slots `rows` of the given parts into a new array whose output row `j` is
/// null iff `validity[j]` is unset. The parts need not have increasing offsets per child.
///
/// # Safety
/// Every row must be in bounds, and the `(type, offset)` of every row that is valid in the output
/// must be in bounds.
pub unsafe fn gather_union_slots(
    dtype: &ArrowDataType,
    types: &[i8],
    offsets: &[i64],
    fields: &[Box<dyn Array>],
    validity: Option<Bitmap>,
    rows: impl ExactSizeIterator<Item = usize>,
) -> LargeUnionArray {
    let len = rows.len();
    let mut out_types = Vec::with_capacity(len);
    let mut out_offsets = Vec::with_capacity(len);
    let mut positions: Vec<ChildPositions> =
        fields.iter().map(|c| ChildPositions::new(c.len())).collect();
    let mut counts = vec![0i64; fields.len()];
    for (j, i) in rows.enumerate() {
        // Null slots get a null placeholder in child 0.
        let (t, pos) = if validity.as_ref().is_none_or(|v| v.get_bit_unchecked(j)) {
            let t = *types.get_unchecked(i) as usize;
            (t, Some(*offsets.get_unchecked(i) as usize))
        } else {
            (0, None)
        };
        out_types.push(t as i8);
        out_offsets.push(counts[t]);
        counts[t] += 1;
        positions[t].push(pos);
    }

    let fields = fields
        .iter()
        .zip(positions)
        .map(|(child, positions)| gather_child(child.as_ref(), positions))
        .collect();

    LargeUnionArray::new_unchecked(
        dtype.clone(),
        out_types.into(),
        out_offsets.into(),
        fields,
        validity,
    )
}
