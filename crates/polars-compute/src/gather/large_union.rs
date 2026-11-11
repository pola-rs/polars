use polars_arrow::array::builder::{ShareStrategy, StaticArrayBuilder};
use polars_arrow::array::{Array, LargeUnionArray, LargeUnionArrayBuilder, new_null_array};
use polars_arrow::bitmap::{Bitmap, BitmapBuilder};
use polars_arrow::compute::utils::combine_validities_and;
use polars_arrow::datatypes::{ArrowDataType, IdxArr};
use polars_utils::IdxSize;

use super::bitmap::take_bitmap_nulls_unchecked;

pub(super) unsafe fn take_unchecked(array: &LargeUnionArray, indices: &IdxArr) -> LargeUnionArray {
    // Child positions must fit in `IdxSize` for the per-child gathers.
    if array
        .fields()
        .iter()
        .any(|c| c.len() > IdxSize::MAX as usize)
    {
        let mut builder = LargeUnionArrayBuilder::new(array.dtype().clone());
        let rows = indices.iter().map(|i| i.map(|i| *i as usize));
        builder.extend_rows(array, rows, ShareStrategy::Always);
        return builder.freeze();
    }

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

/// Gathers the union slots `rows` of the given parts into a new array whose output row `j` is
/// null iff `validity[j]` is unset. The parts need not have increasing offsets per child.
///
/// # Safety
/// Every row must be in bounds, and the `(type, offset)` of every row that is valid in the output
/// must be in bounds. Every child must be at most `IdxSize::MAX` long.
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
    let mut child_idxs: Vec<Vec<IdxSize>> = vec![Vec::new(); fields.len()];
    // Null slots get a null placeholder in child 0, gathered through a null index.
    let mut child0_validity = BitmapBuilder::new();
    for (j, i) in rows.enumerate() {
        let (t, pos) = if validity.as_ref().is_none_or(|v| v.get_bit_unchecked(j)) {
            let t = *types.get_unchecked(i) as usize;
            (t, Some(*offsets.get_unchecked(i)))
        } else {
            (0, None)
        };
        out_types.push(t as i8);
        out_offsets.push(child_idxs[t].len() as i64);
        // Checked: a position past `IdxSize::MAX` must take the builder path.
        child_idxs[t].push(IdxSize::try_from(pos.unwrap_or(0)).unwrap());
        if t == 0 {
            child0_validity.push(pos.is_some());
        }
    }

    let fields = fields
        .iter()
        .zip(child_idxs)
        .enumerate()
        .map(|(t, (child, idxs))| {
            let mut idxs = IdxArr::from_vec(idxs);
            if t == 0 {
                let child0_validity = std::mem::take(&mut child0_validity).into_opt_validity();
                if child0_validity.is_some() {
                    idxs = idxs.with_validity(child0_validity);
                    if child.is_empty() {
                        let placeholder = new_null_array(child.dtype().clone(), 1);
                        return super::take_unchecked(placeholder.as_ref(), &idxs);
                    }
                }
            }
            super::take_unchecked(child.as_ref(), &idxs)
        })
        .collect();

    LargeUnionArray::new_unchecked(
        dtype.clone(),
        out_types.into(),
        out_offsets.into(),
        fields,
        validity,
    )
}
