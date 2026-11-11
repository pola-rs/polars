use polars_arrow::array::builder::{ShareStrategy, StaticArrayBuilder};
use polars_arrow::array::{Array, LargeUnionArray, LargeUnionArrayBuilder, new_null_array};
use polars_arrow::bitmap::BitmapBuilder;
use polars_arrow::compute::utils::combine_validities_and;
use polars_arrow::datatypes::IdxArr;
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

    let n = array.fields().len();
    let mut types = Vec::with_capacity(indices.len());
    let mut offsets = Vec::with_capacity(indices.len());
    let mut child_idxs: Vec<Vec<IdxSize>> = vec![Vec::new(); n];
    // Null slots get a null placeholder in child 0, gathered through a null index.
    let mut child0_validity = BitmapBuilder::new();
    for (j, &i) in indices.values().iter().enumerate() {
        let (t, pos) = if validity.as_ref().is_none_or(|v| v.get_bit_unchecked(j)) {
            let (t, pos) = array.index_unchecked(i as usize);
            (t, Some(pos))
        } else {
            (0, None)
        };
        types.push(t as i8);
        offsets.push(child_idxs[t].len() as i64);
        // Checked: a position past `IdxSize::MAX` must take the builder path above.
        child_idxs[t].push(IdxSize::try_from(pos.unwrap_or(0)).unwrap());
        if t == 0 {
            child0_validity.push(pos.is_some());
        }
    }

    let fields = array
        .fields()
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
        array.dtype().clone(),
        types.into(),
        offsets.into(),
        fields,
        validity,
    )
}
