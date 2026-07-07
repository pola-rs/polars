use arrow::array::{Array, StructArray};
use arrow::bitmap::Bitmap;

use super::{SpanPlan, build_inner_values, build_values_validity};

/// Concatenate the struct inner values of a list column horizontally.
///
/// Mirrors the approach in [`crate::horizontal_flatten`]'s struct kernel: the
/// concatenation is applied independently to every struct field (recursing
/// through [`build_inner_values`]) as well as to the struct's own validity,
/// then the parts are reassembled. The span plan describes the row-major
/// element order and is replayed for every field.
///
/// # Safety
/// All preconditions in [`super::horizontal_concat_list_unchecked`].
pub(super) unsafe fn build_struct_values(
    arrays: &[StructArray],
    plan: &SpanPlan<'_>,
) -> StructArray {
    let dtype = arrays[0].dtype().clone();
    let n_fields = arrays[0].values().len();

    let mut scratch: Vec<Box<dyn Array>> = Vec::with_capacity(arrays.len());
    let field_arrays = (0..n_fields)
        .map(|f| {
            scratch.clear();
            scratch.extend(arrays.iter().map(|a| a.values()[f].clone()));
            build_inner_values(&scratch, plan)
        })
        .collect::<Vec<_>>();

    let per_col_validity: Vec<Option<&Bitmap>> = arrays.iter().map(|a| a.validity()).collect();
    let validity = build_values_validity(&per_col_validity, plan);

    StructArray::new(dtype, plan.out_values_len, field_arrays, validity)
}
