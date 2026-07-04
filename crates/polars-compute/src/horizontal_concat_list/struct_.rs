use arrow::array::{Array, StructArray};
use arrow::bitmap::Bitmap;

use super::{Span, build_inner_values, build_values_validity};

/// Concatenate the struct inner values of a list column horizontally.
///
/// Mirrors the approach in [`crate::horizontal_flatten`]'s struct kernel: the
/// concatenation is applied independently to every struct field (recursing
/// through [`build_inner_values`]) as well as to the struct's own validity,
/// then the parts are reassembled. The `spans` describe the row-major element
/// order and are shared across all fields.
///
/// # Safety
/// All preconditions in [`super::horizontal_concat_list_unchecked`].
pub(super) unsafe fn build_struct_values(
    arrays: &[StructArray],
    spans: &[Span],
    out_values_len: usize,
) -> StructArray {
    let dtype = arrays[0].dtype().clone();
    let n_fields = arrays[0].values().len();

    let mut scratch: Vec<Box<dyn Array>> = Vec::with_capacity(arrays.len());
    let field_arrays = (0..n_fields)
        .map(|f| {
            scratch.clear();
            scratch.extend(arrays.iter().map(|a| a.values()[f].clone()));
            build_inner_values(&scratch, spans, out_values_len)
        })
        .collect::<Vec<_>>();

    let per_col_validity: Vec<Option<&Bitmap>> = arrays.iter().map(|a| a.validity()).collect();
    let validity = build_values_validity(&per_col_validity, spans, out_values_len);

    StructArray::new(dtype, out_values_len, field_arrays, validity)
}
