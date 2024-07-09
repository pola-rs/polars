use self::primitive::take_values_and_validity_unchecked;
use super::*;
use crate::array::BinaryViewArray;

/// # Safety
/// No bound checks
pub(super) unsafe fn take_binview_unchecked(
    arr: &BinaryViewArray,
    indices: &IdxArr,
) -> BinaryViewArray {
    let (views, validity) =
        take_values_and_validity_unchecked(arr.views(), arr.validity(), indices);

    BinaryViewArray::new_unchecked_unknown_md(
        arr.data_type().clone(),
        views.into(),
        arr.data_buffers().clone(),
        validity,
        Some(arr.total_buffer_len()),
    )
    .maybe_gc()
}
