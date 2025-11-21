use arrow::array::{BinaryViewArrayGeneric, ViewType};

use self::primitive::take_values_and_validity_unchecked;
use super::*;

/// # Safety
/// No bound checks
pub unsafe fn take_binview_unchecked<V: ViewType + ?Sized>(
    arr: &BinaryViewArrayGeneric<V>,
    indices: &IdxArr,
) -> BinaryViewArrayGeneric<V> {
    let (views, validity) =
        take_values_and_validity_unchecked(arr.views(), arr.validity(), indices);

    BinaryViewArrayGeneric::new_unchecked_unknown_md(
        arr.dtype().clone(),
        views.into(),
        arr.data_buffers().clone(),
        validity,
        Some(arr.total_buffer_len()),
    )
    .maybe_gc()
}
