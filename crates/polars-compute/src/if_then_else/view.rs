use std::sync::Arc;

use arrow::array::{Array, BinaryViewArray, Utf8ViewArray, View};
use arrow::bitmap::Bitmap;
use arrow::buffer::Buffer;
use arrow::datatypes::ArrowDataType;

use super::scalar::{if_then_else_scalar_64, if_then_else_scalar_rest};
use super::IfThenElseKernel;
use crate::if_then_else::scalar::{if_then_else_broadcast_both_scalar_64, if_then_else_broadcast_false_scalar_64};

impl IfThenElseKernel for BinaryViewArray {
    type Scalar<'a> = &'a [u8];

    fn if_then_else(mask: &Bitmap, if_true: &Self, if_false: &Self) -> Self {
        let combined_buffers: Arc<_>;
        let combined_buffer_len: usize;
        let false_buffer_idx_offset: u32;
        if Arc::ptr_eq(if_true.data_buffers(), if_false.data_buffers()) {
            // Share exact same buffers, no need to combine.
            combined_buffers = if_true.data_buffers().clone();
            combined_buffer_len = if_true.total_buffer_len();
            false_buffer_idx_offset = 0;
        } else {
            // Put false buffers after true buffers.
            let true_buffers = if_true.data_buffers().iter().cloned();
            let false_buffers = if_false.data_buffers().iter().cloned();
            combined_buffers = true_buffers.chain(false_buffers).collect();
            combined_buffer_len = if_true.total_buffer_len() + if_false.total_buffer_len();
            false_buffer_idx_offset = if_true.data_buffers().len() as u32;
        }

        let map_false_view = |mut v: View| -> View {
            v.buffer_idx += false_buffer_idx_offset;
            v
        };
        let views = super::if_then_else_loop(
            mask,
            if_true.views(),
            if_false.views(),
            |m, t, f, o| if_then_else_scalar_rest(m, t, f, map_false_view, o),
            |m, t, f, o| if_then_else_scalar_64(m, t, f, map_false_view, o),
        );

        let validity = super::if_then_else_validity(mask, if_true.validity(), if_false.validity());
        unsafe {
            BinaryViewArray::new_unchecked_unknown_md(
                if_true.data_type().clone(),
                views.into(),
                combined_buffers,
                validity,
                Some(combined_buffer_len),
            )
        }
    }

    fn if_then_else_broadcast_true(mask: &Bitmap, if_true: Self::Scalar<'_>, if_false: &Self) -> Self {
        // It's cheaper if we put the false buffers first, that way we don't need to modify any views in the loop.
        let false_buffers = if_false.data_buffers().iter().cloned();
        let true_buffer: Buffer<u8> = if_true.to_owned().into();
        let combined_buffers: Arc<_> = false_buffers.chain(std::iter::once(true_buffer)).collect();
        let combined_buffer_len = if_false.total_buffer_len() + if_true.len();

        let true_buffer_idx_offset: u32 = if_false.data_buffers().len() as u32;
        let true_view = View::new_from_bytes(if_true, true_buffer_idx_offset, 0);
        let views = super::if_then_else_loop_broadcast_false(
            true, // Invert the mask so we effectively broadcast true.
            mask,
            if_false.views(),
            true_view,
            if_then_else_broadcast_false_scalar_64,
        );

        let validity = super::if_then_else_validity(mask, None, if_false.validity());
        unsafe {
            BinaryViewArray::new_unchecked_unknown_md(
                if_false.data_type().clone(),
                views.into(),
                combined_buffers,
                validity,
                Some(combined_buffer_len),
            )
        }
    }

    fn if_then_else_broadcast_false(
        mask: &Bitmap,
        if_true: &Self,
        if_false: Self::Scalar<'_>,
    ) -> Self {
        // It's cheaper if we put the true buffers first, that way we don't need to modify any views in the loop.
        let true_buffers = if_true.data_buffers().iter().cloned();
        let false_buffer: Buffer<u8> = if_false.to_owned().into();
        let combined_buffers: Arc<_> = true_buffers.chain(std::iter::once(false_buffer)).collect();
        let combined_buffer_len = if_true.total_buffer_len() + if_false.len();

        let false_buffer_idx_offset: u32 = if_true.data_buffers().len() as u32;
        let false_view = View::new_from_bytes(if_false, false_buffer_idx_offset, 0);
        let views = super::if_then_else_loop_broadcast_false(
            false,
            mask,
            if_true.views(),
            false_view,
            if_then_else_broadcast_false_scalar_64,
        );

        let validity = super::if_then_else_validity(mask, if_true.validity(), None);
        unsafe {
            BinaryViewArray::new_unchecked_unknown_md(
                if_true.data_type().clone(),
                views.into(),
                combined_buffers,
                validity,
                Some(combined_buffer_len),
            )
        }
    }

    fn if_then_else_broadcast_both(dtype: ArrowDataType, mask: &Bitmap, if_true: Self::Scalar<'_>, if_false: Self::Scalar<'_>) -> Self {
        let total_len = if_true.len() + if_false.len();
        let buffer: Buffer<u8> = [if_true, if_false].concat().into();
        let buffers: Arc<_> = std::iter::once(buffer).collect();
        let true_view = View::new_from_bytes(if_true, 0, 0);
        let false_view = View::new_from_bytes(if_false, 0, if_true.len().try_into().unwrap());
        let views = super::if_then_else_loop_broadcast_both(
            mask,
            true_view,
            false_view,
            if_then_else_broadcast_both_scalar_64,
        );
        unsafe {
            BinaryViewArray::new_unchecked(
                dtype,
                views.into(),
                buffers,
                None,
                total_len,
                total_len,
            )
        }
    }
}

impl IfThenElseKernel for Utf8ViewArray {
    type Scalar<'a> = &'a str;

    fn if_then_else(mask: &Bitmap, if_true: &Self, if_false: &Self) -> Self {
        let ret =
            IfThenElseKernel::if_then_else(mask, &if_true.to_binview(), &if_false.to_binview());
        unsafe { ret.to_utf8view_unchecked() }
    }

    fn if_then_else_broadcast_true(mask: &Bitmap, if_true: Self::Scalar<'_>, if_false: &Self) -> Self {
        let ret = IfThenElseKernel::if_then_else_broadcast_true(
            mask,
            if_true.as_bytes(),
            &if_false.to_binview(),
        );
        unsafe { ret.to_utf8view_unchecked() }
    }

    fn if_then_else_broadcast_false(
        mask: &Bitmap,
        if_true: &Self,
        if_false: Self::Scalar<'_>,
    ) -> Self {
        let ret = IfThenElseKernel::if_then_else_broadcast_false(
            mask,
            &if_true.to_binview(),
            if_false.as_bytes(),
        );
        unsafe { ret.to_utf8view_unchecked() }
    }

    fn if_then_else_broadcast_both(dtype: ArrowDataType, mask: &Bitmap, if_true: Self::Scalar<'_>, if_false: Self::Scalar<'_>) -> Self {
        let ret: BinaryViewArray = IfThenElseKernel::if_then_else_broadcast_both(
            dtype,
            mask,
            if_true.as_bytes(),
            if_false.as_bytes(),
        );
        unsafe { ret.to_utf8view_unchecked() }
    }
}
