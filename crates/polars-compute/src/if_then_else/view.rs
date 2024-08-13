use std::mem::MaybeUninit;
use std::ops::Deref;
use std::sync::Arc;

use arrow::array::{Array, BinaryViewArray, MutablePlBinary, Utf8ViewArray, View};
use arrow::bitmap::Bitmap;
use arrow::buffer::Buffer;
use arrow::datatypes::ArrowDataType;
use polars_utils::aliases::{InitHashMaps, PlHashSet};

use super::IfThenElseKernel;
use crate::if_then_else::scalar::if_then_else_broadcast_both_scalar_64;

// Makes a buffer and a set of views into that buffer from a set of strings.
// Does not allocate a buffer if not necessary.
fn make_buffer_and_views<const N: usize>(
    strings: [&[u8]; N],
    buffer_idx: u32,
) -> ([View; N], Option<Buffer<u8>>) {
    let mut buf_data = Vec::new();
    let views = strings.map(|s| {
        let offset = buf_data.len().try_into().unwrap();
        if s.len() > 12 {
            buf_data.extend(s);
        }
        View::new_from_bytes(s, buffer_idx, offset)
    });
    let buf = (!buf_data.is_empty()).then(|| buf_data.into());
    (views, buf)
}

fn has_duplicate_buffers(bufs: &[Buffer<u8>]) -> bool {
    let mut has_duplicate_buffers = false;
    let mut bufset = PlHashSet::new();
    for buf in bufs {
        if !bufset.insert(buf.as_ptr()) {
            has_duplicate_buffers = true;
            break;
        }
    }
    has_duplicate_buffers
}

impl IfThenElseKernel for BinaryViewArray {
    type Scalar<'a> = &'a [u8];

    fn if_then_else(mask: &Bitmap, if_true: &Self, if_false: &Self) -> Self {
        let combined_buffers: Arc<_>;
        let false_buffer_idx_offset: u32;
        let mut has_duplicate_bufs = false;
        if Arc::ptr_eq(if_true.data_buffers(), if_false.data_buffers()) {
            // Share exact same buffers, no need to combine.
            combined_buffers = if_true.data_buffers().clone();
            false_buffer_idx_offset = 0;
        } else {
            // Put false buffers after true buffers.
            let true_buffers = if_true.data_buffers().iter().cloned();
            let false_buffers = if_false.data_buffers().iter().cloned();

            combined_buffers = true_buffers.chain(false_buffers).collect();
            has_duplicate_bufs = has_duplicate_buffers(&combined_buffers);
            false_buffer_idx_offset = if_true.data_buffers().len() as u32;
        }

        let views = super::if_then_else_loop(
            mask,
            if_true.views(),
            if_false.views(),
            |m, t, f, o| if_then_else_view_rest(m, t, f, o, false_buffer_idx_offset),
            |m, t, f, o| if_then_else_view_64(m, t, f, o, false_buffer_idx_offset),
        );

        let validity = super::if_then_else_validity(mask, if_true.validity(), if_false.validity());

        let mut builder = MutablePlBinary::with_capacity(views.len());

        if has_duplicate_bufs {
            unsafe {
                builder.extend_non_null_views_unchecked_dedupe(
                    views.into_iter(),
                    combined_buffers.deref(),
                )
            };
        } else {
            unsafe {
                builder.extend_non_null_views_unchecked(views.into_iter(), combined_buffers.deref())
            };
        }
        builder
            .freeze_with_dtype(if_true.data_type().clone())
            .with_validity(validity)
    }

    fn if_then_else_broadcast_true(
        mask: &Bitmap,
        if_true: Self::Scalar<'_>,
        if_false: &Self,
    ) -> Self {
        // It's cheaper if we put the false buffers first, that way we don't need to modify any views in the loop.
        let false_buffers = if_false.data_buffers().iter().cloned();
        let true_buffer_idx_offset: u32 = if_false.data_buffers().len() as u32;
        let ([true_view], true_buffer) = make_buffer_and_views([if_true], true_buffer_idx_offset);
        let combined_buffers: Arc<_> = false_buffers.chain(true_buffer).collect();

        let views = super::if_then_else_loop_broadcast_false(
            true, // Invert the mask so we effectively broadcast true.
            mask,
            if_false.views(),
            true_view,
            if_then_else_broadcast_false_view_64,
        );

        let validity = super::if_then_else_validity(mask, None, if_false.validity());

        let mut builder = MutablePlBinary::with_capacity(views.len());

        unsafe {
            if has_duplicate_buffers(&combined_buffers) {
                builder.extend_non_null_views_unchecked_dedupe(
                    views.into_iter(),
                    combined_buffers.deref(),
                )
            } else {
                builder.extend_non_null_views_unchecked(views.into_iter(), combined_buffers.deref())
            }
        }
        builder
            .freeze_with_dtype(if_false.data_type().clone())
            .with_validity(validity)
    }

    fn if_then_else_broadcast_false(
        mask: &Bitmap,
        if_true: &Self,
        if_false: Self::Scalar<'_>,
    ) -> Self {
        // It's cheaper if we put the true buffers first, that way we don't need to modify any views in the loop.
        let true_buffers = if_true.data_buffers().iter().cloned();
        let false_buffer_idx_offset: u32 = if_true.data_buffers().len() as u32;
        let ([false_view], false_buffer) =
            make_buffer_and_views([if_false], false_buffer_idx_offset);
        let combined_buffers: Arc<_> = true_buffers.chain(false_buffer).collect();

        let views = super::if_then_else_loop_broadcast_false(
            false,
            mask,
            if_true.views(),
            false_view,
            if_then_else_broadcast_false_view_64,
        );

        let validity = super::if_then_else_validity(mask, if_true.validity(), None);

        let mut builder = MutablePlBinary::with_capacity(views.len());
        unsafe {
            if has_duplicate_buffers(&combined_buffers) {
                builder.extend_non_null_views_unchecked_dedupe(
                    views.into_iter(),
                    combined_buffers.deref(),
                )
            } else {
                builder.extend_non_null_views_unchecked(views.into_iter(), combined_buffers.deref())
            }
        };
        builder
            .freeze_with_dtype(if_true.data_type().clone())
            .with_validity(validity)
    }

    fn if_then_else_broadcast_both(
        dtype: ArrowDataType,
        mask: &Bitmap,
        if_true: Self::Scalar<'_>,
        if_false: Self::Scalar<'_>,
    ) -> Self {
        let ([true_view, false_view], buffer) = make_buffer_and_views([if_true, if_false], 0);
        let buffers: Arc<_> = buffer.into_iter().collect();
        let views = super::if_then_else_loop_broadcast_both(
            mask,
            true_view,
            false_view,
            if_then_else_broadcast_both_scalar_64,
        );

        let mut builder = MutablePlBinary::with_capacity(views.len());
        unsafe {
            if has_duplicate_buffers(&buffers) {
                builder.extend_non_null_views_unchecked_dedupe(views.into_iter(), buffers.deref())
            } else {
                builder.extend_non_null_views_unchecked(views.into_iter(), buffers.deref())
            }
        };
        builder.freeze_with_dtype(dtype)
    }
}

impl IfThenElseKernel for Utf8ViewArray {
    type Scalar<'a> = &'a str;

    fn if_then_else(mask: &Bitmap, if_true: &Self, if_false: &Self) -> Self {
        let ret =
            IfThenElseKernel::if_then_else(mask, &if_true.to_binview(), &if_false.to_binview());
        unsafe { ret.to_utf8view_unchecked() }
    }

    fn if_then_else_broadcast_true(
        mask: &Bitmap,
        if_true: Self::Scalar<'_>,
        if_false: &Self,
    ) -> Self {
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

    fn if_then_else_broadcast_both(
        dtype: ArrowDataType,
        mask: &Bitmap,
        if_true: Self::Scalar<'_>,
        if_false: Self::Scalar<'_>,
    ) -> Self {
        let ret: BinaryViewArray = IfThenElseKernel::if_then_else_broadcast_both(
            dtype,
            mask,
            if_true.as_bytes(),
            if_false.as_bytes(),
        );
        unsafe { ret.to_utf8view_unchecked() }
    }
}

pub fn if_then_else_view_rest(
    mask: u64,
    if_true: &[View],
    if_false: &[View],
    out: &mut [MaybeUninit<View>],
    false_buffer_idx_offset: u32,
) {
    assert!(if_true.len() <= out.len()); // Removes bounds checks in inner loop.
    let true_it = if_true.iter();
    let false_it = if_false.iter();
    for (i, (t, f)) in true_it.zip(false_it).enumerate() {
        // Written like this, this loop *should* be branchless.
        // Unfortunately we're still dependent on the compiler.
        let m = (mask >> i) & 1 != 0;
        let src = if m { t } else { f };
        let mut v = *src;
        let offset = if m | (v.length <= 12) {
            // Yes, | instead of || is intentional.
            0
        } else {
            false_buffer_idx_offset
        };
        v.buffer_idx += offset;
        out[i] = MaybeUninit::new(v);
    }
}

pub fn if_then_else_view_64(
    mask: u64,
    if_true: &[View; 64],
    if_false: &[View; 64],
    out: &mut [MaybeUninit<View>; 64],
    false_buffer_idx_offset: u32,
) {
    if_then_else_view_rest(mask, if_true, if_false, out, false_buffer_idx_offset)
}

// Using the scalar variant of this works, but was slower, we want to select a source pointer and
// then copy it. Using this version for the integers results in branches.
pub fn if_then_else_broadcast_false_view_64(
    mask: u64,
    if_true: &[View; 64],
    if_false: View,
    out: &mut [MaybeUninit<View>; 64],
) {
    assert!(if_true.len() == out.len()); // Removes bounds checks in inner loop.
    for (i, t) in if_true.iter().enumerate() {
        let src = if (mask >> i) & 1 != 0 { t } else { &if_false };
        out[i] = MaybeUninit::new(*src);
    }
}
