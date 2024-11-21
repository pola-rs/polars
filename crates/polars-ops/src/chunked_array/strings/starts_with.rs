use arrow::array::View;
use arrow::buffer::Buffer;
use polars_utils::slice::SliceAble;

/// Checks if the string starts with the prefix
/// When the prefix is smaller than View::MAX_INLINE_SIZE then this will be very fast
pub(crate) fn starts_with_str(view: View, prefix: &str, buffers: &[Buffer<u8>]) -> bool {
    unsafe {
        if view.length <= View::MAX_INLINE_SIZE {
            view.get_inlined_slice_unchecked()
                .starts_with(prefix.as_bytes())
        } else {
            let starts = view
                .prefix
                .to_le_bytes()
                .starts_with(prefix.as_bytes().slice_unchecked(0..4));
            if starts {
                return view
                    .get_slice_unchecked(buffers)
                    .starts_with(prefix.as_bytes());
            }
            false
        }
    }
}

/// Checks if the string starts with the prefix
/// If you call this in a loop and the prefix doesn't change then prefer starts_with_str()
pub(crate) fn starts_with_view(
    view: View,
    prefix: View,
    left_buffers: &[Buffer<u8>],
    right_buffers: &[Buffer<u8>],
) -> bool {
    unsafe {
        if !view.prefix.to_le_bytes()[0..view.length.min(4) as usize]
            .starts_with(&prefix.prefix.to_le_bytes()[..view.length.min(4) as usize])
        {
            return false;
        }

        let left_buffer = view.get_slice_unchecked(left_buffers);
        let right_buffer = prefix.get_slice_unchecked(right_buffers);

        left_buffer.starts_with(right_buffer)
    }
}
