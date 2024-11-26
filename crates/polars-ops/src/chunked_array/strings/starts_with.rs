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
                .to_ne_bytes()
                .starts_with(prefix.as_bytes().slice_unchecked(0..4.min(prefix.len())));
            if starts {
                return view
                    .get_slice_unchecked(buffers)
                    .starts_with(prefix.as_bytes());
            }
            false
        }
    }
}
