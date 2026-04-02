use arrow::types::AlignedBytes;

use super::ArrayChunks;
use crate::parquet::error::ParquetResult;

#[inline(never)]
pub fn decode<B: AlignedBytes>(
    values: ArrayChunks<'_, B>,
    target: &mut Vec<B>,
) -> ParquetResult<()> {
    if values.is_empty() {
        return Ok(());
    }

    target.reserve(values.len());

    // SAFETY: Vec guarantees if the `capacity != 0` the pointer to valid since we just reserve
    // that pointer.
    let dst = unsafe { target.as_mut_ptr().add(target.len()) };
    let src = values.as_ptr();

    // SAFETY:
    // - `src` is valid for read of values.len() elements.
    // - `dst` is valid for writes of values.len() elements, it was just reserved.
    // - B::Unaligned is always aligned, since it has an alignment of 1
    // - The ranges for src and dst do not overlap
    unsafe {
        std::ptr::copy_nonoverlapping::<B::Unaligned>(src.cast(), dst.cast(), values.len());
        target.set_len(target.len() + values.len());
    };

    Ok(())
}
