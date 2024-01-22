use super::*;

pub(super) fn filter_bitmap_and_validity(
    values: &Bitmap,
    validity: Option<&Bitmap>,
    mask: &Bitmap,
) -> (MutableBitmap, Option<MutableBitmap>) {
    if let Some(validity) = validity {
        let (values, validity) = null_filter(values, validity, mask);
        (values, Some(validity))
    } else {
        (nonnull_filter(values, mask), None)
    }
}

/// # Safety
/// This assumes that the `mask_chunks` contains a number of set/true items equal
/// to `filter_count`
unsafe fn nonnull_filter_impl<I>(
    values: &Bitmap,
    mut mask_chunks: I,
    filter_count: usize,
) -> MutableBitmap
where
    I: BitChunkIterExact<u64>,
{
    // TOOO! we might use ChunksExact here if offset = 0.
    let mut chunks = values.chunks::<u64>();
    let mut new = MutableBitmap::with_capacity(filter_count);

    chunks
        .by_ref()
        .zip(mask_chunks.by_ref())
        .for_each(|(chunk, mask_chunk)| {
            let ones = mask_chunk.count_ones();
            let leading_ones = get_leading_ones(mask_chunk);

            if ones == leading_ones {
                let size = leading_ones as usize;
                unsafe { new.extend_from_slice_unchecked(chunk.to_ne_bytes().as_ref(), 0, size) };
                return;
            }

            let ones_iter = BitChunkOnes::from_known_count(mask_chunk, ones as usize);
            for pos in ones_iter {
                new.push_unchecked(chunk & (1 << pos) > 0);
            }
        });

    chunks
        .remainder_iter()
        .zip(mask_chunks.remainder_iter())
        .for_each(|(value, is_selected)| {
            if is_selected {
                unsafe {
                    new.push_unchecked(value);
                };
            }
        });

    new
}

/// # Safety
/// This assumes that the `mask_chunks` contains a number of set/true items equal
/// to `filter_count`
unsafe fn null_filter_impl<I>(
    values: &Bitmap,
    validity: &Bitmap,
    mut mask_chunks: I,
    filter_count: usize,
) -> (MutableBitmap, MutableBitmap)
where
    I: BitChunkIterExact<u64>,
{
    let mut chunks = values.chunks::<u64>();
    let mut validity_chunks = validity.chunks::<u64>();

    let mut new = MutableBitmap::with_capacity(filter_count);
    let mut new_validity = MutableBitmap::with_capacity(filter_count);

    chunks
        .by_ref()
        .zip(validity_chunks.by_ref())
        .zip(mask_chunks.by_ref())
        .for_each(|((chunk, validity_chunk), mask_chunk)| {
            let ones = mask_chunk.count_ones();
            let leading_ones = get_leading_ones(mask_chunk);

            if ones == leading_ones {
                let size = leading_ones as usize;

                unsafe {
                    new.extend_from_slice_unchecked(chunk.to_ne_bytes().as_ref(), 0, size);

                    // safety: invariant offset + length <= slice.len()
                    new_validity.extend_from_slice_unchecked(
                        validity_chunk.to_ne_bytes().as_ref(),
                        0,
                        size,
                    );
                }
                return;
            }

            // this triggers a bitcount
            let ones_iter = BitChunkOnes::from_known_count(mask_chunk, ones as usize);
            for pos in ones_iter {
                new.push_unchecked(chunk & (1 << pos) > 0);
                new_validity.push_unchecked(validity_chunk & (1 << pos) > 0);
            }
        });

    chunks
        .remainder_iter()
        .zip(validity_chunks.remainder_iter())
        .zip(mask_chunks.remainder_iter())
        .for_each(|((value, is_valid), is_selected)| {
            if is_selected {
                unsafe {
                    new.push_unchecked(value);
                    new_validity.push_unchecked(is_valid);
                };
            }
        });

    (new, new_validity)
}

fn null_filter(
    values: &Bitmap,
    validity: &Bitmap,
    mask: &Bitmap,
) -> (MutableBitmap, MutableBitmap) {
    assert_eq!(values.len(), mask.len());
    let filter_count = mask.len() - mask.unset_bits();

    let (slice, offset, length) = mask.as_slice();
    if offset == 0 {
        let mask_chunks = BitChunksExact::<u64>::new(slice, length);
        unsafe { null_filter_impl(values, validity, mask_chunks, filter_count) }
    } else {
        let mask_chunks = mask.chunks::<u64>();
        unsafe { null_filter_impl(values, validity, mask_chunks, filter_count) }
    }
}

fn nonnull_filter(values: &Bitmap, mask: &Bitmap) -> MutableBitmap {
    assert_eq!(values.len(), mask.len());
    let filter_count = mask.len() - mask.unset_bits();

    let (slice, offset, length) = mask.as_slice();
    if offset == 0 {
        let mask_chunks = BitChunksExact::<u64>::new(slice, length);
        unsafe { nonnull_filter_impl(values, mask_chunks, filter_count) }
    } else {
        let mask_chunks = mask.chunks::<u64>();
        unsafe { nonnull_filter_impl(values, mask_chunks, filter_count) }
    }
}
