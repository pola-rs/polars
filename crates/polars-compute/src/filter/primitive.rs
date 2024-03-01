use super::*;

pub(super) fn filter_values_and_validity<T: NativeType>(
    values: &[T],
    validity: Option<&Bitmap>,
    mask: &Bitmap,
) -> (Vec<T>, Option<MutableBitmap>) {
    if let Some(validity) = validity {
        let (values, validity) = null_filter(values, validity, mask);
        (values, Some(validity))
    } else {
        (nonnull_filter(values, mask), None)
    }
}

pub(super) fn filter_primitive<T: NativeType + Simd>(
    array: &PrimitiveArray<T>,
    mask: &Bitmap,
) -> PrimitiveArray<T> {
    assert_eq!(array.len(), mask.len());
    let (values, validity) = filter_values_and_validity(array.values(), array.validity(), mask);
    let validity = validity.map(|validity| validity.freeze());
    unsafe {
        PrimitiveArray::<T>::new_unchecked(array.data_type().clone(), values.into(), validity)
    }
}

/// # Safety
/// This assumes that the `mask_chunks` contains a number of set/true items equal
/// to `filter_count`
unsafe fn nonnull_filter_impl<T, I>(values: &[T], mut mask_chunks: I, filter_count: usize) -> Vec<T>
where
    T: NativeType,
    I: BitChunkIterExact<u64>,
{
    let mut chunks = values.chunks_exact(64);
    let mut new = Vec::<T>::with_capacity(filter_count);
    let mut dst = new.as_mut_ptr();

    chunks
        .by_ref()
        .zip(mask_chunks.by_ref())
        .for_each(|(chunk, mask_chunk)| {
            let ones = mask_chunk.count_ones();
            let leading_ones = get_leading_ones(mask_chunk);

            if ones == leading_ones {
                let size = leading_ones as usize;
                unsafe {
                    std::ptr::copy(chunk.as_ptr(), dst, size);
                    dst = dst.add(size);
                }
                return;
            }

            let ones_iter = BitChunkOnes::from_known_count(mask_chunk, ones as usize);
            for pos in ones_iter {
                dst.write(*chunk.get_unchecked(pos));
                dst = dst.add(1);
            }
        });

    chunks
        .remainder()
        .iter()
        .zip(mask_chunks.remainder_iter())
        .for_each(|(value, b)| {
            if b {
                unsafe {
                    dst.write(*value);
                    dst = dst.add(1);
                };
            }
        });

    unsafe { new.set_len(filter_count) };
    new
}

/// # Safety
/// This assumes that the `mask_chunks` contains a number of set/true items equal
/// to `filter_count`
unsafe fn null_filter_impl<T, I>(
    values: &[T],
    validity: &Bitmap,
    mut mask_chunks: I,
    filter_count: usize,
) -> (Vec<T>, MutableBitmap)
where
    T: NativeType,
    I: BitChunkIterExact<u64>,
{
    let mut chunks = values.chunks_exact(64);

    let mut validity_chunks = validity.chunks::<u64>();

    let mut new = Vec::<T>::with_capacity(filter_count);
    let mut dst = new.as_mut_ptr();
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
                    std::ptr::copy(chunk.as_ptr(), dst, size);
                    dst = dst.add(size);

                    // SAFETY: invariant offset + length <= slice.len()
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
                dst.write(*chunk.get_unchecked(pos));
                dst = dst.add(1);
                new_validity.push_unchecked(validity_chunk & (1 << pos) > 0);
            }
        });

    chunks
        .remainder()
        .iter()
        .zip(validity_chunks.remainder_iter())
        .zip(mask_chunks.remainder_iter())
        .for_each(|((value, is_valid), is_selected)| {
            if is_selected {
                unsafe {
                    dst.write(*value);
                    dst = dst.add(1);
                    new_validity.push_unchecked(is_valid);
                };
            }
        });

    unsafe { new.set_len(filter_count) };
    (new, new_validity)
}

fn null_filter<T: NativeType>(
    values: &[T],
    validity: &Bitmap,
    mask: &Bitmap,
) -> (Vec<T>, MutableBitmap) {
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

fn nonnull_filter<T: NativeType>(values: &[T], mask: &Bitmap) -> Vec<T> {
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
