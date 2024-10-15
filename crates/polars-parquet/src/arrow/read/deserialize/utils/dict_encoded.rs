use arrow::bitmap::bitmask::BitMask;
use arrow::bitmap::{Bitmap, MutableBitmap};
use arrow::types::NativeType;
use bytemuck::Pod;
use polars_compute::filter::filter_boolean_kernel;

use super::filter_from_range;
use crate::parquet::encoding::hybrid_rle::{HybridRleChunk, HybridRleDecoder};
use crate::parquet::error::ParquetResult;
use crate::read::{Filter, ParquetError};

pub fn decode_dict<T: std::fmt::Debug + NativeType>(
    values: HybridRleDecoder<'_>,
    dict: &[T],
    is_optional: bool,
    page_validity: Option<&Bitmap>,
    filter: Option<Filter>,
    validity: &mut MutableBitmap,
    target: &mut Vec<T>,
) -> ParquetResult<()> {
    // @TODO: It would be really nice to reduce monomorphizations here. All decode kernels only
    // dependent on the alignment and size of `T` so we could make it downcast here to types that
    // are `Pod` and have the same alignment and size.

    decode_dict_dispatch(
        values,
        dict,
        is_optional,
        page_validity,
        filter,
        validity,
        target,
    )
}

#[inline(never)]
fn decode_dict_dispatch<T: std::fmt::Debug + Pod>(
    values: HybridRleDecoder<'_>,
    dict: &[T],
    is_optional: bool,
    page_validity: Option<&Bitmap>,
    filter: Option<Filter>,
    validity: &mut MutableBitmap,
    target: &mut Vec<T>,
) -> ParquetResult<()> {
    if is_optional {
        match (page_validity.as_ref(), filter.as_ref()) {
            (None, None) => validity.extend_constant(values.len(), true),
            (None, Some(f)) => validity.extend_constant(f.num_rows(), true),
            (Some(page_validity), None) => validity.extend_from_bitmap(page_validity),
            (Some(page_validity), Some(Filter::Range(rng))) => {
                let pv = (*page_validity).clone();
                validity.extend_from_bitmap(&pv.sliced(rng.start, rng.len()))
            },
            (Some(page_validity), Some(Filter::Mask(mask))) => {
                validity.extend_from_bitmap(&filter_boolean_kernel(page_validity, &mask))
            },
        }
    }

    match (filter, page_validity) {
        (None, None) => decode_required_dict(values, dict, None, target),
        (Some(Filter::Range(rng)), None) if rng.start == 0 => {
            decode_required_dict(values, dict, Some(rng.end), target)
        },
        (None, Some(page_validity)) => decode_optional_dict(values, dict, &page_validity, target),
        (Some(Filter::Range(rng)), Some(page_validity)) if rng.start == 0 => {
            decode_optional_dict(values, dict, &page_validity, target)
        },
        (Some(Filter::Mask(filter)), None) => {
            decode_masked_required_dict(values, dict, &filter, target)
        },
        (Some(Filter::Mask(filter)), Some(page_validity)) => {
            decode_masked_optional_dict(values, dict, &filter, &page_validity, target)
        },
        (Some(Filter::Range(rng)), None) => {
            decode_masked_required_dict(values, dict, &filter_from_range(rng.clone()), target)
        },
        (Some(Filter::Range(rng)), Some(page_validity)) => decode_masked_optional_dict(
            values,
            dict,
            &filter_from_range(rng.clone()),
            &page_validity,
            target,
        ),
    }
}

#[cold]
#[inline(always)]
fn oob_dict_idx() -> ParquetError {
    ParquetError::oos("Dictionary Index is out-of-bounds")
}

#[inline(always)]
fn verify_dict_indices(indices: &[u32; 32], dict_size: usize) -> ParquetResult<()> {
    let mut is_valid = true;
    for &idx in indices {
        is_valid &= (idx as usize) < dict_size;
    }

    if is_valid {
        return Ok(());
    }

    Err(oob_dict_idx())
}

#[inline(never)]
pub fn decode_required_dict<T: Pod>(
    mut values: HybridRleDecoder<'_>,
    dict: &[T],
    limit: Option<usize>,
    target: &mut Vec<T>,
) -> ParquetResult<()> {
    if dict.is_empty() && values.len() > 0 {
        return Err(oob_dict_idx());
    }

    let mut limit = limit.unwrap_or(values.len());
    assert!(limit <= values.len());
    let start_length = target.len();
    let end_length = start_length + limit;

    target.reserve(limit);
    let mut target_ptr = unsafe { target.as_mut_ptr().add(start_length) };

    while limit > 0 {
        let chunk = values.next_chunk()?.unwrap();

        match chunk {
            HybridRleChunk::Rle(value, length) => {
                let length = length.min(limit);

                let Some(&value) = dict.get(value as usize) else {
                    return Err(oob_dict_idx());
                };
                let target_slice;
                // SAFETY:
                // 1. `target_ptr..target_ptr + limit` is allocated
                // 2. `length <= limit`
                unsafe {
                    target_slice = std::slice::from_raw_parts_mut(target_ptr, length);
                    target_ptr = target_ptr.add(length);
                }

                target_slice.fill(value);
                limit -= length;
            },
            HybridRleChunk::Bitpacked(mut decoder) => {
                let mut chunked = decoder.chunked();
                loop {
                    if limit < 32 {
                        break;
                    }

                    let Some(chunk) = chunked.next() else {
                        break;
                    };

                    verify_dict_indices(&chunk, dict.len())?;

                    for i in 0..32 {
                        let v = unsafe { dict.get_unchecked(chunk[i] as usize) };
                        unsafe { target_ptr.add(i).write(*v) };
                    }

                    unsafe {
                        target_ptr = target_ptr.add(32);
                    }
                    limit -= 32;
                }

                if let Some((chunk, chunk_size)) = chunked.next_inexact() {
                    let chunk_size = chunk_size.min(limit);

                    let highest_idx = chunk[..chunk_size].iter().copied().max().unwrap();
                    assert!((highest_idx as usize) < dict.len());

                    for i in 0..chunk_size {
                        let v = unsafe { dict.get_unchecked(chunk[i] as usize) };
                        unsafe { target_ptr.add(i).write(*v) };
                    }

                    unsafe {
                        target_ptr = target_ptr.add(chunk_size);
                    }

                    limit -= chunk_size;
                }
            },
        }
    }

    unsafe {
        target.set_len(end_length);
    }

    Ok(())
}

#[inline(never)]
pub fn decode_optional_dict<T: Pod>(
    mut values: HybridRleDecoder<'_>,
    dict: &[T],
    validity: &Bitmap,
    target: &mut Vec<T>,
) -> ParquetResult<()> {
    let mut limit = validity.len();
    let num_valid_values = validity.set_bits();

    // Dispatch to the required kernel if all rows are valid anyway.
    if num_valid_values == validity.len() {
        return decode_required_dict(values, dict, Some(validity.len()), target);
    }

    if dict.is_empty() && num_valid_values > 0 {
        return Err(oob_dict_idx());
    }

    assert!(num_valid_values <= values.len());
    let start_length = target.len();
    let end_length = start_length + validity.len();

    target.reserve(validity.len());
    let mut target_ptr = unsafe { target.as_mut_ptr().add(start_length) };

    let mut validity = BitMask::from_bitmap(validity);
    let mut values_buffer = [0u32; 128];
    let values_buffer = &mut values_buffer;

    loop {
        if limit == 0 {
            break;
        }

        let Some(chunk) = values.next_chunk()? else {
            break;
        };

        match chunk {
            HybridRleChunk::Rle(value, size) => {
                // If we know that we have `size` times `value` that we can append, but there might
                // be nulls in between those values.
                //
                // 1. See how many `num_rows = valid + invalid` values `size` would entail. This is
                //    done with `num_bits_before_nth_one` on the validity mask.
                // 2. Fill `num_rows` values into the target buffer.
                // 3. Advance the validity mask by `num_rows` values.

                let num_chunk_rows = validity.nth_set_bit_idx(size, 0).unwrap_or(validity.len());

                (_, validity) = unsafe { validity.split_at_unchecked(num_chunk_rows) };

                let Some(&value) = dict.get(value as usize) else {
                    return Err(oob_dict_idx());
                };
                let target_slice;
                // SAFETY:
                // Given `validity_iter` before the `advance_by_bits`
                //
                // 1. `target_ptr..target_ptr + validity_iter.bits_left()` is allocated
                // 2. `num_chunk_rows <= validity_iter.bits_left()`
                unsafe {
                    target_slice = std::slice::from_raw_parts_mut(target_ptr, num_chunk_rows);
                    target_ptr = target_ptr.add(num_chunk_rows);
                }

                target_slice.fill(value);
                limit -= num_chunk_rows;
            },
            HybridRleChunk::Bitpacked(mut decoder) => {
                let mut chunked = decoder.chunked();

                let mut buffer_part_idx = 0;
                let mut values_offset = 0;
                let mut num_buffered: usize = 0;

                {
                    let mut num_done = 0;
                    let mut validity_iter = validity.fast_iter_u56();

                    'outer: while limit >= 64 {
                        let v = validity_iter.next().unwrap();

                        while num_buffered < v.count_ones() as usize {
                            let buffer_part = <&mut [u32; 32]>::try_from(
                                &mut values_buffer[buffer_part_idx * 32..][..32],
                            )
                            .unwrap();
                            let Some(num_added) = chunked.next_into(buffer_part) else {
                                break 'outer;
                            };

                            verify_dict_indices(&buffer_part, dict.len())?;

                            num_buffered += num_added;

                            buffer_part_idx += 1;
                            buffer_part_idx %= 4;
                        }

                        let mut num_read = 0;

                        for i in 0..56 {
                            let idx = values_buffer[(values_offset + num_read) % 128];

                            // SAFETY:
                            // 1. `values_buffer` starts out as only zeros, which we know is in the
                            //    dictionary following the original `dict.is_empty` check.
                            // 2. Each time we write to `values_buffer`, it is followed by a
                            //    `verify_dict_indices`.
                            let value = unsafe { dict.get_unchecked(idx as usize) };
                            unsafe { target_ptr.add(i).write(*value) };
                            num_read += ((v >> i) & 1) as usize;
                        }

                        values_offset += num_read;
                        values_offset %= 128;
                        num_buffered -= num_read;
                        unsafe {
                            target_ptr = target_ptr.add(56);
                        }
                        num_done += 56;
                        limit -= 56;
                    }

                    (_, validity) = unsafe { validity.split_at_unchecked(num_done) };
                }

                let num_decoder_remaining = num_buffered + chunked.decoder.len();
                let decoder_limit = validity
                    .nth_set_bit_idx(num_decoder_remaining, 0)
                    .unwrap_or(validity.len());

                let num_remaining = limit.min(decoder_limit);
                let current_validity;
                (current_validity, validity) =
                    unsafe { validity.split_at_unchecked(num_remaining) };
                let (v, _) = current_validity.fast_iter_u56().remainder();

                while num_buffered < v.count_ones() as usize {
                    let buffer_part = <&mut [u32; 32]>::try_from(
                        &mut values_buffer[buffer_part_idx * 32..][..32],
                    )
                    .unwrap();
                    let num_added = chunked.next_into(buffer_part).unwrap();

                    verify_dict_indices(&buffer_part, dict.len())?;

                    num_buffered += num_added;

                    buffer_part_idx += 1;
                    buffer_part_idx %= 4;
                }

                let mut num_read = 0;

                for i in 0..num_remaining {
                    let idx = values_buffer[(values_offset + num_read) % 128];
                    let value = unsafe { dict.get_unchecked(idx as usize) }.clone();
                    unsafe { *target_ptr.add(i) = value };
                    num_read += ((v >> i) & 1) as usize;
                }

                unsafe {
                    target_ptr = target_ptr.add(num_remaining);
                }
                limit -= num_remaining;
            },
        }
    }

    if cfg!(debug_assertions) {
        assert_eq!(validity.set_bits(), 0);
    }

    let target_slice;
    unsafe {
        target_slice = std::slice::from_raw_parts_mut(target_ptr, limit);
    }

    target_slice.fill(T::zeroed());

    unsafe {
        target.set_len(end_length);
    }

    Ok(())
}

#[inline(never)]
pub fn decode_masked_optional_dict<T: Pod>(
    mut values: HybridRleDecoder<'_>,
    dict: &[T],
    filter: &Bitmap,
    validity: &Bitmap,
    target: &mut Vec<T>,
) -> ParquetResult<()> {
    let num_rows = filter.set_bits();
    let num_valid_values = validity.set_bits();

    // Dispatch to the non-filter kernel if all rows are needed anyway.
    if num_rows == filter.len() {
        return decode_optional_dict(values, dict, validity, target);
    }

    // Dispatch to the required kernel if all rows are valid anyway.
    if num_valid_values == validity.len() {
        return decode_masked_required_dict(values, dict, filter, target);
    }

    if dict.is_empty() && num_valid_values > 0 {
        return Err(oob_dict_idx());
    }

    debug_assert_eq!(filter.len(), validity.len());
    assert!(num_valid_values <= values.len());
    let start_length = target.len();

    target.reserve(num_rows);
    let mut target_ptr = unsafe { target.as_mut_ptr().add(start_length) };

    let mut filter = BitMask::from_bitmap(filter);
    let mut validity = BitMask::from_bitmap(validity);

    let mut values_buffer = [0u32; 128];
    let values_buffer = &mut values_buffer;

    let mut num_rows_left = num_rows;

    loop {
        // Early stop if we have no more rows to load.
        if num_rows_left == 0 {
            break;
        }

        let Some(chunk) = values.next_chunk()? else {
            break;
        };

        match chunk {
            HybridRleChunk::Rle(value, size) => {
                if value as usize >= dict.len() {
                    return Err(oob_dict_idx());
                }

                // If we know that we have `size` times `value` that we can append, but there might
                // be nulls in between those values.
                //
                // 1. See how many `num_rows = valid + invalid` values `size` would entail. This is
                //    done with `num_bits_before_nth_one` on the validity mask.
                // 2. Fill `num_rows` values into the target buffer.
                // 3. Advance the validity mask by `num_rows` values.

                let num_chunk_values = validity.nth_set_bit_idx(size, 0).unwrap_or(validity.len());

                let current_filter;
                (_, validity) = unsafe { validity.split_at_unchecked(num_chunk_values) };
                (current_filter, filter) = unsafe { validity.split_at_unchecked(num_chunk_values) };

                let num_chunk_rows = current_filter.set_bits();

                if num_chunk_rows > 0 {
                    // SAFETY: Bounds check done before.
                    let value = unsafe { dict.get_unchecked(value as usize) };

                    let target_slice;
                    // SAFETY:
                    // Given `filter_iter` before the `advance_by_bits`.
                    //
                    // 1. `target_ptr..target_ptr + filter_iter.count_ones()` is allocated
                    // 2. `num_chunk_rows < filter_iter.count_ones()`
                    unsafe {
                        target_slice = std::slice::from_raw_parts_mut(target_ptr, num_chunk_rows);
                        target_ptr = target_ptr.add(num_chunk_rows);
                    }

                    target_slice.fill(*value);
                    num_rows_left -= num_chunk_rows;
                }
            },
            HybridRleChunk::Bitpacked(mut decoder) => {
                // For bitpacked we do the following:
                // 1. See how many rows are encoded by this `decoder`.
                // 2. Go through the filter and validity 56 bits at a time and:
                //    0. If filter bits are 0, skip the chunk entirely.
                //    1. Buffer enough values so that we can branchlessly decode with the filter
                //       and validity.
                //    2. Decode with filter and validity.
                // 3. Decode remainder.

                let size = decoder.len();
                let mut chunked = decoder.chunked();

                let num_chunk_values = validity.nth_set_bit_idx(size, 0).unwrap_or(validity.len());

                let mut buffer_part_idx = 0;
                let mut values_offset = 0;
                let mut num_buffered: usize = 0;
                let mut skip_values = 0;

                let current_filter;
                let current_validity;

                (current_filter, filter) = unsafe { filter.split_at_unchecked(num_chunk_values) };
                (current_validity, validity) =
                    unsafe { validity.split_at_unchecked(num_chunk_values) };

                let mut iter = |mut f: u64, mut v: u64| {
                    // Skip chunk if we don't any values from here.
                    if f == 0 {
                        skip_values += v.count_ones() as usize;
                        return ParquetResult::Ok(());
                    }

                    // Skip over already buffered items.
                    let num_buffered_skipped = skip_values.min(num_buffered);
                    values_offset += num_buffered_skipped;
                    num_buffered -= num_buffered_skipped;
                    skip_values -= num_buffered_skipped;

                    // If we skipped plenty already, just skip decoding those chunks instead of
                    // decoding them and throwing them away.
                    chunked.decoder.skip_chunks((skip_values / 32) as usize);
                    // The leftovers we have to decode but we can also just skip.
                    skip_values %= 32;

                    while num_buffered < v.count_ones() as usize {
                        let buffer_part = <&mut [u32; 32]>::try_from(
                            &mut values_buffer[buffer_part_idx * 32..][..32],
                        )
                        .unwrap();
                        let num_added = chunked.next_into(buffer_part).unwrap();

                        verify_dict_indices(&buffer_part, dict.len())?;

                        let skip_chunk_values = (skip_values as usize).min(num_added);

                        values_offset += skip_chunk_values;
                        num_buffered += num_added - skip_chunk_values;
                        skip_values -= skip_chunk_values;

                        buffer_part_idx += 1;
                        buffer_part_idx %= 4;
                    }

                    let mut num_read = 0;
                    let mut num_written = 0;

                    while f != 0 {
                        let offset = f.trailing_zeros();

                        num_read += (v & (1u64 << offset).wrapping_sub(1)).count_ones() as usize;
                        v >>= offset;

                        let idx = values_buffer[(values_offset + num_read) % 128];
                        // SAFETY:
                        // 1. `values_buffer` starts out as only zeros, which we know is in the
                        //    dictionary following the original `dict.is_empty` check.
                        // 2. Each time we write to `values_buffer`, it is followed by a
                        //    `verify_dict_indices`.
                        let value = unsafe { dict.get_unchecked(idx as usize) }.clone();
                        unsafe { target_ptr.add(num_written).write(value) };

                        num_written += 1;
                        num_read += (v & 1) as usize;

                        f >>= offset + 1; // Clear least significant bit.
                        v >>= 1;
                    }

                    num_read += v.count_ones() as usize;

                    values_offset += num_read;
                    values_offset %= 128;
                    num_buffered -= num_read;
                    unsafe {
                        target_ptr = target_ptr.add(num_written);
                    }
                    num_rows_left -= num_written;

                    ParquetResult::Ok(())
                };

                let mut f_iter = current_filter.fast_iter_u56();
                let mut v_iter = current_validity.fast_iter_u56();

                for (f, v) in f_iter.by_ref().zip(v_iter.by_ref()) {
                    iter(f, v)?;
                }

                let (f, fl) = f_iter.remainder();
                let (v, vl) = v_iter.remainder();

                assert_eq!(fl, vl);

                iter(f, v)?;
            },
        }
    }

    if cfg!(debug_assertions) {
        assert_eq!(validity.set_bits(), 0);
    }

    let target_slice;
    unsafe {
        target_slice = std::slice::from_raw_parts_mut(target_ptr, num_rows_left);
    }

    target_slice.fill(T::zeroed());


    unsafe {
        target.set_len(start_length + num_rows);
    }

    Ok(())
}

#[inline(never)]
pub fn decode_masked_required_dict<T: Pod>(
    mut values: HybridRleDecoder<'_>,
    dict: &[T],
    filter: &Bitmap,
    target: &mut Vec<T>,
) -> ParquetResult<()> {
    let num_rows = filter.set_bits();

    // Dispatch to the non-filter kernel if all rows are needed anyway.
    if num_rows == filter.len() {
        return decode_required_dict(values, dict, Some(filter.len()), target);
    }

    if dict.is_empty() && values.len() > 0 {
        return Err(oob_dict_idx());
    }

    let start_length = target.len();

    target.reserve(num_rows);
    let mut target_ptr = unsafe { target.as_mut_ptr().add(start_length) };

    let mut filter = BitMask::from_bitmap(filter);

    let mut values_buffer = [0u32; 128];
    let values_buffer = &mut values_buffer;

    let mut num_rows_left = num_rows;

    loop {
        if num_rows_left == 0 {
            break;
        }

        let Some(chunk) = values.next_chunk()? else {
            break;
        };

        match chunk {
            HybridRleChunk::Rle(value, size) => {
                if value as usize >= dict.len() {
                    return Err(oob_dict_idx());
                }

                let size = size.min(filter.len());

                // If we know that we have `size` times `value` that we can append, but there might
                // be nulls in between those values.
                //
                // 1. See how many `num_rows = valid + invalid` values `size` would entail. This is
                //    done with `num_bits_before_nth_one` on the validity mask.
                // 2. Fill `num_rows` values into the target buffer.
                // 3. Advance the validity mask by `num_rows` values.

                let current_filter;

                (current_filter, filter) = unsafe { filter.split_at_unchecked(size) };
                let num_chunk_rows = current_filter.set_bits();

                if num_chunk_rows > 0 {
                    // SAFETY: Bounds check done before.
                    let value = unsafe { dict.get_unchecked(value as usize) };

                    let target_slice;
                    // SAFETY:
                    // Given `filter_iter` before the `advance_by_bits`.
                    //
                    // 1. `target_ptr..target_ptr + filter_iter.count_ones()` is allocated
                    // 2. `num_chunk_rows < filter_iter.count_ones()`
                    unsafe {
                        target_slice = std::slice::from_raw_parts_mut(target_ptr, num_chunk_rows);
                        target_ptr = target_ptr.add(num_chunk_rows);
                    }

                    target_slice.fill(*value);
                    num_rows_left -= num_chunk_rows;
                }
            },
            HybridRleChunk::Bitpacked(mut decoder) => {
                let size = decoder.len().min(filter.len());
                let mut chunked = decoder.chunked();

                let mut buffer_part_idx = 0;
                let mut values_offset = 0;
                let mut num_buffered: usize = 0;
                let mut skip_values = 0;

                let current_filter;

                (current_filter, filter) = unsafe { filter.split_at_unchecked(size) };

                let mut iter = |mut f: u64, len: usize| {
                    debug_assert!(len <= 64);

                    // Skip chunk if we don't any values from here.
                    if f == 0 {
                        skip_values += len;
                        return ParquetResult::Ok(());
                    }

                    // Skip over already buffered items.
                    let num_buffered_skipped = skip_values.min(num_buffered);
                    values_offset += num_buffered_skipped;
                    num_buffered -= num_buffered_skipped;
                    skip_values -= num_buffered_skipped;

                    // If we skipped plenty already, just skip decoding those chunks instead of
                    // decoding them and throwing them away.
                    chunked.decoder.skip_chunks((skip_values / 32) as usize);
                    // The leftovers we have to decode but we can also just skip.
                    skip_values %= 32;

                    while num_buffered < len {
                        let buffer_part = <&mut [u32; 32]>::try_from(
                            &mut values_buffer[buffer_part_idx * 32..][..32],
                        )
                        .unwrap();
                        let num_added = chunked.next_into(buffer_part).unwrap();

                        verify_dict_indices(&buffer_part, dict.len())?;

                        let skip_chunk_values = (skip_values as usize).min(num_added);

                        values_offset += skip_chunk_values;
                        num_buffered += num_added - skip_chunk_values;
                        skip_values -= skip_chunk_values;

                        buffer_part_idx += 1;
                        buffer_part_idx %= 4;
                    }

                    let mut num_read = 0;
                    let mut num_written = 0;

                    while f != 0 {
                        let offset = f.trailing_zeros() as usize;

                        num_read += offset;

                        let idx = values_buffer[(values_offset + num_read) % 128];
                        // SAFETY:
                        // 1. `values_buffer` starts out as only zeros, which we know is in the
                        //    dictionary following the original `dict.is_empty` check.
                        // 2. Each time we write to `values_buffer`, it is followed by a
                        //    `verify_dict_indices`.
                        let value = *unsafe { dict.get_unchecked(idx as usize) };
                        unsafe { target_ptr.add(num_written).write(value) };

                        num_written += 1;
                        num_read += 1;

                        f >>= offset + 1; // Clear least significant bit.
                    }

                    values_offset += len;
                    values_offset %= 128;
                    num_buffered -= len;
                    unsafe {
                        target_ptr = target_ptr.add(num_written);
                    }
                    num_rows_left -= num_written;

                    ParquetResult::Ok(())
                };

                let mut f_iter = current_filter.fast_iter_u56();

                for f in f_iter.by_ref() {
                    iter(f, 56)?;
                }

                let (f, fl) = f_iter.remainder();

                iter(f, fl)?;
            },
        }
    }

    unsafe {
        target.set_len(start_length + num_rows);
    }

    Ok(())
}
