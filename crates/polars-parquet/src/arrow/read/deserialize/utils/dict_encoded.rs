use std::ops::Range;

use arrow::bitmap::{Bitmap, MutableBitmap};
use bytemuck::Zeroable;
use polars_compute::filter::filter_boolean_kernel;

use crate::parquet::encoding::hybrid_rle::gatherer::HybridRleGatherer;
use crate::parquet::encoding::hybrid_rle::{HybridRleChunk, HybridRleDecoder};
use crate::parquet::error::ParquetResult;
use crate::read::Filter;

fn filter_from_range(rng: Range<usize>) -> Bitmap {
    let mut bm = MutableBitmap::with_capacity(rng.end);

    bm.extend_constant(rng.start, false);
    bm.extend_constant(rng.len(), true);

    bm.freeze()
}

fn decode_page_validity(
    mut page_validity: HybridRleDecoder<'_>,
    limit: Option<usize>,
) -> ParquetResult<Bitmap> {
    struct BitmapGatherer;

    impl HybridRleGatherer<bool> for BitmapGatherer {
        type Target = MutableBitmap;

        fn target_reserve(&self, target: &mut Self::Target, n: usize) {
            target.reserve(n);
        }

        fn target_num_elements(&self, target: &Self::Target) -> usize {
            target.len()
        }

        fn hybridrle_to_target(&self, value: u32) -> ParquetResult<bool> {
            Ok(value != 0)
        }

        fn gather_one(&self, target: &mut Self::Target, value: bool) -> ParquetResult<()> {
            target.push(value);
            Ok(())
        }

        fn gather_repeated(
            &self,
            target: &mut Self::Target,
            value: bool,
            n: usize,
        ) -> ParquetResult<()> {
            target.extend_constant(n, value);
            Ok(())
        }
    }

    let mut bm = MutableBitmap::with_capacity(limit.unwrap_or(page_validity.len()));

    let gatherer = BitmapGatherer;

    match limit {
        None => page_validity.gather_into(&mut bm, &gatherer)?,
        Some(limit) => page_validity.gather_n_into(&mut bm, limit, &gatherer)?,
    }

    ParquetResult::Ok(bm.freeze())
}

pub fn decode_dict<T: Clone + Zeroable>(
    values: HybridRleDecoder<'_>,
    dict: &[T],
    is_optional: bool,
    page_validity: Option<HybridRleDecoder<'_>>,
    filter: Option<Filter>,
    validity: &mut MutableBitmap,
    target: &mut Vec<T>,
) -> ParquetResult<()> {
    let page_validity = match (page_validity, filter.as_ref()) {
        (None, _) => None,
        (Some(page_validity), None) => Some(decode_page_validity(page_validity, None)?),
        (Some(page_validity), Some(filter)) => Some(decode_page_validity(
            page_validity,
            Some(filter.max_offset()),
        )?),
    };

    if is_optional {
        match (page_validity.as_ref(), filter.as_ref()) {
            (None, None) => validity.extend_constant(values.len(), true),
            (None, Some(f)) => validity.extend_constant(f.num_rows(), true),
            (Some(page_validity), None) => validity.extend_from_bitmap(page_validity),
            (Some(page_validity), Some(Filter::Range(rng))) => {
                validity.extend_from_bitmap(&page_validity.clone().sliced(rng.start, rng.len()))
            },
            (Some(page_validity), Some(Filter::Mask(mask))) => {
                validity.extend_from_bitmap(&filter_boolean_kernel(page_validity, &mask))
            },
        }
    }

    match (filter, page_validity) {
        (None, None) => decode_non_optional_dict(values, dict, None, target),
        (Some(Filter::Range(rng)), None) if rng.start == 0 => {
            decode_non_optional_dict(values, dict, Some(rng.end), target)
        },
        (None, Some(page_validity)) => decode_optional_dict(values, dict, &page_validity, target),
        (Some(Filter::Range(rng)), Some(page_validity)) if rng.start == 0 => {
            decode_optional_dict(values, dict, &page_validity, target)
        },
        (Some(Filter::Mask(filter)), None) => decode_masked_dict(
            values,
            dict,
            &filter,
            &Bitmap::new_with_value(true, filter.len()),
            target,
        ),
        (Some(Filter::Mask(filter)), Some(page_validity)) => {
            decode_masked_dict(values, dict, &filter, &page_validity, target)
        },
        (Some(Filter::Range(rng)), None) => decode_masked_dict(
            values,
            dict,
            &filter_from_range(rng.clone()),
            &Bitmap::new_with_value(true, rng.end),
            target,
        ),
        (Some(Filter::Range(rng)), Some(page_validity)) => decode_masked_dict(
            values,
            dict,
            &filter_from_range(rng.clone()),
            &page_validity,
            target,
        ),
    }
}

fn decode_non_optional_dict<T: Clone + Zeroable>(
    values: HybridRleDecoder<'_>,
    dict: &[T],
    limit: Option<usize>,
    target: &mut Vec<T>,
) -> ParquetResult<()> {
    let limit = limit.unwrap_or(values.len());
    assert!(limit <= values.len());
    let start_length = target.len();
    let end_length = start_length + limit;

    target.reserve(limit);
    let mut target_ptr = unsafe { target.as_mut_ptr().add(start_length) };

    let mut limit = limit;
    let mut values = values;

    while limit > 0 {
        let chunk = values.next_chunk()?.unwrap();

        match chunk {
            HybridRleChunk::Rle(value, length) => {
                let length = length.min(limit);

                let v = dict.get(value as usize).unwrap();
                let target_slice = unsafe { std::slice::from_raw_parts_mut(target_ptr, length) };
                target_slice.fill(v.clone());

                unsafe {
                    target_ptr = target_ptr.add(length);
                }
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

                    let highest_idx = chunk.iter().copied().max().unwrap();
                    assert!((highest_idx as usize) < dict.len());

                    for i in 0..32 {
                        let v = unsafe { dict.get_unchecked(chunk[i] as usize) }.clone();
                        unsafe { target_ptr.add(i).write(v) };
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
                        let v = unsafe { dict.get_unchecked(chunk[i] as usize) }.clone();
                        unsafe { target_ptr.add(i).write(v) };
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

fn decode_optional_dict<T: Clone + Zeroable>(
    values: HybridRleDecoder<'_>,
    dict: &[T],
    validity: &Bitmap,
    target: &mut Vec<T>,
) -> ParquetResult<()> {
    let mut limit = validity.len();
    let num_valid_values = validity.set_bits();

    assert!(num_valid_values <= values.len());
    let start_length = target.len();

    target.reserve(validity.len());
    let mut target_ptr = unsafe { target.as_mut_ptr().add(start_length) };

    let mut chunk_iter = values.into_chunk_iter();
    let mut validity_iter = validity.fast_iter_u56();
    let values_buffer = [0u32; 128];
    let mut values_offset = 0;
    let mut num_buffered: usize = 0;
    let mut rridx = 0;

    loop {
        if limit == 0 {
            break;
        }

        let Some(chunk) = chunk_iter.next() else {
            break;
        };

        let chunk = chunk?;

        match chunk {
            HybridRleChunk::Rle(value, size) => {
                let size = size;

                let num_rows = validity_iter.num_bits_until_n_ones(size);
                let num_rows = num_rows.min(limit);

                let value = unsafe { dict.get_unchecked(value as usize) }.clone();
                let target_slice = unsafe { std::slice::from_raw_parts_mut(target_ptr, num_rows) };
                target_slice.fill(value);

                unsafe {
                    target_ptr = target_ptr.add(num_rows);
                }
                validity_iter.advance_by_bits(num_rows);
                limit -= num_rows;
            },
            HybridRleChunk::Bitpacked(mut decoder) => {
                let mut chunked = decoder.chunked();

                'outer: while limit > 56 {
                    let v = validity_iter.next().unwrap();

                    while num_buffered < v.count_ones() as usize {
                        rridx %= 4;

                        let buffer_part =
                            &mut values_buffer[rridx * 32..][..32].try_into().unwrap();
                        let Some(num_added) = chunked.next_into(buffer_part) else {
                            break 'outer;
                        };

                        let highest_idx = buffer_part.iter().copied().max().unwrap();
                        assert!((highest_idx as usize) < dict.len());

                        num_buffered += num_added;
                        rridx += 1;
                    }

                    let mut num_read = 0;

                    for i in 0..56 {
                        let idx = values_buffer[(values_offset + num_read) % 128];
                        let value = unsafe { dict.get_unchecked(idx as usize) }.clone();
                        unsafe { *target_ptr.add(i) = value };
                        num_read += ((v >> i) & 1) as usize;
                    }

                    values_offset += num_read;
                    values_offset %= 128;
                    num_buffered -= num_read;
                    unsafe {
                        target_ptr = target_ptr.add(56);
                    }
                    limit -= 56;
                }

                let num_remaining = limit.min(num_buffered + chunked.decoder.len());
                debug_assert!(num_remaining < 56);
                let (v, _) = validity_iter.limit_to_bits(num_remaining).remainder();
                validity_iter.advance_by_bits(num_remaining);

                while num_buffered < v.count_ones() as usize {
                    rridx %= 4;

                    let buffer_part = &mut values_buffer[rridx * 32..][..32].try_into().unwrap();
                    let num_added = chunked.next_into(buffer_part).unwrap();

                    let highest_idx = buffer_part.iter().copied().max().unwrap();
                    assert!((highest_idx as usize) < dict.len());

                    num_buffered += num_added;
                    rridx += 1;
                }

                let mut num_read = 0;

                for i in 0..num_remaining {
                    let idx = values_buffer[(values_offset + num_read) % 128];
                    let value = unsafe { dict.get_unchecked(idx as usize) }.clone();
                    unsafe { *target_ptr.add(i) = value };
                    num_read += ((v >> i) & 1) as usize;
                }

                values_offset += num_read;
                values_offset %= 128;
                num_buffered -= num_read;
                unsafe {
                    target_ptr = target_ptr.add(num_remaining);
                }
                limit -= num_remaining;

                debug_assert!(limit == 0 || num_buffered == 0);
            },
        }
    }

    unsafe {
        target.set_len(start_length + validity.len());
    }

    Ok(())
}

fn decode_masked_dict<T: Clone + Zeroable>(
    values: HybridRleDecoder<'_>,
    dict: &[T],
    filter: &Bitmap,
    validity: &Bitmap,
    target: &mut Vec<T>,
) -> ParquetResult<()> {
    let start_length = target.len();
    let num_rows = filter.set_bits();
    let num_valid_values = validity.set_bits();

    if dict.is_empty() {
        assert_eq!(num_valid_values, 0);
        target.resize(start_length + num_rows, T::zeroed());
    }

    assert_eq!(filter.len(), validity.len());
    assert!(values.len() >= num_valid_values);

    let mut filter_iter = filter.fast_iter_u56();
    let mut validity_iter = validity.fast_iter_u56();

    let mut values = values;
    let mut values_offset = 0;
    // @NOTE:
    // Since we asserted before that dict is not empty and every time we copy values into this
    // buffer we verify the indexes. These will now always be valid offsets into the dict array.
    let mut values_buffer = [0u32; 64];

    target.reserve(num_rows);
    let mut target_ptr = unsafe { target.as_mut_ptr().add(start_length) };

    let mut num_buffered = 0;
    let mut buffer_side = false;

    let mut intermediate_buffer = Vec::with_capacity(32);

    let mut iter_u32 = |f: u32, v: u32| {
        while num_buffered < v.count_ones() as usize {
            intermediate_buffer.clear();
            let num_added_values = values.len().min(32);
            values.collect_n_into(&mut intermediate_buffer, num_added_values)?;

            // @TODO: Make this into a chunk by chunk operation.
            // let (added_buffer, num_added_values) = values.next_chunk().unwrap();

            values_buffer[usize::from(buffer_side) * 32..][..num_added_values]
                .copy_from_slice(&intermediate_buffer[..num_added_values]);

            let highest_idx = (0..num_added_values)
                .map(|i| values_buffer[(values_offset + i) % 64])
                .max()
                .unwrap();
            assert!((highest_idx as usize) < dict.len());

            buffer_side = !buffer_side;
            num_buffered += num_added_values;
        }

        // @NOTE: We have to cast to u64 here to avoid the `shr` overflow on the filter.
        let mut f = f as u64;
        let mut v = v as u64;
        let mut num_read = 0;
        let mut num_written = 0;

        while f != 0 {
            let offset = f.trailing_zeros();

            num_read += (v & (1u64 << offset).wrapping_sub(1)).count_ones() as usize;
            v >>= offset;

            let idx = values_buffer[(values_offset + num_read) % 64];
            let value = unsafe { dict.get_unchecked(idx as usize) }.clone();
            unsafe { *target_ptr.add(num_written) = value };

            num_written += 1;
            num_read += (v & 1) as usize;

            f >>= offset + 1;
            v >>= 1;
        }

        num_read += v.count_ones() as usize;

        values_offset = (values_offset + num_read) % 64;
        num_buffered -= num_read;
        target_ptr = unsafe { target_ptr.add(num_written) };

        ParquetResult::Ok(())
    };
    let mut iter = |f: u64, v: u64| {
        iter_u32((f & 0xFFFF_FFFF) as u32, (v & 0xFFFF_FFFF) as u32)?;
        iter_u32((f >> 32) as u32, (v >> 32) as u32)?;

        ParquetResult::Ok(())
    };

    for (f, v) in filter_iter.by_ref().zip(validity_iter.by_ref()) {
        iter(f, v)?;
    }

    let (f, fl) = filter_iter.remainder();
    let (v, vl) = validity_iter.remainder();

    assert_eq!(fl, vl);

    iter(f, v)?;

    unsafe { target.set_len(start_length + num_rows) };
    Ok(())
}
