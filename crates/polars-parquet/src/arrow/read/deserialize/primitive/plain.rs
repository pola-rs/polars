use arrow::bitmap::{Bitmap, MutableBitmap};
use arrow::types::NativeType;
use polars_compute::filter::filter_boolean_kernel;

use super::DecoderFunction;
use crate::parquet::error::ParquetResult;
use crate::parquet::types::NativeType as ParquetNativeType;
use crate::read::deserialize::utils::array_chunks::ArrayChunks;
use crate::read::deserialize::utils::filter_from_range;
use crate::read::Filter;

pub fn decode<P: ParquetNativeType, T: NativeType, D: DecoderFunction<P, T>>(
    values: ArrayChunks<'_, P>,
    is_optional: bool,
    page_validity: Option<&Bitmap>,
    filter: Option<Filter>,
    validity: &mut MutableBitmap,
    target: &mut Vec<T>,
    dfn: D,
) -> ParquetResult<()> {
    // @TODO: It would be really nice to reduce monomorphizations here. All decode kernels only
    // dependent on the alignment and size of `T` so we could make it downcast here to types that
    // are `Pod` and have the same alignment and size.

    decode_plain_dispatch(
        values,
        is_optional,
        page_validity,
        filter,
        validity,
        target,
        dfn,
    )
}

#[inline(never)]
fn decode_plain_dispatch<P: ParquetNativeType, T: NativeType, D: DecoderFunction<P, T>>(
    values: ArrayChunks<'_, P>,
    is_optional: bool,
    page_validity: Option<&Bitmap>,
    filter: Option<Filter>,
    validity: &mut MutableBitmap,
    target: &mut Vec<T>,
    dfn: D,
) -> ParquetResult<()> {
    if is_optional {
        match (page_validity, filter.as_ref()) {
            (None, None) => validity.extend_constant(values.len(), true),
            (None, Some(f)) => validity.extend_constant(f.num_rows(), true),
            (Some(page_validity), None) => validity.extend_from_bitmap(page_validity),
            (Some(page_validity), Some(Filter::Range(rng))) => {
                let page_validity = page_validity.clone();
                validity.extend_from_bitmap(&page_validity.clone().sliced(rng.start, rng.len()))
            },
            (Some(page_validity), Some(Filter::Mask(mask))) => {
                validity.extend_from_bitmap(&filter_boolean_kernel(page_validity, mask))
            },
        }
    }

    let num_unfiltered_rows = match (filter.as_ref(), page_validity) {
        (None, _) => values.len(),
        (Some(f), v) => {
            if cfg!(debug_assertions) {
                if let Some(v) = v {
                    assert!(v.len() >= f.max_offset());
                }
            }

            f.max_offset()
        },
    };

    let page_validity = page_validity.map(|pv| {
        if pv.len() > num_unfiltered_rows {
            pv.clone().sliced(0, num_unfiltered_rows)
        } else {
            pv.clone()
        }
    });

    match (filter, page_validity) {
        (None, None) => decode_required(values, None, target, dfn),
        (Some(Filter::Range(rng)), None) if rng.start == 0 => {
            decode_required(values, Some(rng.end), target, dfn)
        },
        (None, Some(page_validity)) => decode_optional(values, &page_validity, target, dfn),
        (Some(Filter::Range(rng)), Some(page_validity)) if rng.start == 0 => {
            decode_optional(values, &page_validity, target, dfn)
        },
        (Some(Filter::Mask(filter)), None) => decode_masked_required(values, &filter, target, dfn),
        (Some(Filter::Mask(filter)), Some(page_validity)) => {
            decode_masked_optional(values, &page_validity, &filter, target, dfn)
        },
        (Some(Filter::Range(rng)), None) => {
            decode_masked_required(values, &filter_from_range(rng.clone()), target, dfn)
        },
        (Some(Filter::Range(rng)), Some(page_validity)) => decode_masked_optional(
            values,
            &page_validity,
            &filter_from_range(rng.clone()),
            target,
            dfn,
        ),
    }
}

#[inline(never)]
fn decode_required<P: ParquetNativeType, T: NativeType, D: DecoderFunction<P, T>>(
    values: ArrayChunks<'_, P>,
    limit: Option<usize>,
    target: &mut Vec<T>,
    dfn: D,
) -> ParquetResult<()> {
    let limit = limit.unwrap_or(values.len());
    assert!(limit <= values.len());

    target.extend((0..limit).map(|i| {
        let v = unsafe { values.get_unchecked(i) };
        dfn.decode(v)
    }));

    Ok(())
}

#[inline(never)]
fn decode_optional<P: ParquetNativeType, T: NativeType, D: DecoderFunction<P, T>>(
    values: ArrayChunks<'_, P>,
    validity: &Bitmap,
    target: &mut Vec<T>,
    dfn: D,
) -> ParquetResult<()> {
    let num_values = validity.set_bits();

    if num_values == validity.len() {
        return decode_required(values, Some(validity.len()), target, dfn);
    }

    let mut limit = validity.len();

    assert!(num_values <= values.len());

    let start_length = target.len();
    let end_length = target.len() + limit;
    target.reserve(limit);
    let mut target_ptr = unsafe { target.as_mut_ptr().add(start_length) };

    let mut validity_iter = validity.fast_iter_u56();
    let mut num_values_remaining = num_values;
    let mut value_offset = 0;

    let mut iter = |mut v: u64, len: usize| {
        debug_assert!(len < 64);

        let num_chunk_values = v.count_ones() as usize;

        if num_values_remaining == num_chunk_values {
            for i in 0..len {
                let is_valid = v & 1 != 0;
                let value = if is_valid {
                    let value = unsafe { values.get_unchecked(value_offset) };
                    dfn.decode(value)
                } else {
                    T::zeroed()
                };
                unsafe { target_ptr.add(i).write(value) };

                value_offset += (v & 1) as usize;
                v >>= 1;
            }
        } else {
            for i in 0..len {
                let value = unsafe { values.get_unchecked(value_offset) };
                let value = dfn.decode(value);
                unsafe { target_ptr.add(i).write(value) };

                value_offset += (v & 1) as usize;
                v >>= 1;
            }
        }

        num_values_remaining -= num_chunk_values;
        unsafe {
            target_ptr = target_ptr.add(len);
        }
    };

    for v in validity_iter.by_ref() {
        if limit < 56 {
            iter(v, limit);
        } else {
            iter(v, 56);
        }
        limit -= 56;
    }

    let (v, vl) = validity_iter.remainder();

    iter(v, vl.min(limit));

    unsafe { target.set_len(end_length) };

    Ok(())
}

#[inline(never)]
fn decode_masked_required<P: ParquetNativeType, T: NativeType, D: DecoderFunction<P, T>>(
    values: ArrayChunks<'_, P>,
    mask: &Bitmap,
    target: &mut Vec<T>,
    dfn: D,
) -> ParquetResult<()> {
    let num_rows = mask.set_bits();

    if num_rows == mask.len() {
        return decode_required(values, Some(num_rows), target, dfn);
    }

    assert!(mask.len() <= values.len());

    let start_length = target.len();
    target.reserve(num_rows);
    let mut target_ptr = unsafe { target.as_mut_ptr().add(start_length) };

    let mut mask_iter = mask.fast_iter_u56();
    let mut num_rows_left = num_rows;
    let mut value_offset = 0;

    let mut iter = |mut f: u64, len: usize| {
        if num_rows_left == 0 {
            return false;
        }

        let mut num_read = 0;
        let mut num_written = 0;

        while f != 0 {
            let offset = f.trailing_zeros() as usize;

            num_read += offset;

            // SAFETY:
            // 1. `values_buffer` starts out as only zeros, which we know is in the
            //    dictionary following the original `dict.is_empty` check.
            // 2. Each time we write to `values_buffer`, it is followed by a
            //    `verify_dict_indices`.
            let value = unsafe { values.get_unchecked(value_offset + num_read) };
            let value = dfn.decode(value);
            unsafe { target_ptr.add(num_written).write(value) };

            num_written += 1;
            num_read += 1;

            f >>= offset + 1; // Clear least significant bit.
        }

        unsafe {
            target_ptr = target_ptr.add(num_written);
        }
        value_offset += len;
        num_rows_left -= num_written;

        true
    };

    for f in mask_iter.by_ref() {
        if !iter(f, 56) {
            break;
        }
    }

    let (f, fl) = mask_iter.remainder();

    iter(f, fl);

    unsafe { target.set_len(start_length + num_rows) };

    Ok(())
}

#[inline(never)]
fn decode_masked_optional<P: ParquetNativeType, T: NativeType, D: DecoderFunction<P, T>>(
    values: ArrayChunks<'_, P>,
    validity: &Bitmap,
    mask: &Bitmap,
    target: &mut Vec<T>,
    dfn: D,
) -> ParquetResult<()> {
    let num_rows = mask.set_bits();
    let num_values = validity.set_bits();

    if num_rows == mask.len() {
        return decode_optional(values, validity, target, dfn);
    }

    if num_values == validity.len() {
        return decode_masked_required(values, mask, target, dfn);
    }

    assert!(mask.len() <= values.len());

    let start_length = target.len();
    target.reserve(num_rows);
    let mut target_ptr = unsafe { target.as_mut_ptr().add(start_length) };

    let mut validity_iter = validity.fast_iter_u56();
    let mut mask_iter = mask.fast_iter_u56();
    let mut num_values_left = num_values;
    let mut num_rows_left = num_rows;
    let mut value_offset = 0;

    let mut iter = |mut f: u64, mut v: u64, len: usize| {
        if num_rows_left == 0 {
            return false;
        }

        let num_chunk_values = v.count_ones() as usize;

        let mut num_read = 0;
        let mut num_written = 0;

        if num_chunk_values == num_values_left {
            while f != 0 {
                let offset = f.trailing_zeros() as usize;

                num_read += (v & (1u64 << offset).wrapping_sub(1)).count_ones() as usize;
                v >>= offset;

                let is_valid = v & 1 != 0;
                let value = if is_valid {
                    let value = unsafe { values.get_unchecked(value_offset + num_read) };
                    dfn.decode(value)
                } else {
                    T::zeroed()
                };
                unsafe { target_ptr.add(num_written).write(value) };

                num_written += 1;
                num_read += (v & 1) as usize;

                f >>= offset + 1; // Clear least significant bit.
                v >>= 1;
            }
        } else {
            while f != 0 {
                let offset = f.trailing_zeros() as usize;

                num_read += (v & (1u64 << offset).wrapping_sub(1)).count_ones() as usize;
                v >>= offset;

                let value = unsafe { values.get_unchecked(value_offset + num_read) };
                let value = dfn.decode(value);
                unsafe { target_ptr.add(num_written).write(value) };

                num_written += 1;
                num_read += (v & 1) as usize;

                f >>= offset + 1; // Clear least significant bit.
                v >>= 1;
            }
        }

        unsafe {
            target_ptr = target_ptr.add(num_written);
        }
        value_offset += len;
        num_rows_left -= num_written;
        num_values_left -= num_chunk_values;

        true
    };

    for (f, v) in mask_iter.by_ref().zip(validity_iter.by_ref()) {
        if !iter(f, v, 56) {
            break;
        }
    }

    let (f, fl) = mask_iter.remainder();
    let (v, vl) = validity_iter.remainder();

    assert_eq!(fl, vl);

    iter(f, v, fl);

    unsafe { target.set_len(start_length + num_rows) };

    Ok(())
}
