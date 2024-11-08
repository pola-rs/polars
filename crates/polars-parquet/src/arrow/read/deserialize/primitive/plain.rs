use arrow::array::Splitable;
use arrow::bitmap::{Bitmap, MutableBitmap};
use arrow::types::{AlignedBytes, NativeType};

use super::DecoderFunction;
use crate::parquet::error::ParquetResult;
use crate::parquet::types::NativeType as ParquetNativeType;
use crate::read::deserialize::dictionary_encoded::{append_validity, constrain_page_validity};
use crate::read::deserialize::utils::array_chunks::ArrayChunks;
use crate::read::{Filter, ParquetError};

#[allow(clippy::too_many_arguments)]
pub fn decode<P: ParquetNativeType, T: NativeType, D: DecoderFunction<P, T>>(
    values: &[u8],
    is_optional: bool,
    page_validity: Option<&Bitmap>,
    filter: Option<Filter>,
    validity: &mut MutableBitmap,
    intermediate: &mut Vec<P>,
    target: &mut Vec<T>,
    dfn: D,
) -> ParquetResult<()> {
    if cfg!(debug_assertions) && is_optional {
        assert_eq!(target.len(), validity.len());
    }

    if D::CAN_TRANSMUTE {
        let values = ArrayChunks::<'_, T::AlignedBytes>::new(values).ok_or_else(|| {
            ParquetError::oos("Page content does not align with expected element size")
        })?;

        let start_length = target.len();
        decode_aligned_bytes_dispatch(
            values,
            is_optional,
            page_validity,
            filter,
            validity,
            <T::AlignedBytes as AlignedBytes>::cast_vec_ref_mut(target),
        )?;

        if D::NEED_TO_DECODE {
            let to_decode: &mut [P] = bytemuck::cast_slice_mut(&mut target[start_length..]);

            for v in to_decode {
                *v = bytemuck::cast(dfn.decode(*v));
            }
        }
    } else {
        let values = ArrayChunks::<'_, P::AlignedBytes>::new(values).ok_or_else(|| {
            ParquetError::oos("Page content does not align with expected element size")
        })?;

        intermediate.clear();
        decode_aligned_bytes_dispatch(
            values,
            is_optional,
            page_validity,
            filter,
            validity,
            <P::AlignedBytes as AlignedBytes>::cast_vec_ref_mut(intermediate),
        )?;

        target.extend(intermediate.iter().copied().map(|v| dfn.decode(v)));
    }

    if cfg!(debug_assertions) && is_optional {
        assert_eq!(target.len(), validity.len());
    }

    Ok(())
}

#[inline(never)]
pub fn decode_aligned_bytes_dispatch<B: AlignedBytes>(
    values: ArrayChunks<'_, B>,
    is_optional: bool,
    page_validity: Option<&Bitmap>,
    filter: Option<Filter>,
    validity: &mut MutableBitmap,
    target: &mut Vec<B>,
) -> ParquetResult<()> {
    if is_optional {
        append_validity(page_validity, filter.as_ref(), validity, values.len());
    }

    let page_validity = constrain_page_validity(values.len(), page_validity, filter.as_ref());

    match (filter, page_validity) {
        (None, None) => decode_required(values, target),
        (None, Some(page_validity)) => decode_optional(values, page_validity, target),

        (Some(Filter::Range(rng)), None) => {
            decode_required(values.slice(rng.start, rng.len()), target)
        },
        (Some(Filter::Range(rng)), Some(mut page_validity)) => {
            let mut values = values;
            if rng.start > 0 {
                let prevalidity;
                (prevalidity, page_validity) = page_validity.split_at(rng.start);
                page_validity.slice(0, rng.len());
                let values_start = prevalidity.set_bits();
                values = values.slice(values_start, values.len() - values_start);
            }

            decode_optional(values, page_validity, target)
        },

        (Some(Filter::Mask(filter)), None) => decode_masked_required(values, filter, target),
        (Some(Filter::Mask(filter)), Some(page_validity)) => {
            decode_masked_optional(values, page_validity, filter, target)
        },
    }?;

    Ok(())
}

#[inline(never)]
fn decode_required<B: AlignedBytes>(
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

#[inline(never)]
fn decode_optional<B: AlignedBytes>(
    values: ArrayChunks<'_, B>,
    mut validity: Bitmap,
    target: &mut Vec<B>,
) -> ParquetResult<()> {
    target.reserve(validity.len());

    // Handle the leading and trailing zeros. This may allow dispatch to a faster kernel or
    // possibly removes iterations from the lower kernel.
    let num_leading_nulls = validity.take_leading_zeros();
    target.resize(target.len() + num_leading_nulls, B::zeroed());
    let num_trailing_nulls = validity.take_trailing_zeros();

    // Dispatch to a faster kernel if possible.
    let num_values = validity.set_bits();
    if num_values == validity.len() {
        decode_required(values.truncate(validity.len()), target)?;
        target.resize(target.len() + num_trailing_nulls, B::zeroed());
        return Ok(());
    }

    assert!(num_values <= values.len());

    let start_length = target.len();
    let end_length = target.len() + validity.len();
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
                    unsafe { values.get_unchecked(value_offset) }
                } else {
                    B::zeroed()
                };
                unsafe { target_ptr.add(i).write(value) };

                value_offset += (v & 1) as usize;
                v >>= 1;
            }
        } else {
            for i in 0..len {
                let value = unsafe { values.get_unchecked(value_offset) };
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

    let mut num_remaining = validity.len();
    for v in validity_iter.by_ref() {
        if num_remaining < 56 {
            iter(v, num_remaining);
        } else {
            iter(v, 56);
        }
        num_remaining -= 56;
    }

    let (v, vl) = validity_iter.remainder();

    iter(v, vl.min(num_remaining));

    unsafe { target.set_len(end_length) };
    target.resize(target.len() + num_trailing_nulls, B::zeroed());

    Ok(())
}

#[inline(never)]
fn decode_masked_required<B: AlignedBytes>(
    values: ArrayChunks<'_, B>,
    mut mask: Bitmap,
    target: &mut Vec<B>,
) -> ParquetResult<()> {
    // Remove leading or trailing filtered values. This may allow dispatch to a faster kernel or
    // may remove iterations from the slower kernel below.
    let num_leading_filtered = mask.take_leading_zeros();
    mask.take_trailing_zeros();
    let values = values.slice(num_leading_filtered, mask.len());

    // Dispatch to a faster kernel if possible.
    let num_rows = mask.set_bits();
    if num_rows == mask.len() {
        return decode_required(values.truncate(num_rows), target);
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
fn decode_masked_optional<B: AlignedBytes>(
    values: ArrayChunks<'_, B>,
    mut validity: Bitmap,
    mut mask: Bitmap,
    target: &mut Vec<B>,
) -> ParquetResult<()> {
    assert_eq!(validity.len(), mask.len());

    let num_leading_filtered = mask.take_leading_zeros();
    mask.take_trailing_zeros();
    let leading_validity;
    (leading_validity, validity) = validity.split_at(num_leading_filtered);
    validity.slice(0, mask.len());

    let num_rows = mask.set_bits();
    let num_values = validity.set_bits();

    let values = values.slice(leading_validity.set_bits(), num_values);

    // Dispatch to a faster kernel if possible.
    if num_rows == mask.len() {
        return decode_optional(values, validity, target);
    }
    if num_values == validity.len() {
        return decode_masked_required(values, mask, target);
    }

    assert!(num_values <= values.len());

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
                    unsafe { values.get_unchecked(value_offset + num_read) }
                } else {
                    B::zeroed()
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
