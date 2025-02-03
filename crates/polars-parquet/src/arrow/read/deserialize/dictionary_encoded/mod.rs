use arrow::bitmap::bitmask::BitMask;
use arrow::bitmap::{Bitmap, BitmapBuilder};
use arrow::types::{AlignedBytes, Bytes4Alignment4, NativeType};
use polars_compute::filter::filter_boolean_kernel;

use super::ParquetError;
use crate::parquet::encoding::hybrid_rle::HybridRleDecoder;
use crate::parquet::error::ParquetResult;
use crate::read::Filter;

mod optional;
mod optional_masked_dense;
mod predicate;
mod required;
mod required_masked_dense;

/// A mapping from a `u32` to a value. This is used in to map dictionary encoding to a value.
pub trait IndexMapping {
    type Output: Copy;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    fn len(&self) -> usize;
    fn get(&self, idx: u32) -> Option<Self::Output> {
        ((idx as usize) < self.len()).then(|| unsafe { self.get_unchecked(idx) })
    }
    unsafe fn get_unchecked(&self, idx: u32) -> Self::Output;
}

// Base mapping used for everything except the CategoricalDecoder.
impl<T: Copy> IndexMapping for &[T] {
    type Output = T;

    #[inline(always)]
    fn len(&self) -> usize {
        <[T]>::len(self)
    }
    #[inline(always)]
    unsafe fn get_unchecked(&self, idx: u32) -> Self::Output {
        *unsafe { <[T]>::get_unchecked(self, idx as usize) }
    }
}

// Unit mapping used in the CategoricalDecoder.
impl IndexMapping for usize {
    type Output = Bytes4Alignment4;

    #[inline(always)]
    fn len(&self) -> usize {
        *self
    }
    #[inline(always)]
    unsafe fn get_unchecked(&self, idx: u32) -> Self::Output {
        bytemuck::must_cast(idx)
    }
}

#[allow(clippy::too_many_arguments)]
pub fn decode_dict<T: NativeType>(
    values: HybridRleDecoder<'_>,
    dict: &[T],
    dict_mask: Option<&Bitmap>,
    is_optional: bool,
    page_validity: Option<&Bitmap>,
    filter: Option<Filter>,
    validity: &mut BitmapBuilder,
    target: &mut Vec<T>,
    pred_true_mask: &mut BitmapBuilder,
) -> ParquetResult<()> {
    decode_dict_dispatch(
        values,
        bytemuck::cast_slice(dict),
        dict_mask,
        is_optional,
        page_validity,
        filter,
        validity,
        <T::AlignedBytes as AlignedBytes>::cast_vec_ref_mut(target),
        pred_true_mask,
    )
}

#[inline(never)]
#[allow(clippy::too_many_arguments)]
pub fn decode_dict_dispatch<B: AlignedBytes, D: IndexMapping<Output = B>>(
    mut values: HybridRleDecoder<'_>,
    dict: D,
    dict_mask: Option<&Bitmap>,
    is_optional: bool,
    page_validity: Option<&Bitmap>,
    filter: Option<Filter>,
    validity: &mut BitmapBuilder,
    target: &mut Vec<B>,
    pred_true_mask: &mut BitmapBuilder,
) -> ParquetResult<()> {
    if is_optional {
        append_validity(page_validity, filter.as_ref(), validity, values.len());
    }

    let page_validity = constrain_page_validity(values.len(), page_validity, filter.as_ref());

    match (filter, page_validity) {
        (None, None) => required::decode(values, dict, target, 0),
        (Some(Filter::Range(rng)), None) => {
            values.limit_to(rng.end);
            required::decode(values, dict, target, rng.start)
        },
        (None, Some(page_validity)) => optional::decode(values, dict, page_validity, target, 0),
        (Some(Filter::Range(rng)), Some(page_validity)) => {
            optional::decode(values, dict, page_validity, target, rng.start)
        },
        (Some(Filter::Mask(filter)), None) => {
            required_masked_dense::decode(values, dict, filter, target)
        },
        (Some(Filter::Mask(filter)), Some(page_validity)) => {
            optional_masked_dense::decode(values, dict, filter, page_validity, target)
        },
        (Some(Filter::Predicate(p)), None) => {
            predicate::decode(values, dict, dict_mask.unwrap(), &p, target, pred_true_mask)
        },
        (Some(Filter::Predicate(_)), Some(_)) => todo!(),
    }?;

    Ok(())
}

pub(crate) fn append_validity(
    page_validity: Option<&Bitmap>,
    filter: Option<&Filter>,
    validity: &mut BitmapBuilder,
    values_len: usize,
) {
    match (page_validity, filter) {
        (None, None) => validity.extend_constant(values_len, true),
        (None, Some(f)) => validity.extend_constant(f.num_rows(values_len), true),
        (Some(page_validity), None) => validity.extend_from_bitmap(page_validity),
        (Some(page_validity), Some(Filter::Range(rng))) => {
            let page_validity = page_validity.clone();
            validity.extend_from_bitmap(&page_validity.clone().sliced(rng.start, rng.len()))
        },
        (Some(page_validity), Some(Filter::Mask(mask))) => {
            validity.extend_from_bitmap(&filter_boolean_kernel(page_validity, mask))
        },
        (_, Some(Filter::Predicate(_))) => todo!(),
    }
}

pub(crate) fn constrain_page_validity(
    values_len: usize,
    page_validity: Option<&Bitmap>,
    filter: Option<&Filter>,
) -> Option<Bitmap> {
    let num_unfiltered_rows = match (filter.as_ref(), page_validity) {
        (None, None) => values_len,
        (None, Some(pv)) => pv.len(),
        (Some(f), Some(pv)) => {
            debug_assert!(pv.len() >= f.max_offset(pv.len()));
            f.max_offset(pv.len())
        },
        (Some(f), None) => f.max_offset(values_len),
    };

    page_validity.map(|pv| {
        if pv.len() > num_unfiltered_rows {
            pv.clone().sliced(0, num_unfiltered_rows)
        } else {
            pv.clone()
        }
    })
}

#[cold]
fn oob_dict_idx() -> ParquetError {
    ParquetError::oos("Dictionary Index is out-of-bounds")
}

#[cold]
fn no_more_bitpacked_values() -> ParquetError {
    ParquetError::oos("Bitpacked Hybrid-RLE ran out before all values were served")
}

#[inline(always)]
fn verify_dict_indices(indices: &[u32], dict_size: usize) -> ParquetResult<()> {
    debug_assert!(dict_size <= u32::MAX as usize);
    let dict_size = dict_size as u32;

    let mut is_valid = true;
    for &idx in indices {
        is_valid &= idx < dict_size;
    }

    if is_valid {
        Ok(())
    } else {
        Err(oob_dict_idx())
    }
}

/// Skip over entire chunks in a [`HybridRleDecoder`] as long as all skipped chunks do not include
/// more than `num_values_to_skip` values.
#[inline(always)]
fn required_skip_whole_chunks(
    values: &mut HybridRleDecoder<'_>,
    num_values_to_skip: &mut usize,
) -> ParquetResult<()> {
    if *num_values_to_skip == 0 {
        return Ok(());
    }

    loop {
        let mut values_clone = values.clone();
        let Some(chunk_len) = values_clone.next_chunk_length()? else {
            break;
        };
        if *num_values_to_skip < chunk_len {
            break;
        }
        *values = values_clone;
        *num_values_to_skip -= chunk_len;
    }

    Ok(())
}

/// Skip over entire chunks in a [`HybridRleDecoder`] as long as all skipped chunks do not include
/// more than `num_values_to_skip` values.
#[inline(always)]
fn optional_skip_whole_chunks(
    values: &mut HybridRleDecoder<'_>,
    validity: &mut BitMask<'_>,
    num_rows_to_skip: &mut usize,
    num_values_to_skip: &mut usize,
) -> ParquetResult<()> {
    if *num_values_to_skip == 0 {
        return Ok(());
    }

    let mut total_num_skipped_values = 0;

    loop {
        let mut values_clone = values.clone();
        let Some(chunk_len) = values_clone.next_chunk_length()? else {
            break;
        };
        if *num_values_to_skip < chunk_len {
            break;
        }
        *values = values_clone;
        *num_values_to_skip -= chunk_len;
        total_num_skipped_values += chunk_len;
    }

    if total_num_skipped_values > 0 {
        let offset = validity
            .nth_set_bit_idx(total_num_skipped_values - 1, 0)
            .map_or(validity.len(), |v| v + 1);
        *num_rows_to_skip -= offset;
        validity.advance_by(offset);
    }

    Ok(())
}
