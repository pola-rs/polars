use arrow::bitmap::{Bitmap, MutableBitmap};
use arrow::types::{AlignedBytes, NativeType};
use polars_compute::filter::filter_boolean_kernel;

use super::ParquetError;
use crate::parquet::encoding::hybrid_rle::HybridRleDecoder;
use crate::parquet::error::ParquetResult;
use crate::read::Filter;

mod optional;
mod optional_masked_dense;
mod required;
mod required_masked_dense;

pub fn decode_dict<T: NativeType>(
    values: HybridRleDecoder<'_>,
    dict: &[T],
    is_optional: bool,
    page_validity: Option<&Bitmap>,
    filter: Option<Filter>,
    validity: &mut MutableBitmap,
    target: &mut Vec<T>,
) -> ParquetResult<()> {
    decode_dict_dispatch(
        values,
        bytemuck::cast_slice(dict),
        is_optional,
        page_validity,
        filter,
        validity,
        <T::AlignedBytes as AlignedBytes>::cast_vec_ref_mut(target),
    )
}

#[inline(never)]
pub fn decode_dict_dispatch<B: AlignedBytes>(
    mut values: HybridRleDecoder<'_>,
    dict: &[B],
    is_optional: bool,
    page_validity: Option<&Bitmap>,
    filter: Option<Filter>,
    validity: &mut MutableBitmap,
    target: &mut Vec<B>,
) -> ParquetResult<()> {
    if cfg!(debug_assertions) && is_optional {
        assert_eq!(target.len(), validity.len());
    }

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
    }?;

    if cfg!(debug_assertions) && is_optional {
        assert_eq!(target.len(), validity.len());
    }

    Ok(())
}

pub(crate) fn append_validity(
    page_validity: Option<&Bitmap>,
    filter: Option<&Filter>,
    validity: &mut MutableBitmap,
    values_len: usize,
) {
    match (page_validity, filter) {
        (None, None) => validity.extend_constant(values_len, true),
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

pub(crate) fn constrain_page_validity(
    values_len: usize,
    page_validity: Option<&Bitmap>,
    filter: Option<&Filter>,
) -> Option<Bitmap> {
    let num_unfiltered_rows = match (filter.as_ref(), page_validity) {
        (None, None) => values_len,
        (None, Some(pv)) => {
            debug_assert!(pv.len() >= values_len);
            pv.len()
        },
        (Some(f), v) => {
            if cfg!(debug_assertions) {
                if let Some(v) = v {
                    assert!(v.len() >= f.max_offset());
                }
            }

            f.max_offset()
        },
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

#[inline(always)]
fn verify_dict_indices_slice(indices: &[u32], dict_size: usize) -> ParquetResult<()> {
    let mut is_valid = true;
    for &idx in indices {
        is_valid &= (idx as usize) < dict_size;
    }

    if is_valid {
        return Ok(());
    }

    Err(oob_dict_idx())
}
