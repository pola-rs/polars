use arrow::array::BooleanArray;
use arrow::bitmap::bitmask::BitMask;
use arrow::bitmap::utils::BitmapIter;
use arrow::bitmap::{Bitmap, MutableBitmap};
use arrow::datatypes::ArrowDataType;
use polars_compute::filter::filter_boolean_kernel;

use super::utils::dict_encoded::{append_validity, constrain_page_validity};
use super::utils::{
    self, decode_hybrid_rle_into_bitmap, filter_from_range, freeze_validity,
    Decoder, ExactSize,
};
use super::Filter;
use crate::parquet::encoding::hybrid_rle::{HybridRleChunk, HybridRleDecoder};
use crate::parquet::encoding::Encoding;
use crate::parquet::error::ParquetResult;
use crate::parquet::page::{split_buffer, DataPage, DictPage};

#[allow(clippy::large_enum_variant)]
#[derive(Debug)]
pub(crate) enum StateTranslation<'a> {
    Plain(BitmapIter<'a>),
    Rle(HybridRleDecoder<'a>),
}

impl<'a> utils::StateTranslation<'a, BooleanDecoder> for StateTranslation<'a> {
    type PlainDecoder = BitmapIter<'a>;

    fn new(
        _decoder: &BooleanDecoder,
        page: &'a DataPage,
        _dict: Option<&'a <BooleanDecoder as Decoder>::Dict>,
        page_validity: Option<&Bitmap>,
    ) -> ParquetResult<Self> {
        let values = split_buffer(page)?.values;

        match page.encoding() {
            Encoding::Plain => {
                let max_num_values = values.len() * u8::BITS as usize;
                let num_values = if page_validity.is_some() {
                    // @NOTE: We overestimate the amount of values here, but in the V1
                    // specification we don't really have a way to know the number of valid items.
                    // Without traversing the list.
                    max_num_values
                } else {
                    // @NOTE: We cannot really trust the value from this as it might relate to the
                    // number of top-level nested values. Therefore, we do a `min` with the maximum
                    // number of possible values.
                    usize::min(page.num_values(), max_num_values)
                };

                Ok(Self::Plain(BitmapIter::new(values, 0, num_values)))
            },
            Encoding::Rle => {
                // @NOTE: For a nullable list, we might very well overestimate the amount of
                // values, but we never collect those items. We don't really have a way to know the
                // number of valid items in the V1 specification.

                // For RLE boolean values the length in bytes is pre-pended.
                // https://github.com/apache/parquet-format/blob/e517ac4dbe08d518eb5c2e58576d4c711973db94/Encodings.md#run-length-encoding--bit-packing-hybrid-rle--3
                let (_len_in_bytes, values) = values.split_at(4);
                Ok(Self::Rle(HybridRleDecoder::new(
                    values,
                    1,
                    page.num_values(),
                )))
            },
            _ => Err(utils::not_implemented(page)),
        }
    }
}

fn decode_required_rle(
    values: HybridRleDecoder<'_>,
    limit: Option<usize>,
    target: &mut MutableBitmap,
) -> ParquetResult<()> {
    decode_hybrid_rle_into_bitmap(values, limit, target)?;
    Ok(())
}

fn decode_optional_rle(
    values: HybridRleDecoder<'_>,
    target: &mut MutableBitmap,
    page_validity: &Bitmap,
) -> ParquetResult<()> {
    debug_assert!(page_validity.set_bits() <= values.len());

    if page_validity.unset_bits() == 0 {
        return decode_required_rle(values, Some(page_validity.len()), target);
    }

    target.reserve(page_validity.len());

    let mut validity_mask = BitMask::from_bitmap(page_validity);

    for chunk in values.into_chunk_iter() {
        let chunk = chunk?;

        match chunk {
            HybridRleChunk::Rle(value, size) => {
                let offset = validity_mask
                    .nth_set_bit_idx(size, 0)
                    .unwrap_or(validity_mask.len());

                let t;
                (t, validity_mask) = validity_mask.split_at(offset);

                target.extend_constant(t.len(), value != 0);
            },
            HybridRleChunk::Bitpacked(decoder) => {
                let decoder_slice = decoder.as_slice();
                let offset = validity_mask
                    .nth_set_bit_idx(decoder.len(), 0)
                    .unwrap_or(validity_mask.len());

                let decoder_validity;
                (decoder_validity, validity_mask) = validity_mask.split_at(offset);

                let mut offset = 0;
                let mut validity_iter = decoder_validity.iter();
                while validity_iter.num_remaining() > 0 {
                    let num_valid = validity_iter.take_leading_ones();
                    target.extend_from_slice(decoder_slice, offset, num_valid);
                    offset += num_valid;

                    let num_invalid = validity_iter.take_leading_zeros();
                    target.extend_constant(num_invalid, false);
                }
            },
        }
    }

    if cfg!(debug_assertions) {
        assert_eq!(validity_mask.set_bits(), 0);
    }
    target.extend_constant(validity_mask.len(), false);

    Ok(())
}

fn decode_masked_required_rle(
    values: HybridRleDecoder<'_>,
    target: &mut MutableBitmap,
    mask: &Bitmap,
) -> ParquetResult<()> {
    debug_assert!(mask.len() <= values.len());

    if mask.unset_bits() == 0 {
        return decode_required_rle(values, Some(mask.len()), target);
    }

    let mut im_target = MutableBitmap::new();
    decode_required_rle(values, Some(mask.len()), &mut im_target)?;

    target.extend_from_bitmap(&filter_boolean_kernel(&im_target.freeze(), mask));

    Ok(())
}

fn decode_masked_optional_rle(
    values: HybridRleDecoder<'_>,
    target: &mut MutableBitmap,
    page_validity: &Bitmap,
    mask: &Bitmap,
) -> ParquetResult<()> {
    debug_assert_eq!(page_validity.len(), mask.len());
    debug_assert!(mask.len() <= values.len());

    if mask.unset_bits() == 0 {
        return decode_optional_rle(values, target, page_validity);
    }

    if page_validity.unset_bits() == 0 {
        return decode_masked_required_rle(values, target, mask);
    }

    let mut im_target = MutableBitmap::new();
    decode_optional_rle(values, &mut im_target, page_validity)?;

    target.extend_from_bitmap(&filter_boolean_kernel(&im_target.freeze(), mask));

    Ok(())
}

fn decode_required_plain(
    mut values: BitmapIter<'_>,
    limit: Option<usize>,
    target: &mut MutableBitmap,
) -> ParquetResult<()> {
    let limit = limit.unwrap_or(values.len());
    values.collect_n_into(target, limit);
    Ok(())
}

fn decode_optional_plain(
    mut values: BitmapIter<'_>,
    target: &mut MutableBitmap,
    page_validity: &Bitmap,
) -> ParquetResult<()> {
    debug_assert!(page_validity.set_bits() <= values.len());

    if page_validity.unset_bits() == 0 {
        return decode_required_plain(values, Some(page_validity.len()), target);
    }

    target.reserve(page_validity.len());

    let mut validity_iter = page_validity.iter();
    while validity_iter.num_remaining() > 0 {
        let num_valid = validity_iter.take_leading_ones();
        values.collect_n_into(target, num_valid);

        let num_invalid = validity_iter.take_leading_zeros();
        target.extend_constant(num_invalid, false);
    }

    Ok(())
}

fn decode_masked_required_plain(
    values: BitmapIter<'_>,
    target: &mut MutableBitmap,
    mask: &Bitmap,
) -> ParquetResult<()> {
    debug_assert!(mask.len() <= values.len());

    if mask.unset_bits() == 0 {
        return decode_required_plain(values, Some(mask.len()), target);
    }

    let mut im_target = MutableBitmap::new();
    decode_required_plain(values, Some(mask.len()), &mut im_target)?;

    target.extend_from_bitmap(&filter_boolean_kernel(&im_target.freeze(), mask));

    Ok(())
}

fn decode_masked_optional_plain(
    values: BitmapIter<'_>,
    target: &mut MutableBitmap,
    page_validity: &Bitmap,
    mask: &Bitmap,
) -> ParquetResult<()> {
    debug_assert_eq!(page_validity.len(), mask.len());
    debug_assert!(page_validity.set_bits() <= values.len());

    if mask.unset_bits() == 0 {
        return decode_optional_plain(values, target, page_validity);
    }

    if page_validity.unset_bits() == 0 {
        return decode_masked_required_plain(values, target, mask);
    }

    let mut im_target = MutableBitmap::new();
    decode_optional_plain(values, &mut im_target, page_validity)?;

    target.extend_from_bitmap(&filter_boolean_kernel(&im_target.freeze(), mask));

    Ok(())
}

impl ExactSize for (MutableBitmap, MutableBitmap) {
    fn len(&self) -> usize {
        self.0.len()
    }
}

impl ExactSize for () {
    fn len(&self) -> usize {
        0
    }
}

pub(crate) struct BooleanDecoder;

impl Decoder for BooleanDecoder {
    type Translation<'a> = StateTranslation<'a>;
    type Dict = ();
    type DecodedState = (MutableBitmap, MutableBitmap);
    type Output = BooleanArray;

    fn with_capacity(&self, capacity: usize) -> Self::DecodedState {
        (
            MutableBitmap::with_capacity(capacity),
            MutableBitmap::with_capacity(capacity),
        )
    }

    fn deserialize_dict(&mut self, _: DictPage) -> ParquetResult<Self::Dict> {
        Ok(())
    }

    fn finalize(
        &self,
        dtype: ArrowDataType,
        _dict: Option<Self::Dict>,
        (values, validity): Self::DecodedState,
    ) -> ParquetResult<Self::Output> {
        let validity = freeze_validity(validity);
        Ok(BooleanArray::new(dtype, values.freeze(), validity))
    }

    fn extend_filtered_with_state(
        &mut self,
        state: utils::State<'_, Self>,
        (target, validity): &mut Self::DecodedState,
        filter: Option<super::Filter>,
    ) -> ParquetResult<()> {
        match state.translation {
            StateTranslation::Plain(values) => {
                if state.is_optional {
                    append_validity(
                        state.page_validity.as_ref(),
                        filter.as_ref(),
                        validity,
                        values.len(),
                    );
                }

                let page_validity = constrain_page_validity(
                    values.len(),
                    state.page_validity.as_ref(),
                    filter.as_ref(),
                );

                match (filter, page_validity) {
                    (None, None) => decode_required_plain(values, None, target),
                    (Some(Filter::Range(rng)), None) if rng.start == 0 => {
                        decode_required_plain(values, Some(rng.end), target)
                    },
                    (None, Some(page_validity)) => {
                        decode_optional_plain(values, target, &page_validity)
                    },
                    (Some(Filter::Range(rng)), Some(page_validity)) if rng.start == 0 => {
                        decode_optional_plain(values, target, &page_validity)
                    },
                    (Some(Filter::Mask(mask)), None) => {
                        decode_masked_required_plain(values, target, &mask)
                    },
                    (Some(Filter::Mask(mask)), Some(page_validity)) => {
                        decode_masked_optional_plain(values, target, &page_validity, &mask)
                    },
                    (Some(Filter::Range(rng)), None) => decode_masked_required_plain(
                        values,
                        target,
                        &filter_from_range(rng.clone()),
                    ),
                    (Some(Filter::Range(rng)), Some(page_validity)) => {
                        decode_masked_optional_plain(
                            values,
                            target,
                            &page_validity,
                            &filter_from_range(rng.clone()),
                        )
                    },
                }?;

                Ok(())
            },
            StateTranslation::Rle(values) => {
                if state.is_optional {
                    append_validity(
                        state.page_validity.as_ref(),
                        filter.as_ref(),
                        validity,
                        values.len(),
                    );
                }

                let page_validity = constrain_page_validity(
                    values.len(),
                    state.page_validity.as_ref(),
                    filter.as_ref(),
                );

                match (filter, page_validity) {
                    (None, None) => decode_required_rle(values, None, target),
                    (Some(Filter::Range(rng)), None) if rng.start == 0 => {
                        decode_required_rle(values, Some(rng.end), target)
                    },
                    (None, Some(page_validity)) => {
                        decode_optional_rle(values, target, &page_validity)
                    },
                    (Some(Filter::Range(rng)), Some(page_validity)) if rng.start == 0 => {
                        decode_optional_rle(values, target, &page_validity)
                    },
                    (Some(Filter::Mask(filter)), None) => {
                        decode_masked_required_rle(values, target, &filter)
                    },
                    (Some(Filter::Mask(filter)), Some(page_validity)) => {
                        decode_masked_optional_rle(values, target, &page_validity, &filter)
                    },
                    (Some(Filter::Range(rng)), None) => {
                        decode_masked_required_rle(values, target, &filter_from_range(rng.clone()))
                    },
                    (Some(Filter::Range(rng)), Some(page_validity)) => decode_masked_optional_rle(
                        values,
                        target,
                        &page_validity,
                        &filter_from_range(rng.clone()),
                    ),
                }?;

                Ok(())
            },
        }
    }
}
