pub(crate) mod array_chunks;
pub(crate) mod filter;

use std::ops::Range;

use arrow::array::{Array, IntoBoxedArray, Splitable};
use arrow::bitmap::{Bitmap, BitmapBuilder};
use arrow::datatypes::ArrowDataType;
use arrow::pushable::Pushable;

use self::filter::Filter;
use super::{BasicDecompressor, PredicateFilter};
use crate::parquet::encoding::hybrid_rle::{self, HybridRleChunk, HybridRleDecoder};
use crate::parquet::error::{ParquetError, ParquetResult};
use crate::parquet::page::{split_buffer, DataPage, DictPage};
use crate::parquet::schema::Repetition;

#[derive(Debug)]
pub(crate) struct State<'a, D: Decoder> {
    pub(crate) dict: Option<&'a D::Dict>,
    pub(crate) dict_mask: Option<&'a Bitmap>,
    pub(crate) is_optional: bool,
    pub(crate) page_validity: Option<Bitmap>,
    pub(crate) translation: D::Translation<'a>,
}

pub(crate) trait StateTranslation<'a, D: Decoder>: Sized {
    type PlainDecoder;

    fn new(
        decoder: &D,
        page: &'a DataPage,
        dict: Option<&'a D::Dict>,
        page_validity: Option<&Bitmap>,
    ) -> ParquetResult<Self>;
    fn num_rows(&self) -> usize;
}

impl<'a, D: Decoder> State<'a, D> {
    pub fn new(
        decoder: &D,
        page: &'a DataPage,
        dict: Option<&'a D::Dict>,
        dict_mask: Option<&'a Bitmap>,
    ) -> ParquetResult<Self> {
        let is_optional =
            page.descriptor.primitive_type.field_info.repetition == Repetition::Optional;

        let mut page_validity = None;

        // Make the page_validity None if there are no nulls in the page
        if is_optional && page.null_count().is_none_or(|nc| nc != 0) {
            let pv = page_validity_decoder(page)?;
            page_validity = decode_page_validity(pv, None)?;
        }

        let translation = D::Translation::new(decoder, page, dict, page_validity.as_ref())?;

        Ok(Self {
            dict,
            dict_mask,
            is_optional,
            page_validity,
            translation,
        })
    }

    pub fn new_nested(
        decoder: &D,
        page: &'a DataPage,
        dict: Option<&'a D::Dict>,
        mut page_validity: Option<Bitmap>,
    ) -> ParquetResult<Self> {
        let translation = D::Translation::new(decoder, page, dict, None)?;

        let is_optional =
            page.descriptor.primitive_type.field_info.repetition == Repetition::Optional;

        if page_validity
            .as_ref()
            .is_some_and(|bm| bm.unset_bits() == 0)
        {
            page_validity = None;
        }

        Ok(Self {
            dict,
            dict_mask: None,
            translation,
            is_optional,
            page_validity,
        })
    }

    pub fn decode(
        self,
        decoder: &mut D,
        decoded: &mut D::DecodedState,
        pred_true_mask: &mut BitmapBuilder,
        filter: Option<Filter>,
    ) -> ParquetResult<()> {
        decoder.extend_filtered_with_state(self, decoded, pred_true_mask, filter)
    }
}

pub fn not_implemented(page: &DataPage) -> ParquetError {
    let is_optional = page.descriptor.primitive_type.field_info.repetition == Repetition::Optional;
    let required = if is_optional { "optional" } else { "required" };
    ParquetError::not_supported(format!(
        "Decoding {:?} \"{:?}\"-encoded {required} parquet pages not yet supported",
        page.descriptor.primitive_type.physical_type,
        page.encoding(),
    ))
}

pub(crate) type PageValidity<'a> = HybridRleDecoder<'a>;
pub(crate) fn page_validity_decoder(page: &DataPage) -> ParquetResult<PageValidity> {
    let validity = split_buffer(page)?.def;
    let decoder = hybrid_rle::HybridRleDecoder::new(validity, 1, page.num_values());
    Ok(decoder)
}

pub(crate) fn unspecialized_decode<T: Default>(
    mut num_rows: usize,

    mut decode_one: impl FnMut() -> ParquetResult<T>,

    mut filter: Option<Filter>,
    mut page_validity: Option<Bitmap>,

    is_optional: bool,

    validity: &mut BitmapBuilder,
    target: &mut impl Pushable<T>,
) -> ParquetResult<()> {
    match &mut filter {
        None => {},
        Some(Filter::Range(range)) => {
            match page_validity.as_mut() {
                None => {
                    for _ in 0..range.start {
                        decode_one()?;
                    }
                },
                Some(pv) => {
                    let c;
                    (c, *pv) = pv.split_at(range.start);
                    for _ in 0..c.set_bits() {
                        decode_one()?;
                    }
                    *pv = std::mem::take(pv).sliced(0, range.len());
                },
            }

            num_rows = range.len();
            filter = None;
        },
        Some(Filter::Mask(mask)) => {
            let leading_zeros = mask.take_leading_zeros();
            mask.take_trailing_zeros();

            match page_validity.as_mut() {
                None => {
                    for _ in 0..leading_zeros {
                        decode_one()?;
                    }
                },
                Some(pv) => {
                    let c;
                    (c, *pv) = pv.split_at(leading_zeros);
                    for _ in 0..c.set_bits() {
                        decode_one()?;
                    }
                    *pv = std::mem::take(pv).sliced(0, mask.len());
                },
            }

            num_rows = mask.len();
            if mask.unset_bits() == 0 {
                filter = None;
            }
        },
        Some(Filter::Predicate(_)) => todo!(),
    };

    page_validity = page_validity.filter(|pv| pv.unset_bits() > 0);

    match (filter, page_validity) {
        (None, None) => {
            target.reserve(num_rows);
            for _ in 0..num_rows {
                target.push(decode_one()?);
            }

            if is_optional {
                validity.extend_constant(num_rows, true);
            }
        },
        (None, Some(page_validity)) => {
            target.reserve(page_validity.len());
            for is_valid in page_validity.iter() {
                let v = if is_valid {
                    decode_one()?
                } else {
                    T::default()
                };
                target.push(v);
            }

            validity.extend_from_bitmap(&page_validity);
        },
        (Some(Filter::Range(_)), _) => unreachable!(),
        (Some(Filter::Mask(mut mask)), None) => {
            target.reserve(num_rows);

            while !mask.is_empty() {
                let num_ones = mask.take_leading_ones();
                for _ in 0..num_ones {
                    target.push(decode_one()?);
                }

                let num_zeros = mask.take_leading_zeros();
                for _ in 0..num_zeros {
                    decode_one()?;
                }
            }

            if is_optional {
                validity.extend_constant(num_rows, true);
            }
        },
        (Some(Filter::Mask(mask)), Some(page_validity)) => {
            assert_eq!(mask.len(), page_validity.len());

            let num_rows = mask.set_bits();
            target.reserve(num_rows);

            let mut mask_iter = mask.fast_iter_u56();
            let mut validity_iter = page_validity.fast_iter_u56();

            let mut iter = |mut f: u64, mut v: u64| {
                while f != 0 {
                    let offset = f.trailing_ones();

                    if (v >> offset) & 1 != 0 {
                        target.push(decode_one()?);
                    } else {
                        target.push(T::default());
                    }

                    let skip = (v & (1u64 << offset).wrapping_sub(1)).count_ones() as usize;
                    for _ in 0..skip {
                        decode_one()?;
                    }

                    v >>= offset + 1;
                    f >>= offset + 1;
                }

                for _ in 0..v.count_ones() as usize {
                    decode_one()?;
                }

                ParquetResult::Ok(())
            };

            for (f, v) in mask_iter.by_ref().zip(validity_iter.by_ref()) {
                iter(f, v)?;
            }

            let (f, fl) = mask_iter.remainder();
            let (v, vl) = validity_iter.remainder();

            assert_eq!(fl, vl);

            iter(f, v)?;

            validity.extend_from_bitmap(&page_validity);
        },
        (Some(Filter::Predicate(_)), _) => todo!(),
    }

    Ok(())
}

/// The state that will be decoded into.
///
/// This is usually an Array and a validity mask as a MutableBitmap.
pub(super) trait Decoded {
    /// The number of items in the container
    fn len(&self) -> usize;
    /// Extend the decoded state with `n` nulls.
    fn extend_nulls(&mut self, n: usize);
}

/// A decoder that knows how to map `State` -> Array
pub(super) trait Decoder: Sized {
    /// The state that this decoder derives from a [`DataPage`]. This is bound to the page.
    type Translation<'a>: StateTranslation<'a, Self>;
    /// The dictionary representation that the decoder uses
    type Dict: Array;
    /// The target state that this Decoder decodes into.
    type DecodedState: Decoded;

    type Output: IntoBoxedArray;

    fn evaluate_dict_predicate(
        &self,
        dict: &Self::Dict,
        predicate: &PredicateFilter,
    ) -> ParquetResult<Bitmap> {
        Ok(predicate.predicate.evaluate(dict))
    }

    /// Initializes a new [`Self::DecodedState`].
    fn with_capacity(&self, capacity: usize) -> Self::DecodedState;

    /// Deserializes a [`DictPage`] into [`Self::Dict`].
    fn deserialize_dict(&mut self, page: DictPage) -> ParquetResult<Self::Dict>;

    fn has_predicate_specialization(
        &self,
        state: &State<'_, Self>,
        predicate: &PredicateFilter,
    ) -> ParquetResult<bool>;

    fn extend_decoded(
        &self,
        decoded: &mut Self::DecodedState,
        additional: &dyn Array,
        is_optional: bool,
    ) -> ParquetResult<()>;

    fn unspecialized_predicate_decode(
        &mut self,
        state: State<'_, Self>,
        decoded: &mut Self::DecodedState,
        pred_true_mask: &mut BitmapBuilder,
        predicate: &PredicateFilter,
        dtype: &ArrowDataType,
    ) -> ParquetResult<()> {
        let is_optional = state.is_optional;

        let mut intermediate_array = self.with_capacity(state.translation.num_rows());
        self.extend_filtered_with_state(
            state,
            &mut intermediate_array,
            &mut BitmapBuilder::new(),
            None,
        )?;
        let intermediate_array = self
            .finalize(dtype.clone(), None, intermediate_array)?
            .into_boxed();

        let mask = if let Some(validity) = intermediate_array.validity() {
            let ignore_validity_array = intermediate_array.with_validity(None);
            let mask = predicate.predicate.evaluate(ignore_validity_array.as_ref());

            if predicate.predicate.evaluate_null() {
                &mask | validity
            } else {
                &mask & validity
            }
        } else {
            predicate.predicate.evaluate(intermediate_array.as_ref())
        };

        let filtered =
            polars_compute::filter::filter_with_bitmap(intermediate_array.as_ref(), &mask);

        pred_true_mask.extend_from_bitmap(&mask);
        self.extend_decoded(decoded, filtered.as_ref(), is_optional)?;

        Ok(())
    }

    fn extend_filtered_with_state(
        &mut self,
        state: State<'_, Self>,
        decoded: &mut Self::DecodedState,
        pred_true_mask: &mut BitmapBuilder,
        filter: Option<Filter>,
    ) -> ParquetResult<()>;

    fn apply_dictionary(
        &mut self,
        _decoded: &mut Self::DecodedState,
        _dict: &Self::Dict,
    ) -> ParquetResult<()> {
        Ok(())
    }

    fn finalize(
        &self,
        dtype: ArrowDataType,
        dict: Option<Self::Dict>,
        decoded: Self::DecodedState,
    ) -> ParquetResult<Self::Output>;
}

pub struct PageDecoder<D: Decoder> {
    pub iter: BasicDecompressor,
    pub dtype: ArrowDataType,
    pub dict: Option<D::Dict>,
    pub decoder: D,
}

impl<D: Decoder> PageDecoder<D> {
    pub fn new(
        mut iter: BasicDecompressor,
        dtype: ArrowDataType,
        mut decoder: D,
    ) -> ParquetResult<Self> {
        let dict_page = iter.read_dict_page()?;
        let dict = dict_page.map(|d| decoder.deserialize_dict(d)).transpose()?;

        Ok(Self {
            iter,
            dtype,
            dict,
            decoder,
        })
    }

    pub fn collect(mut self, mut filter: Option<Filter>) -> ParquetResult<(D::Output, Bitmap)> {
        let mut num_rows_remaining = Filter::opt_num_rows(&filter, self.iter.total_num_values());

        // @TODO: Don't allocate if include_values == false
        let mut target = self.decoder.with_capacity(num_rows_remaining);
        let mut pred_true_mask = BitmapBuilder::new();

        let mut pred_tracks_nulls = true;
        let mut dict_mask = None;
        if let Some(dict) = self.dict.as_ref() {
            self.decoder.apply_dictionary(&mut target, dict)?;

            if let Some(Filter::Predicate(p)) = &filter {
                pred_tracks_nulls = p.predicate.evaluate_null();
                pred_true_mask.reserve(num_rows_remaining);
                dict_mask = Some(self.decoder.evaluate_dict_predicate(dict, p)?);
            }
        }

        while num_rows_remaining > 0 {
            let Some(page) = self.iter.next() else {
                break;
            };
            let page = page?;

            let page_num_values = page.num_values();

            let state_filter;
            (state_filter, filter) = Filter::opt_split_at(&filter, page_num_values);

            // Skip the whole page if we don't need any rows from it
            if state_filter
                .as_ref()
                .is_some_and(|f| f.num_rows(page_num_values) == 0)
            {
                continue;
            }

            // Skip a dictionary encoded page if none of the dictionary values match the predicate.
            // This is essentially a slower version of statistics skipping.
            if dict_mask.as_ref().is_some_and(|dm| dm.set_bits() == 0)
                && page.page().header().is_dictionary_encoded()
                && (page.page().descriptor.primitive_type.field_info.repetition
                    != Repetition::Optional
                    || !pred_tracks_nulls)
            {
                pred_true_mask.extend_constant(page.num_values(), false);
                continue;
            }

            let page = page.decompress(&mut self.iter)?;

            let state = State::new(&self.decoder, &page, self.dict.as_ref(), dict_mask.as_ref())?;

            let start_length = target.len();
            match &state_filter {
                // Handle the case where column is held equal to Null. This can be the same for all
                // non-nested columns.
                Some(Filter::Predicate(p))
                    if p.predicate
                        .to_equals_scalar()
                        .is_some_and(|sc| sc.is_null()) =>
                {
                    if state.is_optional {
                        match &state.page_validity {
                            None => pred_true_mask.extend_constant(page.num_values(), false),
                            Some(v) => {
                                pred_true_mask.extend_from_bitmap(v);
                                if p.include_values {
                                    target.extend_nulls(v.set_bits());
                                }
                            },
                        }
                    } else {
                        pred_true_mask.extend_constant(page.num_values(), false);
                    }
                    drop(state);
                },

                // For now, we have a function that indicates whether the predicate can actually be
                // handled in the kernels. If it cannot be handled in the kernels, catch it here
                // and load it as if it weren't filtered.
                Some(Filter::Predicate(p))
                    if !self.decoder.has_predicate_specialization(&state, p)? =>
                {
                    self.decoder.unspecialized_predicate_decode(
                        state,
                        &mut target,
                        &mut pred_true_mask,
                        p,
                        &self.dtype,
                    )?
                },
                _ => state.decode(
                    &mut self.decoder,
                    &mut target,
                    &mut pred_true_mask,
                    state_filter,
                )?,
            }

            let end_length = target.len();

            num_rows_remaining -= end_length - start_length;

            self.iter.reuse_page_buffer(page);
        }

        let array = self.decoder.finalize(self.dtype, self.dict, target)?;
        Ok((array, pred_true_mask.freeze()))
    }

    pub fn collect_boxed(self, filter: Option<Filter>) -> ParquetResult<(Box<dyn Array>, Bitmap)> {
        self.collect(filter)
            .map(|(arr, ptm)| (arr.into_boxed(), ptm))
    }
}

#[inline]
pub(super) fn dict_indices_decoder(
    page: &DataPage,
    null_count: usize,
) -> ParquetResult<hybrid_rle::HybridRleDecoder> {
    let indices_buffer = split_buffer(page)?.values;

    // SPEC: Data page format: the bit width used to encode the entry ids stored as 1 byte (max bit width = 32),
    // SPEC: followed by the values encoded using RLE/Bit packed described above (with the given bit width).
    let bit_width = indices_buffer[0];
    let indices_buffer = &indices_buffer[1..];

    Ok(hybrid_rle::HybridRleDecoder::new(
        indices_buffer,
        bit_width as u32,
        page.num_values() - null_count,
    ))
}

/// Freeze a [`MutableBitmap`] into a `Option<Bitmap>`.
///
/// This will turn the several instances where `None` (representing "all valid") suffices.
pub fn freeze_validity(validity: BitmapBuilder) -> Option<Bitmap> {
    if validity.is_empty() || validity.unset_bits() == 0 {
        return None;
    }

    let validity = validity.freeze();
    Some(validity)
}

pub(crate) fn filter_from_range(rng: Range<usize>) -> Bitmap {
    let mut bm = BitmapBuilder::with_capacity(rng.end);

    bm.extend_constant(rng.start, false);
    bm.extend_constant(rng.len(), true);

    bm.freeze()
}

pub(crate) fn decode_hybrid_rle_into_bitmap(
    mut page_validity: HybridRleDecoder<'_>,
    limit: Option<usize>,
    bitmap: &mut BitmapBuilder,
) -> ParquetResult<()> {
    assert!(page_validity.num_bits() <= 1);

    let mut limit = limit.unwrap_or(page_validity.len());
    bitmap.reserve(limit);

    while let Some(chunk) = page_validity.next_chunk()? {
        if limit == 0 {
            break;
        }

        match chunk {
            HybridRleChunk::Rle(value, size) => {
                let size = size.min(limit);
                bitmap.extend_constant(size, value != 0);
                limit -= size;
            },
            HybridRleChunk::Bitpacked(decoder) => {
                let len = decoder.len().min(limit);
                bitmap.extend_from_slice(decoder.as_slice(), 0, len);
                limit -= len;
            },
        }
    }

    Ok(())
}

pub(crate) fn decode_page_validity(
    mut page_validity: HybridRleDecoder<'_>,
    limit: Option<usize>,
) -> ParquetResult<Option<Bitmap>> {
    assert!(page_validity.num_bits() <= 1);

    let mut num_ones = 0;

    let mut bm = BitmapBuilder::new();
    let limit = limit.unwrap_or(page_validity.len());
    page_validity.limit_to(limit);
    let num_values = page_validity.len();

    // If all values are valid anyway, we will return a None so don't allocate until we disprove
    // that that is the case.
    while let Some(chunk) = page_validity.next_chunk()? {
        match chunk {
            HybridRleChunk::Rle(value, size) if value != 0 => num_ones += size,
            HybridRleChunk::Rle(value, size) => {
                bm.reserve(num_values);
                bm.extend_constant(num_ones, true);
                bm.extend_constant(size, value != 0);
                break;
            },
            HybridRleChunk::Bitpacked(decoder) => {
                let len = decoder.len();
                bm.reserve(num_values);
                bm.extend_constant(num_ones, true);
                bm.extend_from_slice(decoder.as_slice(), 0, len);
                break;
            },
        }
    }

    if page_validity.len() == 0 && bm.is_empty() {
        return Ok(None);
    }

    decode_hybrid_rle_into_bitmap(page_validity, None, &mut bm)?;
    Ok(Some(bm.freeze()))
}
