pub(crate) mod array_chunks;
pub(crate) mod dict_encoded;
pub(crate) mod filter;

use std::ops::Range;

use arrow::array::{DictionaryArray, DictionaryKey, PrimitiveArray};
use arrow::bitmap::{Bitmap, MutableBitmap};
use arrow::datatypes::ArrowDataType;
use arrow::pushable::Pushable;

use self::filter::Filter;
use super::BasicDecompressor;
use crate::parquet::encoding::hybrid_rle::gatherer::{
    HybridRleGatherer, ZeroCount, ZeroCountGatherer,
};
use crate::parquet::encoding::hybrid_rle::{self, HybridRleChunk, HybridRleDecoder};
use crate::parquet::error::{ParquetError, ParquetResult};
use crate::parquet::page::{split_buffer, DataPage, DictPage};
use crate::parquet::schema::Repetition;

#[derive(Debug)]
pub(crate) struct State<'a, D: Decoder> {
    pub(crate) dict: Option<&'a D::Dict>,
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
    fn len_when_not_nullable(&self) -> usize;
    fn skip_in_place(&mut self, n: usize) -> ParquetResult<()>;

    /// extends [`Self::DecodedState`] by deserializing items in [`Self::State`].
    /// It guarantees that the length of `decoded` is at most `decoded.len() + additional`.
    fn extend_from_state(
        &mut self,
        decoder: &mut D,
        decoded: &mut D::DecodedState,
        is_optional: bool,
        page_validity: &mut Option<Bitmap>,
        dict: Option<&'a D::Dict>,
        additional: usize,
    ) -> ParquetResult<()>;
}

impl<'a, D: Decoder> State<'a, D> {
    pub fn new(decoder: &D, page: &'a DataPage, dict: Option<&'a D::Dict>) -> ParquetResult<Self> {
        let is_optional =
            page.descriptor.primitive_type.field_info.repetition == Repetition::Optional;

        let mut page_validity = None;

        // Make the page_validity None if there are no nulls in the page
        if is_optional && !page.null_count().is_some_and(|nc| nc == 0) {
            let pv = page_validity_decoder(page)?;
            let pv = decode_page_validity(pv, None)?;

            if pv.unset_bits() > 0 {
                page_validity = Some(pv);
            }
        }

        let translation = D::Translation::new(decoder, page, dict, page_validity.as_ref())?;

        Ok(Self {
            dict,
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
            translation,
            is_optional,
            page_validity,
        })
    }

    pub fn len(&self) -> usize {
        match &self.page_validity {
            Some(v) => v.len(),
            None => self.translation.len_when_not_nullable(),
        }
    }

    pub fn skip_in_place(&mut self, n: usize) -> ParquetResult<()> {
        if n == 0 {
            return Ok(());
        }

        let n = self.page_validity.as_mut().map_or(n, |page_validity| {
            let mut pv = page_validity.clone();
            pv.slice(0, n);
            pv.unset_bits()
        });

        self.translation.skip_in_place(n)
    }

    pub fn decode(
        self,
        decoder: &mut D,
        decoded: &mut D::DecodedState,
        filter: Option<Filter>,
    ) -> ParquetResult<()> {
        decoder.extend_filtered_with_state(self, decoded, filter)
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

pub trait BatchableCollector<I, T> {
    fn reserve(target: &mut T, n: usize);
    fn push_n(&mut self, target: &mut T, n: usize) -> ParquetResult<()>;
    fn push_n_nulls(&mut self, target: &mut T, n: usize) -> ParquetResult<()>;
    fn skip_in_place(&mut self, n: usize) -> ParquetResult<()>;
}

/// This batches sequential collect operations to try and prevent unnecessary buffering and
/// `Iterator::next` polling.
#[must_use]
pub struct BatchedCollector<'a, I, T, C: BatchableCollector<I, T>> {
    pub(crate) num_waiting_valids: usize,
    pub(crate) num_waiting_invalids: usize,

    target: &'a mut T,
    collector: C,
    _pd: std::marker::PhantomData<I>,
}

impl<'a, I, T, C: BatchableCollector<I, T>> BatchedCollector<'a, I, T, C> {
    pub fn new(collector: C, target: &'a mut T) -> Self {
        Self {
            num_waiting_valids: 0,
            num_waiting_invalids: 0,
            target,
            collector,
            _pd: Default::default(),
        }
    }

    #[inline]
    pub fn push_valid(&mut self) -> ParquetResult<()> {
        self.push_n_valids(1)
    }

    #[inline]
    pub fn push_invalid(&mut self) {
        self.push_n_invalids(1)
    }

    #[inline]
    pub fn push_n_valids(&mut self, n: usize) -> ParquetResult<()> {
        if self.num_waiting_invalids == 0 {
            self.num_waiting_valids += n;
            return Ok(());
        }

        self.collector
            .push_n(self.target, self.num_waiting_valids)?;
        self.collector
            .push_n_nulls(self.target, self.num_waiting_invalids)?;

        self.num_waiting_valids = n;
        self.num_waiting_invalids = 0;

        Ok(())
    }

    #[inline]
    pub fn push_n_invalids(&mut self, n: usize) {
        self.num_waiting_invalids += n;
    }

    #[inline]
    pub fn skip_in_place(&mut self, n: usize) -> ParquetResult<()> {
        if self.num_waiting_valids > 0 {
            self.collector
                .push_n(self.target, self.num_waiting_valids)?;
            self.num_waiting_valids = 0;
        }
        if self.num_waiting_invalids > 0 {
            self.collector
                .push_n_nulls(self.target, self.num_waiting_invalids)?;
            self.num_waiting_invalids = 0;
        }

        self.collector.skip_in_place(n)?;

        Ok(())
    }

    #[inline]
    pub fn finalize(mut self) -> ParquetResult<()> {
        self.collector
            .push_n(self.target, self.num_waiting_valids)?;
        self.collector
            .push_n_nulls(self.target, self.num_waiting_invalids)?;
        Ok(())
    }
}

pub(crate) type PageValidity<'a> = HybridRleDecoder<'a>;
pub(crate) fn page_validity_decoder(page: &DataPage) -> ParquetResult<PageValidity> {
    let validity = split_buffer(page)?.def;
    let decoder = hybrid_rle::HybridRleDecoder::new(validity, 1, page.num_values());
    Ok(decoder)
}

#[derive(Default)]
pub(crate) struct BatchGatherer<'a, I, T, C: BatchableCollector<I, T>>(
    std::marker::PhantomData<&'a (I, T, C)>,
);
impl<'a, I, T, C: BatchableCollector<I, T>> HybridRleGatherer<u32> for BatchGatherer<'a, I, T, C> {
    type Target = (&'a mut MutableBitmap, BatchedCollector<'a, I, T, C>);

    fn target_reserve(&self, _target: &mut Self::Target, _n: usize) {}

    fn target_num_elements(&self, target: &Self::Target) -> usize {
        target.0.len()
    }

    fn hybridrle_to_target(&self, value: u32) -> ParquetResult<u32> {
        Ok(value)
    }

    fn gather_one(&self, (validity, values): &mut Self::Target, value: u32) -> ParquetResult<()> {
        if value == 0 {
            values.push_invalid();
            validity.extend_constant(1, false);
        } else {
            values.push_valid()?;
            validity.extend_constant(1, true);
        }

        Ok(())
    }

    fn gather_repeated(
        &self,
        (validity, values): &mut Self::Target,
        value: u32,
        n: usize,
    ) -> ParquetResult<()> {
        if value == 0 {
            values.push_n_invalids(n);
            validity.extend_constant(n, false);
        } else {
            values.push_n_valids(n)?;
            validity.extend_constant(n, true);
        }

        Ok(())
    }

    fn gather_slice(&self, target: &mut Self::Target, source: &[u32]) -> ParquetResult<()> {
        let mut prev = 0u32;
        let mut len = 0usize;

        for v in source {
            let v = *v;

            if v == prev {
                len += 1;
            } else {
                if len != 0 {
                    self.gather_repeated(target, prev, len)?;
                }
                prev = v;
                len = 1;
            }
        }

        if len != 0 {
            self.gather_repeated(target, prev, len)?;
        }

        Ok(())
    }
}

/// Extends a [`Pushable`] from an iterator of non-null values and an hybrid-rle decoder
pub(super) fn extend_from_decoder<I, T, C: BatchableCollector<I, T>>(
    validity: &mut MutableBitmap,
    page_validity: &mut Bitmap,
    limit: Option<usize>,
    target: &mut T,
    collector: C,
) -> ParquetResult<()> {
    let num_elements = limit.map_or(page_validity.len(), |limit| limit.min(page_validity.len()));

    validity.reserve(num_elements);
    C::reserve(target, num_elements);

    let mut batched_collector = BatchedCollector::new(collector, target);

    let mut pv = page_validity.clone();
    pv.slice(0, num_elements);

    // @TODO: This is terribly slow now.
    validity.extend_from_bitmap(&pv);
    let mut iter = pv.iter();
    while iter.num_remaining() > 0 {
        batched_collector.push_n_valids(iter.take_leading_ones())?;
        batched_collector.push_n_invalids(iter.take_leading_zeros());
    }

    batched_collector.finalize()?;

    Ok(())
}

impl<T, P: Pushable<T>, I: Iterator<Item = T>> BatchableCollector<T, P> for I {
    #[inline]
    fn reserve(target: &mut P, n: usize) {
        target.reserve(n);
    }

    #[inline]
    fn push_n(&mut self, target: &mut P, n: usize) -> ParquetResult<()> {
        target.extend_n(n, self);
        Ok(())
    }

    #[inline]
    fn push_n_nulls(&mut self, target: &mut P, n: usize) -> ParquetResult<()> {
        target.extend_null_constant(n);
        Ok(())
    }

    #[inline]
    fn skip_in_place(&mut self, n: usize) -> ParquetResult<()> {
        if n > 0 {
            _ = self.nth(n - 1);
        }
        Ok(())
    }
}

/// An item with a known size
pub(super) trait ExactSize {
    /// The number of items in the container
    fn len(&self) -> usize;
}

/// A decoder that knows how to map `State` -> Array
pub(super) trait Decoder: Sized {
    /// The state that this decoder derives from a [`DataPage`]. This is bound to the page.
    type Translation<'a>: StateTranslation<'a, Self>;
    /// The dictionary representation that the decoder uses
    type Dict: ExactSize;
    /// The target state that this Decoder decodes into.
    type DecodedState: ExactSize;

    type Output;

    /// Initializes a new [`Self::DecodedState`].
    fn with_capacity(&self, capacity: usize) -> Self::DecodedState;

    /// Deserializes a [`DictPage`] into [`Self::Dict`].
    fn deserialize_dict(&mut self, page: DictPage) -> ParquetResult<Self::Dict>;

    fn extend_filtered_with_state(
        &mut self,
        state: State<'_, Self>,
        decoded: &mut Self::DecodedState,
        filter: Option<Filter>,
    ) -> ParquetResult<()> {
        self.extend_filtered_with_state_default(state, decoded, filter)
    }

    fn extend_filtered_with_state_default(
        &mut self,
        mut state: State<'_, Self>,
        decoded: &mut Self::DecodedState,
        filter: Option<Filter>,
    ) -> ParquetResult<()> {
        match filter {
            None => {
                let num_rows = state.len();

                if num_rows == 0 {
                    return Ok(());
                }

                state.translation.extend_from_state(
                    self,
                    decoded,
                    state.is_optional,
                    &mut state.page_validity,
                    state.dict,
                    num_rows,
                )
            },
            Some(filter) => match filter {
                Filter::Range(range) => {
                    let start = range.start;
                    let end = range.end;

                    state.skip_in_place(start)?;
                    debug_assert!(end - start <= state.len());

                    if end - start > 0 {
                        state.translation.extend_from_state(
                            self,
                            decoded,
                            state.is_optional,
                            &mut state.page_validity,
                            state.dict,
                            end - start,
                        )?;
                    }

                    Ok(())
                },
                Filter::Mask(bitmap) => {
                    let mut iter = bitmap.iter();
                    while iter.num_remaining() > 0 && state.len() > 0 {
                        let prev_state_len = state.len();

                        let num_ones = iter.take_leading_ones();

                        if num_ones > 0 {
                            state.translation.extend_from_state(
                                self,
                                decoded,
                                state.is_optional,
                                &mut state.page_validity,
                                state.dict,
                                num_ones,
                            )?;
                        }

                        if iter.num_remaining() == 0 || state.len() == 0 {
                            break;
                        }

                        let num_zeros = iter.take_leading_zeros();
                        state.skip_in_place(num_zeros)?;

                        assert!(
                            prev_state_len != state.len(),
                            "No forward progress was booked in a filtered parquet file."
                        );
                    }

                    Ok(())
                },
            },
        }
    }

    fn apply_dictionary(
        &mut self,
        _decoded: &mut Self::DecodedState,
        _dict: &Self::Dict,
    ) -> ParquetResult<()> {
        Ok(())
    }

    fn decode_plain_encoded<'a>(
        &mut self,
        decoded: &mut Self::DecodedState,
        page_values: &mut <Self::Translation<'a> as StateTranslation<'a, Self>>::PlainDecoder,
        is_optional: bool,
        page_validity: Option<&mut Bitmap>,
        limit: usize,
    ) -> ParquetResult<()>;
    fn decode_dictionary_encoded(
        &mut self,
        decoded: &mut Self::DecodedState,
        page_values: &mut HybridRleDecoder<'_>,
        is_optional: bool,
        page_validity: Option<&mut Bitmap>,
        dict: &Self::Dict,
        limit: usize,
    ) -> ParquetResult<()>;

    fn finalize(
        &self,
        dtype: ArrowDataType,
        dict: Option<Self::Dict>,
        decoded: Self::DecodedState,
    ) -> ParquetResult<Self::Output>;
}

pub trait DictDecodable: Decoder {
    fn finalize_dict_array<K: DictionaryKey>(
        &self,
        dtype: ArrowDataType,
        dict: Self::Dict,
        keys: PrimitiveArray<K>,
    ) -> ParquetResult<DictionaryArray<K>>;
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

    pub fn collect_n(mut self, mut filter: Option<Filter>) -> ParquetResult<D::Output> {
        let mut num_rows_remaining = Filter::opt_num_rows(&filter, self.iter.total_num_values());

        let mut target = self.decoder.with_capacity(num_rows_remaining);

        if let Some(dict) = self.dict.as_ref() {
            self.decoder.apply_dictionary(&mut target, dict)?;
        }

        while num_rows_remaining > 0 {
            let Some(page) = self.iter.next() else {
                break;
            };
            let page = page?;

            let state_filter;
            (state_filter, filter) = Filter::opt_split_at(&filter, page.num_values());

            // Skip the whole page if we don't need any rows from it
            if state_filter.as_ref().is_some_and(|f| f.num_rows() == 0) {
                continue;
            }

            let page = page.decompress(&mut self.iter)?;

            let state = State::new(&self.decoder, &page, self.dict.as_ref())?;

            let start_length = target.len();
            state.decode(&mut self.decoder, &mut target, state_filter)?;
            let end_length = target.len();

            num_rows_remaining -= end_length - start_length;

            self.iter.reuse_page_buffer(page);
        }

        self.decoder.finalize(self.dtype, self.dict, target)
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
pub fn freeze_validity(validity: MutableBitmap) -> Option<Bitmap> {
    if validity.is_empty() {
        return None;
    }

    let validity = validity.freeze();

    if validity.unset_bits() == 0 {
        return None;
    }

    Some(validity)
}

pub(crate) fn hybrid_rle_count_zeros(
    decoder: &hybrid_rle::HybridRleDecoder<'_>,
) -> ParquetResult<usize> {
    let mut count = ZeroCount::default();
    decoder
        .clone()
        .gather_into(&mut count, &ZeroCountGatherer)?;
    Ok(count.num_zero)
}

pub(crate) fn filter_from_range(rng: Range<usize>) -> Bitmap {
    let mut bm = MutableBitmap::with_capacity(rng.end);

    bm.extend_constant(rng.start, false);
    bm.extend_constant(rng.len(), true);

    bm.freeze()
}

pub(crate) fn decode_page_validity(
    mut page_validity: HybridRleDecoder<'_>,
    limit: Option<usize>,
) -> ParquetResult<Bitmap> {
    let mut limit = limit.unwrap_or(page_validity.len());
    let mut bm = MutableBitmap::with_capacity(limit);

    while let Some(chunk) = page_validity.next_chunk()? {
        if limit == 0 {
            break;
        }

        match chunk {
            HybridRleChunk::Rle(value, size) => {
                let size = size.min(limit);
                bm.extend_constant(size, value != 0);
                limit -= size;
            },
            HybridRleChunk::Bitpacked(decoder) => {
                let len = decoder.len().min(limit);
                bm.extend_from_slice(decoder.as_slice(), 0, len);
                limit -= len;
            },
        }
    }

    Ok(bm.freeze())
}
