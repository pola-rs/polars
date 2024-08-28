pub(crate) mod array_chunks;
pub(crate) mod filter;

use arrow::array::{DictionaryArray, DictionaryKey, MutableBinaryViewArray, PrimitiveArray, View};
use arrow::bitmap::{Bitmap, MutableBitmap};
use arrow::datatypes::ArrowDataType;
use arrow::pushable::Pushable;

use self::filter::Filter;
use super::BasicDecompressor;
use crate::parquet::encoding::hybrid_rle::gatherer::{
    HybridRleGatherer, ZeroCount, ZeroCountGatherer,
};
use crate::parquet::encoding::hybrid_rle::{self, HybridRleDecoder, Translator};
use crate::parquet::error::{ParquetError, ParquetResult};
use crate::parquet::page::{split_buffer, DataPage, DictPage};
use crate::parquet::schema::Repetition;

#[derive(Debug)]
pub(crate) struct State<'a, D: Decoder> {
    pub(crate) dict: Option<&'a D::Dict>,
    pub(crate) is_optional: bool,
    pub(crate) page_validity: Option<PageValidity<'a>>,
    pub(crate) translation: D::Translation<'a>,
}

pub(crate) trait StateTranslation<'a, D: Decoder>: Sized {
    type PlainDecoder;

    fn new(
        decoder: &D,
        page: &'a DataPage,
        dict: Option<&'a D::Dict>,
        page_validity: Option<&PageValidity<'a>>,
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
        page_validity: &mut Option<PageValidity<'a>>,
        dict: Option<&'a D::Dict>,
        additional: usize,
    ) -> ParquetResult<()>;
}

impl<'a, D: Decoder> State<'a, D> {
    pub fn new(decoder: &D, page: &'a DataPage, dict: Option<&'a D::Dict>) -> ParquetResult<Self> {
        let is_optional =
            page.descriptor.primitive_type.field_info.repetition == Repetition::Optional;

        let mut page_validity = is_optional
            .then(|| page_validity_decoder(page))
            .transpose()?;

        // Make the page_validity None if there are no nulls in the page
        let null_count = page
            .null_count()
            .map(Ok)
            .or_else(|| page_validity.as_ref().map(hybrid_rle_count_zeros))
            .transpose()?;
        if null_count == Some(0) {
            page_validity = None;
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
    ) -> ParquetResult<Self> {
        let translation = D::Translation::new(decoder, page, dict, None)?;

        Ok(Self {
            dict,
            translation,

            // Nested values may be optional, but all that is handled elsewhere.
            is_optional: false,
            page_validity: None,
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

        let n = self
            .page_validity
            .as_mut()
            .map_or(ParquetResult::Ok(n), |page_validity| {
                let mut zc = ZeroCount::default();
                page_validity.gather_n_into(&mut zc, n, &ZeroCountGatherer)?;
                Ok(zc.num_nonzero)
            })?;

        self.translation.skip_in_place(n)
    }

    pub fn extend_from_state(
        &mut self,
        decoder: &mut D,
        decoded: &mut D::DecodedState,
        filter: Option<Filter>,
    ) -> ParquetResult<()> {
        match filter {
            None => {
                let num_rows = self.len();

                if num_rows == 0 {
                    return Ok(());
                }

                self.translation.extend_from_state(
                    decoder,
                    decoded,
                    self.is_optional,
                    &mut self.page_validity,
                    self.dict,
                    num_rows,
                )
            },
            Some(filter) => match filter {
                Filter::Range(range) => {
                    let start = range.start;
                    let end = range.end;

                    self.skip_in_place(start)?;
                    debug_assert!(end - start <= self.len());

                    if end - start > 0 {
                        self.translation.extend_from_state(
                            decoder,
                            decoded,
                            self.is_optional,
                            &mut self.page_validity,
                            self.dict,
                            end - start,
                        )?;
                    }

                    Ok(())
                },
                Filter::Mask(bitmap) => {
                    debug_assert!(bitmap.len() == self.len());

                    let mut iter = bitmap.iter();
                    while iter.num_remaining() > 0 && self.len() > 0 {
                        let prev_state_len = self.len();

                        let num_ones = iter.take_leading_ones();

                        if num_ones > 0 {
                            self.translation.extend_from_state(
                                decoder,
                                decoded,
                                self.is_optional,
                                &mut self.page_validity,
                                self.dict,
                                num_ones,
                            )?;
                        }

                        if iter.num_remaining() == 0 || self.len() == 0 {
                            break;
                        }

                        let num_zeros = iter.take_leading_zeros();
                        self.skip_in_place(num_zeros)?;

                        assert!(
                            prev_state_len != self.len(),
                            "No forward progress was booked in a filtered parquet file."
                        );
                    }

                    Ok(())
                },
            },
        }
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

struct BatchGatherer<'a, I, T, C: BatchableCollector<I, T>>(
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
    page_validity: &mut PageValidity,
    limit: Option<usize>,
    target: &mut T,
    collector: C,
) -> ParquetResult<()> {
    let num_elements = limit.map_or(page_validity.len(), |limit| limit.min(page_validity.len()));

    validity.reserve(num_elements);
    C::reserve(target, num_elements);

    let batched_collector = BatchedCollector::new(collector, target);
    let mut target = (validity, batched_collector);
    let gatherer = BatchGatherer(Default::default());

    page_validity.gather_n_into(&mut target, num_elements, &gatherer)?;

    target.1.finalize()?;

    Ok(())
}

/// This translates and collects items from a [`HybridRleDecoder`] into a target [`Vec`].
///
/// This batches sequential collect operations to try and prevent unnecessary buffering.
pub struct TranslatedHybridRle<'a, 'b, 'c, O, T>
where
    O: Clone + Default,
    T: Translator<O>,
{
    decoder: &'a mut HybridRleDecoder<'b>,
    translator: &'c T,
    _pd: std::marker::PhantomData<O>,
}

impl<'a, 'b, 'c, O, T> TranslatedHybridRle<'a, 'b, 'c, O, T>
where
    O: Clone + Default,
    T: Translator<O>,
{
    pub fn new(decoder: &'a mut HybridRleDecoder<'b>, translator: &'c T) -> Self {
        Self {
            decoder,
            translator,
            _pd: Default::default(),
        }
    }
}

impl<'a, 'b, 'c, O, T> BatchableCollector<u32, Vec<O>> for TranslatedHybridRle<'a, 'b, 'c, O, T>
where
    O: Clone + Default,
    T: Translator<O>,
{
    #[inline]
    fn reserve(target: &mut Vec<O>, n: usize) {
        target.reserve(n);
    }

    #[inline]
    fn push_n(&mut self, target: &mut Vec<O>, n: usize) -> ParquetResult<()> {
        self.decoder
            .translate_and_collect_n_into(target, n, self.translator)
    }

    #[inline]
    fn push_n_nulls(&mut self, target: &mut Vec<O>, n: usize) -> ParquetResult<()> {
        target.resize(target.len() + n, O::default());
        Ok(())
    }

    #[inline]
    fn skip_in_place(&mut self, n: usize) -> ParquetResult<()> {
        self.decoder.skip_in_place(n)
    }
}

pub struct GatheredHybridRle<'a, 'b, 'c, O, G>
where
    O: Clone,
    G: HybridRleGatherer<O>,
{
    decoder: &'a mut HybridRleDecoder<'b>,
    gatherer: &'c G,
    null_value: O,
    _pd: std::marker::PhantomData<O>,
}

impl<'a, 'b, 'c, O, G> GatheredHybridRle<'a, 'b, 'c, O, G>
where
    O: Clone,
    G: HybridRleGatherer<O>,
{
    pub fn new(decoder: &'a mut HybridRleDecoder<'b>, gatherer: &'c G, null_value: O) -> Self {
        Self {
            decoder,
            gatherer,
            null_value,
            _pd: Default::default(),
        }
    }
}

impl<'a, 'b, 'c, O, G> BatchableCollector<u8, Vec<u8>> for GatheredHybridRle<'a, 'b, 'c, O, G>
where
    O: Clone,
    G: HybridRleGatherer<O, Target = Vec<u8>>,
{
    #[inline]
    fn reserve(target: &mut Vec<u8>, n: usize) {
        target.reserve(n);
    }

    #[inline]
    fn push_n(&mut self, target: &mut Vec<u8>, n: usize) -> ParquetResult<()> {
        self.decoder.gather_n_into(target, n, self.gatherer)?;
        Ok(())
    }

    #[inline]
    fn push_n_nulls(&mut self, target: &mut Vec<u8>, n: usize) -> ParquetResult<()> {
        self.gatherer
            .gather_repeated(target, self.null_value.clone(), n)?;
        Ok(())
    }

    #[inline]
    fn skip_in_place(&mut self, n: usize) -> ParquetResult<()> {
        self.decoder.skip_in_place(n)
    }
}

impl<'a, 'b, 'c, T> BatchableCollector<u32, MutableBinaryViewArray<[u8]>>
    for TranslatedHybridRle<'a, 'b, 'c, View, T>
where
    T: Translator<View>,
{
    #[inline]
    fn reserve(target: &mut MutableBinaryViewArray<[u8]>, n: usize) {
        target.reserve(n);
    }

    #[inline]
    fn push_n(&mut self, target: &mut MutableBinaryViewArray<[u8]>, n: usize) -> ParquetResult<()> {
        self.decoder.translate_and_collect_n_into(
            unsafe { target.views_mut() },
            n,
            self.translator,
        )?;

        if let Some(validity) = target.validity() {
            validity.extend_constant(n, true);
        }

        Ok(())
    }

    #[inline]
    fn push_n_nulls(
        &mut self,
        target: &mut MutableBinaryViewArray<[u8]>,
        n: usize,
    ) -> ParquetResult<()> {
        target.extend_null(n);
        Ok(())
    }

    #[inline]
    fn skip_in_place(&mut self, n: usize) -> ParquetResult<()> {
        self.decoder.skip_in_place(n)
    }
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
    fn deserialize_dict(&self, page: DictPage) -> ParquetResult<Self::Dict>;

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
        page_validity: Option<&mut PageValidity<'a>>,
        limit: usize,
    ) -> ParquetResult<()>;
    fn decode_dictionary_encoded<'a>(
        &mut self,
        decoded: &mut Self::DecodedState,
        page_values: &mut HybridRleDecoder<'a>,
        is_optional: bool,
        page_validity: Option<&mut PageValidity<'a>>,
        dict: &Self::Dict,
        limit: usize,
    ) -> ParquetResult<()>;

    fn finalize(
        &self,
        data_type: ArrowDataType,
        dict: Option<Self::Dict>,
        decoded: Self::DecodedState,
    ) -> ParquetResult<Self::Output>;
}

pub(crate) trait NestedDecoder: Decoder {
    fn validity_extend(
        state: &mut State<'_, Self>,
        decoded: &mut Self::DecodedState,
        value: bool,
        n: usize,
    );
    fn values_extend_nulls(state: &mut State<'_, Self>, decoded: &mut Self::DecodedState, n: usize);

    fn push_n_valids(
        &mut self,
        state: &mut State<'_, Self>,
        decoded: &mut Self::DecodedState,
        n: usize,
    ) -> ParquetResult<()> {
        state.extend_from_state(self, decoded, Some(Filter::new_limited(n)))?;
        Self::validity_extend(state, decoded, true, n);

        Ok(())
    }

    fn push_n_nulls(
        &self,
        state: &mut State<'_, Self>,
        decoded: &mut Self::DecodedState,
        n: usize,
    ) {
        Self::validity_extend(state, decoded, false, n);
        Self::values_extend_nulls(state, decoded, n);
    }
}

pub trait DictDecodable: Decoder {
    fn finalize_dict_array<K: DictionaryKey>(
        &self,
        data_type: ArrowDataType,
        dict: Self::Dict,
        keys: PrimitiveArray<K>,
    ) -> ParquetResult<DictionaryArray<K>>;
}

pub struct PageDecoder<D: Decoder> {
    pub iter: BasicDecompressor,
    pub data_type: ArrowDataType,
    pub dict: Option<D::Dict>,
    pub decoder: D,
}

impl<D: Decoder> PageDecoder<D> {
    pub fn new(
        mut iter: BasicDecompressor,
        data_type: ArrowDataType,
        decoder: D,
    ) -> ParquetResult<Self> {
        let dict_page = iter.read_dict_page()?;
        let dict = dict_page.map(|d| decoder.deserialize_dict(d)).transpose()?;

        Ok(Self {
            iter,
            data_type,
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

            let mut state = State::new(&self.decoder, &page, self.dict.as_ref())?;

            let start_length = target.len();
            state.extend_from_state(&mut self.decoder, &mut target, state_filter)?;
            let end_length = target.len();

            num_rows_remaining -= end_length - start_length;

            debug_assert!(state.len() == 0 || num_rows_remaining == 0);

            drop(state);
            self.iter.reuse_page_buffer(page);
        }

        self.decoder.finalize(self.data_type, self.dict, target)
    }
}

#[inline]
pub(super) fn dict_indices_decoder(page: &DataPage) -> ParquetResult<hybrid_rle::HybridRleDecoder> {
    let indices_buffer = split_buffer(page)?.values;

    // SPEC: Data page format: the bit width used to encode the entry ids stored as 1 byte (max bit width = 32),
    // SPEC: followed by the values encoded using RLE/Bit packed described above (with the given bit width).
    let bit_width = indices_buffer[0];
    let indices_buffer = &indices_buffer[1..];

    Ok(hybrid_rle::HybridRleDecoder::new(
        indices_buffer,
        bit_width as u32,
        page.num_values(),
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
