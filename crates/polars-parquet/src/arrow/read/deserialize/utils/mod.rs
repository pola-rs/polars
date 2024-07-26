pub(crate) mod array_chunks;
pub(crate) mod filter;

use arrow::array::{
    Array, BinaryArray, DictionaryArray, DictionaryKey, MutableBinaryViewArray, View,
};
use arrow::bitmap::{Bitmap, MutableBitmap};
use arrow::datatypes::ArrowDataType;
use arrow::pushable::Pushable;
use arrow::types::Offset;
use polars_error::{polars_err, PolarsError, PolarsResult};

use self::filter::Filter;
use super::binary::utils::Binary;
use super::{BasicDecompressor, CompressedPagesIter, ParquetError};
use crate::parquet::encoding::hybrid_rle::gatherer::{
    HybridRleGatherer, ZeroCount, ZeroCountGatherer,
};
use crate::parquet::encoding::hybrid_rle::{self, HybridRleDecoder, Translator};
use crate::parquet::error::ParquetResult;
use crate::parquet::page::{split_buffer, DataPage, DictPage, Page};
use crate::parquet::schema::Repetition;
use crate::read::deserialize::dictionary::DictionaryDecoder;

#[derive(Debug)]
pub(crate) struct State<'a, D: Decoder> {
    pub(crate) page_validity: Option<PageValidity<'a>>,
    pub(crate) translation: D::Translation<'a>,
    pub(crate) filter: Option<Filter<'a>>,
}

pub(crate) trait StateTranslation<'a, D: Decoder>: Sized {
    type PlainDecoder;

    fn new(
        decoder: &D,
        page: &'a DataPage,
        dict: Option<&'a D::Dict>,
        page_validity: Option<&PageValidity<'a>>,
        filter: Option<&Filter<'a>>,
    ) -> PolarsResult<Self>;
    fn len_when_not_nullable(&self) -> usize;
    fn skip_in_place(&mut self, n: usize) -> ParquetResult<()>;

    /// extends [`Self::DecodedState`] by deserializing items in [`Self::State`].
    /// It guarantees that the length of `decoded` is at most `decoded.len() + additional`.
    fn extend_from_state(
        &mut self,
        decoder: &mut D,
        decoded: &mut D::DecodedState,
        page_validity: &mut Option<PageValidity<'a>>,
        additional: usize,
    ) -> ParquetResult<()>;
}

impl<'a, D: Decoder> State<'a, D> {
    pub fn new(decoder: &D, page: &'a DataPage, dict: Option<&'a D::Dict>) -> PolarsResult<Self> {
        let is_optional =
            page.descriptor.primitive_type.field_info.repetition == Repetition::Optional;
        let is_filtered = page.selected_rows().is_some();

        let page_validity = is_optional
            .then(|| page_validity_decoder(page))
            .transpose()?;
        let filter = is_filtered.then(|| Filter::new(page)).flatten();

        let translation =
            D::Translation::new(decoder, page, dict, page_validity.as_ref(), filter.as_ref())?;

        Ok(Self {
            page_validity,
            translation,
            filter,
        })
    }

    pub fn new_nested(
        decoder: &D,
        page: &'a DataPage,
        dict: Option<&'a D::Dict>,
    ) -> PolarsResult<Self> {
        let translation = D::Translation::new(decoder, page, dict, None, None)?;

        Ok(Self {
            translation,
            page_validity: None,
            filter: None,
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
        additional: usize,
    ) -> ParquetResult<()> {
        // @TODO: Taking the filter here is a bit unfortunate. Since each error leaves the filter
        // empty.
        let filter = self.filter.take();

        match filter {
            None => self.translation.extend_from_state(
                decoder,
                decoded,
                &mut self.page_validity,
                additional,
            ),
            Some(mut filter) => {
                let mut n = additional;
                while n > 0 && self.len() > 0 {
                    let prev_n = n;
                    let prev_state_len = self.len();

                    // Skip over all intervals that we have already passed or that are length == 0.
                    while filter
                        .selected_rows
                        .get(filter.current_interval)
                        .is_some_and(|iv| {
                            iv.length == 0 || iv.start + iv.length <= filter.current_index
                        })
                    {
                        filter.current_interval += 1;
                    }

                    let Some(iv) = filter.selected_rows.get(filter.current_interval) else {
                        self.skip_in_place(self.len())?;
                        self.filter = Some(filter);
                        return Ok(());
                    };

                    // Move to at least the start of the interval
                    if filter.current_index < iv.start {
                        self.skip_in_place(iv.start - filter.current_index)?;
                        filter.current_index = iv.start;
                    }

                    let n_this_round = usize::min(iv.start + iv.length - filter.current_index, n);

                    self.translation.extend_from_state(
                        decoder,
                        decoded,
                        &mut self.page_validity,
                        n_this_round,
                    )?;

                    let iv = &filter.selected_rows[filter.current_interval];
                    filter.current_index += n_this_round;
                    if filter.current_index >= iv.start + iv.length {
                        filter.current_interval += 1;
                    }

                    n -= n_this_round;

                    assert!(
                        prev_n != n || prev_state_len != self.len(),
                        "No forward progress was booked in a filtered parquet file."
                    );
                }

                self.filter = Some(filter);
                Ok(())
            },
        }
    }
}

pub fn not_implemented(page: &DataPage) -> PolarsError {
    let is_optional = page.descriptor.primitive_type.field_info.repetition == Repetition::Optional;
    let is_filtered = page.selected_rows().is_some();
    let required = if is_optional { "optional" } else { "required" };
    let is_filtered = if is_filtered { ", index-filtered" } else { "" };
    polars_err!(ComputeError:
        "Decoding {:?} \"{:?}\"-encoded {} {} parquet pages not yet implemented",
        page.descriptor.primitive_type.physical_type,
        page.encoding(),
        required,
        is_filtered,
    )
}

pub trait BatchableCollector<I, T> {
    fn reserve(target: &mut T, n: usize);
    fn push_n(&mut self, target: &mut T, n: usize) -> ParquetResult<()>;
    fn push_n_nulls(&mut self, target: &mut T, n: usize) -> ParquetResult<()>;
}

/// This batches sequential collect operations to try and prevent unnecessary buffering and
/// `Iterator::next` polling.
#[must_use]
pub struct BatchedCollector<'a, I, T, C: BatchableCollector<I, T>> {
    num_waiting_valids: usize,
    num_waiting_invalids: usize,

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
}

impl<'a, 'b, 'c, O, Out, G> BatchableCollector<u8, Binary<O>>
    for GatheredHybridRle<'a, 'b, 'c, Out, G>
where
    O: Offset,
    Out: Clone,
    G: HybridRleGatherer<Out, Target = Binary<O>>,
{
    #[inline]
    fn reserve(target: &mut Binary<O>, n: usize) {
        target.offsets.reserve(n);
        target.values.reserve(n);
    }

    #[inline]
    fn push_n(&mut self, target: &mut Binary<O>, n: usize) -> ParquetResult<()> {
        self.decoder.gather_n_into(target, n, self.gatherer)?;
        Ok(())
    }

    #[inline]
    fn push_n_nulls(&mut self, target: &mut Binary<O>, n: usize) -> ParquetResult<()> {
        self.gatherer
            .gather_repeated(target, self.null_value.clone(), n)?;
        Ok(())
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
        self.decoder
            .translate_and_collect_n_into(target.views_mut(), n, self.translator)?;

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

    /// Initializes a new [`Self::DecodedState`].
    fn with_capacity(&self, capacity: usize) -> Self::DecodedState;

    /// Deserializes a [`DictPage`] into [`Self::Dict`].
    fn deserialize_dict(&self, page: DictPage) -> Self::Dict;

    fn decode_plain_encoded<'a>(
        &mut self,
        decoded: &mut Self::DecodedState,
        page_values: &mut <Self::Translation<'a> as StateTranslation<'a, Self>>::PlainDecoder,
        page_validity: Option<&mut PageValidity<'a>>,
        limit: usize,
    ) -> ParquetResult<()>;
    fn decode_dictionary_encoded<'a>(
        &mut self,
        decoded: &mut Self::DecodedState,
        page_values: &mut HybridRleDecoder<'a>,
        page_validity: Option<&mut PageValidity<'a>>,
        dict: &Self::Dict,
        limit: usize,
    ) -> ParquetResult<()>;

    fn finalize(
        &self,
        data_type: ArrowDataType,
        decoded: Self::DecodedState,
    ) -> ParquetResult<Box<dyn Array>>;

    /// Turn the collected arrays into the final dictionary array.
    fn finalize_dict_array<K: DictionaryKey>(
        &self,
        data_type: ArrowDataType,
        dict: Self::Dict,
        decoded: (Vec<K>, Option<Bitmap>),
    ) -> ParquetResult<DictionaryArray<K>>;
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
        state.extend_from_state(self, decoded, n)?;
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

pub struct PageDecoder<I: CompressedPagesIter, D: Decoder> {
    pub iter: BasicDecompressor<I>,
    pub data_type: ArrowDataType,
    pub dict: Option<D::Dict>,
    pub decoder: D,
}

impl<I: CompressedPagesIter, D: Decoder> PageDecoder<I, D> {
    pub fn new(
        mut iter: BasicDecompressor<I>,
        data_type: ArrowDataType,
        decoder: D,
    ) -> ParquetResult<Self> {
        let dict_page = iter.read_dict_page()?;
        let dict = dict_page.map(|d| decoder.deserialize_dict(d));

        Ok(Self {
            iter,
            data_type,
            dict,
            decoder,
        })
    }

    pub fn collect_n(mut self, limit: usize) -> ParquetResult<Box<dyn Array>> {
        let mut target = self.decoder.with_capacity(limit);
        self.collect_n_into(&mut target, limit)?;
        self.decoder.finalize(self.data_type, target)
    }

    pub fn collect_n_into(
        &mut self,
        target: &mut D::DecodedState,
        mut limit: usize,
    ) -> ParquetResult<usize> {
        use streaming_decompression::FallibleStreamingIterator;

        if limit == 0 {
            return Ok(0);
        }

        let start_limit = limit;

        while limit > 0 {
            let Some(page) = self.iter.next()? else {
                return Ok(start_limit - limit);
            };

            let Page::Data(page) = page else {
                // @TODO This should be removed
                unreachable!();
            };

            let mut state = State::new(&self.decoder, page, self.dict.as_ref())?;
            let start_length = target.len();
            state.extend_from_state(&mut self.decoder, target, limit)?;
            let end_length = target.len();

            limit -= end_length - start_length;

            debug_assert!(state.len() == 0 || limit == 0);
        }

        Ok(start_limit - limit)
    }
}

pub struct PageDictArrayDecoder<I: CompressedPagesIter, K: DictionaryKey, D: Decoder> {
    pub iter: BasicDecompressor<I>,
    pub data_type: ArrowDataType,
    pub dict: D::Dict,
    pub decoder: D,
    _pd: std::marker::PhantomData<K>,
}

impl<I: CompressedPagesIter, K: DictionaryKey, D: Decoder> PageDictArrayDecoder<I, K, D> {
    pub fn new(
        mut iter: BasicDecompressor<I>,
        data_type: ArrowDataType,
        decoder: D,
    ) -> ParquetResult<Self> {
        let dict_page = iter
            .read_dict_page()?
            .ok_or(ParquetError::FeatureNotSupported(
                "Dictionary array without a dictionary page".to_string(),
            ))?;
        let dict = decoder.deserialize_dict(dict_page);

        Ok(Self {
            iter,
            data_type,
            dict,
            decoder,
            _pd: std::marker::PhantomData,
        })
    }

    pub fn collect_n(mut self, limit: usize) -> ParquetResult<DictionaryArray<K>> {
        let mut target = (
            Vec::with_capacity(limit),
            MutableBitmap::with_capacity(limit),
        );
        self.collect_n_into(&mut target, limit)?;
        let (values, validity) = target;
        let validity = if !validity.is_empty() {
            Some(validity.freeze())
        } else {
            None
        };
        self.decoder
            .finalize_dict_array(self.data_type, self.dict, (values, validity))
    }

    pub fn collect_n_into(
        &mut self,
        target: &mut (Vec<K>, MutableBitmap),
        mut limit: usize,
    ) -> ParquetResult<usize> {
        use streaming_decompression::FallibleStreamingIterator;

        if limit == 0 {
            return Ok(0);
        }

        let start_limit = limit;

        while limit > 0 {
            let Some(page) = self.iter.next()? else {
                return Ok(start_limit - limit);
            };

            let Page::Data(page) = page else {
                // @TODO This should be removed
                unreachable!();
            };

            let mut dictionary_decoder = DictionaryDecoder::new(self.dict.len());
            let mut state = State::new(&dictionary_decoder, page, Some(&()))?;
            let start_length = target.len();
            state.extend_from_state(&mut dictionary_decoder, target, limit)?;
            let end_length = target.len();

            limit -= end_length - start_length;

            debug_assert!(state.len() == 0 || limit == 0);
        }

        Ok(start_limit - limit)
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

/// Generate a look-up table of views from a look-up table of values into a `BinaryViewArray`.
///
/// This makes sure to only allocate the necessary buffer space in the `BinaryViewArray` if it is
/// desperately needed.
#[inline]
pub(super) fn binary_views_dict(
    values: &mut MutableBinaryViewArray<[u8]>,
    dict: &BinaryArray<i64>,
) -> Vec<View> {
    // We create a dictionary of views here, so that the views only have be calculated
    // once and are then just a lookup. We also only push the dictionary buffer when we
    // see the first View that cannot be inlined.
    //
    // @TODO: Maybe we can do something smarter here by only pushing the items that are larger than
    // 12 bytes. Maybe, we say if the num_inlined < dict.len() / 2 then push the whole buffer.
    // Otherwise, only push the non-inlinable items.

    let mut buffer_idx = None;
    dict.values_iter()
        .enumerate()
        .map(|(i, value)| {
            if value.len() <= View::MAX_INLINE_SIZE as usize {
                View::new_inline(value)
            } else {
                let (offset_start, offset_end) = dict.offsets().start_end(i);
                debug_assert_eq!(value.len(), offset_end - offset_start);

                let buffer_idx =
                    buffer_idx.get_or_insert_with(|| values.push_buffer(dict.values().clone()));

                debug_assert!(offset_start <= u32::MAX as usize);
                View::new_from_bytes(value, *buffer_idx, offset_start as u32)
            }
        })
        .collect()
}
