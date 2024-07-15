use std::collections::VecDeque;

pub(crate) mod array_chunks;
pub(crate) mod filter;

use arrow::array::{BinaryArray, MutableBinaryViewArray, View};
use arrow::bitmap::MutableBitmap;
use arrow::pushable::Pushable;
use polars_error::{polars_err, PolarsError, PolarsResult};

use self::filter::Filter;
use super::super::PagesIter;
use crate::parquet::encoding::hybrid_rle::gatherer::{
    HybridRleGatherer, ZeroCount, ZeroCountGatherer,
};
use crate::parquet::encoding::hybrid_rle::{self, HybridRleDecoder, Translator};
use crate::parquet::error::ParquetResult;
use crate::parquet::page::{split_buffer, DataPage, DictPage, Page};
use crate::parquet::schema::Repetition;

#[derive(Debug)]
pub(crate) struct State<'a, D: Decoder<'a>, T: StateTranslation<'a, D>> {
    pub(crate) page_validity: Option<PageValidity<'a>>,
    pub(crate) translation: T,
    pub(crate) filter: Option<Filter<'a>>,
    _pd: std::marker::PhantomData<D>,
}

pub(crate) trait StateTranslation<'a, D: Decoder<'a>>: Sized {
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
    /// It guarantees that the length of `decoded` is at most `decoded.len() + remaining`.
    fn extend_from_state(
        &mut self,
        decoder: &D,
        decoded: &mut D::DecodedState,
        page_validity: &mut Option<PageValidity<'a>>,
        additional: usize,
    ) -> ParquetResult<()>;
}

impl<'a, D: Decoder<'a>, T: StateTranslation<'a, D>> State<'a, D, T> {
    pub fn new(decoder: &D, page: &'a DataPage, dict: Option<&'a D::Dict>) -> PolarsResult<Self> {
        let is_optional =
            page.descriptor.primitive_type.field_info.repetition == Repetition::Optional;
        let is_filtered = page.selected_rows().is_some();

        let page_validity = is_optional
            .then(|| page_validity_decoder(page))
            .transpose()?;
        let filter = is_filtered.then(|| Filter::new(page)).flatten();

        let translation = T::new(decoder, page, dict, page_validity.as_ref(), filter.as_ref())?;

        Ok(Self {
            page_validity,
            translation,
            filter,
            _pd: std::marker::PhantomData,
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
        decoder: &D,
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

/// The state of a partially deserialized page
pub(super) trait PageState<'a>: std::fmt::Debug {
    fn len(&self) -> usize;
}

/// The state of a partially deserialized page
pub(super) trait DecodedState: std::fmt::Debug {
    // the number of values that the state already has
    fn len(&self) -> usize;
}

/// A decoder that knows how to map `State` -> Array
pub(super) trait Decoder<'a>: Sized {
    // @TODO: Remove Translation
    /// The state that this decoder derives from a [`DataPage`]. This is bound to the page.
    type Translation: StateTranslation<'a, Self>;
    /// The dictionary representation that the decoder uses
    type Dict;
    /// The target state that this Decoder decodes into.
    type DecodedState: DecodedState;

    /// Initializes a new [`Self::DecodedState`].
    fn with_capacity(&self, capacity: usize) -> Self::DecodedState;

    /// Deserializes a [`DictPage`] into [`Self::Dict`].
    fn deserialize_dict(&self, page: &DictPage) -> Self::Dict;
}

pub(super) fn extend_from_new_page<'a, T: Decoder<'a>>(
    mut page: State<'a, T, T::Translation>,
    chunk_size: Option<usize>,
    items: &mut VecDeque<T::DecodedState>,
    remaining: &mut usize,
    decoder: &T,
) -> PolarsResult<()> {
    let capacity = std::cmp::min(chunk_size.unwrap_or(0), *remaining);
    let chunk_size = chunk_size.unwrap_or(usize::MAX);

    let mut decoded = if let Some(decoded) = items.pop_back() {
        decoded
    } else {
        // there is no state => initialize it
        decoder.with_capacity(capacity)
    };
    let existing = decoded.len();

    let additional = (chunk_size - existing).min(*remaining);

    page.extend_from_state(decoder, &mut decoded, additional)?;
    *remaining -= decoded.len() - existing;
    items.push_back(decoded);

    while page.len() > 0 && *remaining > 0 {
        let additional = chunk_size.min(*remaining);

        let mut decoded = decoder.with_capacity(additional);
        let len_before = decoded.len();
        page.extend_from_state(decoder, &mut decoded, additional)?;
        assert!(
            len_before != decoded.len() || additional == 0,
            "No progress booked"
        );
        *remaining -= decoded.len();
        items.push_back(decoded)
    }
    Ok(())
}

/// Represents what happened when a new page was consumed
#[derive(Debug)]
pub enum MaybeNext<P> {
    /// Whether the page was sufficient to fill `chunk_size`
    Some(P),
    /// whether there are no more pages or intermediary decoded states
    None,
    /// Whether the page was insufficient to fill `chunk_size` and a new page is required
    More,
}

#[inline]
pub(super) fn next<'a, I: PagesIter, D: Decoder<'a>>(
    iter: &'a mut I,
    items: &'a mut VecDeque<D::DecodedState>,
    dict: &'a mut Option<D::Dict>,
    remaining: &'a mut usize,
    chunk_size: Option<usize>,
    decoder: &'a D,
) -> MaybeNext<PolarsResult<D::DecodedState>> {
    // front[a1, a2, a3, ...]back
    if items.len() > 1 {
        return MaybeNext::Some(Ok(items.pop_front().unwrap()));
    }
    if (items.len() == 1) && items.front().unwrap().len() == chunk_size.unwrap_or(usize::MAX) {
        return MaybeNext::Some(Ok(items.pop_front().unwrap()));
    }
    if *remaining == 0 {
        return match items.pop_front() {
            Some(decoded) => MaybeNext::Some(Ok(decoded)),
            None => MaybeNext::None,
        };
    }

    match iter.next() {
        Err(e) => MaybeNext::Some(Err(e.into())),
        Ok(Some(page)) => {
            let page = match page {
                Page::Data(ref page) => page,
                Page::Dict(ref dict_page) => {
                    *dict = Some(decoder.deserialize_dict(dict_page));
                    return MaybeNext::More;
                },
            };

            // there is a new page => consume the page from the start
            let maybe_page = State::new(decoder, page, dict.as_ref());
            let page = match maybe_page {
                Ok(page) => page,
                Err(e) => return MaybeNext::Some(Err(e)),
            };

            if let Err(e) = extend_from_new_page(page, chunk_size, items, remaining, decoder) {
                return MaybeNext::Some(Err(e));
            }

            if (items.len() == 1) && items.front().unwrap().len() < chunk_size.unwrap_or(usize::MAX)
            {
                MaybeNext::More
            } else {
                let decoded = items.pop_front().unwrap();
                MaybeNext::Some(Ok(decoded))
            }
        },
        Ok(None) => {
            if let Some(decoded) = items.pop_front() {
                // we have a populated item and no more pages
                // the only case where an item's length may be smaller than chunk_size
                debug_assert!(decoded.len() <= chunk_size.unwrap_or(usize::MAX));
                MaybeNext::Some(Ok(decoded))
            } else {
                MaybeNext::None
            }
        },
    }
}

#[inline]
pub(super) fn dict_indices_decoder(page: &DataPage) -> PolarsResult<hybrid_rle::HybridRleDecoder> {
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
                let (offset, _) = dict.offsets().start_end(i);
                let buffer_idx =
                    buffer_idx.get_or_insert_with(|| values.push_buffer(dict.values().clone()));

                debug_assert!(offset <= u32::MAX as usize);
                View::new_from_bytes(value, *buffer_idx, offset as u32)
            }
        })
        .collect()
}

pub(super) fn page_is_optional(page: &DataPage) -> bool {
    page.descriptor.primitive_type.field_info.repetition == Repetition::Optional
}

pub(super) fn page_is_filtered(page: &DataPage) -> bool {
    page.selected_rows().is_some()
}
