use std::collections::VecDeque;

use arrow::array::{MutableBinaryViewArray, View};
use arrow::bitmap::utils::BitmapIter;
use arrow::bitmap::MutableBitmap;
use arrow::pushable::Pushable;
use polars_error::{polars_err, PolarsError, PolarsResult};

use super::super::PagesIter;
use crate::parquet::deserialize::{
    FilteredHybridEncoded, FilteredHybridRleDecoderIter, HybridDecoderBitmapIter, HybridEncoded,
};
use crate::parquet::encoding::hybrid_rle::{self, HybridRleDecoder, Translator};
use crate::parquet::error::ParquetResult;
use crate::parquet::indexes::Interval;
use crate::parquet::page::{split_buffer, DataPage, DictPage, Page};
use crate::parquet::schema::Repetition;

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

/// The state of a partially deserialized page
pub(super) trait PageValidity<'a> {
    fn next_limited(&mut self, limit: usize) -> Option<FilteredHybridEncoded<'a>>;
}

pub trait BatchableCollector<I, T> {
    fn reserve(target: &mut T, n: usize);
    fn push_n(&mut self, target: &mut T, n: usize) -> ParquetResult<()>;
    fn push_n_nulls(&mut self, target: &mut T, n: usize) -> ParquetResult<()>;
    fn skip_n(&mut self, n: usize) -> ParquetResult<()>;
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
        if n == 0 {
            return Ok(());
        }

        if self.num_waiting_valids > 0 {
            self.collector
                .push_n(self.target, self.num_waiting_valids)?;
        }
        if self.num_waiting_invalids > 0 {
            self.collector
                .push_n_nulls(self.target, self.num_waiting_invalids)?;
        }
        self.collector.skip_n(n)?;

        self.num_waiting_valids = 0;
        self.num_waiting_invalids = 0;

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

#[derive(Debug, Clone)]
pub struct FilteredOptionalPageValidity<'a> {
    iter: FilteredHybridRleDecoderIter<'a>,
    current: Option<(FilteredHybridEncoded<'a>, usize)>,
}

impl<'a> FilteredOptionalPageValidity<'a> {
    pub fn try_new(page: &'a DataPage) -> PolarsResult<Self> {
        let validity = split_buffer(page)?.def;

        let iter = hybrid_rle::Decoder::new(validity, 1);
        let iter = HybridDecoderBitmapIter::new(iter, page.num_values());
        let selected_rows = get_selected_rows(page);
        let iter = FilteredHybridRleDecoderIter::new(iter, selected_rows);

        Ok(Self {
            iter,
            current: None,
        })
    }

    pub fn len(&self) -> usize {
        self.iter.len()
    }
}

pub fn get_selected_rows(page: &DataPage) -> VecDeque<Interval> {
    page.selected_rows()
        .unwrap_or(&[Interval::new(0, page.num_values())])
        .iter()
        .copied()
        .collect()
}

impl<'a> PageValidity<'a> for FilteredOptionalPageValidity<'a> {
    fn next_limited(&mut self, limit: usize) -> Option<FilteredHybridEncoded<'a>> {
        let (run, own_offset) = if let Some((run, offset)) = self.current {
            (run, offset)
        } else {
            // a new run
            let run = self.iter.next()?; // no run -> None
            self.current = Some((run, 0));
            return self.next_limited(limit);
        };

        match run {
            FilteredHybridEncoded::Bitmap {
                values,
                offset,
                length,
            } => {
                let run_length = length - own_offset;

                let length = limit.min(run_length);

                if length == run_length {
                    self.current = None;
                } else {
                    self.current = Some((run, own_offset + length));
                }

                Some(FilteredHybridEncoded::Bitmap {
                    values,
                    offset,
                    length,
                })
            },
            FilteredHybridEncoded::Repeated { is_set, length } => {
                let run_length = length - own_offset;

                let length = limit.min(run_length);

                if length == run_length {
                    self.current = None;
                } else {
                    self.current = Some((run, own_offset + length));
                }

                Some(FilteredHybridEncoded::Repeated { is_set, length })
            },
            FilteredHybridEncoded::Skipped(set) => {
                self.current = None;
                Some(FilteredHybridEncoded::Skipped(set))
            },
        }
    }
}

#[derive(Debug, Clone)]
pub struct OptionalPageValidity<'a> {
    iter: HybridDecoderBitmapIter<'a>,
    current: Option<(HybridEncoded<'a>, usize)>,
}

impl<'a> OptionalPageValidity<'a> {
    pub fn try_new(page: &'a DataPage) -> PolarsResult<Self> {
        let validity = split_buffer(page)?.def;

        let iter = hybrid_rle::Decoder::new(validity, 1);
        let iter = HybridDecoderBitmapIter::new(iter, page.num_values());
        Ok(Self {
            iter,
            current: None,
        })
    }

    /// Number of items remaining
    pub fn len(&self) -> usize {
        self.iter.len()
            + self
                .current
                .as_ref()
                .map(|(run, offset)| run.len() - offset)
                .unwrap_or_default()
    }

    fn next_limited(&mut self, limit: usize) -> Option<FilteredHybridEncoded<'a>> {
        let (run, offset) = if let Some((run, offset)) = self.current {
            (run, offset)
        } else {
            // a new run
            let run = self.iter.next()?; // no run -> None
            self.current = Some((run, 0));
            return self.next_limited(limit);
        };

        match run {
            HybridEncoded::Bitmap(values, length) => {
                let run_length = length - offset;

                let length = limit.min(run_length);

                if length == run_length {
                    self.current = None;
                } else {
                    self.current = Some((run, offset + length));
                }

                Some(FilteredHybridEncoded::Bitmap {
                    values,
                    offset,
                    length,
                })
            },
            HybridEncoded::Repeated(is_set, run_length) => {
                let run_length = run_length - offset;

                let length = limit.min(run_length);

                if length == run_length {
                    self.current = None;
                } else {
                    self.current = Some((run, offset + length));
                }

                Some(FilteredHybridEncoded::Repeated { is_set, length })
            },
        }
    }
}

impl<'a> PageValidity<'a> for OptionalPageValidity<'a> {
    fn next_limited(&mut self, limit: usize) -> Option<FilteredHybridEncoded<'a>> {
        self.next_limited(limit)
    }
}

fn reserve_pushable_and_validity<'a, I, T, C: BatchableCollector<I, T>>(
    validity: &mut MutableBitmap,
    page_validity: &'a mut dyn PageValidity,
    limit: Option<usize>,
    target: &mut T,
) -> Vec<FilteredHybridEncoded<'a>> {
    let limit = limit.unwrap_or(usize::MAX);

    let mut runs = vec![];
    let mut remaining = limit;
    let mut reserve_pushable = 0;

    // first do a scan so that we know how much to reserve up front
    while remaining > 0 {
        let run = page_validity.next_limited(remaining);
        let run = if let Some(run) = run { run } else { break };

        match run {
            FilteredHybridEncoded::Bitmap { length, .. } => {
                reserve_pushable += length;
                remaining -= length;
            },
            FilteredHybridEncoded::Repeated { length, .. } => {
                reserve_pushable += length;
                remaining -= length;
            },
            _ => {},
        };
        runs.push(run)
    }
    C::reserve(target, reserve_pushable);
    validity.reserve(reserve_pushable);
    runs
}

/// Extends a [`Pushable`] from an iterator of non-null values and an hybrid-rle decoder
pub(super) fn extend_from_decoder<I, T, C: BatchableCollector<I, T>>(
    validity: &mut MutableBitmap,
    page_validity: &mut dyn PageValidity,
    limit: Option<usize>,
    target: &mut T,
    collector: C,
) -> ParquetResult<()> {
    let runs = reserve_pushable_and_validity::<I, T, C>(validity, page_validity, limit, target);

    let mut batched_collector = BatchedCollector::new(collector, target);

    // then a second loop to really fill the buffers
    for run in runs {
        match run {
            FilteredHybridEncoded::Bitmap {
                values,
                offset,
                length,
            } => {
                // consume `length` items
                let mut validity_iter = BitmapIter::new(values, offset, length);

                let mut bit_sum = 0;
                while validity_iter.num_remaining() != 0 {
                    let num_valid = validity_iter.take_leading_ones();
                    bit_sum += num_valid;
                    batched_collector.push_n_valids(num_valid)?;

                    let num_invalid = validity_iter.take_leading_zeros();
                    bit_sum += num_invalid;
                    batched_collector.push_n_invalids(num_invalid);
                }

                debug_assert_eq!(bit_sum, length);

                validity.extend_from_slice(values, offset, length);
            },
            FilteredHybridEncoded::Repeated { is_set, length } => {
                validity.extend_constant(length, is_set);
                if is_set {
                    batched_collector.push_n_valids(length)?;
                } else {
                    batched_collector.push_n_invalids(length);
                }
            },
            FilteredHybridEncoded::Skipped(valids) => batched_collector.skip_in_place(valids)?,
        };
    }

    batched_collector.finalize()?;

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
    fn skip_n(&mut self, n: usize) -> ParquetResult<()> {
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

    #[inline]
    fn skip_n(&mut self, n: usize) -> ParquetResult<()> {
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
    fn skip_n(&mut self, n: usize) -> ParquetResult<()> {
        if n == 0 {
            return Ok(());
        }

        _ = self.nth(n);
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
pub(super) trait Decoder<'a> {
    /// The state that this decoder derives from a [`DataPage`]. This is bound to the page.
    type State: PageState<'a>;
    /// The dictionary representation that the decoder uses
    type Dict;
    /// The target state that this Decoder decodes into.
    type DecodedState: DecodedState;

    /// Creates a new `Self::State`
    fn build_state(
        &self,
        page: &'a DataPage,
        dict: Option<&'a Self::Dict>,
    ) -> PolarsResult<Self::State>;

    /// Initializes a new [`Self::DecodedState`].
    fn with_capacity(&self, capacity: usize) -> Self::DecodedState;

    /// extends [`Self::DecodedState`] by deserializing items in [`Self::State`].
    /// It guarantees that the length of `decoded` is at most `decoded.len() + remaining`.
    fn extend_from_state(
        &self,
        page: &mut Self::State,
        decoded: &mut Self::DecodedState,
        additional: usize,
    ) -> PolarsResult<()>;

    /// Deserializes a [`DictPage`] into [`Self::Dict`].
    fn deserialize_dict(&self, page: &DictPage) -> Self::Dict;
}

pub(super) fn extend_from_new_page<'a, T: Decoder<'a>>(
    mut page: T::State,
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

    decoder.extend_from_state(&mut page, &mut decoded, additional)?;
    *remaining -= decoded.len() - existing;
    items.push_back(decoded);

    while page.len() > 0 && *remaining > 0 {
        let additional = chunk_size.min(*remaining);

        let mut decoded = decoder.with_capacity(additional);
        decoder.extend_from_state(&mut page, &mut decoded, additional)?;
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
            let maybe_page = decoder.build_state(page, dict.as_ref());
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

pub(super) fn page_is_optional(page: &DataPage) -> bool {
    page.descriptor.primitive_type.field_info.repetition == Repetition::Optional
}

pub(super) fn page_is_filtered(page: &DataPage) -> bool {
    page.selected_rows().is_some()
}
