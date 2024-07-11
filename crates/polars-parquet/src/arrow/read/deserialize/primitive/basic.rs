use std::collections::VecDeque;

use arrow::array::MutablePrimitiveArray;
use arrow::bitmap::MutableBitmap;
use arrow::datatypes::ArrowDataType;
use arrow::types::NativeType;
use polars_error::PolarsResult;

use super::super::utils::{MaybeNext, OptionalPageValidity};
use super::super::{utils, PagesIter};
use crate::parquet::encoding::hybrid_rle::DictionaryTranslator;
use crate::parquet::encoding::{byte_stream_split, hybrid_rle, Encoding};
use crate::parquet::error::ParquetResult;
use crate::parquet::page::{split_buffer, DataPage, DictPage};
use crate::parquet::types::{decode, NativeType as ParquetNativeType};
use crate::read::deserialize::utils::filter::{
    extend_from_state_with_opt_filter, Filter, SkipInPlace,
};
use crate::read::deserialize::utils::{PageState, TranslatedHybridRle};

#[derive(Debug)]
pub(super) struct Values<'a> {
    pub values: std::slice::ChunksExact<'a, u8>,
}

impl<'a> Values<'a> {
    pub fn try_new<P: ParquetNativeType>(page: &'a DataPage) -> PolarsResult<Self> {
        let values = split_buffer(page)?.values;
        assert_eq!(values.len() % std::mem::size_of::<P>(), 0);
        Ok(Self {
            values: values.chunks_exact(std::mem::size_of::<P>()),
        })
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.values.size_hint().0
    }
}

#[derive(Debug)]
pub(super) struct ValuesDictionary<'a, T: NativeType> {
    pub values: hybrid_rle::HybridRleDecoder<'a>,
    pub dict: &'a [T],
}

impl<'a, T: NativeType> ValuesDictionary<'a, T> {
    pub fn try_new(page: &'a DataPage, dict: &'a [T]) -> PolarsResult<Self> {
        let values = utils::dict_indices_decoder(page)?;

        Ok(Self { dict, values })
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.values.len()
    }
}

#[derive(Debug)]
pub(super) struct State<'a, T: NativeType> {
    pub page_validity: Option<OptionalPageValidity<'a>>,
    pub translation: StateTranslation<'a, T>,
    pub filter: Option<Filter<'a>>,
}

#[allow(clippy::large_enum_variant)]
#[derive(Debug)]
pub(super) enum StateTranslation<'a, T: NativeType> {
    Unit(Values<'a>),
    Dictionary(ValuesDictionary<'a, T>),
    ByteStreamSplit(byte_stream_split::Decoder<'a>),
}

impl<'a, T: NativeType> SkipInPlace for State<'a, T> {
    fn skip_in_place(&mut self, n: usize) -> ParquetResult<()> {
        if n == 0 {
            return Ok(());
        }

        let mut num_valid = 0;

        if let Some(page_validity) = self.page_validity.as_mut() {
            let mut n = n;
            while n > 0 {
                let Some(next) = page_validity.next_limited(n) else {
                    break;
                };

                num_valid += next.count_ones();
                n -= next.len();
            }
        }

        // @Q: Do we need to do the same we did for Unit for Dictionary and ByteStreamSplit as
        // well?
        //
        // We just throw a `not_implemented` this in `build_state` for now.
        match &mut self.translation {
            StateTranslation::Unit(t) if self.page_validity.is_some() => {
                if num_valid > 0 {
                    _ = t.values.by_ref().nth(num_valid - 1);
                }
            },
            StateTranslation::Unit(t) => _ = t.values.by_ref().nth(n - 1),
            StateTranslation::Dictionary(t) => t.values.skip_in_place(n)?,
            StateTranslation::ByteStreamSplit(t) => _ = t.iter_converted(|_| ()).nth(n - 1),
        }

        Ok(())
    }
}

impl<'a, T: NativeType> PageState<'a> for State<'a, T> {
    fn len(&self) -> usize {
        match &self.translation {
            StateTranslation::Unit(n) => n.len(),
            StateTranslation::Dictionary(n) => n.len(),
            StateTranslation::ByteStreamSplit(n) => n.len(),
        }
    }
}

#[derive(Debug)]
pub(super) struct PrimitiveDecoder<T, P, F>
where
    T: NativeType,
    P: ParquetNativeType,
    F: Fn(P) -> T,
{
    phantom: std::marker::PhantomData<T>,
    phantom_p: std::marker::PhantomData<P>,
    pub op: F,
}

impl<T, P, F> PrimitiveDecoder<T, P, F>
where
    T: NativeType,
    P: ParquetNativeType,
    F: Fn(P) -> T,
{
    #[inline]
    pub(super) fn new(op: F) -> Self {
        Self {
            phantom: std::marker::PhantomData,
            phantom_p: std::marker::PhantomData,
            op,
        }
    }
}

impl<T: std::fmt::Debug> utils::DecodedState for (Vec<T>, MutableBitmap) {
    fn len(&self) -> usize {
        self.0.len()
    }
}

impl<'a, T, P, F> utils::Decoder<'a> for PrimitiveDecoder<T, P, F>
where
    T: NativeType,
    P: ParquetNativeType,
    F: Copy + Fn(P) -> T,
{
    type State = State<'a, T>;
    type Dict = Vec<T>;
    type DecodedState = (Vec<T>, MutableBitmap);

    fn build_state(
        &self,
        page: &'a DataPage,
        dict: Option<&'a Self::Dict>,
    ) -> PolarsResult<Self::State> {
        let is_optional = utils::page_is_optional(page);
        let is_filtered = utils::page_is_filtered(page);

        let page_validity = is_optional
            .then(|| OptionalPageValidity::try_new(page))
            .transpose()?;
        let filter = is_filtered.then(|| Filter::new(page)).flatten();

        let translation = match (page.encoding(), dict) {
            (Encoding::PlainDictionary | Encoding::RleDictionary, Some(dict)) => {
                StateTranslation::Dictionary(ValuesDictionary::try_new(page, dict)?)
            },
            (Encoding::Plain, _) => StateTranslation::Unit(Values::try_new::<P>(page)?),
            (Encoding::ByteStreamSplit, _) => {
                let values = split_buffer(page)?.values;
                StateTranslation::ByteStreamSplit(byte_stream_split::Decoder::try_new(
                    values,
                    std::mem::size_of::<P>(),
                )?)
            },
            _ => return Err(utils::not_implemented(page)),
        };

        // @TODO: For now we just catch this here because I don't really now what to do with this.
        // See Q in `skip_in_place`.
        if is_filtered
            && matches!(
                translation,
                StateTranslation::Dictionary(_) | StateTranslation::ByteStreamSplit(_)
            )
        {
            return Err(utils::not_implemented(page));
        }

        Ok(State {
            page_validity,
            filter,
            translation,
        })
    }

    fn with_capacity(&self, capacity: usize) -> Self::DecodedState {
        (
            Vec::<T>::with_capacity(capacity),
            MutableBitmap::with_capacity(capacity),
        )
    }

    fn extend_from_state(
        &self,
        state: &mut Self::State,
        decoded: &mut Self::DecodedState,
        remaining: usize,
    ) -> PolarsResult<()> {
        let mut filter = state.filter.take();

        extend_from_state_with_opt_filter(
            state,
            &mut filter,
            remaining,
            |state: &mut Self::State, n| {
                let (values, validity) = decoded;

                match (&mut state.translation, &mut state.page_validity) {
                    (StateTranslation::Unit(page), Some(page_validity)) => {
                        utils::extend_from_decoder(
                            validity,
                            page_validity,
                            Some(n),
                            values,
                            &mut page.values.by_ref().map(decode).map(self.op),
                        )?
                    },
                    (StateTranslation::Unit(page), None) => {
                        values.extend(page.values.by_ref().map(decode).map(self.op).take(n));
                    },
                    (StateTranslation::Dictionary(page), Some(page_validity)) => {
                        let translator = DictionaryTranslator(page.dict);
                        let translated_hybridrle =
                            TranslatedHybridRle::new(&mut page.values, &translator);

                        utils::extend_from_decoder(
                            validity,
                            page_validity,
                            Some(n),
                            values,
                            translated_hybridrle,
                        )?;
                    },
                    (StateTranslation::Dictionary(page), None) => {
                        let translator = DictionaryTranslator(page.dict);
                        page.values
                            .translate_and_collect_n_into(values, n, &translator)?;
                    },
                    (StateTranslation::ByteStreamSplit(decoder), Some(page_validity)) => {
                        utils::extend_from_decoder(
                            validity,
                            page_validity,
                            Some(n),
                            values,
                            &mut decoder.iter_converted(decode).map(self.op),
                        )?
                    },
                    (StateTranslation::ByteStreamSplit(decoder), None) => {
                        values.extend(decoder.iter_converted(decode).map(self.op).take(n));
                    },
                }

                Ok(())
            },
        )?;

        state.filter = filter;

        Ok(())
    }

    fn deserialize_dict(&self, page: &DictPage) -> Self::Dict {
        deserialize_plain(&page.buffer, self.op)
    }
}

pub(super) fn finish<T: NativeType>(
    data_type: &ArrowDataType,
    values: Vec<T>,
    validity: MutableBitmap,
) -> MutablePrimitiveArray<T> {
    let validity = if validity.is_empty() {
        None
    } else {
        Some(validity)
    };
    MutablePrimitiveArray::try_new(data_type.clone(), values, validity).unwrap()
}

/// An [`Iterator`] adapter over [`PagesIter`] assumed to be encoded as primitive arrays
#[derive(Debug)]
pub struct Iter<T, I, P, F>
where
    I: PagesIter,
    T: NativeType,
    P: ParquetNativeType,
    F: Fn(P) -> T,
{
    iter: I,
    data_type: ArrowDataType,
    items: VecDeque<(Vec<T>, MutableBitmap)>,
    remaining: usize,
    chunk_size: Option<usize>,
    dict: Option<Vec<T>>,
    op: F,
    phantom: std::marker::PhantomData<P>,
}

impl<T, I, P, F> Iter<T, I, P, F>
where
    I: PagesIter,
    T: NativeType,

    P: ParquetNativeType,
    F: Copy + Fn(P) -> T,
{
    pub fn new(
        iter: I,
        data_type: ArrowDataType,
        num_rows: usize,
        chunk_size: Option<usize>,
        op: F,
    ) -> Self {
        Self {
            iter,
            data_type,
            items: VecDeque::new(),
            dict: None,
            remaining: num_rows,
            chunk_size,
            op,
            phantom: Default::default(),
        }
    }
}

impl<T, I, P, F> Iterator for Iter<T, I, P, F>
where
    I: PagesIter,
    T: NativeType,
    P: ParquetNativeType,
    F: Copy + Fn(P) -> T,
{
    type Item = PolarsResult<MutablePrimitiveArray<T>>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let maybe_state = utils::next(
                &mut self.iter,
                &mut self.items,
                &mut self.dict,
                &mut self.remaining,
                self.chunk_size,
                &PrimitiveDecoder::new(self.op),
            );
            match maybe_state {
                MaybeNext::Some(Ok((values, validity))) => {
                    return Some(Ok(finish(&self.data_type, values, validity)))
                },
                MaybeNext::Some(Err(e)) => return Some(Err(e)),
                MaybeNext::None => return None,
                MaybeNext::More => continue,
            }
        }
    }
}

pub(super) fn deserialize_plain<T, P, F>(values: &[u8], op: F) -> Vec<T>
where
    T: NativeType,
    P: ParquetNativeType,
    F: Copy + Fn(P) -> T,
{
    values
        .chunks_exact(std::mem::size_of::<P>())
        .map(decode)
        .map(op)
        .collect::<Vec<_>>()
}
