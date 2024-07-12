use std::collections::VecDeque;

use arrow::array::MutablePrimitiveArray;
use arrow::bitmap::MutableBitmap;
use arrow::datatypes::ArrowDataType;
use arrow::types::NativeType;
use polars_error::PolarsResult;

use super::super::utils::MaybeNext;
use super::super::{utils, PagesIter};
use crate::parquet::encoding::hybrid_rle::DictionaryTranslator;
use crate::parquet::encoding::{byte_stream_split, hybrid_rle, Encoding};
use crate::parquet::error::ParquetResult;
use crate::parquet::page::{split_buffer, DataPage, DictPage};
use crate::parquet::types::{decode, NativeType as ParquetNativeType};
use crate::read::deserialize::utils::filter::Filter;
use crate::read::deserialize::utils::{PageValidity, TranslatedHybridRle};

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

#[allow(clippy::large_enum_variant)]
#[derive(Debug)]
pub(super) enum StateTranslation<'a, T: NativeType> {
    Unit(Values<'a>),
    Dictionary(ValuesDictionary<'a, T>),
    ByteStreamSplit(byte_stream_split::Decoder<'a>),
}

impl<'a, T, P, F> utils::StateTranslation<'a, PrimitiveDecoder<T, P, F>> for StateTranslation<'a, T>
where
    T: NativeType,
    P: ParquetNativeType,
    F: Copy + Fn(P) -> T,
{
    fn new(
        _decoder: &PrimitiveDecoder<T, P, F>,
        page: &'a DataPage,
        dict: Option<&'a <PrimitiveDecoder<T, P, F> as utils::Decoder<'a>>::Dict>,
        _page_validity: Option<&PageValidity<'a>>,
        _filter: Option<&Filter<'a>>,
    ) -> PolarsResult<Self> {
        match (page.encoding(), dict) {
            (Encoding::PlainDictionary | Encoding::RleDictionary, Some(dict)) => {
                Ok(Self::Dictionary(ValuesDictionary::try_new(page, dict)?))
            },
            (Encoding::Plain, _) => Ok(Self::Unit(Values::try_new::<P>(page)?)),
            (Encoding::ByteStreamSplit, _) => {
                let values = split_buffer(page)?.values;
                Ok(Self::ByteStreamSplit(byte_stream_split::Decoder::try_new(
                    values,
                    std::mem::size_of::<P>(),
                )?))
            },
            _ => Err(utils::not_implemented(page)),
        }
    }

    fn len_when_not_nullable(&self) -> usize {
        match self {
            Self::Unit(n) => n.len(),
            Self::Dictionary(n) => n.len(),
            Self::ByteStreamSplit(n) => n.len(),
        }
    }

    fn skip_in_place(&mut self, n: usize) -> ParquetResult<()> {
        if n == 0 {
            return Ok(());
        }

        match self {
            Self::Unit(t) => _ = t.values.by_ref().nth(n - 1),
            Self::Dictionary(t) => t.values.skip_in_place(n)?,
            Self::ByteStreamSplit(t) => _ = t.iter_converted(|_| ()).nth(n - 1),
        }

        Ok(())
    }

    fn extend_from_state(
        &mut self,
        decoder: &PrimitiveDecoder<T, P, F>,
        decoded: &mut <PrimitiveDecoder<T, P, F> as utils::Decoder<'a>>::DecodedState,
        page_validity: &mut Option<PageValidity<'a>>,
        additional: usize,
    ) -> ParquetResult<()> {
        let (values, validity) = decoded;

        match (self, page_validity) {
            (Self::Unit(page), Some(page_validity)) => utils::extend_from_decoder(
                validity,
                page_validity,
                Some(additional),
                values,
                &mut page.values.by_ref().map(decode).map(decoder.op),
            )?,
            (Self::Unit(page), None) => {
                values.extend(
                    page.values
                        .by_ref()
                        .map(decode)
                        .map(decoder.op)
                        .take(additional),
                );
            },
            (Self::Dictionary(page), Some(page_validity)) => {
                let translator = DictionaryTranslator(page.dict);
                let translated_hybridrle = TranslatedHybridRle::new(&mut page.values, &translator);

                utils::extend_from_decoder(
                    validity,
                    page_validity,
                    Some(additional),
                    values,
                    translated_hybridrle,
                )?;
            },
            (Self::Dictionary(page), None) => {
                let translator = DictionaryTranslator(page.dict);
                page.values
                    .translate_and_collect_n_into(values, additional, &translator)?;
            },
            (Self::ByteStreamSplit(page_values), Some(page_validity)) => {
                utils::extend_from_decoder(
                    validity,
                    page_validity,
                    Some(additional),
                    values,
                    &mut page_values.iter_converted(decode).map(decoder.op),
                )?
            },
            (Self::ByteStreamSplit(page_values), None) => {
                values.extend(
                    page_values
                        .iter_converted(decode)
                        .map(decoder.op)
                        .take(additional),
                );
            },
        }

        Ok(())
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
    type Translation = StateTranslation<'a, T>;
    type Dict = Vec<T>;
    type DecodedState = (Vec<T>, MutableBitmap);

    fn with_capacity(&self, capacity: usize) -> Self::DecodedState {
        (
            Vec::<T>::with_capacity(capacity),
            MutableBitmap::with_capacity(capacity),
        )
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
