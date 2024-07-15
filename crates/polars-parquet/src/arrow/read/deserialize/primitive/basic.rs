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
use crate::read::deserialize::utils::array_chunks::ArrayChunks;
use crate::read::deserialize::utils::filter::Filter;
use crate::read::deserialize::utils::{BatchableCollector, PageValidity, TranslatedHybridRle};

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

/// A function that defines how to decode from the
/// [`parquet::types::NativeType`][ParquetNativeType] to the [`arrow::types::NativeType`].
///
/// This should almost always be inlined.
pub(crate) trait DecoderFunction<P, T>: Copy
where
    T: NativeType,
    P: ParquetNativeType,
{
    fn decode(self, x: P) -> T;
}

#[derive(Default, Clone, Copy)]
pub(crate) struct IntoDecoderFunction<P, T>(std::marker::PhantomData<(P, T)>);
impl<P: Into<T>, T> DecoderFunction<P, T> for IntoDecoderFunction<P, T>
where
    P: ParquetNativeType,
    T: NativeType,
{
    #[inline(always)]
    fn decode(self, x: P) -> T {
        x.into()
    }
}

#[derive(Default, Clone, Copy)]
pub(crate) struct AsDecoderFunction<P, T>(std::marker::PhantomData<(P, T)>);
macro_rules! as_decoder_impl {
    ($($p:ty => $t:ty,)+) => {
        $(
        impl DecoderFunction<$p, $t> for AsDecoderFunction<$p, $t> {
            #[inline(always)]
            fn decode(self, x : $p) -> $t {
                x as $t
            }
        }
        )+
    };
}

as_decoder_impl![
    i32 => i8,
    i32 => i16,
    i32 => u8,
    i32 => u16,
    i32 => u32,
    i64 => i32,
    i64 => u32,
    i64 => u64,
];

#[derive(Default, Clone, Copy)]
pub(crate) struct UnitDecoderFunction<T>(std::marker::PhantomData<T>);
impl<T> DecoderFunction<T, T> for UnitDecoderFunction<T>
where
    T: NativeType + ParquetNativeType,
{
    #[inline(always)]
    fn decode(self, x: T) -> T {
        x
    }
}

struct BatchDecoder<'a, 'b, P, T, D>
where
    T: NativeType,
    P: ParquetNativeType,
    D: DecoderFunction<P, T>,
{
    chunks: &'b mut ArrayChunks<'a, P>,
    decoder: D,
    _pd: std::marker::PhantomData<T>,
}

impl<'a, 'b, P, T, D: DecoderFunction<P, T>> BatchableCollector<(), Vec<T>>
    for BatchDecoder<'a, 'b, P, T, D>
where
    T: NativeType,
    P: ParquetNativeType,
    D: DecoderFunction<P, T>,
{
    fn reserve(target: &mut Vec<T>, n: usize) {
        target.reserve(n);
    }

    fn push_n(&mut self, target: &mut Vec<T>, n: usize) -> ParquetResult<()> {
        let n = usize::min(self.chunks.len(), n);
        let (items, remainder) = self.chunks.bytes.split_at(n);
        let decoder = self.decoder;
        target.extend(
            items
                .iter()
                .map(|chunk| decoder.decode(P::from_le_bytes(*chunk))),
        );
        self.chunks.bytes = remainder;
        Ok(())
    }

    fn push_n_nulls(&mut self, target: &mut Vec<T>, n: usize) -> ParquetResult<()> {
        target.resize(target.len() + n, T::default());
        Ok(())
    }
}

#[allow(clippy::large_enum_variant)]
#[derive(Debug)]
pub(super) enum StateTranslation<'a, P: ParquetNativeType, T: NativeType> {
    Unit(ArrayChunks<'a, P>),
    Dictionary(ValuesDictionary<'a, T>),
    ByteStreamSplit(byte_stream_split::Decoder<'a>),
}

impl<'a, P, T, D> utils::StateTranslation<'a, PrimitiveDecoder<P, T, D>>
    for StateTranslation<'a, P, T>
where
    T: NativeType,
    P: ParquetNativeType,
    D: DecoderFunction<P, T>,
{
    fn new(
        _decoder: &PrimitiveDecoder<P, T, D>,
        page: &'a DataPage,
        dict: Option<&'a <PrimitiveDecoder<P, T, D> as utils::Decoder<'a>>::Dict>,
        _page_validity: Option<&PageValidity<'a>>,
        _filter: Option<&Filter<'a>>,
    ) -> PolarsResult<Self> {
        match (page.encoding(), dict) {
            (Encoding::PlainDictionary | Encoding::RleDictionary, Some(dict)) => {
                Ok(Self::Dictionary(ValuesDictionary::try_new(page, dict)?))
            },
            (Encoding::Plain, _) => {
                let values = split_buffer(page)?.values;
                Ok(Self::Unit(ArrayChunks::new(values).unwrap()))
            },
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
            Self::Unit(t) => _ = t.nth(n - 1),
            Self::Dictionary(t) => t.values.skip_in_place(n)?,
            Self::ByteStreamSplit(t) => _ = t.iter_converted(|_| ()).nth(n - 1),
        }

        Ok(())
    }

    fn extend_from_state(
        &mut self,
        decoder: &PrimitiveDecoder<P, T, D>,
        decoded: &mut <PrimitiveDecoder<P, T, D> as utils::Decoder<'a>>::DecodedState,
        page_validity: &mut Option<PageValidity<'a>>,
        additional: usize,
    ) -> ParquetResult<()> {
        let (values, validity) = decoded;

        match (self, page_validity) {
            (Self::Unit(page), None) => {
                values.extend(
                    page.by_ref()
                        .map(|v| decoder.decoder.decode(P::from_le_bytes(*v)))
                        .take(additional),
                );
            },
            (Self::Unit(page), Some(page_validity)) => {
                let batched = BatchDecoder {
                    chunks: page,
                    decoder: decoder.decoder,
                    _pd: std::marker::PhantomData,
                };

                utils::extend_from_decoder(
                    validity,
                    page_validity,
                    Some(additional),
                    values,
                    batched,
                )?
            },
            (Self::Dictionary(page), None) => {
                let translator = DictionaryTranslator(page.dict);
                page.values
                    .translate_and_collect_n_into(values, additional, &translator)?;
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
            (Self::ByteStreamSplit(page_values), None) => {
                values.extend(
                    page_values
                        .iter_converted(|v| decoder.decoder.decode(decode(v)))
                        .take(additional),
                );
            },
            (Self::ByteStreamSplit(page_values), Some(page_validity)) => {
                utils::extend_from_decoder(
                    validity,
                    page_validity,
                    Some(additional),
                    values,
                    &mut page_values.iter_converted(|v| decoder.decoder.decode(decode(v))),
                )?
            },
        }

        Ok(())
    }
}

#[derive(Debug)]
pub(super) struct PrimitiveDecoder<P, T, D>
where
    T: NativeType,
    P: ParquetNativeType,
    D: DecoderFunction<P, T>,
{
    pub(crate) decoder: D,
    _pd: std::marker::PhantomData<(P, T)>,
}

impl<P, T, D> PrimitiveDecoder<P, T, D>
where
    T: NativeType,
    P: ParquetNativeType,
    D: DecoderFunction<P, T>,
{
    #[inline]
    pub(super) fn new(decoder: D) -> Self {
        Self {
            decoder,
            _pd: std::marker::PhantomData,
        }
    }
}

impl<T: std::fmt::Debug> utils::DecodedState for (Vec<T>, MutableBitmap) {
    fn len(&self) -> usize {
        self.0.len()
    }
}

impl<'a, P, T, D> utils::Decoder<'a> for PrimitiveDecoder<P, T, D>
where
    T: NativeType,
    P: ParquetNativeType,
    D: DecoderFunction<P, T>,
{
    type Translation = StateTranslation<'a, P, T>;
    type Dict = Vec<T>;
    type DecodedState = (Vec<T>, MutableBitmap);

    fn with_capacity(&self, capacity: usize) -> Self::DecodedState {
        (
            Vec::<T>::with_capacity(capacity),
            MutableBitmap::with_capacity(capacity),
        )
    }

    fn deserialize_dict(&self, page: &DictPage) -> Self::Dict {
        deserialize_plain::<P, T, D>(&page.buffer, self.decoder)
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
pub struct Iter<T, I, P, D>
where
    I: PagesIter,
    T: NativeType,
    P: ParquetNativeType,
    D: DecoderFunction<P, T>,
{
    iter: I,
    data_type: ArrowDataType,
    items: VecDeque<(Vec<T>, MutableBitmap)>,
    remaining: usize,
    chunk_size: Option<usize>,
    dict: Option<Vec<T>>,
    decoder: D,
    phantom: std::marker::PhantomData<P>,
}

impl<T, I, P, D> Iter<T, I, P, D>
where
    I: PagesIter,
    T: NativeType,

    P: ParquetNativeType,
    D: DecoderFunction<P, T>,
{
    pub fn new(
        iter: I,
        data_type: ArrowDataType,
        num_rows: usize,
        chunk_size: Option<usize>,
        decoder: D,
    ) -> Self {
        Self {
            iter,
            data_type,
            items: VecDeque::new(),
            dict: None,
            remaining: num_rows,
            chunk_size,
            decoder,
            phantom: Default::default(),
        }
    }
}

impl<T, I, P, D> Iterator for Iter<T, I, P, D>
where
    I: PagesIter,
    T: NativeType,
    P: ParquetNativeType,
    D: DecoderFunction<P, T>,
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
                &PrimitiveDecoder::new(self.decoder),
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

pub(super) fn deserialize_plain<P, T, D>(values: &[u8], decoder: D) -> Vec<T>
where
    T: NativeType,
    P: ParquetNativeType,
    D: DecoderFunction<P, T>,
{
    values
        .chunks_exact(std::mem::size_of::<P>())
        .map(decode)
        .map(|v| decoder.decode(v))
        .collect::<Vec<_>>()
}
