use arrow::array::MutablePrimitiveArray;
use arrow::bitmap::MutableBitmap;
use arrow::datatypes::ArrowDataType;
use arrow::types::NativeType;
use polars_error::PolarsResult;

use super::super::{utils, PagesIter};
use crate::parquet::encoding::hybrid_rle::DictionaryTranslator;
use crate::parquet::encoding::{byte_stream_split, hybrid_rle, Encoding};
use crate::parquet::error::ParquetResult;
use crate::parquet::page::{split_buffer, DataPage, DictPage};
use crate::parquet::types::{decode, NativeType as ParquetNativeType};
use crate::read::deserialize::utils::array_chunks::ArrayChunks;
use crate::read::deserialize::utils::filter::Filter;
use crate::read::deserialize::utils::{
    BasicDecodeIterator, BatchableCollector, PageValidity, TranslatedHybridRle,
};

#[derive(Debug)]
pub(super) struct ValuesDictionary<'pages, T: NativeType> {
    pub values: hybrid_rle::HybridRleDecoder<'pages>,
    pub dict: &'pages [T],
}

impl<'pages, T: NativeType> ValuesDictionary<'pages, T> {
    pub fn try_new(page: &'pages DataPage, dict: &'pages [T]) -> PolarsResult<Self> {
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
pub(super) enum StateTranslation<'pages, P: ParquetNativeType, T: NativeType> {
    Unit(ArrayChunks<'pages, P>),
    Dictionary(ValuesDictionary<'pages, T>),
    ByteStreamSplit(byte_stream_split::Decoder<'pages>),
}

impl<'pages, 'mmap: 'pages, P, T, D> utils::StateTranslation<'pages, 'mmap, PrimitiveDecoder<P, T, D>>
    for StateTranslation<'pages, P, T>
where
    T: NativeType,
    P: ParquetNativeType,
    D: DecoderFunction<P, T>,
{
    fn new(
        _decoder: &PrimitiveDecoder<P, T, D>,
        page: &'pages DataPage<'mmap>,
        dict: Option<&'pages <PrimitiveDecoder<P, T, D> as utils::Decoder<'pages, 'mmap>>::Dict>,
        _page_validity: Option<&PageValidity<'pages>>,
        _filter: Option<&Filter<'pages>>,
    ) -> PolarsResult<Self> {
        match (page.encoding(), dict) {
            (Encoding::PlainDictionary | Encoding::RleDictionary, Some(dict)) => {
                Ok(Self::Dictionary(ValuesDictionary::try_new(page, dict.as_ref())?))
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
        decoded: &mut <PrimitiveDecoder<P, T, D> as utils::Decoder<'pages, 'mmap>>::DecodedState,
        page_validity: &mut Option<PageValidity<'pages>>,
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
                let translator = DictionaryTranslator(page.dict.as_ref());
                page.values
                    .translate_and_collect_n_into(values, additional, &translator)?;
            },
            (Self::Dictionary(page), Some(page_validity)) => {
                let translator = DictionaryTranslator(page.dict.as_ref());
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

impl<'pages, 'mmap: 'pages, P, T, D> utils::Decoder<'pages, 'mmap> for PrimitiveDecoder<P, T, D>
where
    T: NativeType,
    P: ParquetNativeType,
    D: DecoderFunction<P, T>,
{
    type Translation = StateTranslation<'pages, P, T>;
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
    (values, validity): (Vec<T>, MutableBitmap),
) -> PolarsResult<MutablePrimitiveArray<T>> {
    let validity = if validity.is_empty() {
        None
    } else {
        Some(validity)
    };
    Ok(MutablePrimitiveArray::try_new(data_type.clone(), values, validity).unwrap())
}

pub struct PrimitiveDecodeIter;

impl PrimitiveDecodeIter {
    pub fn new<'pages, 'mmap: 'pages, T, I, P, D>(
        iter: I,
        data_type: ArrowDataType,
        num_rows: usize,
        chunk_size: Option<usize>,
        decoder: D,
    ) -> BasicDecodeIterator<
        'pages,
        'mmap,
        MutablePrimitiveArray<T>,
        I,
        PrimitiveDecoder<P, T, D>,
        fn(
            &ArrowDataType,
            <PrimitiveDecoder<P, T, D> as utils::Decoder<'pages, 'mmap>>::DecodedState,
        ) -> PolarsResult<MutablePrimitiveArray<T>>,
    >
    where
        I: PagesIter<'mmap>,
        T: NativeType,

        P: ParquetNativeType,
        D: DecoderFunction<P, T>,
    {
        BasicDecodeIterator::new(
            iter,
            data_type,
            chunk_size,
            num_rows,
            PrimitiveDecoder::new(decoder),
            finish,
        )
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
