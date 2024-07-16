use arrow::array::MutablePrimitiveArray;
use arrow::bitmap::MutableBitmap;
use arrow::datatypes::ArrowDataType;
use arrow::types::NativeType;
use num_traits::AsPrimitive;
use polars_error::PolarsResult;

use super::super::{utils, PagesIter};
use super::basic::{finish, DecoderFunction, PrimitiveDecoder, ValuesDictionary};
use crate::parquet::encoding::hybrid_rle::DictionaryTranslator;
use crate::parquet::encoding::{byte_stream_split, delta_bitpacked, Encoding};
use crate::parquet::error::ParquetResult;
use crate::parquet::page::{split_buffer, DataPage, DictPage};
use crate::parquet::types::{decode, NativeType as ParquetNativeType};
use crate::read::deserialize::utils::array_chunks::ArrayChunks;
use crate::read::deserialize::utils::filter::Filter;
use crate::read::deserialize::utils::{BasicDecodeIterator, PageValidity, TranslatedHybridRle};

#[allow(clippy::large_enum_variant)]
#[derive(Debug)]
pub(super) enum StateTranslation<'pages, P: ParquetNativeType, T: NativeType> {
    Unit(ArrayChunks<'pages, P>),
    Dictionary(ValuesDictionary<'pages, T>),
    ByteStreamSplit(byte_stream_split::Decoder<'pages>),
    DeltaBinaryPacked(delta_bitpacked::Decoder<'pages>),
}

impl<'pages, 'mmap: 'pages, P, T, D> utils::StateTranslation<'pages, 'mmap, IntDecoder<P, T, D>>
    for StateTranslation<'pages, P, T>
where
    T: NativeType,
    P: ParquetNativeType,
    i64: num_traits::AsPrimitive<P>,
    D: DecoderFunction<P, T>,
{
    fn new(
        _decoder: &IntDecoder<P, T, D>,
        page: &'pages DataPage<'mmap>,
        dict: Option<&'pages <IntDecoder<P, T, D> as utils::Decoder>::Dict>,
        _page_validity: Option<&PageValidity<'pages>>,
        _filter: Option<&Filter<'pages>>,
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
            (Encoding::DeltaBinaryPacked, _) => {
                let values = split_buffer(page)?.values;
                Ok(Self::DeltaBinaryPacked(delta_bitpacked::Decoder::try_new(
                    values,
                )?))
            },
            _ => Err(utils::not_implemented(page)),
        }
    }

    fn len_when_not_nullable(&self) -> usize {
        match self {
            Self::Unit(v) => v.len(),
            Self::Dictionary(v) => v.len(),
            Self::ByteStreamSplit(v) => v.len(),
            Self::DeltaBinaryPacked(v) => v.size_hint().0,
        }
    }

    fn skip_in_place(&mut self, n: usize) -> ParquetResult<()> {
        if n == 0 {
            return Ok(());
        }

        match self {
            Self::Unit(v) => _ = v.nth(n - 1),
            Self::Dictionary(v) => v.values.skip_in_place(n)?,
            Self::ByteStreamSplit(v) => _ = v.iter_converted(|_| ()).nth(n - 1),
            Self::DeltaBinaryPacked(v) => _ = v.nth(n - 1),
        }

        Ok(())
    }

    fn extend_from_state(
        &mut self,
        decoder: &IntDecoder<P, T, D>,
        decoded: &mut <IntDecoder<P, T, D> as utils::Decoder<'pages, 'mmap>>::DecodedState,
        page_validity: &mut Option<PageValidity<'pages>>,
        additional: usize,
    ) -> ParquetResult<()> {
        let (values, validity) = decoded;
        match (self, page_validity) {
            (Self::Unit(page), Some(page_validity)) => utils::extend_from_decoder(
                validity,
                page_validity,
                Some(additional),
                values,
                &mut page
                    .by_ref()
                    .map(|v| decoder.0.decoder.decode(P::from_le_bytes(*v))),
            )?,
            (Self::Unit(page), None) => {
                values.extend(
                    page.by_ref()
                        .map(|v| decoder.0.decoder.decode(P::from_le_bytes(*v)))
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
                    &mut page_values.iter_converted(|v| decoder.0.decoder.decode(decode(v))),
                )?
            },
            (Self::ByteStreamSplit(page_values), None) => {
                values.extend(
                    page_values
                        .iter_converted(|v| decoder.0.decoder.decode(decode(v)))
                        .take(additional),
                );
            },
            (Self::DeltaBinaryPacked(page_values), None) => {
                values.extend(
                    page_values
                        .by_ref()
                        .map(|x| decoder.0.decoder.decode(x.unwrap().as_()))
                        .take(additional),
                );
            },
            (Self::DeltaBinaryPacked(page_values), Some(page_validity)) => {
                utils::extend_from_decoder(
                    validity,
                    page_validity,
                    Some(additional),
                    values,
                    &mut page_values
                        .by_ref()
                        .map(|x| decoder.0.decoder.decode(x.unwrap().as_())),
                )?
            },
        }

        Ok(())
    }
}

/// Decoder of integer parquet type
#[derive(Debug)]
struct IntDecoder<P, T, D>(PrimitiveDecoder<P, T, D>)
where
    T: NativeType,
    P: ParquetNativeType,
    i64: num_traits::AsPrimitive<P>,
    D: DecoderFunction<P, T>;

impl<T, P, D> IntDecoder<P, T, D>
where
    T: NativeType,
    P: ParquetNativeType,
    i64: num_traits::AsPrimitive<P>,
    D: DecoderFunction<P, T>,
{
    #[inline]
    fn new(decoder: D) -> Self {
        Self(PrimitiveDecoder::new(decoder))
    }
}

impl<'pages, 'mmap: 'pages, P, T, D> utils::Decoder<'pages, 'mmap> for IntDecoder<P, T, D>
where
    T: NativeType,
    P: ParquetNativeType,
    i64: num_traits::AsPrimitive<P>,
    D: DecoderFunction<P, T>,
{
    type Translation = StateTranslation<'pages, P, T>;
    type Dict = Vec<T>;
    type DecodedState = (Vec<T>, MutableBitmap);

    fn with_capacity(&self, capacity: usize) -> Self::DecodedState {
        self.0.with_capacity(capacity)
    }

    fn deserialize_dict(&self, page: &'pages DictPage<'mmap>) -> Self::Dict {
        self.0.deserialize_dict(page)
    }
}

pub struct IntegerDecodeIter;

impl IntegerDecodeIter {
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
        IntDecoder<P, T, D>,
        fn(
            &ArrowDataType,
            <IntDecoder<P, T, D> as utils::Decoder<'pages, 'mmap>>::DecodedState,
        ) -> PolarsResult<MutablePrimitiveArray<T>>,
    >
    where
        I: PagesIter<'mmap>,
        T: NativeType,

        i64: num_traits::AsPrimitive<P>,
        P: ParquetNativeType,
        D: DecoderFunction<P, T>,
    {
        BasicDecodeIterator::new(
            iter,
            data_type,
            chunk_size,
            num_rows,
            IntDecoder::new(decoder),
            finish,
        )
    }
}
