use arrow::array::{Array, MutablePrimitiveArray};
use arrow::bitmap::MutableBitmap;
use arrow::datatypes::ArrowDataType;
use arrow::types::NativeType;
use num_traits::AsPrimitive;
use polars_error::PolarsResult;

use super::super::utils;
use super::basic::{DecoderFunction, PlainDecoderFnCollector, PrimitiveDecoder, ValuesDictionary};
use crate::parquet::encoding::hybrid_rle::DictionaryTranslator;
use crate::parquet::encoding::{byte_stream_split, delta_bitpacked, Encoding};
use crate::parquet::error::ParquetResult;
use crate::parquet::page::{split_buffer, DataPage, DictPage};
use crate::parquet::types::{decode, NativeType as ParquetNativeType};
use crate::read::deserialize::utils::array_chunks::ArrayChunks;
use crate::read::deserialize::utils::filter::Filter;
use crate::read::deserialize::utils::{BatchableCollector, PageValidity, TranslatedHybridRle};

#[allow(clippy::large_enum_variant)]
#[derive(Debug)]
pub(crate) enum StateTranslation<'a, P: ParquetNativeType, T: NativeType> {
    Plain(ArrayChunks<'a, P>),
    Dictionary(ValuesDictionary<'a, T>),
    ByteStreamSplit(byte_stream_split::Decoder<'a>),
    DeltaBinaryPacked(delta_bitpacked::Decoder<'a>),
}

impl<'a, P, T, D> utils::StateTranslation<'a, IntDecoder<P, T, D>> for StateTranslation<'a, P, T>
where
    T: NativeType,
    P: ParquetNativeType,
    i64: num_traits::AsPrimitive<P>,
    D: DecoderFunction<P, T>,
{
    fn new(
        _decoder: &IntDecoder<P, T, D>,
        page: &'a DataPage,
        dict: Option<&'a <IntDecoder<P, T, D> as utils::Decoder>::Dict>,
        _page_validity: Option<&PageValidity<'a>>,
        _filter: Option<&Filter<'a>>,
    ) -> PolarsResult<Self> {
        match (page.encoding(), dict) {
            (Encoding::PlainDictionary | Encoding::RleDictionary, Some(dict)) => {
                Ok(Self::Dictionary(ValuesDictionary::try_new(page, dict)?))
            },
            (Encoding::Plain, _) => {
                let values = split_buffer(page)?.values;
                Ok(Self::Plain(ArrayChunks::new(values).unwrap()))
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
            Self::Plain(v) => v.len(),
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
            Self::Plain(v) => _ = v.nth(n - 1),
            Self::Dictionary(v) => v.values.skip_in_place(n)?,
            Self::ByteStreamSplit(v) => _ = v.iter_converted(|_| ()).nth(n - 1),
            Self::DeltaBinaryPacked(v) => _ = v.nth(n - 1),
        }

        Ok(())
    }

    fn extend_from_state(
        &mut self,
        decoder: &IntDecoder<P, T, D>,
        decoded: &mut <IntDecoder<P, T, D> as utils::Decoder>::DecodedState,
        page_validity: &mut Option<PageValidity<'a>>,
        additional: usize,
    ) -> ParquetResult<()> {
        let (values, validity) = decoded;
        match (self, page_validity) {
            (Self::Plain(page), Some(page_validity)) => {
                let collector = PlainDecoderFnCollector {
                    chunks: page,
                    decoder: decoder.0.decoder,
                    _pd: Default::default(),
                };

                utils::extend_from_decoder(
                    validity,
                    page_validity,
                    Some(additional),
                    values,
                    collector,
                )?;
            },
            (Self::Plain(page), None) => {
                PlainDecoderFnCollector {
                    chunks: page,
                    decoder: decoder.0.decoder,
                    _pd: Default::default(),
                }
                .push_n(values, additional)?;
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
pub(crate) struct IntDecoder<P, T, D>(PrimitiveDecoder<P, T, D>)
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
    pub(crate) fn new(decoder: D) -> Self {
        Self(PrimitiveDecoder::new(decoder))
    }
}

impl<P, T, D> utils::Decoder for IntDecoder<P, T, D>
where
    T: NativeType,
    P: ParquetNativeType,
    i64: num_traits::AsPrimitive<P>,
    D: DecoderFunction<P, T>,
{
    type Translation<'a> = StateTranslation<'a, P, T>;
    type Dict = Vec<T>;
    type DecodedState = (Vec<T>, MutableBitmap);

    fn with_capacity(&self, capacity: usize) -> Self::DecodedState {
        self.0.with_capacity(capacity)
    }

    fn deserialize_dict(&self, page: DictPage) -> Self::Dict {
        self.0.deserialize_dict(page)
    }

    fn finalize(
        &self,
        data_type: ArrowDataType,
        (values, validity): Self::DecodedState,
    ) -> ParquetResult<Box<dyn Array>> {
        let validity = if validity.is_empty() {
            None
        } else {
            Some(validity)
        };

        Ok(Box::new(
            MutablePrimitiveArray::try_new(data_type, values, validity)
                .unwrap()
                .freeze(),
        ))
    }
}
