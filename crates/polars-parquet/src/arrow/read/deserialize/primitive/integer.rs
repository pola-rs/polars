use arrow::array::{DictionaryArray, DictionaryKey, PrimitiveArray};
use arrow::bitmap::MutableBitmap;
use arrow::datatypes::ArrowDataType;
use arrow::types::NativeType;
use num_traits::AsPrimitive;
use polars_error::PolarsResult;

use super::super::utils;
use super::basic::{
    AsDecoderFunction, ClosureDecoderFunction, DecoderFunction, IntoDecoderFunction,
    PlainDecoderFnCollector, PrimitiveDecoder, UnitDecoderFunction, ValuesDictionary,
};
use crate::parquet::encoding::hybrid_rle::{self, DictionaryTranslator};
use crate::parquet::encoding::{byte_stream_split, delta_bitpacked, Encoding};
use crate::parquet::error::ParquetResult;
use crate::parquet::page::{split_buffer, DataPage, DictPage};
use crate::parquet::types::{decode, NativeType as ParquetNativeType};
use crate::read::deserialize::utils::array_chunks::ArrayChunks;
use crate::read::deserialize::utils::{
    freeze_validity, BatchableCollector, Decoder, PageValidity, TranslatedHybridRle,
};

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
    type PlainDecoder = ArrayChunks<'a, P>;

    fn new(
        _decoder: &IntDecoder<P, T, D>,
        page: &'a DataPage,
        dict: Option<&'a <IntDecoder<P, T, D> as utils::Decoder>::Dict>,
        _page_validity: Option<&PageValidity<'a>>,
    ) -> PolarsResult<Self> {
        match (page.encoding(), dict) {
            (Encoding::PlainDictionary | Encoding::RleDictionary, Some(dict)) => {
                Ok(Self::Dictionary(ValuesDictionary::try_new(page, dict)?))
            },
            (Encoding::Plain, _) => {
                let values = split_buffer(page)?.values;
                let chunks = ArrayChunks::new(values).unwrap();
                Ok(Self::Plain(chunks))
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
        decoder: &mut IntDecoder<P, T, D>,
        decoded: &mut <IntDecoder<P, T, D> as utils::Decoder>::DecodedState,
        page_validity: &mut Option<PageValidity<'a>>,
        additional: usize,
    ) -> ParquetResult<()> {
        match self {
            Self::Plain(page_values) => decoder.decode_plain_encoded(
                decoded,
                page_values,
                page_validity.as_mut(),
                additional,
            )?,
            Self::Dictionary(page) => decoder.decode_dictionary_encoded(
                decoded,
                &mut page.values,
                page_validity.as_mut(),
                page.dict,
                additional,
            )?,
            Self::ByteStreamSplit(page_values) => {
                let (values, validity) = decoded;

                match page_validity {
                    None => {
                        values.extend(
                            page_values
                                .iter_converted(|v| decoder.0.decoder.decode(decode(v)))
                                .take(additional),
                        );
                    },
                    Some(page_validity) => {
                        utils::extend_from_decoder(
                            validity,
                            page_validity,
                            Some(additional),
                            values,
                            &mut page_values
                                .iter_converted(|v| decoder.0.decoder.decode(decode(v))),
                        )?;
                    },
                }
            },
            Self::DeltaBinaryPacked(page_values) => {
                let (values, validity) = decoded;

                match page_validity {
                    None => {
                        values.extend(
                            page_values
                                .by_ref()
                                .map(|x| decoder.0.decoder.decode(x.unwrap().as_()))
                                .take(additional),
                        );
                    },
                    Some(page_validity) => utils::extend_from_decoder(
                        validity,
                        page_validity,
                        Some(additional),
                        values,
                        &mut page_values
                            .by_ref()
                            .map(|x| decoder.0.decoder.decode(x.unwrap().as_())),
                    )?,
                }
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

impl<P, T, D> IntDecoder<P, T, D>
where
    P: ParquetNativeType,
    T: NativeType,
    i64: num_traits::AsPrimitive<P>,
    D: DecoderFunction<P, T>,
{
    #[inline]
    fn new(decoder: PrimitiveDecoder<P, T, D>) -> Self {
        Self(decoder)
    }
}

impl<T> IntDecoder<T, T, UnitDecoderFunction<T>>
where
    T: NativeType + ParquetNativeType,
    i64: num_traits::AsPrimitive<T>,
    UnitDecoderFunction<T>: Default + DecoderFunction<T, T>,
{
    pub(crate) fn unit() -> Self {
        Self::new(PrimitiveDecoder::unit())
    }
}

impl<P, T> IntDecoder<P, T, AsDecoderFunction<P, T>>
where
    P: ParquetNativeType,
    T: NativeType,
    i64: num_traits::AsPrimitive<P>,
    AsDecoderFunction<P, T>: Default + DecoderFunction<P, T>,
{
    pub(crate) fn cast_as() -> Self {
        Self::new(PrimitiveDecoder::cast_as())
    }
}

impl<P, T> IntDecoder<P, T, IntoDecoderFunction<P, T>>
where
    P: ParquetNativeType,
    T: NativeType,
    i64: num_traits::AsPrimitive<P>,
    IntoDecoderFunction<P, T>: Default + DecoderFunction<P, T>,
{
    pub(crate) fn cast_into() -> Self {
        Self::new(PrimitiveDecoder::cast_into())
    }
}

impl<P, T, F> IntDecoder<P, T, ClosureDecoderFunction<P, T, F>>
where
    P: ParquetNativeType,
    T: NativeType,
    i64: num_traits::AsPrimitive<P>,
    F: Copy + Fn(P) -> T,
{
    pub(crate) fn closure(f: F) -> Self {
        Self::new(PrimitiveDecoder::closure(f))
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
    type Output = PrimitiveArray<T>;

    fn with_capacity(&self, capacity: usize) -> Self::DecodedState {
        self.0.with_capacity(capacity)
    }

    fn deserialize_dict(&self, page: DictPage) -> Self::Dict {
        self.0.deserialize_dict(page)
    }

    fn decode_plain_encoded<'a>(
        &mut self,
        (values, validity): &mut Self::DecodedState,
        page_values: &mut <Self::Translation<'a> as utils::StateTranslation<'a, Self>>::PlainDecoder,
        page_validity: Option<&mut PageValidity<'a>>,
        limit: usize,
    ) -> ParquetResult<()> {
        match page_validity {
            None => {
                PlainDecoderFnCollector {
                    chunks: page_values,
                    decoder: self.0.decoder,
                    _pd: Default::default(),
                }
                .push_n(values, limit)?;
            },
            Some(page_validity) => {
                let collector = PlainDecoderFnCollector {
                    chunks: page_values,
                    decoder: self.0.decoder,
                    _pd: Default::default(),
                };

                utils::extend_from_decoder(
                    validity,
                    page_validity,
                    Some(limit),
                    values,
                    collector,
                )?;
            },
        }

        Ok(())
    }

    fn decode_dictionary_encoded<'a>(
        &mut self,
        (values, validity): &mut Self::DecodedState,
        page_values: &mut hybrid_rle::HybridRleDecoder<'a>,
        page_validity: Option<&mut PageValidity<'a>>,
        dict: &Self::Dict,
        limit: usize,
    ) -> ParquetResult<()> {
        match page_validity {
            Some(page_validity) => {
                let translator = DictionaryTranslator(dict);
                let translated_hybridrle = TranslatedHybridRle::new(page_values, &translator);

                utils::extend_from_decoder(
                    validity,
                    page_validity,
                    Some(limit),
                    values,
                    translated_hybridrle,
                )?;
            },
            None => {
                let translator = DictionaryTranslator(dict);
                page_values.translate_and_collect_n_into(values, limit, &translator)?;
            },
        }

        Ok(())
    }

    fn finalize(
        &self,
        data_type: ArrowDataType,
        _dict: Option<Self::Dict>,
        (values, validity): Self::DecodedState,
    ) -> ParquetResult<Self::Output> {
        let validity = freeze_validity(validity);
        Ok(PrimitiveArray::try_new(data_type, values.into(), validity).unwrap())
    }
}

impl<P, T, D> utils::DictDecodable for IntDecoder<P, T, D>
where
    T: NativeType,
    P: ParquetNativeType,
    i64: num_traits::AsPrimitive<P>,
    D: DecoderFunction<P, T>,
{
    fn finalize_dict_array<K: DictionaryKey>(
        &self,
        data_type: ArrowDataType,
        dict: Self::Dict,
        keys: PrimitiveArray<K>,
    ) -> ParquetResult<DictionaryArray<K>> {
        let value_type = match &data_type {
            ArrowDataType::Dictionary(_, value, _) => value.as_ref().clone(),
            _ => T::PRIMITIVE.into(),
        };

        let dict = Box::new(PrimitiveArray::new(value_type, dict.into(), None));

        Ok(DictionaryArray::try_new(data_type, keys, dict).unwrap())
    }
}

impl<P, T, D> utils::NestedDecoder for IntDecoder<P, T, D>
where
    T: NativeType,
    P: ParquetNativeType,
    i64: num_traits::AsPrimitive<P>,
    D: DecoderFunction<P, T>,
{
    fn validity_extend(
        _: &mut utils::State<'_, Self>,
        (_, validity): &mut Self::DecodedState,
        value: bool,
        n: usize,
    ) {
        validity.extend_constant(n, value);
    }

    fn values_extend_nulls(
        _: &mut utils::State<'_, Self>,
        (values, _): &mut Self::DecodedState,
        n: usize,
    ) {
        values.resize(values.len() + n, T::default());
    }
}
