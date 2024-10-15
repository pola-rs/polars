use arrow::array::{DictionaryArray, DictionaryKey, PrimitiveArray};
use arrow::bitmap::{Bitmap, MutableBitmap};
use arrow::datatypes::ArrowDataType;
use arrow::types::NativeType;

use super::super::utils;
use super::{
    AsDecoderFunction, ClosureDecoderFunction, DecoderFunction, DeltaCollector, DeltaTranslator,
    IntoDecoderFunction, PrimitiveDecoder, UnitDecoderFunction,
};
use crate::parquet::encoding::{byte_stream_split, delta_bitpacked, hybrid_rle, Encoding};
use crate::parquet::error::ParquetResult;
use crate::parquet::page::{split_buffer, DataPage, DictPage};
use crate::parquet::types::{decode, NativeType as ParquetNativeType};
use crate::read::deserialize::utils::array_chunks::ArrayChunks;
use crate::read::deserialize::utils::{dict_indices_decoder, freeze_validity};
use crate::read::{Filter, ParquetError};

#[allow(clippy::large_enum_variant)]
#[derive(Debug)]
pub(crate) enum StateTranslation<'a, P: ParquetNativeType> {
    Plain(ArrayChunks<'a, P>),
    Dictionary(hybrid_rle::HybridRleDecoder<'a>),
    ByteStreamSplit(byte_stream_split::Decoder<'a>),
    DeltaBinaryPacked(delta_bitpacked::Decoder<'a>),
}

impl<'a, P, T, D> utils::StateTranslation<'a, IntDecoder<P, T, D>> for StateTranslation<'a, P>
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
        page_validity: Option<&Bitmap>,
    ) -> ParquetResult<Self> {
        match (page.encoding(), dict) {
            (Encoding::PlainDictionary | Encoding::RleDictionary, Some(_)) => {
                let values =
                    dict_indices_decoder(page, page_validity.map_or(0, |bm| bm.unset_bits()))?;
                Ok(Self::Dictionary(values))
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
                    size_of::<P>(),
                )?))
            },
            (Encoding::DeltaBinaryPacked, _) => {
                let values = split_buffer(page)?.values;
                Ok(Self::DeltaBinaryPacked(
                    delta_bitpacked::Decoder::try_new(values)?.0,
                ))
            },
            _ => Err(utils::not_implemented(page)),
        }
    }

    fn len_when_not_nullable(&self) -> usize {
        match self {
            Self::Plain(v) => v.len(),
            Self::Dictionary(v) => v.len(),
            Self::ByteStreamSplit(v) => v.len(),
            Self::DeltaBinaryPacked(v) => v.len(),
        }
    }

    fn skip_in_place(&mut self, n: usize) -> ParquetResult<()> {
        if n == 0 {
            return Ok(());
        }

        match self {
            Self::Plain(v) => v.skip_in_place(n),
            Self::Dictionary(v) => v.skip_in_place(n)?,
            Self::ByteStreamSplit(v) => _ = v.iter_converted(|_| ()).nth(n - 1),
            Self::DeltaBinaryPacked(v) => v.skip_in_place(n)?,
        }

        Ok(())
    }

    fn extend_from_state(
        &mut self,
        decoder: &mut IntDecoder<P, T, D>,
        decoded: &mut <IntDecoder<P, T, D> as utils::Decoder>::DecodedState,
        is_optional: bool,
        page_validity: &mut Option<Bitmap>,
        _dict: Option<&'a <IntDecoder<P, T, D> as utils::Decoder>::Dict>,
        additional: usize,
    ) -> ParquetResult<()> {
        match self {
            Self::ByteStreamSplit(page_values) => {
                let (values, validity) = decoded;

                match page_validity {
                    None => {
                        values.extend(
                            page_values
                                .iter_converted(|v| decoder.0.decoder.decode(decode(v)))
                                .take(additional),
                        );

                        if is_optional {
                            validity.extend_constant(additional, true);
                        }
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

                let mut gatherer = DeltaTranslator {
                    dfn: decoder.0.decoder,
                    _pd: std::marker::PhantomData,
                };

                match page_validity {
                    None => {
                        page_values.gather_n_into(values, additional, &mut gatherer)?;

                        if is_optional {
                            validity.extend_constant(additional, true);
                        }
                    },
                    Some(page_validity) => utils::extend_from_decoder(
                        validity,
                        page_validity,
                        Some(additional),
                        values,
                        DeltaCollector {
                            decoder: page_values,
                            gatherer,
                        },
                    )?,
                }
            },
            _ => unreachable!(),
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
    fn new(decoder: D) -> Self {
        Self(PrimitiveDecoder::new(decoder))
    }
}

impl<T> IntDecoder<T, T, UnitDecoderFunction<T>>
where
    T: NativeType + ParquetNativeType,
    i64: num_traits::AsPrimitive<T>,
    UnitDecoderFunction<T>: Default + DecoderFunction<T, T>,
{
    pub(crate) fn unit() -> Self {
        Self::new(UnitDecoderFunction::<T>::default())
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
        Self::new(AsDecoderFunction::<P, T>::default())
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
        Self::new(IntoDecoderFunction::<P, T>::default())
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
        Self::new(ClosureDecoderFunction(f, std::marker::PhantomData))
    }
}

impl<P, T, D> utils::Decoder for IntDecoder<P, T, D>
where
    T: NativeType,
    P: ParquetNativeType,
    i64: num_traits::AsPrimitive<P>,
    D: DecoderFunction<P, T>,
{
    type Translation<'a> = StateTranslation<'a, P>;
    type Dict = Vec<T>;
    type DecodedState = (Vec<T>, MutableBitmap);
    type Output = PrimitiveArray<T>;

    fn with_capacity(&self, capacity: usize) -> Self::DecodedState {
        (
            Vec::<T>::with_capacity(capacity),
            MutableBitmap::with_capacity(capacity),
        )
    }

    fn deserialize_dict(&self, page: DictPage) -> ParquetResult<Self::Dict> {
        let Some(values) = ArrayChunks::<P>::new(page.buffer.as_ref()) else {
            return Err(ParquetError::oos(
                "Primitive dictionary page size is not a multiple of primitive size",
            ));
        };

        let mut target = Vec::new();
        super::plain::decode(
            values,
            false,
            None,
            None,
            &mut MutableBitmap::new(),
            &mut target,
            self.0.decoder,
        )?;
        Ok(target)
    }

    fn decode_plain_encoded<'a>(
        &mut self,
        _decoded: &mut Self::DecodedState,
        _page_values: &mut <Self::Translation<'a> as utils::StateTranslation<'a, Self>>::PlainDecoder,
        _is_optional: bool,
        _page_validity: Option<&mut Bitmap>,
        _limit: usize,
    ) -> ParquetResult<()> {
        unreachable!()
    }

    fn decode_dictionary_encoded<'a>(
        &mut self,
        _decoded: &mut Self::DecodedState,
        _page_values: &mut hybrid_rle::HybridRleDecoder<'a>,
        _is_optional: bool,
        _page_validity: Option<&mut Bitmap>,
        _dict: &Self::Dict,
        _limit: usize,
    ) -> ParquetResult<()> {
        unreachable!()
    }

    fn finalize(
        &self,
        dtype: ArrowDataType,
        _dict: Option<Self::Dict>,
        (values, validity): Self::DecodedState,
    ) -> ParquetResult<Self::Output> {
        let validity = freeze_validity(validity);
        Ok(PrimitiveArray::try_new(dtype, values.into(), validity).unwrap())
    }

    fn extend_filtered_with_state<'a>(
        &mut self,
        mut state: utils::State<'a, Self>,
        decoded: &mut Self::DecodedState,
        filter: Option<Filter>,
    ) -> ParquetResult<()> {
        match state.translation {
            StateTranslation::Plain(ref mut values) => super::plain::decode(
                values.clone(),
                state.is_optional,
                state.page_validity.as_ref(),
                filter,
                &mut decoded.1,
                &mut decoded.0,
                self.0.decoder,
            ),
            StateTranslation::Dictionary(ref mut indexes) => utils::dict_encoded::decode_dict(
                indexes.clone(),
                state.dict.unwrap(),
                state.is_optional,
                state.page_validity.as_ref(),
                filter,
                &mut decoded.1,
                &mut decoded.0,
            ),
            _ => self.extend_filtered_with_state_default(state, decoded, filter),
        }
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
        dtype: ArrowDataType,
        dict: Self::Dict,
        keys: PrimitiveArray<K>,
    ) -> ParquetResult<DictionaryArray<K>> {
        let value_type = match &dtype {
            ArrowDataType::Dictionary(_, value, _) => value.as_ref().clone(),
            _ => T::PRIMITIVE.into(),
        };

        let dict = Box::new(PrimitiveArray::new(value_type, dict.into(), None));

        Ok(DictionaryArray::try_new(dtype, keys, dict).unwrap())
    }
}
