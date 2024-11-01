use arrow::array::{DictionaryArray, DictionaryKey, PrimitiveArray};
use arrow::bitmap::{Bitmap, MutableBitmap};
use arrow::datatypes::ArrowDataType;
use arrow::types::NativeType;

use super::super::utils;
use super::{
    AsDecoderFunction, ClosureDecoderFunction, DecoderFunction, PrimitiveDecoder,
    UnitDecoderFunction,
};
use crate::parquet::encoding::{byte_stream_split, hybrid_rle, Encoding};
use crate::parquet::error::ParquetResult;
use crate::parquet::page::{split_buffer, DataPage, DictPage};
use crate::parquet::types::{decode, NativeType as ParquetNativeType};
use crate::read::deserialize::utils::{
    dict_indices_decoder, freeze_validity, unspecialized_decode,
};
use crate::read::Filter;

#[allow(clippy::large_enum_variant)]
#[derive(Debug)]
pub(crate) enum StateTranslation<'a> {
    Plain(&'a [u8]),
    Dictionary(hybrid_rle::HybridRleDecoder<'a>),
    ByteStreamSplit(byte_stream_split::Decoder<'a>),
}

impl<'a, P, T, D> utils::StateTranslation<'a, FloatDecoder<P, T, D>> for StateTranslation<'a>
where
    T: NativeType,
    P: ParquetNativeType,
    D: DecoderFunction<P, T>,
{
    type PlainDecoder = &'a [u8];

    fn new(
        _decoder: &FloatDecoder<P, T, D>,
        page: &'a DataPage,
        dict: Option<&'a <FloatDecoder<P, T, D> as utils::Decoder>::Dict>,
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
                Ok(Self::Plain(values))
            },
            (Encoding::ByteStreamSplit, _) => {
                let values = split_buffer(page)?.values;
                Ok(Self::ByteStreamSplit(byte_stream_split::Decoder::try_new(
                    values,
                    size_of::<P>(),
                )?))
            },
            _ => Err(utils::not_implemented(page)),
        }
    }
}

#[derive(Debug)]
pub(crate) struct FloatDecoder<P, T, D>(PrimitiveDecoder<P, T, D>)
where
    P: ParquetNativeType,
    T: NativeType,
    D: DecoderFunction<P, T>;

impl<P, T, D> FloatDecoder<P, T, D>
where
    P: ParquetNativeType,
    T: NativeType,
    D: DecoderFunction<P, T>,
{
    #[inline]
    fn new(decoder: D) -> Self {
        Self(PrimitiveDecoder::new(decoder))
    }
}

impl<T> FloatDecoder<T, T, UnitDecoderFunction<T>>
where
    T: NativeType + ParquetNativeType,
    UnitDecoderFunction<T>: Default + DecoderFunction<T, T>,
{
    pub(crate) fn unit() -> Self {
        Self::new(UnitDecoderFunction::<T>::default())
    }
}

impl<P, T> FloatDecoder<P, T, AsDecoderFunction<P, T>>
where
    P: ParquetNativeType,
    T: NativeType,
    AsDecoderFunction<P, T>: Default + DecoderFunction<P, T>,
{
    pub(crate) fn cast_as() -> Self {
        Self::new(AsDecoderFunction::<P, T>::default())
    }
}

impl<P, T, F> FloatDecoder<P, T, ClosureDecoderFunction<P, T, F>>
where
    P: ParquetNativeType,
    T: NativeType,
    F: Copy + Fn(P) -> T,
{
    pub(crate) fn closure(f: F) -> Self {
        Self::new(ClosureDecoderFunction(f, std::marker::PhantomData))
    }
}

impl<T> utils::ExactSize for (Vec<T>, MutableBitmap) {
    fn len(&self) -> usize {
        self.0.len()
    }
}

impl<P, T, D> utils::Decoder for FloatDecoder<P, T, D>
where
    T: NativeType,
    P: ParquetNativeType,
    D: DecoderFunction<P, T>,
{
    type Translation<'a> = StateTranslation<'a>;
    type Dict = Vec<T>;
    type DecodedState = (Vec<T>, MutableBitmap);
    type Output = PrimitiveArray<T>;

    fn with_capacity(&self, capacity: usize) -> Self::DecodedState {
        (
            Vec::<T>::with_capacity(capacity),
            MutableBitmap::with_capacity(capacity),
        )
    }

    fn deserialize_dict(&mut self, page: DictPage) -> ParquetResult<Self::Dict> {
        let values = page.buffer.as_ref();

        let mut target = Vec::with_capacity(page.num_values);
        super::plain::decode(
            values,
            false,
            None,
            None,
            &mut MutableBitmap::new(),
            &mut self.0.intermediate,
            &mut target,
            self.0.decoder,
        )?;
        Ok(target)
    }

    fn extend_filtered_with_state(
        &mut self,
        mut state: utils::State<'_, Self>,
        decoded: &mut Self::DecodedState,
        filter: Option<Filter>,
    ) -> ParquetResult<()> {
        match state.translation {
            StateTranslation::Plain(ref mut values) => super::plain::decode(
                values,
                state.is_optional,
                state.page_validity.as_ref(),
                filter,
                &mut decoded.1,
                &mut self.0.intermediate,
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
            StateTranslation::ByteStreamSplit(mut decoder) => {
                let num_rows = decoder.len();
                let mut iter = decoder.iter_converted(|v| self.0.decoder.decode(decode(v)));

                unspecialized_decode(
                    num_rows,
                    || Ok(iter.next().unwrap()),
                    filter,
                    state.page_validity,
                    state.is_optional,
                    &mut decoded.1,
                    &mut decoded.0,
                )
            },
        }
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
}

impl<P, T, D> utils::DictDecodable for FloatDecoder<P, T, D>
where
    T: NativeType,
    P: ParquetNativeType,
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
