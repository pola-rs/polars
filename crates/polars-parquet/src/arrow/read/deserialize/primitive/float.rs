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
use crate::read::deserialize::utils::array_chunks::ArrayChunks;
use crate::read::deserialize::utils::{
    dict_indices_decoder, freeze_validity, Decoder,
};
use crate::read::{Filter, ParquetError};

#[allow(clippy::large_enum_variant)]
#[derive(Debug)]
pub(crate) enum StateTranslation<'a, P: ParquetNativeType> {
    Plain(ArrayChunks<'a, P>),
    Dictionary(hybrid_rle::HybridRleDecoder<'a>),
    ByteStreamSplit(byte_stream_split::Decoder<'a>),
}

impl<'a, P, T, D> utils::StateTranslation<'a, FloatDecoder<P, T, D>> for StateTranslation<'a, P>
where
    T: NativeType,
    P: ParquetNativeType,
    D: DecoderFunction<P, T>,
{
    type PlainDecoder = ArrayChunks<'a, P>;

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
            _ => Err(utils::not_implemented(page)),
        }
    }

    fn len_when_not_nullable(&self) -> usize {
        match self {
            Self::Plain(n) => n.len(),
            Self::Dictionary(n) => n.len(),
            Self::ByteStreamSplit(n) => n.len(),
        }
    }

    fn skip_in_place(&mut self, n: usize) -> ParquetResult<()> {
        if n == 0 {
            return Ok(());
        }

        match self {
            Self::Plain(t) => t.skip_in_place(n),
            Self::Dictionary(t) => t.skip_in_place(n)?,
            Self::ByteStreamSplit(t) => _ = t.iter_converted(|_| ()).nth(n - 1),
        }

        Ok(())
    }

    fn extend_from_state(
        &mut self,
        decoder: &mut FloatDecoder<P, T, D>,
        decoded: &mut <FloatDecoder<P, T, D> as utils::Decoder>::DecodedState,
        is_optional: bool,
        page_validity: &mut Option<Bitmap>,
        dict: Option<&'a <FloatDecoder<P, T, D> as utils::Decoder>::Dict>,
        additional: usize,
    ) -> ParquetResult<()> {
        match self {
            Self::Plain(page_values) => decoder.decode_plain_encoded(
                decoded,
                page_values,
                is_optional,
                page_validity.as_mut(),
                additional,
            )?,
            Self::Dictionary(ref mut page) => decoder.decode_dictionary_encoded(
                decoded,
                page,
                is_optional,
                page_validity.as_mut(),
                dict.unwrap(),
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

                        if is_optional {
                            validity.extend_constant(additional, true);
                        }
                    },
                    Some(page_validity) => utils::extend_from_decoder(
                        validity,
                        page_validity,
                        Some(additional),
                        values,
                        &mut page_values.iter_converted(|v| decoder.0.decoder.decode(decode(v))),
                    )?,
                }
            },
        }

        Ok(())
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

    fn extend_filtered_with_state<'a>(
        &mut self,
        mut state: utils::State<'a, Self>,
        decoded: &mut Self::DecodedState,
        filter: Option<Filter>,
    ) -> ParquetResult<()> {
        let num_rows = state.len();
        let mut max_offset = num_rows;

        if let Some(ref filter) = filter {
            max_offset = filter.max_offset();
            assert!(filter.max_offset() <= num_rows);
        }

        match state.translation {
            StateTranslation::Plain(ref mut values) => {
                super::plain::decode(
                    values.clone(),
                    state.is_optional,
                    state.page_validity.as_ref(),
                    filter,
                    &mut decoded.1,
                    &mut decoded.0,
                    self.0.decoder,
                )?;

                // @NOTE: Needed for compatibility now.
                values.skip_in_place(max_offset);
                if let Some(ref mut page_validity) = state.page_validity {
                    page_validity.slice(max_offset, page_validity.len() - max_offset);
                }

                Ok(())
            },
            StateTranslation::Dictionary(ref mut indexes) => {
                utils::dict_encoded::decode_dict(
                    indexes.clone(),
                    state.dict.unwrap(),
                    state.is_optional,
                    state.page_validity.as_ref(),
                    filter,
                    &mut decoded.1,
                    &mut decoded.0,
                )?;

                // @NOTE: Needed for compatibility now.
                indexes.skip_in_place(max_offset)?;
                if let Some(ref mut page_validity) = state.page_validity {
                    page_validity.slice(max_offset, page_validity.len() - max_offset);
                }

                Ok(())
            },
            _ => self.extend_filtered_with_state_default(state, decoded, filter),
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
