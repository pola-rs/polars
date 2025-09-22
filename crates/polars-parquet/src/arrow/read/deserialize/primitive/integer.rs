use arrow::array::PrimitiveArray;
use arrow::bitmap::{Bitmap, BitmapBuilder};
use arrow::datatypes::ArrowDataType;
use arrow::types::{AlignedBytes, NativeType};
use bytemuck::Zeroable;

use super::super::utils;
use super::{
    AsDecoderFunction, ClosureDecoderFunction, DecoderFunction, IntoDecoderFunction,
    PrimitiveDecoder, UnitDecoderFunction,
};
use crate::parquet::encoding::{Encoding, byte_stream_split, delta_bitpacked, hybrid_rle};
use crate::parquet::error::ParquetResult;
use crate::parquet::page::{DataPage, DictPage, split_buffer};
use crate::parquet::types::{NativeType as ParquetNativeType, decode};
use crate::read::Filter;
use crate::read::deserialize::dictionary_encoded;
use crate::read::deserialize::utils::array_chunks::ArrayChunks;
use crate::read::deserialize::utils::{
    dict_indices_decoder, freeze_validity, unspecialized_decode,
};
use crate::read::expr::{ParquetScalar, SpecializedParquetColumnExpr};

#[allow(clippy::large_enum_variant)]
#[derive(Debug)]
pub(crate) enum StateTranslation<'a> {
    Plain(&'a [u8]),
    Dictionary(hybrid_rle::HybridRleDecoder<'a>),
    ByteStreamSplit(byte_stream_split::Decoder<'a>),
    DeltaBinaryPacked(delta_bitpacked::Decoder<'a>),
}

impl<'a, P, T, D> utils::StateTranslation<'a, IntDecoder<P, T, D>> for StateTranslation<'a>
where
    T: NativeType,
    P: ParquetNativeType,
    i64: num_traits::AsPrimitive<P>,
    D: DecoderFunction<P, T>,
{
    type PlainDecoder = &'a [u8];

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
                Ok(Self::Plain(values))
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
    fn num_rows(&self) -> usize {
        match self {
            Self::Plain(v) => v.len() / size_of::<P>(),
            Self::Dictionary(i) => i.len(),
            Self::ByteStreamSplit(i) => i.len(),
            Self::DeltaBinaryPacked(i) => i.len(),
        }
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
    type Translation<'a> = StateTranslation<'a>;
    type Dict = PrimitiveArray<T>;
    type DecodedState = (Vec<T>, BitmapBuilder);
    type Output = PrimitiveArray<T>;

    fn with_capacity(&self, capacity: usize) -> Self::DecodedState {
        (
            Vec::<T>::with_capacity(capacity),
            BitmapBuilder::with_capacity(capacity),
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
            &mut BitmapBuilder::new(),
            &mut self.0.intermediate,
            &mut target,
            self.0.decoder,
        )?;
        Ok(PrimitiveArray::new(
            T::PRIMITIVE.into(),
            target.into(),
            None,
        ))
    }

    fn evaluate_predicate(
        &mut self,
        state: &utils::State<'_, Self>,
        predicate: Option<&SpecializedParquetColumnExpr>,
        pred_true_mask: &mut BitmapBuilder,
        dict_mask: Option<&Bitmap>,
    ) -> ParquetResult<bool> {
        // @Performance: This should be added
        if state.page_validity.is_some() {
            return Ok(false);
        }

        if let StateTranslation::Dictionary(values) = &state.translation {
            let dict_mask = dict_mask.unwrap();
            super::super::dictionary_encoded::predicate::decode(
                values.clone(),
                dict_mask,
                pred_true_mask,
            )?;
            return Ok(true);
        }

        if !D::CAN_TRANSMUTE || D::NEED_TO_DECODE {
            return Ok(false);
        }

        let Some(predicate) = predicate else {
            return Ok(false);
        };

        use SpecializedParquetColumnExpr as S;
        match (&state.translation, predicate) {
            (StateTranslation::Plain(values), S::Equal(needle)) => {
                let values = ArrayChunks::new(values).unwrap();
                let needle = needle.to_aligned_bytes::<T::AlignedBytes>().unwrap();
                super::plain::predicate::decode_equals(values, needle, pred_true_mask);
            },
            (StateTranslation::Plain(values), S::Between(low, high)) => {
                let values = ArrayChunks::new(values).unwrap();
                use arrow::types::PrimitiveType as PT;
                let is_signed = match T::PRIMITIVE {
                    PT::Int8 | PT::Int16 | PT::Int32 | PT::Int64 => true,
                    PT::UInt8 | PT::UInt16 | PT::UInt32 | PT::UInt64 => false,
                    PT::Int128
                    | PT::Int256
                    | PT::UInt128
                    | PT::Float16
                    | PT::Float32
                    | PT::Float64
                    | PT::DaysMs
                    | PT::MonthDayNano
                    | PT::MonthDayMillis => return Ok(false),
                };

                let Some(low) = low.to_aligned_bytes::<T::AlignedBytes>() else {
                    return Ok(false);
                };
                let Some(high) = high.to_aligned_bytes::<T::AlignedBytes>() else {
                    return Ok(false);
                };

                let mut low1 = low;
                let mut high1 = high;
                let mut low2 = low;
                let mut high2 = high;

                if is_signed && !low.unsigned_leq(high) {
                    low1 = low;
                    high1 = T::AlignedBytes::ones();

                    low2 = T::AlignedBytes::zeros();
                    high2 = high;
                }

                super::plain::predicate::decode_between(
                    values,
                    low1,
                    high1,
                    low2,
                    high2,
                    pred_true_mask,
                );
            },
            (StateTranslation::Plain(values), S::EqualOneOf(needles))
                if (1..=8).contains(&needles.len()) =>
            {
                let values = ArrayChunks::new(values).unwrap();
                let mut needles_array = [<T::AlignedBytes>::zeroed(); 8];
                for i in 0..8 {
                    needles_array[i] = needles[i.min(needles.len() - 1)]
                        .to_aligned_bytes::<T::AlignedBytes>()
                        .unwrap();
                }
                super::plain::predicate::decode_is_in(values, &needles_array, pred_true_mask);
            },
            _ => return Ok(false),
        }

        Ok(true)
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

    fn extend_decoded(
        &self,
        decoded: &mut Self::DecodedState,
        additional: &dyn arrow::array::Array,
        is_optional: bool,
    ) -> ParquetResult<()> {
        let additional = additional
            .as_any()
            .downcast_ref::<PrimitiveArray<T>>()
            .unwrap();
        decoded.0.extend(additional.values().iter().copied());
        match additional.validity() {
            Some(v) => decoded.1.extend_from_bitmap(v),
            None if is_optional => decoded.1.extend_constant(additional.len(), true),
            None => {},
        }

        Ok(())
    }

    fn extend_constant(
        &mut self,
        decoded: &mut Self::DecodedState,
        length: usize,
        value: &ParquetScalar,
    ) -> ParquetResult<()> {
        self.0.extend_constant(decoded, length, value)
    }

    fn extend_filtered_with_state(
        &mut self,
        mut state: utils::State<'_, Self>,
        decoded: &mut Self::DecodedState,
        filter: Option<Filter>,
        _chunks: &mut Vec<Self::Output>,
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
            StateTranslation::Dictionary(ref mut indexes) => dictionary_encoded::decode_dict(
                indexes.clone(),
                state.dict.unwrap().values().as_slice(),
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
            StateTranslation::DeltaBinaryPacked(decoder) => {
                let num_rows = decoder.len();
                let values = decoder.collect::<Vec<i64>>()?;

                let mut i = 0;
                unspecialized_decode(
                    num_rows,
                    || {
                        use num_traits::AsPrimitive;
                        let value = values[i];
                        i += 1;
                        Ok(self.0.decoder.decode(value.as_()))
                    },
                    filter,
                    state.page_validity,
                    state.is_optional,
                    &mut decoded.1,
                    &mut decoded.0,
                )
            },
        }
    }
}
