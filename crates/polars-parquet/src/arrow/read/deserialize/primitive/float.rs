use arrow::array::PrimitiveArray;
use arrow::bitmap::{Bitmap, BitmapBuilder};
use arrow::datatypes::ArrowDataType;
use arrow::types::NativeType;

use super::super::utils;
use super::{ClosureDecoderFunction, DecoderFunction, PrimitiveDecoder, UnitDecoderFunction};
use crate::parquet::encoding::{byte_stream_split, hybrid_rle, Encoding};
use crate::parquet::error::ParquetResult;
use crate::parquet::page::{split_buffer, DataPage, DictPage};
use crate::parquet::types::{decode, NativeType as ParquetNativeType};
use crate::read::deserialize::dictionary_encoded;
use crate::read::deserialize::utils::{
    dict_indices_decoder, freeze_validity, unspecialized_decode,
};
use crate::read::{Filter, PredicateFilter};

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
    fn num_rows(&self) -> usize {
        match self {
            Self::Plain(v) => v.len() / size_of::<P>(),
            Self::Dictionary(i) => i.len(),
            Self::ByteStreamSplit(i) => i.len(),
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

impl<T: NativeType> utils::Decoded for (Vec<T>, BitmapBuilder) {
    fn len(&self) -> usize {
        self.0.len()
    }
    fn extend_nulls(&mut self, n: usize) {
        self.0.resize(self.0.len() + n, T::default());
        self.1.extend_constant(n, false);
    }
}

impl<P, T, D> utils::Decoder for FloatDecoder<P, T, D>
where
    T: NativeType,
    P: ParquetNativeType,
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
            &mut BitmapBuilder::new(),
            self.0.decoder,
        )?;
        Ok(PrimitiveArray::new(
            T::PRIMITIVE.into(),
            target.into(),
            None,
        ))
    }

    fn has_predicate_specialization(
        &self,
        state: &utils::State<'_, Self>,
        predicate: &PredicateFilter,
    ) -> ParquetResult<bool> {
        let mut has_predicate_specialization = false;

        has_predicate_specialization |=
            matches!(state.translation, StateTranslation::Dictionary(_));
        has_predicate_specialization |= matches!(state.translation, StateTranslation::Plain(_))
            && predicate.predicate.to_equals_scalar().is_some();

        // @TODO: This should be implemented
        has_predicate_specialization &= state.page_validity.is_none();

        Ok(has_predicate_specialization)
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

    fn extend_filtered_with_state(
        &mut self,
        mut state: utils::State<'_, Self>,
        decoded: &mut Self::DecodedState,
        pred_true_mask: &mut BitmapBuilder,
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
                pred_true_mask,
                self.0.decoder,
            ),
            StateTranslation::Dictionary(ref mut indexes) => dictionary_encoded::decode_dict(
                indexes.clone(),
                state.dict.unwrap().values().as_slice(),
                state.dict_mask,
                state.is_optional,
                state.page_validity.as_ref(),
                filter,
                &mut decoded.1,
                &mut decoded.0,
                pred_true_mask,
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
