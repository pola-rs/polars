use arrow::array::{DictionaryArray, DictionaryKey, PrimitiveArray};
use arrow::bitmap::MutableBitmap;
use arrow::datatypes::ArrowDataType;
use arrow::types::NativeType;

use super::super::utils;
use crate::parquet::encoding::hybrid_rle::DictionaryTranslator;
use crate::parquet::encoding::{byte_stream_split, hybrid_rle, Encoding};
use crate::parquet::error::ParquetResult;
use crate::parquet::page::{split_buffer, DataPage, DictPage};
use crate::parquet::types::{decode, NativeType as ParquetNativeType};
use crate::read::deserialize::utils::array_chunks::ArrayChunks;
use crate::read::deserialize::utils::{
    dict_indices_decoder, freeze_validity, BatchableCollector, Decoder, PageValidity,
    TranslatedHybridRle,
};

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
pub(crate) struct UnitDecoderFunction<T>(std::marker::PhantomData<T>);
impl<T: NativeType + ParquetNativeType> DecoderFunction<T, T> for UnitDecoderFunction<T> {
    #[inline(always)]
    fn decode(self, x: T) -> T {
        x
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
pub(crate) struct IntoDecoderFunction<P, T>(std::marker::PhantomData<(P, T)>);
impl<P, T> DecoderFunction<P, T> for IntoDecoderFunction<P, T>
where
    P: ParquetNativeType + Into<T>,
    T: NativeType,
{
    #[inline(always)]
    fn decode(self, x: P) -> T {
        x.into()
    }
}

#[derive(Clone, Copy)]
pub(crate) struct ClosureDecoderFunction<P, T, F>(F, std::marker::PhantomData<(P, T)>);
impl<P, T, F> DecoderFunction<P, T> for ClosureDecoderFunction<P, T, F>
where
    P: ParquetNativeType,
    T: NativeType,
    F: Copy + Fn(P) -> T,
{
    #[inline(always)]
    fn decode(self, x: P) -> T {
        (self.0)(x)
    }
}

pub(crate) struct PlainDecoderFnCollector<'a, 'b, P, T, D>
where
    T: NativeType,
    P: ParquetNativeType,
    D: DecoderFunction<P, T>,
{
    pub(crate) chunks: &'b mut ArrayChunks<'a, P>,
    pub(crate) decoder: D,
    pub(crate) _pd: std::marker::PhantomData<T>,
}

impl<'a, 'b, P, T, D: DecoderFunction<P, T>> BatchableCollector<(), Vec<T>>
    for PlainDecoderFnCollector<'a, 'b, P, T, D>
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

    fn skip_in_place(&mut self, n: usize) -> ParquetResult<()> {
        self.chunks.skip_in_place(n);
        Ok(())
    }
}

#[allow(clippy::large_enum_variant)]
#[derive(Debug)]
pub(crate) enum StateTranslation<'a, P: ParquetNativeType> {
    Plain(ArrayChunks<'a, P>),
    Dictionary(hybrid_rle::HybridRleDecoder<'a>),
    ByteStreamSplit(byte_stream_split::Decoder<'a>),
}

impl<'a, P, T, D> utils::StateTranslation<'a, PrimitiveDecoder<P, T, D>> for StateTranslation<'a, P>
where
    T: NativeType,
    P: ParquetNativeType,
    D: DecoderFunction<P, T>,
{
    type PlainDecoder = ArrayChunks<'a, P>;

    fn new(
        _decoder: &PrimitiveDecoder<P, T, D>,
        page: &'a DataPage,
        dict: Option<&'a <PrimitiveDecoder<P, T, D> as utils::Decoder>::Dict>,
        _page_validity: Option<&PageValidity<'a>>,
    ) -> ParquetResult<Self> {
        match (page.encoding(), dict) {
            (Encoding::PlainDictionary | Encoding::RleDictionary, Some(_)) => {
                let values = dict_indices_decoder(page)?;
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
                    std::mem::size_of::<P>(),
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
        decoder: &mut PrimitiveDecoder<P, T, D>,
        decoded: &mut <PrimitiveDecoder<P, T, D> as utils::Decoder>::DecodedState,
        is_optional: bool,
        page_validity: &mut Option<PageValidity<'a>>,
        dict: Option<&'a <PrimitiveDecoder<P, T, D> as utils::Decoder>::Dict>,
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
                                .iter_converted(|v| decoder.decoder.decode(decode(v)))
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
                        &mut page_values.iter_converted(|v| decoder.decoder.decode(decode(v))),
                    )?,
                }
            },
        }

        Ok(())
    }
}

#[derive(Debug)]
pub(crate) struct PrimitiveDecoder<P, T, D>
where
    P: ParquetNativeType,
    T: NativeType,
    D: DecoderFunction<P, T>,
{
    pub(crate) decoder: D,
    _pd: std::marker::PhantomData<(P, T)>,
}

impl<P, T, D> PrimitiveDecoder<P, T, D>
where
    P: ParquetNativeType,
    T: NativeType,
    D: DecoderFunction<P, T>,
{
    #[inline]
    fn new(decoder: D) -> Self {
        Self {
            decoder,
            _pd: std::marker::PhantomData,
        }
    }
}

impl<T> PrimitiveDecoder<T, T, UnitDecoderFunction<T>>
where
    T: NativeType + ParquetNativeType,
    UnitDecoderFunction<T>: Default + DecoderFunction<T, T>,
{
    pub(crate) fn unit() -> Self {
        Self::new(UnitDecoderFunction::<T>::default())
    }
}

impl<P, T> PrimitiveDecoder<P, T, AsDecoderFunction<P, T>>
where
    P: ParquetNativeType,
    T: NativeType,
    AsDecoderFunction<P, T>: Default + DecoderFunction<P, T>,
{
    pub(crate) fn cast_as() -> Self {
        Self::new(AsDecoderFunction::<P, T>::default())
    }
}

impl<P, T> PrimitiveDecoder<P, T, IntoDecoderFunction<P, T>>
where
    P: ParquetNativeType,
    T: NativeType,
    IntoDecoderFunction<P, T>: Default + DecoderFunction<P, T>,
{
    pub(crate) fn cast_into() -> Self {
        Self::new(IntoDecoderFunction::<P, T>::default())
    }
}

impl<P, T, F> PrimitiveDecoder<P, T, ClosureDecoderFunction<P, T, F>>
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

impl<P, T, D> utils::Decoder for PrimitiveDecoder<P, T, D>
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
        Ok(deserialize_plain::<P, T, D>(&page.buffer, self.decoder))
    }

    fn decode_plain_encoded<'a>(
        &mut self,
        (values, validity): &mut Self::DecodedState,
        page_values: &mut <Self::Translation<'a> as utils::StateTranslation<'a, Self>>::PlainDecoder,
        is_optional: bool,
        page_validity: Option<&mut PageValidity<'a>>,
        limit: usize,
    ) -> ParquetResult<()> {
        match page_validity {
            None => {
                PlainDecoderFnCollector {
                    chunks: page_values,
                    decoder: self.decoder,
                    _pd: std::marker::PhantomData,
                }
                .push_n(values, limit)?;

                if is_optional {
                    validity.extend_constant(limit, true);
                }
            },
            Some(page_validity) => {
                let collector = PlainDecoderFnCollector {
                    chunks: page_values,
                    decoder: self.decoder,
                    _pd: std::marker::PhantomData,
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
        is_optional: bool,
        page_validity: Option<&mut PageValidity<'a>>,
        dict: &Self::Dict,
        limit: usize,
    ) -> ParquetResult<()> {
        let translator = DictionaryTranslator(dict);

        match page_validity {
            None => {
                page_values.translate_and_collect_n_into(values, limit, &translator)?;

                if is_optional {
                    validity.extend_constant(limit, true);
                }
            },
            Some(page_validity) => {
                let translated_hybridrle = TranslatedHybridRle::new(page_values, &translator);

                utils::extend_from_decoder(
                    validity,
                    page_validity,
                    Some(limit),
                    values,
                    translated_hybridrle,
                )?;
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

impl<P, T, D> utils::DictDecodable for PrimitiveDecoder<P, T, D>
where
    T: NativeType,
    P: ParquetNativeType,
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

impl<P, T, D> utils::NestedDecoder for PrimitiveDecoder<P, T, D>
where
    T: NativeType,
    P: ParquetNativeType,
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
