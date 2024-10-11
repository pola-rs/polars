use arrow::array::{DictionaryArray, DictionaryKey, PrimitiveArray};
use arrow::bitmap::{Bitmap, MutableBitmap};
use arrow::datatypes::ArrowDataType;
use arrow::types::NativeType;
use polars_compute::filter::filter_boolean_kernel;

use super::super::utils;
use super::{
    deserialize_plain, AsDecoderFunction, ClosureDecoderFunction, DecoderFunction, DeltaCollector,
    DeltaTranslator, IntoDecoderFunction, PlainDecoderFnCollector, PrimitiveDecoder,
    UnitDecoderFunction,
};
use crate::parquet::encoding::hybrid_rle::gatherer::HybridRleGatherer;
use crate::parquet::encoding::hybrid_rle::{self, DictionaryTranslator};
use crate::parquet::encoding::{byte_stream_split, delta_bitpacked, Encoding};
use crate::parquet::error::ParquetResult;
use crate::parquet::page::{split_buffer, DataPage, DictPage};
use crate::parquet::types::{decode, NativeType as ParquetNativeType};
use crate::read::deserialize::utils::array_chunks::ArrayChunks;
use crate::read::deserialize::utils::{
    dict_indices_decoder, freeze_validity, BatchableCollector, Decoder, PageValidity,
    TranslatedHybridRle,
};
use crate::read::Filter;

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
        page_validity: &mut Option<PageValidity<'a>>,
        dict: Option<&'a <IntDecoder<P, T, D> as utils::Decoder>::Dict>,
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
        Ok(deserialize_plain::<P, T, D>(&page.buffer, self.0.decoder))
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
                    decoder: self.0.decoder,
                    _pd: Default::default(),
                }
                .push_n(values, limit)?;

                if is_optional {
                    validity.extend_constant(limit, true);
                }
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
        is_optional: bool,
        page_validity: Option<&mut PageValidity<'a>>,
        dict: &Self::Dict,
        limit: usize,
    ) -> ParquetResult<()> {
        match page_validity {
            None => {
                let translator = DictionaryTranslator(dict);
                page_values.translate_and_collect_n_into(values, limit, &translator)?;

                if is_optional {
                    validity.extend_constant(limit, true);
                }
            },
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
        }

        Ok(())
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
        state: &mut utils::State<'a, Self>,
        decoded: &mut Self::DecodedState,
        filter: Option<Filter>,
    ) -> ParquetResult<()> {
        struct BitmapGatherer;

        impl HybridRleGatherer<bool> for BitmapGatherer {
            type Target = MutableBitmap;

            fn target_reserve(&self, target: &mut Self::Target, n: usize) {
                target.reserve(n);
            }

            fn target_num_elements(&self, target: &Self::Target) -> usize {
                target.len()
            }

            fn hybridrle_to_target(&self, value: u32) -> ParquetResult<bool> {
                Ok(value != 0)
            }

            fn gather_one(&self, target: &mut Self::Target, value: bool) -> ParquetResult<()> {
                target.push(value);
                Ok(())
            }

            fn gather_repeated(
                &self,
                target: &mut Self::Target,
                value: bool,
                n: usize,
            ) -> ParquetResult<()> {
                target.extend_constant(n, value);
                Ok(())
            }
        }

        let num_rows = state.len();
        let mut max_offset = num_rows;

        if let Some(ref filter) = filter {
            max_offset = filter.max_offset();
            assert!(filter.max_offset() <= num_rows);
        }

        match state.translation {
            StateTranslation::Plain(ref mut chunks) => {
                let page_validity = state
                    .page_validity
                    .take()
                    .map(|v| {
                        let mut bm = MutableBitmap::with_capacity(max_offset);

                        let gatherer = BitmapGatherer;

                        v.clone().gather_n_into(&mut bm, max_offset, &gatherer)?;

                        ParquetResult::Ok(bm.freeze())
                    })
                    .transpose()?;

                let filter = match filter {
                    None => Bitmap::new_with_value(true, max_offset),
                    Some(Filter::Range(rng)) => {
                        let mut bm = MutableBitmap::with_capacity(max_offset);

                        bm.extend_constant(rng.start, false);
                        bm.extend_constant(rng.len(), true);

                        bm.freeze()
                    },
                    Some(Filter::Mask(bm)) => bm,
                };
                let validity =
                    page_validity.unwrap_or_else(|| Bitmap::new_with_value(true, max_offset));

                let start_len = decoded.0.len();
                decode_masked_plain(*chunks, &filter, &validity, &mut decoded.0, self.0.decoder);
                debug_assert_eq!(decoded.0.len() - start_len, filter.set_bits());
                if state.is_optional {
                    let validity = filter_boolean_kernel(&validity, &filter);
                    decoded.1.extend_from_bitmap(&validity);
                }

                chunks.skip_in_place(max_offset);

                Ok(())
            },
            StateTranslation::Dictionary(ref mut indexes) => {
                utils::dict_encoded::decode_dict(
                    indexes.clone(),
                    state.dict.unwrap(),
                    state.is_optional,
                    state.page_validity.clone(),
                    filter,
                    &mut decoded.1,
                    &mut decoded.0,
                )?;

                // @NOTE: Needed for compatibility now.
                indexes.skip_in_place(max_offset)?;
                if let Some(ref mut page_validity) = state.page_validity {
                    page_validity.skip_in_place(max_offset)?;
                }

                Ok(())
            },
            _ => self.extend_filtered_with_state_default(state, decoded, filter),
        }
    }
}

fn decode_masked_plain<P, T, D>(
    values: ArrayChunks<'_, P>,
    filter: &Bitmap,
    validity: &Bitmap,
    target: &mut Vec<T>,
    dfn: D,
) where
    T: NativeType,
    P: ParquetNativeType,
    i64: num_traits::AsPrimitive<P>,
    D: DecoderFunction<P, T>,
{
    /// # Safety
    ///
    /// All of the below need to hold:
    /// - `values.len() >= validity.count_ones()`
    /// - it needs to be safe to write to out[..filter.count_ones()]
    unsafe fn decode_iter_allow_last<P, T, D>(
        values: ArrayChunks<'_, P>,
        mut filter: u64,
        mut validity: u64,
        out: *mut T,
        dfn: D,
    ) -> (usize, usize)
    where
        T: NativeType,
        P: ParquetNativeType,
        i64: num_traits::AsPrimitive<P>,
        D: DecoderFunction<P, T>,
    {
        let mut num_read = 0;
        let mut num_written = 0;

        while filter != 0 {
            let offset = filter.trailing_zeros();

            num_read += (validity & (1u64 << offset).wrapping_sub(1)).count_ones() as usize;
            validity >>= offset;

            let v = if validity & 1 != 0 {
                dfn.decode(unsafe { values.get_unchecked(num_read) })
            } else {
                T::zeroed()
            };

            *out.add(num_written) = v;

            num_written += 1;
            num_read += (validity & 1) as usize;

            filter >>= offset + 1;
            validity >>= 1;
        }

        num_read += validity.count_ones() as usize;

        (num_read, num_written)
    }

    let start_length = target.len();
    let num_rows = filter.set_bits();
    let num_values = validity.set_bits();

    assert_eq!(filter.len(), validity.len());
    assert!(values.len() >= num_values);

    let mut filter_iter = filter.fast_iter_u56();
    let mut validity_iter = validity.fast_iter_u56();

    let mut values = values;

    target.reserve(num_rows);
    let mut target_ptr = unsafe { target.as_mut_ptr().add(start_length) };

    let mut num_rows_done = 0;
    let mut num_values_done = 0;

    for (f, v) in filter_iter.by_ref().zip(validity_iter.by_ref()) {
        let nv = v.count_ones() as usize;
        let nr = f.count_ones() as usize;

        // @NOTE:
        // If we cannot guarantee that there are more values than we need, we run the branching
        // variant and know we can write the remaining values as dummy values because they will be
        // None.
        if num_values_done + nv == num_values {
            // We don't really need to update `values` or `target_ptr` because we won't be
            // using them after this.
            _ = unsafe { decode_iter_allow_last::<P, T, D>(values, f, v, target_ptr, dfn) };

            num_rows_done += nr;
            unsafe { target.set_len(start_length + num_rows_done) };
            target.resize(start_length + num_rows, T::zeroed());
            return;
        }

        let mut v = v;
        let mut f = f;
        let mut num_read = 0;
        let mut num_written = 0;

        while f != 0 {
            let offset = f.trailing_zeros();

            num_read += (v & (1u64 << offset).wrapping_sub(1)).count_ones() as usize;
            v >>= offset;

            unsafe {
                *target_ptr.add(num_written) = dfn.decode(values.get_unchecked(num_read));
            }

            num_written += 1;
            num_read += (v & 1) as usize;

            f >>= offset + 1;
            v >>= 1;
        }

        num_read += v.count_ones() as usize;

        values = ArrayChunks {
            bytes: unsafe { values.bytes.get_unchecked(num_read..) },
        };
        target_ptr = unsafe { target_ptr.add(num_written) };

        num_values_done += nv;
        num_rows_done += nr;
    }

    let (f, fl) = filter_iter.remainder();
    let (v, vl) = validity_iter.remainder();

    assert_eq!(fl, vl);

    _ = unsafe { decode_iter_allow_last::<P, T, D>(values, f, v, target_ptr, dfn) };

    unsafe { target.set_len(start_length + num_rows) };
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
