use std::marker::PhantomData;

use polars_arrow::array::{DictionaryArray, DictionaryKey, MutableBinaryViewArray, PrimitiveArray};
use polars_arrow::bitmap::{Bitmap, BitmapBuilder, MutableBitmap};
use polars_arrow::datatypes::ArrowDataType;
use polars_error::PolarsResult;
use polars_utils::aliases::{InitHashMaps, PlHashMap};
use polars_utils::vec::with_cast_mut_vec;

use super::PredicateFilter;
use super::binview::{BinViewDecoder, BinaryIter};
use super::utils::{
    self, Decoder, DictPredicate, StateTranslation, dict_indices_decoder, freeze_validity,
    unspecialized_decode,
};
use crate::parquet::encoding::Encoding;
use crate::parquet::encoding::hybrid_rle::HybridRleDecoder;
use crate::parquet::error::ParquetResult;
use crate::parquet::page::{DataPage, DictPage, split_buffer};
use crate::read::ParquetError;
use crate::read::deserialize::dictionary_encoded::IndexMapping;
use crate::read::expr::SpecializedParquetColumnExpr;

#[derive(Clone)]
pub enum CategoricalTranslation<'a> {
    Dictionary(HybridRleDecoder<'a>),
    Plain(BinaryIter<'a>, usize),
}

impl<'a, T: DictionaryKey + IndexMapping<Output = T::AlignedBytes>>
    StateTranslation<'a, CategoricalDecoder<T>> for CategoricalTranslation<'a>
{
    type PlainDecoder = HybridRleDecoder<'a>;

    fn new(
        _decoder: &CategoricalDecoder<T>,
        page: &'a DataPage,
        _dict: Option<&'a <CategoricalDecoder<T> as Decoder>::Dict>,
        page_validity: Option<&Bitmap>,
    ) -> ParquetResult<Self> {
        match page.encoding() {
            Encoding::PlainDictionary | Encoding::RleDictionary => Ok(Self::Dictionary(
                dict_indices_decoder(page, page_validity.map_or(0, |bm| bm.unset_bits()))?,
            )),
            Encoding::Plain => {
                let values = split_buffer(page)?.values;
                let num_values = page.num_values();
                Ok(Self::Plain(BinaryIter::new(values, num_values), num_values))
            },
            _ => Err(utils::not_implemented(page)),
        }
    }

    fn num_rows(&self) -> usize {
        match self {
            Self::Dictionary(i) => i.len(),
            Self::Plain(_, num_rows) => *num_rows,
        }
    }
}

/// Special decoder for Polars Enum and Categorical's.
///
/// These are marked as special in the Arrow Field Metadata and they have the properly that for a
/// given row group all the values are in the dictionary page and all data pages are dictionary
/// encoded. This makes the job of decoding them extremely simple and fast.
pub struct CategoricalDecoder<T> {
    dict_size: usize,
    decoder: BinViewDecoder,
    plain_page_dict: MutableBinaryViewArray<[u8]>,
    plain_page_dict_lookup: PlHashMap<Box<[u8]>, T>,

    /// The predicate result for each value in `plain_page_dict` evaluated so far.
    plain_page_dict_mask: MutableBitmap,

    key_type: PhantomData<T>,
}

impl<T> CategoricalDecoder<T> {
    pub fn new() -> Self {
        Self {
            dict_size: usize::MAX,
            decoder: BinViewDecoder::new_string(),
            plain_page_dict: MutableBinaryViewArray::new(),
            plain_page_dict_lookup: PlHashMap::new(),
            plain_page_dict_mask: MutableBitmap::new(),
            key_type: PhantomData,
        }
    }
}

impl<T: DictionaryKey + IndexMapping<Output = T::AlignedBytes>> utils::Decoder
    for CategoricalDecoder<T>
{
    type Translation<'a> = CategoricalTranslation<'a>;
    type Dict = <BinViewDecoder as utils::Decoder>::Dict;
    type DecodedState = (Vec<T>, BitmapBuilder);
    type Output = DictionaryArray<T>;

    fn with_capacity(&self, capacity: usize) -> Self::DecodedState {
        (
            Vec::<T>::with_capacity(capacity),
            BitmapBuilder::with_capacity(capacity),
        )
    }

    fn evaluate_dict_predicate(
        &self,
        dict: &Self::Dict,
        predicate: &PredicateFilter,
    ) -> PolarsResult<Bitmap> {
        // SAFETY: The dictionary was checked to be UTF-8 when it was deserialized.
        let dict = unsafe { dict.to_utf8view_unchecked() };
        predicate.predicate.try_evaluate(&dict)
    }

    fn evaluate_predicate(
        &mut self,
        state: &utils::State<'_, Self>,
        _predicate: Option<&SpecializedParquetColumnExpr>,
        pred_true_mask: &mut BitmapBuilder,
        dict_mask: Option<&Bitmap>,
    ) -> ParquetResult<bool> {
        let CategoricalTranslation::Dictionary(translation) = &state.translation else {
            return Ok(false);
        };

        if state.page_validity.is_some() {
            // @Performance: implement validity aware
            return Ok(false);
        }

        let dict_mask = dict_mask.unwrap();
        super::dictionary_encoded::predicate::decode(
            translation.clone(),
            dict_mask,
            pred_true_mask,
        )?;

        Ok(true)
    }

    fn unspecialized_predicate_decode(
        &mut self,
        state: utils::State<'_, Self>,
        decoded: &mut Self::DecodedState,
        pred_true_mask: &mut BitmapBuilder,
        predicate: &PredicateFilter,
        dict: Option<Self::Dict>,
        dict_predicate: Option<&DictPredicate>,
        _dtype: &ArrowDataType,
    ) -> ParquetResult<()> {
        let is_optional = state.is_optional;
        let mut page = self.with_capacity(StateTranslation::<Self>::num_rows(&state.translation));
        self.extend_filtered_with_state(state, &mut page, None, &mut Vec::new())?;
        let (keys, validity) = page;

        let dict_len = dict.as_ref().map_or(0, |dict| dict.len());
        let dict_mask = match dict_predicate {
            Some(DictPredicate::Mask(mask)) => Some(mask),
            _ => None,
        };
        let mut evaluate_rows = matches!(dict_predicate, Some(DictPredicate::EvaluateRows));
        let num_evaluated = dict_len + self.plain_page_dict_mask.len();
        if !evaluate_rows && num_evaluated < dict_len + self.plain_page_dict.len() {
            let mut new_values = MutableBinaryViewArray::<[u8]>::new();
            for value in self
                .plain_page_dict
                .values_iter()
                .skip(num_evaluated - dict_len)
            {
                new_values.push_value(value);
            }
            // SAFETY: Plain page values are treated as UTF-8 when the array is finalized.
            let new_values = unsafe { new_values.freeze().to_utf8view_unchecked() };
            match predicate.predicate.try_evaluate(&new_values) {
                Ok(mask) => self.plain_page_dict_mask.extend_from_bitmap(&mask),
                Err(_) => evaluate_rows = true,
            }
        }

        let validity = is_optional.then(|| validity.freeze());

        // The predicate result for each valid row.
        let row_mask = if evaluate_rows {
            let mut values = MutableBinaryViewArray::<[u8]>::with_capacity(keys.len());
            for (i, key) in keys.iter().enumerate() {
                if validity.as_ref().is_none_or(|v| v.get_bit(i)) {
                    // SAFETY: The keys are unsigned.
                    let key = unsafe { key.as_usize() };
                    values.push_value(match &dict {
                        Some(dict) if key < dict_len => dict.value(key),
                        _ => self.plain_page_dict.value(key - dict_len),
                    });
                }
            }
            // SAFETY: The values are treated as UTF-8 when the array is finalized.
            let values = unsafe { values.freeze().to_utf8view_unchecked() };
            let mask = predicate
                .predicate
                .try_evaluate(&values)
                .map_err(|e| ParquetError::InvalidParameter(e.to_string()))?;
            Some(mask)
        } else {
            None
        };
        let mut num_valid = 0;

        let null_is_true = predicate.predicate.evaluate_null();
        let start_len = pred_true_mask.len();
        for (i, key) in keys.iter().enumerate() {
            // SAFETY: The validity has one bit per key.
            let is_valid = validity
                .as_ref()
                .is_none_or(|v| unsafe { v.get_bit_unchecked(i) });
            let keep = if !is_valid {
                null_is_true
            } else if let Some(row_mask) = &row_mask {
                num_valid += 1;
                row_mask.get_bit(num_valid - 1)
            } else {
                // SAFETY: The keys are unsigned.
                let key = unsafe { key.as_usize() };
                match dict_mask {
                    // SAFETY: The mask has one bit per dictionary value.
                    Some(dict_mask) if key < dict_len => unsafe {
                        dict_mask.get_bit_unchecked(key)
                    },
                    _ => self.plain_page_dict_mask.get(key - dict_len),
                }
            };
            pred_true_mask.push(keep);
            if keep && predicate.include_values {
                decoded.0.push(*key);
                if is_optional {
                    decoded.1.push(is_valid);
                }
            }
        }
        debug_assert_eq!(pred_true_mask.len() - start_len, keys.len());

        Ok(())
    }

    fn deserialize_dict(&mut self, page: DictPage) -> ParquetResult<Self::Dict> {
        let dict = self.decoder.deserialize_dict(page)?;
        self.dict_size = dict.len();
        Ok(dict)
    }

    fn extend_decoded(
        &self,
        decoded: &mut Self::DecodedState,
        additional: &dyn polars_arrow::array::Array,
        is_optional: bool,
    ) -> ParquetResult<()> {
        let additional = additional
            .as_any()
            .downcast_ref::<DictionaryArray<T>>()
            .unwrap();
        decoded.0.extend(additional.keys().values().iter().copied());
        match additional.validity() {
            Some(v) => decoded.1.extend_from_bitmap(v),
            None if is_optional => decoded.1.extend_constant(additional.len(), true),
            None => {},
        }

        Ok(())
    }

    fn finalize(
        &self,
        dtype: ArrowDataType,
        dict: Option<Self::Dict>,
        (values, validity): Self::DecodedState,
    ) -> ParquetResult<DictionaryArray<T>> {
        let validity = freeze_validity(validity);
        let keys = PrimitiveArray::new(T::PRIMITIVE.into(), values.into(), validity);

        let mut view_dict = MutableBinaryViewArray::with_capacity(
            dict.as_ref().map_or(0, |dict| dict.len()) + self.plain_page_dict.len(),
        );

        if let Some(dict) = dict {
            let (views, buffers, _, _, _) = dict.into_inner();

            for buffer in buffers.iter() {
                view_dict.push_buffer(buffer.clone());
            }
            unsafe { view_dict.views_mut().extend(views.iter()) };
            unsafe { view_dict.set_total_bytes_len(views.iter().map(|v| v.length as usize).sum()) };
        }

        for value in self.plain_page_dict.values_iter() {
            view_dict.push_value(value);
        }

        let view_dict = view_dict.freeze();

        // SAFETY: This was checked during construction of the dictionary
        let dict = unsafe { view_dict.to_utf8view_unchecked() }.boxed();

        // SAFETY: This was checked during decoding
        Ok(unsafe { DictionaryArray::try_new_unchecked(dtype, keys, dict) }.unwrap())
    }

    fn extend_filtered_with_state(
        &mut self,
        state: utils::State<'_, Self>,
        decoded: &mut Self::DecodedState,
        filter: Option<super::Filter>,
        _chunks: &mut Vec<Self::Output>,
    ) -> ParquetResult<()> {
        match state.translation {
            CategoricalTranslation::Dictionary(translation) => {
                with_cast_mut_vec::<T, T::AlignedBytes, _, _>(&mut decoded.0, |aligned_bytes_vec| {
                    super::dictionary_encoded::decode_dict_dispatch(
                        translation,
                        T::try_from(self.dict_size).ok().unwrap(),
                        state.is_optional,
                        state.page_validity.as_ref(),
                        filter,
                        &mut decoded.1,
                        aligned_bytes_vec,
                    )
                })
            },
            CategoricalTranslation::Plain(mut values, num_rows) => {
                let dict_len = state.dict.map_or(0, |dict| dict.len());
                let plain_page_dict = &mut self.plain_page_dict;
                let lookup = &mut self.plain_page_dict_lookup;

                unspecialized_decode(
                    num_rows,
                    || {
                        let value = values.next().unwrap();

                        let key = match lookup.get(value) {
                            Some(key) => *key,
                            None => {
                                let key =
                                    T::try_from(dict_len + plain_page_dict.len()).ok().unwrap();
                                plain_page_dict.push_value(value);
                                lookup.insert(value.into(), key);
                                key
                            },
                        };
                        Ok(key)
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

    fn extend_constant(
        &mut self,
        _decoded: &mut Self::DecodedState,
        _length: usize,
        _value: &crate::read::expr::ParquetScalar,
    ) -> ParquetResult<()> {
        Err(ParquetError::not_supported(
            "categorical with pushed-down equality filter",
        ))
    }
}
