use std::marker::PhantomData;

use arrow::array::{DictionaryArray, DictionaryKey, MutableBinaryViewArray, PrimitiveArray};
use arrow::bitmap::{Bitmap, BitmapBuilder};
use arrow::datatypes::ArrowDataType;
use polars_utils::aliases::{InitHashMaps, PlHashMap};
use polars_utils::vec::with_cast_mut_vec;

use super::binview::{BinViewDecoder, BinaryIter};
use super::utils::{
    self, Decoder, StateTranslation, dict_indices_decoder, freeze_validity, unspecialized_decode,
};
use crate::parquet::encoding::Encoding;
use crate::parquet::encoding::hybrid_rle::HybridRleDecoder;
use crate::parquet::error::ParquetResult;
use crate::parquet::page::{DataPage, DictPage, split_buffer};
use crate::read::ParquetError;
use crate::read::deserialize::dictionary_encoded::IndexMapping;
use crate::read::expr::SpecializedParquetColumnExpr;

/// The translation for a Categorical/Enum column.
///
/// Polars-written files always have every data page in a row group dictionary-encoded, but
/// third-party writers may re-export a Polars-tagged Arrow dictionary schema while writing plain
/// (non-dictionary-encoded) pages, e.g. once a writer's dictionary-page size limit is hit. The
/// `Plain` variant covers that case by decoding literal values and appending them to a
/// locally-grown extension of the dictionary (see `CategoricalDecoder::extra_dict`).
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
    /// Values decoded from `Encoding::Plain` pages, appended past the end of the real
    /// dictionary (if any). See [`CategoricalTranslation`] for why these can occur.
    extra_dict: MutableBinaryViewArray<[u8]>,
    /// Maps a value already pushed onto `extra_dict` back to its key, so that repeated values
    /// across (or within) `Encoding::Plain` pages don't each grow the dictionary.
    extra_dict_lookup: PlHashMap<Box<[u8]>, T>,

    key_type: PhantomData<T>,
}

impl<T> CategoricalDecoder<T> {
    pub fn new() -> Self {
        Self {
            dict_size: usize::MAX,
            decoder: BinViewDecoder::new_string(),
            extra_dict: MutableBinaryViewArray::new(),
            extra_dict_lookup: PlHashMap::new(),
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

    fn evaluate_predicate(
        &mut self,
        state: &utils::State<'_, Self>,
        _predicate: Option<&SpecializedParquetColumnExpr>,
        pred_true_mask: &mut BitmapBuilder,
        dict_mask: Option<&Bitmap>,
    ) -> ParquetResult<bool> {
        // Plain-encoded pages have no dictionary indices to evaluate against a dict mask; fall
        // back to the generic predicate path.
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

    fn deserialize_dict(&mut self, page: DictPage) -> ParquetResult<Self::Dict> {
        let dict = self.decoder.deserialize_dict(page)?;
        self.dict_size = dict.len();
        Ok(dict)
    }

    fn extend_decoded(
        &self,
        decoded: &mut Self::DecodedState,
        additional: &dyn arrow::array::Array,
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
            dict.as_ref().map_or(0, |dict| dict.len()) + self.extra_dict.len(),
        );

        if let Some(dict) = dict {
            let (views, buffers, _, _, _) = dict.into_inner();

            for buffer in buffers.iter() {
                view_dict.push_buffer(buffer.clone());
            }
            unsafe { view_dict.views_mut().extend(views.iter()) };
            unsafe { view_dict.set_total_bytes_len(views.iter().map(|v| v.length as usize).sum()) };
        }

        // Values from `Encoding::Plain` pages that had no dictionary page to reference; see
        // `CategoricalTranslation`.
        for value in self.extra_dict.values_iter() {
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
                // No dictionary page (or no more room in it) for these values: grow the
                // dictionary with whatever new literal values are found in this page. Values
                // are deduplicated against `extra_dict_lookup` so that repeats don't each
                // consume a new key (which matters for small physical key types like `u8`).
                let dict_len = state.dict.map_or(0, |dict| dict.len());
                let extra_dict = &mut self.extra_dict;
                let lookup = &mut self.extra_dict_lookup;

                unspecialized_decode(
                    num_rows,
                    || {
                        let value = values.next().unwrap();

                        let key = match lookup.get(value) {
                            Some(key) => *key,
                            None => {
                                let key = T::try_from(dict_len + extra_dict.len()).ok().unwrap();
                                extra_dict.push_value(value);
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
