use arrow::array::{DictionaryArray, MutableBinaryViewArray, PrimitiveArray};
use arrow::bitmap::{Bitmap, BitmapBuilder};
use arrow::datatypes::ArrowDataType;
use arrow::types::{AlignedBytes, NativeType};

use super::binview::BinViewDecoder;
use super::utils::{self, dict_indices_decoder, freeze_validity, Decoder, StateTranslation};
use super::PredicateFilter;
use crate::parquet::encoding::hybrid_rle::HybridRleDecoder;
use crate::parquet::encoding::Encoding;
use crate::parquet::error::ParquetResult;
use crate::parquet::page::{DataPage, DictPage};

impl<'a> StateTranslation<'a, CategoricalDecoder> for HybridRleDecoder<'a> {
    type PlainDecoder = HybridRleDecoder<'a>;

    fn new(
        _decoder: &CategoricalDecoder,
        page: &'a DataPage,
        _dict: Option<&'a <CategoricalDecoder as Decoder>::Dict>,
        page_validity: Option<&Bitmap>,
    ) -> ParquetResult<Self> {
        if !matches!(
            page.encoding(),
            Encoding::PlainDictionary | Encoding::RleDictionary
        ) {
            return Err(utils::not_implemented(page));
        }

        dict_indices_decoder(page, page_validity.map_or(0, |bm| bm.unset_bits()))
    }
    fn num_rows(&self) -> usize {
        self.len()
    }
}

/// Special decoder for Polars Enum and Categorical's.
///
/// These are marked as special in the Arrow Field Metadata and they have the properly that for a
/// given row group all the values are in the dictionary page and all data pages are dictionary
/// encoded. This makes the job of decoding them extremely simple and fast.
pub struct CategoricalDecoder {
    dict_size: usize,
    decoder: BinViewDecoder,
}

impl CategoricalDecoder {
    pub fn new() -> Self {
        Self {
            dict_size: usize::MAX,
            decoder: BinViewDecoder::new_string(),
        }
    }
}

impl utils::Decoder for CategoricalDecoder {
    type Translation<'a> = HybridRleDecoder<'a>;
    type Dict = <BinViewDecoder as utils::Decoder>::Dict;
    type DecodedState = (Vec<u32>, BitmapBuilder);
    type Output = DictionaryArray<u32>;

    fn with_capacity(&self, capacity: usize) -> Self::DecodedState {
        (
            Vec::<u32>::with_capacity(capacity),
            BitmapBuilder::with_capacity(capacity),
        )
    }

    fn has_predicate_specialization(
        &self,
        state: &utils::State<'_, Self>,
        _predicate: &PredicateFilter,
    ) -> ParquetResult<bool> {
        Ok(state.page_validity.is_none())
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
            .downcast_ref::<DictionaryArray<u32>>()
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
    ) -> ParquetResult<DictionaryArray<u32>> {
        let validity = freeze_validity(validity);
        let dict = dict.unwrap();
        let keys = PrimitiveArray::new(ArrowDataType::UInt32, values.into(), validity);

        let mut view_dict = MutableBinaryViewArray::with_capacity(dict.len());
        let (views, buffers, _, _, _) = dict.into_inner();

        for buffer in buffers.iter() {
            view_dict.push_buffer(buffer.clone());
        }
        unsafe { view_dict.views_mut().extend(views.iter()) };
        unsafe { view_dict.set_total_bytes_len(views.iter().map(|v| v.length as usize).sum()) };
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
        pred_true_mask: &mut BitmapBuilder,
        filter: Option<super::Filter>,
    ) -> ParquetResult<()> {
        super::dictionary_encoded::decode_dict_dispatch(
            state.translation,
            self.dict_size,
            state.dict_mask,
            state.is_optional,
            state.page_validity.as_ref(),
            filter,
            &mut decoded.1,
            <<u32 as NativeType>::AlignedBytes as AlignedBytes>::cast_vec_ref_mut(&mut decoded.0),
            pred_true_mask,
        )
    }
}
