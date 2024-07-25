use std::collections::VecDeque;

use arrow::array::{Array, DictionaryArray, DictionaryKey, FixedSizeBinaryArray, PrimitiveArray};
use arrow::bitmap::{Bitmap, MutableBitmap};
use arrow::datatypes::ArrowDataType;
use polars_error::PolarsResult;

use super::super::dictionary::*;
use super::super::utils::MaybeNext;
use crate::arrow::read::deserialize::nested_utils::{InitNested, NestedState};
use crate::parquet::error::ParquetResult;
use crate::parquet::page::DictPage;
use crate::parquet::read::BasicDecompressor;
use crate::read::deserialize::utils::{DictArrayDecoder, ExactSize};
use crate::read::CompressedPagesIter;

impl ExactSize for FixedSizeBinaryArray {
    fn len(&self) -> usize {
        FixedSizeBinaryArray::len(self)
    }
}

pub(crate) struct FixedSizeBinaryDictArrayDecoder {
    pub(crate) size: usize,
}

impl<K: DictionaryKey> DictArrayDecoder<K> for FixedSizeBinaryDictArrayDecoder {
    type Translation<'a> = super::super::primitive::dictionary::StateTranslation<'a, K, Self>;
    type Dict = FixedSizeBinaryArray;

    fn deserialize_dict(&self, page: DictPage) -> Self::Dict {
        let values = page.buffer.into_vec();

        FixedSizeBinaryArray::try_new(
            ArrowDataType::FixedSizeBinary(self.size),
            values.into(),
            None,
        )
        .unwrap()
    }

    fn finalize(
        &self,
        data_type: ArrowDataType,
        dict: Self::Dict,
        (values, validity): (Vec<K>, Option<Bitmap>),
    ) -> ParquetResult<DictionaryArray<K>> {
        let array = PrimitiveArray::<K>::new(K::PRIMITIVE.into(), values.into(), validity);
        Ok(DictionaryArray::try_new(data_type, array, Box::new(dict)).unwrap())
    }
}

fn read_dict(data_type: ArrowDataType, dict: &DictPage) -> Box<dyn Array> {
    let data_type = match data_type {
        ArrowDataType::Dictionary(_, values, _) => *values,
        _ => data_type,
    };

    let values = dict.buffer.clone();

    FixedSizeBinaryArray::try_new(data_type, values.to_vec().into(), None)
        .unwrap()
        .boxed()
}

/// An iterator adapter that converts [`DataPages`] into an [`Iterator`] of [`DictionaryArray`].
pub struct NestedDictIter<K, I>
where
    I: CompressedPagesIter,
    K: DictionaryKey,
{
    iter: BasicDecompressor<I>,
    init: Vec<InitNested>,
    data_type: ArrowDataType,
    values: Option<Box<dyn Array>>,
    items: VecDeque<(NestedState, (Vec<K>, MutableBitmap))>,
    remaining: usize,
    chunk_size: Option<usize>,
}

impl<K, I> NestedDictIter<K, I>
where
    I: CompressedPagesIter,
    K: DictionaryKey,
{
    pub fn new(
        iter: BasicDecompressor<I>,
        init: Vec<InitNested>,
        data_type: ArrowDataType,
        num_rows: usize,
        chunk_size: Option<usize>,
    ) -> Self {
        Self {
            iter,
            init,
            data_type,
            values: None,
            remaining: num_rows,
            items: VecDeque::new(),
            chunk_size,
        }
    }
}

impl<K, I> Iterator for NestedDictIter<K, I>
where
    I: CompressedPagesIter,
    K: DictionaryKey,
{
    type Item = PolarsResult<(NestedState, DictionaryArray<K>)>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let maybe_state = nested_next_dict(
                &mut self.iter,
                &mut self.items,
                &mut self.remaining,
                &self.init,
                &mut self.values,
                self.data_type.clone(),
                self.chunk_size,
                |dict| read_dict(self.data_type.clone(), dict),
            );
            match maybe_state {
                MaybeNext::Some(Ok(dict)) => return Some(Ok(dict)),
                MaybeNext::Some(Err(e)) => return Some(Err(e)),
                MaybeNext::None => return None,
                MaybeNext::More => continue,
            }
        }
    }
}
