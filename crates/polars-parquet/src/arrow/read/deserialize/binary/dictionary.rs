use std::collections::VecDeque;

use arrow::array::{Array, BinaryArray, DictionaryArray, DictionaryKey, PrimitiveArray, Utf8Array};
use arrow::bitmap::{Bitmap, MutableBitmap};
use arrow::datatypes::{ArrowDataType, PhysicalType};
use arrow::offset::Offset;
use polars_error::PolarsResult;

use super::super::dictionary::*;
use super::super::utils::MaybeNext;
use super::utils::Binary;
use crate::arrow::read::deserialize::nested_utils::{InitNested, NestedState};
use crate::parquet::error::ParquetResult;
use crate::parquet::page::DictPage;
use crate::parquet::read::BasicDecompressor;
use crate::read::deserialize::binary::utils::BinaryIter;
use crate::read::deserialize::utils::{self, DictArrayDecoder};
use crate::read::CompressedPagesIter;

#[derive(Default)]
pub(crate) struct BinaryDictArrayDecoder<O: Offset>(std::marker::PhantomData<O>);

impl<O: Offset> utils::ExactSize for Binary<O> {
    fn len(&self) -> usize {
        Binary::len(self)
    }
}

impl<O: Offset, K: DictionaryKey> DictArrayDecoder<K> for BinaryDictArrayDecoder<O> {
    type Translation<'a> = super::super::primitive::dictionary::StateTranslation<'a, K, Self>;
    type Dict = Binary<O>;

    fn deserialize_dict(&self, page: DictPage) -> Self::Dict {
        let values = BinaryIter::new(&page.buffer, page.num_values);

        let mut data = Binary::<O>::with_capacity(page.num_values);
        data.values = Vec::with_capacity(page.buffer.len() - 4 * page.num_values);
        for item in values {
            data.push(item)
        }

        data
    }

    fn finalize(
        &self,
        data_type: ArrowDataType,
        dict: Self::Dict,
        (values, validity): (Vec<K>, Option<Bitmap>),
    ) -> ParquetResult<DictionaryArray<K>> {
        let value_data_type = match data_type.clone() {
            ArrowDataType::Dictionary(_, values, _) => *values,
            v => v,
        };

        let dict = match value_data_type.to_physical_type() {
            PhysicalType::Utf8 | PhysicalType::LargeUtf8 => Utf8Array::<O>::new(
                value_data_type,
                dict.offsets.into(),
                dict.values.into(),
                None,
            )
            .boxed(),
            PhysicalType::Binary | PhysicalType::LargeBinary => BinaryArray::<O>::new(
                value_data_type,
                dict.offsets.into(),
                dict.values.into(),
                None,
            )
            .boxed(),
            _ => unreachable!(),
        };

        let indices = PrimitiveArray::new(K::PRIMITIVE.into(), values.into(), validity);

        // @TODO: Is this datatype correct?
        Ok(DictionaryArray::try_new(data_type, indices, dict).unwrap())
    }
}

fn read_dict<O: Offset>(data_type: ArrowDataType, dict: &DictPage) -> Box<dyn Array> {
    let data_type = match data_type {
        ArrowDataType::Dictionary(_, values, _) => *values,
        _ => data_type,
    };

    let values = BinaryIter::new(&dict.buffer, dict.num_values);

    let mut data = Binary::<O>::with_capacity(dict.num_values);
    data.values = Vec::with_capacity(dict.buffer.len() - 4 * dict.num_values);
    for item in values {
        data.push(item)
    }

    match data_type.to_physical_type() {
        PhysicalType::Utf8 | PhysicalType::LargeUtf8 => {
            Utf8Array::<O>::new(data_type, data.offsets.into(), data.values.into(), None).boxed()
        },
        PhysicalType::Binary | PhysicalType::LargeBinary => {
            BinaryArray::<O>::new(data_type, data.offsets.into(), data.values.into(), None).boxed()
        },
        _ => unreachable!(),
    }
}

/// An iterator adapter that converts [`DataPages`] into an [`Iterator`] of [`DictionaryArray`]
pub struct NestedDictIter<K, O, I>
where
    I: CompressedPagesIter,
    O: Offset,
    K: DictionaryKey,
{
    iter: BasicDecompressor<I>,
    init: Vec<InitNested>,
    data_type: ArrowDataType,
    values: Option<Box<dyn Array>>,
    items: VecDeque<(NestedState, (Vec<K>, MutableBitmap))>,
    remaining: usize,
    chunk_size: Option<usize>,
    phantom: std::marker::PhantomData<O>,
}

impl<K, O, I> NestedDictIter<K, O, I>
where
    I: CompressedPagesIter,
    O: Offset,
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
            items: VecDeque::new(),
            remaining: num_rows,
            chunk_size,
            phantom: Default::default(),
        }
    }
}

impl<K, O, I> Iterator for NestedDictIter<K, O, I>
where
    I: CompressedPagesIter,
    O: Offset,
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
                |dict| read_dict::<O>(self.data_type.clone(), dict),
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
