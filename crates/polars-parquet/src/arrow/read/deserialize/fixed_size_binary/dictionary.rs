use std::collections::VecDeque;

use arrow::array::{Array, DictionaryArray, DictionaryKey, FixedSizeBinaryArray};
use arrow::bitmap::MutableBitmap;
use arrow::datatypes::ArrowDataType;
use polars_error::PolarsResult;

use super::super::dictionary::*;
use super::super::utils::MaybeNext;
use super::super::Pages;
use crate::arrow::read::deserialize::nested_utils::{InitNested, NestedState};
use crate::parquet::page::DictPage;

/// An iterator adapter over [`Pages`] assumed to be encoded as parquet's dictionary-encoded binary representation
#[derive(Debug)]
pub struct DictIter<K, I>
where
    I: Pages,
    K: DictionaryKey,
{
    iter: I,
    data_type: ArrowDataType,
    values: Option<Box<dyn Array>>,
    items: VecDeque<(Vec<K>, MutableBitmap)>,
    remaining: usize,
    chunk_size: Option<usize>,
}

impl<K, I> DictIter<K, I>
where
    K: DictionaryKey,
    I: Pages,
{
    pub fn new(
        iter: I,
        data_type: ArrowDataType,
        num_rows: usize,
        chunk_size: Option<usize>,
    ) -> Self {
        Self {
            iter,
            data_type,
            values: None,
            items: VecDeque::new(),
            remaining: num_rows,
            chunk_size,
        }
    }
}

fn read_dict(data_type: ArrowDataType, dict: &DictPage) -> Box<dyn Array> {
    let data_type = match data_type {
        ArrowDataType::Dictionary(_, values, _) => *values,
        _ => data_type,
    };

    let values = dict.buffer.clone();

    FixedSizeBinaryArray::try_new(data_type, values.into(), None)
        .unwrap()
        .boxed()
}

impl<K, I> Iterator for DictIter<K, I>
where
    I: Pages,
    K: DictionaryKey,
{
    type Item = PolarsResult<DictionaryArray<K>>;

    fn next(&mut self) -> Option<Self::Item> {
        let maybe_state = next_dict(
            &mut self.iter,
            &mut self.items,
            &mut self.values,
            self.data_type.clone(),
            &mut self.remaining,
            self.chunk_size,
            |dict| read_dict(self.data_type.clone(), dict),
        );
        match maybe_state {
            MaybeNext::Some(Ok(dict)) => Some(Ok(dict)),
            MaybeNext::Some(Err(e)) => Some(Err(e)),
            MaybeNext::None => None,
            MaybeNext::More => self.next(),
        }
    }
}

/// An iterator adapter that converts [`DataPages`] into an [`Iterator`] of [`DictionaryArray`].
#[derive(Debug)]
pub struct NestedDictIter<K, I>
where
    I: Pages,
    K: DictionaryKey,
{
    iter: I,
    init: Vec<InitNested>,
    data_type: ArrowDataType,
    values: Option<Box<dyn Array>>,
    items: VecDeque<(NestedState, (Vec<K>, MutableBitmap))>,
    remaining: usize,
    chunk_size: Option<usize>,
}

impl<K, I> NestedDictIter<K, I>
where
    I: Pages,
    K: DictionaryKey,
{
    pub fn new(
        iter: I,
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
    I: Pages,
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
