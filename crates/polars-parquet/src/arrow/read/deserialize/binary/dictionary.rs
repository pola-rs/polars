use std::collections::VecDeque;

use arrow::array::{Array, BinaryArray, DictionaryArray, DictionaryKey, Utf8Array};
use arrow::bitmap::MutableBitmap;
use arrow::datatypes::{ArrowDataType, PhysicalType};
use arrow::offset::Offset;
use polars_error::PolarsResult;

use super::super::dictionary::*;
use super::super::utils::MaybeNext;
use super::super::PagesIter;
use super::utils::Binary;
use crate::arrow::read::deserialize::nested_utils::{InitNested, NestedState};
use crate::parquet::page::DictPage;
use crate::read::deserialize::binary::utils::BinaryIter;

/// An iterator adapter over [`PagesIter`] assumed to be encoded as parquet's dictionary-encoded binary representation
#[derive(Debug)]
pub struct DictIter<K, O, I>
where
    I: PagesIter,
    O: Offset,
    K: DictionaryKey,
{
    iter: I,
    data_type: ArrowDataType,
    values: Option<Box<dyn Array>>,
    items: VecDeque<(Vec<K>, MutableBitmap)>,
    remaining: usize,
    chunk_size: Option<usize>,
    phantom: std::marker::PhantomData<O>,
}

impl<K, O, I> DictIter<K, O, I>
where
    K: DictionaryKey,
    O: Offset,
    I: PagesIter,
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
            phantom: std::marker::PhantomData,
        }
    }
}

fn read_dict<O: Offset>(data_type: ArrowDataType, dict: &DictPage) -> Box<dyn Array> {
    let data_type = match data_type {
        ArrowDataType::Dictionary(_, values, _) => *values,
        _ => data_type,
    };

    let values = BinaryIter::new(&dict.buffer).take(dict.num_values);

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

impl<K, O, I> Iterator for DictIter<K, O, I>
where
    I: PagesIter,
    O: Offset,
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
            |dict| read_dict::<O>(self.data_type.clone(), dict),
        );
        match maybe_state {
            MaybeNext::Some(Ok(dict)) => Some(Ok(dict)),
            MaybeNext::Some(Err(e)) => Some(Err(e)),
            MaybeNext::None => None,
            MaybeNext::More => self.next(),
        }
    }
}

/// An iterator adapter that converts [`DataPages`] into an [`Iterator`] of [`DictionaryArray`]
#[derive(Debug)]
pub struct NestedDictIter<K, O, I>
where
    I: PagesIter,
    O: Offset,
    K: DictionaryKey,
{
    iter: I,
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
    I: PagesIter,
    O: Offset,
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
            items: VecDeque::new(),
            remaining: num_rows,
            chunk_size,
            phantom: Default::default(),
        }
    }
}

impl<K, O, I> Iterator for NestedDictIter<K, O, I>
where
    I: PagesIter,
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
