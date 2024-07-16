use std::collections::VecDeque;

use arrow::array::{Array, DictionaryArray, DictionaryKey, MutableBinaryViewArray};
use arrow::bitmap::MutableBitmap;
use arrow::datatypes::{ArrowDataType, PhysicalType};
use polars_error::PolarsResult;

use super::super::dictionary::*;
use super::super::PagesIter;
use crate::arrow::read::deserialize::nested_utils::{InitNested, NestedState};
use crate::parquet::page::DictPage;
use crate::read::deserialize::binary::utils::BinaryIter;

/// An iterator adapter over [`PagesIter`] assumed to be encoded as parquet's dictionary-encoded binary representation
#[derive(Debug)]
pub struct DictIter<'a, K, I>
where
    I: PagesIter<'a>,
    K: DictionaryKey,
{
    iter: I,
    data_type: ArrowDataType,
    values: Option<Box<dyn Array>>,
    items: VecDeque<(Vec<K>, MutableBitmap)>,
    remaining: usize,
    chunk_size: Option<usize>,
    _pd: std::marker::PhantomData<&'a ()>,
}

impl<'a, K, I> DictIter<'a, K, I>
where
    K: DictionaryKey,
    I: PagesIter<'a>,
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
            _pd: std::marker::PhantomData,
        }
    }
}

fn read_dict(data_type: ArrowDataType, dict: &DictPage) -> Box<dyn Array> {
    let data_type = match data_type {
        ArrowDataType::Dictionary(_, values, _) => *values,
        _ => data_type,
    };

    let values = BinaryIter::new(&dict.buffer, dict.num_values);

    let mut data = MutableBinaryViewArray::<[u8]>::with_capacity(dict.num_values);
    for item in values {
        data.push_value(item)
    }

    match data_type.to_physical_type() {
        PhysicalType::Utf8View => data.freeze().to_utf8view().unwrap().boxed(),
        PhysicalType::BinaryView => data.freeze().boxed(),
        _ => unreachable!(),
    }
}

impl<'a, K, I> Iterator for DictIter<'a,K, I>
where
    I: PagesIter<'a>,
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

/// An iterator adapter that converts [`DataPages`] into an [`Iterator`] of [`DictionaryArray`]
#[derive(Debug)]
pub struct NestedDictIter<'a, K, I>
where
    I: PagesIter<'a>,
    K: DictionaryKey,
{
    iter: I,
    init: Vec<InitNested>,
    data_type: ArrowDataType,
    values: Option<Box<dyn Array>>,
    items: VecDeque<(NestedState, (Vec<K>, MutableBitmap))>,
    remaining: usize,
    chunk_size: Option<usize>,
            _pd: std::marker::PhantomData<&'a ()>,
}

impl<'a, K, I> NestedDictIter<'a, K, I>
where
    I: PagesIter<'a>,
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
            _pd: std::marker::PhantomData,
        }
    }
}

impl<'a, K, I> Iterator for NestedDictIter<'a, K, I>
where
    I: PagesIter<'a>,
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
