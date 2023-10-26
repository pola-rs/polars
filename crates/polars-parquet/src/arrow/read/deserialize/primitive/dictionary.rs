use std::collections::VecDeque;

use arrow::array::{Array, DictionaryArray, DictionaryKey, PrimitiveArray};
use arrow::bitmap::MutableBitmap;
use arrow::datatypes::DataType;
use arrow::types::NativeType;
use polars_error::PolarsResult;

use super::super::dictionary::{nested_next_dict, *};
use super::super::nested_utils::{InitNested, NestedState};
use super::super::utils::MaybeNext;
use super::super::Pages;
use super::basic::deserialize_plain;
use crate::parquet::page::DictPage;
use crate::parquet::types::NativeType as ParquetNativeType;

fn read_dict<P, T, F>(data_type: DataType, op: F, dict: &DictPage) -> Box<dyn Array>
where
    T: NativeType,
    P: ParquetNativeType,
    F: Copy + Fn(P) -> T,
{
    let data_type = match data_type {
        DataType::Dictionary(_, values, _) => *values,
        _ => data_type,
    };
    let values = deserialize_plain(&dict.buffer, op);

    Box::new(PrimitiveArray::new(data_type, values.into(), None))
}

/// An iterator adapter over [`Pages`] assumed to be encoded as boolean arrays
#[derive(Debug)]
pub struct DictIter<K, T, I, P, F>
where
    I: Pages,
    T: NativeType,
    K: DictionaryKey,
    P: ParquetNativeType,
    F: Fn(P) -> T,
{
    iter: I,
    data_type: DataType,
    values: Option<Box<dyn Array>>,
    items: VecDeque<(Vec<K>, MutableBitmap)>,
    remaining: usize,
    chunk_size: Option<usize>,
    op: F,
    phantom: std::marker::PhantomData<P>,
}

impl<K, T, I, P, F> DictIter<K, T, I, P, F>
where
    K: DictionaryKey,
    I: Pages,
    T: NativeType,

    P: ParquetNativeType,
    F: Copy + Fn(P) -> T,
{
    pub fn new(
        iter: I,
        data_type: DataType,
        num_rows: usize,
        chunk_size: Option<usize>,
        op: F,
    ) -> Self {
        Self {
            iter,
            data_type,
            values: None,
            items: VecDeque::new(),
            chunk_size,
            remaining: num_rows,
            op,
            phantom: Default::default(),
        }
    }
}

impl<K, T, I, P, F> Iterator for DictIter<K, T, I, P, F>
where
    I: Pages,
    T: NativeType,
    K: DictionaryKey,
    P: ParquetNativeType,
    F: Copy + Fn(P) -> T,
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
            |dict| read_dict::<P, T, _>(self.data_type.clone(), self.op, dict),
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
pub struct NestedDictIter<K, T, I, P, F>
where
    I: Pages,
    T: NativeType,
    K: DictionaryKey,
    P: ParquetNativeType,
    F: Fn(P) -> T,
{
    iter: I,
    init: Vec<InitNested>,
    data_type: DataType,
    values: Option<Box<dyn Array>>,
    items: VecDeque<(NestedState, (Vec<K>, MutableBitmap))>,
    remaining: usize,
    chunk_size: Option<usize>,
    op: F,
    phantom: std::marker::PhantomData<P>,
}

impl<K, T, I, P, F> NestedDictIter<K, T, I, P, F>
where
    K: DictionaryKey,
    I: Pages,
    T: NativeType,

    P: ParquetNativeType,
    F: Copy + Fn(P) -> T,
{
    pub fn new(
        iter: I,
        init: Vec<InitNested>,
        data_type: DataType,
        num_rows: usize,
        chunk_size: Option<usize>,
        op: F,
    ) -> Self {
        Self {
            iter,
            init,
            data_type,
            values: None,
            items: VecDeque::new(),
            remaining: num_rows,
            chunk_size,
            op,
            phantom: Default::default(),
        }
    }
}

impl<K, T, I, P, F> Iterator for NestedDictIter<K, T, I, P, F>
where
    I: Pages,
    T: NativeType,
    K: DictionaryKey,
    P: ParquetNativeType,
    F: Copy + Fn(P) -> T,
{
    type Item = PolarsResult<(NestedState, DictionaryArray<K>)>;

    fn next(&mut self) -> Option<Self::Item> {
        let maybe_state = nested_next_dict(
            &mut self.iter,
            &mut self.items,
            &mut self.remaining,
            &self.init,
            &mut self.values,
            self.data_type.clone(),
            self.chunk_size,
            |dict| read_dict::<P, T, _>(self.data_type.clone(), self.op, dict),
        );
        match maybe_state {
            MaybeNext::Some(Ok(dict)) => Some(Ok(dict)),
            MaybeNext::Some(Err(e)) => Some(Err(e)),
            MaybeNext::None => None,
            MaybeNext::More => self.next(),
        }
    }
}
