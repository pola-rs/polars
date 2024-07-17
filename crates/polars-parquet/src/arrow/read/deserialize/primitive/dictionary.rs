use std::collections::VecDeque;

use arrow::array::{Array, DictionaryArray, DictionaryKey, PrimitiveArray};
use arrow::bitmap::MutableBitmap;
use arrow::datatypes::ArrowDataType;
use arrow::types::NativeType;
use polars_error::PolarsResult;

use super::super::dictionary::*;
use super::super::nested_utils::{InitNested, NestedState};
use super::super::utils::MaybeNext;
use super::super::PagesIter;
use super::basic::deserialize_plain;
use super::DecoderFunction;
use crate::parquet::page::DictPage;
use crate::parquet::types::NativeType as ParquetNativeType;

fn read_dict<P, T, D>(data_type: ArrowDataType, dict: &DictPage, decoder: D) -> Box<dyn Array>
where
    T: NativeType,
    P: ParquetNativeType,
    D: DecoderFunction<P, T>,
{
    let data_type = match data_type {
        ArrowDataType::Dictionary(_, values, _) => *values,
        _ => data_type,
    };
    let values = deserialize_plain::<P, T, D>(&dict.buffer, decoder);

    Box::new(PrimitiveArray::new(data_type, values.into(), None))
}

/// An iterator adapter over [`PagesIter`] assumed to be encoded as boolean arrays
#[derive(Debug)]
pub struct DictIter<K, T, I, P, D>
where
    I: PagesIter,
    T: NativeType,
    K: DictionaryKey,
    P: ParquetNativeType,
    D: DecoderFunction<P, T>,
{
    iter: I,
    data_type: ArrowDataType,
    values: Option<Box<dyn Array>>,
    items: VecDeque<(Vec<K>, MutableBitmap)>,
    remaining: usize,
    chunk_size: Option<usize>,
    decoder: D,
    phantom: std::marker::PhantomData<(P, T)>,
}

impl<K, T, I, P, D> DictIter<K, T, I, P, D>
where
    K: DictionaryKey,
    I: PagesIter,
    T: NativeType,

    P: ParquetNativeType,
    D: DecoderFunction<P, T>,
{
    pub fn new(
        iter: I,
        data_type: ArrowDataType,
        num_rows: usize,
        chunk_size: Option<usize>,
        decoder: D,
    ) -> Self {
        Self {
            iter,
            data_type,
            values: None,
            items: VecDeque::new(),
            chunk_size,
            remaining: num_rows,
            decoder,
            phantom: Default::default(),
        }
    }
}

impl<K, T, I, P, D> Iterator for DictIter<K, T, I, P, D>
where
    I: PagesIter,
    T: NativeType,
    K: DictionaryKey,
    P: ParquetNativeType,
    D: DecoderFunction<P, T>,
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
            |dict| read_dict(self.data_type.clone(), dict, self.decoder),
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
pub struct NestedDictIter<K, T, I, P, D>
where
    I: PagesIter,
    T: NativeType,
    K: DictionaryKey,
    P: ParquetNativeType,
    D: DecoderFunction<P, T>,
{
    iter: I,
    init: Vec<InitNested>,
    data_type: ArrowDataType,
    values: Option<Box<dyn Array>>,
    items: VecDeque<(NestedState, (Vec<K>, MutableBitmap))>,
    remaining: usize,
    chunk_size: Option<usize>,
    decoder: D,
    phantom: std::marker::PhantomData<(P, T)>,
}

impl<K, T, I, P, D> NestedDictIter<K, T, I, P, D>
where
    K: DictionaryKey,
    I: PagesIter,
    T: NativeType,

    P: ParquetNativeType,
    D: DecoderFunction<P, T>,
{
    pub fn new(
        iter: I,
        init: Vec<InitNested>,
        data_type: ArrowDataType,
        num_rows: usize,
        chunk_size: Option<usize>,
        decoder: D,
    ) -> Self {
        Self {
            iter,
            init,
            data_type,
            values: None,
            items: VecDeque::new(),
            remaining: num_rows,
            chunk_size,
            decoder,
            phantom: Default::default(),
        }
    }
}

impl<K, T, I, P, D> Iterator for NestedDictIter<K, T, I, P, D>
where
    I: PagesIter,
    T: NativeType,
    K: DictionaryKey,
    P: ParquetNativeType,
    D: DecoderFunction<P, T>,
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
                |dict| read_dict(self.data_type.clone(), dict, self.decoder),
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
