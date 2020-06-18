use crate::prelude::*;
use arrow::array::{PrimitiveBuilder, StringBuilder};
use arrow::datatypes::{ArrowPrimitiveType, Field};
use itertools::Itertools;
use std::cmp::Ordering;
use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};
use std::sync::Arc;

pub struct PrimitiveChunkedBuilder<T>
where
    T: ArrowPrimitiveType,
{
    pub builder: PrimitiveBuilder<T>,
    capacity: usize,
    field: Field,
}

impl<T> PrimitiveChunkedBuilder<T>
where
    T: ArrowPrimitiveType,
{
    pub fn new(name: &str, capacity: usize) -> Self {
        PrimitiveChunkedBuilder {
            builder: PrimitiveBuilder::<T>::new(capacity),
            capacity,
            field: Field::new(name, T::get_data_type(), true),
        }
    }

    pub fn finish(mut self) -> ChunkedArray<T> {
        ChunkedArray {
            field: self.field,
            chunks: vec![Arc::new(self.builder.finish())],
            chunk_id: format!("{}-", self.capacity).to_string(),
            phantom: PhantomData,
        }
    }
}

impl<T: ArrowPrimitiveType> Deref for PrimitiveChunkedBuilder<T> {
    type Target = PrimitiveBuilder<T>;

    fn deref(&self) -> &Self::Target {
        &self.builder
    }
}

impl<T: ArrowPrimitiveType> DerefMut for PrimitiveChunkedBuilder<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.builder
    }
}

pub struct Utf8ChunkedBuilder {
    pub builder: StringBuilder,
    capacity: usize,
    field: Field,
}

impl Utf8ChunkedBuilder {
    pub fn new(name: &str, capacity: usize) -> Self {
        Utf8ChunkedBuilder {
            builder: StringBuilder::new(capacity),
            capacity,
            field: Field::new(name, ArrowDataType::Utf8, true),
        }
    }

    pub fn finish(mut self) -> Utf8Chunked {
        ChunkedArray {
            field: self.field,
            chunks: vec![Arc::new(self.builder.finish())],
            chunk_id: format!("{}-", self.capacity).to_string(),
            phantom: PhantomData,
        }
    }
}

impl Deref for Utf8ChunkedBuilder {
    type Target = StringBuilder;

    fn deref(&self) -> &Self::Target {
        &self.builder
    }
}

impl DerefMut for Utf8ChunkedBuilder {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.builder
    }
}
