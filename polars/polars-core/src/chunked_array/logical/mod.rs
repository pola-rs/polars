mod date;
mod datetime;
mod time;

pub use {date::*, datetime::*, time::*};

use crate::prelude::*;
use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};

/// Maps a logical type to a a chunked array implementation of the physical type.
/// This saves a lot of compiler bloat and allows us to reuse functionality.
pub struct Logical<K: PolarsDataType, T: PolarsDataType>(pub ChunkedArray<T>, PhantomData<K>);

impl<K: PolarsDataType, T: PolarsDataType> Clone for Logical<K, T> {
    fn clone(&self) -> Self {
        Logical::<K, _>::new(self.0.clone())
    }
}

impl<K: PolarsDataType, T: PolarsDataType> Deref for Logical<K, T> {
    type Target = ChunkedArray<T>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<K: PolarsDataType, T: PolarsDataType> DerefMut for Logical<K, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<K: PolarsDataType, T: PolarsDataType> Logical<K, T> {
    pub fn new<J: PolarsDataType>(ca: ChunkedArray<T>) -> Logical<J, T> {
        Logical(ca, PhantomData)
    }
}

pub trait LogicalType {
    /// Get data type of ChunkedArray.
    fn dtype(&self) -> &'static DataType;

    fn get_any_value(&self, _i: usize) -> AnyValue<'_> {
        unimplemented!()
    }
}

impl<K: PolarsDataType, T: PolarsDataType> Logical<K, T>
where
    Self: LogicalType,
{
    pub fn field(&self) -> Field {
        let name = self.0.ref_field().name();
        Field::new(name, LogicalType::dtype(self).clone())
    }
}
