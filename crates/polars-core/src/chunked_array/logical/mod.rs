#[cfg(feature = "dtype-date")]
mod date;
#[cfg(feature = "dtype-date")]
pub use date::*;
#[cfg(feature = "dtype-datetime")]
mod datetime;
#[cfg(feature = "dtype-datetime")]
pub use datetime::*;
#[cfg(feature = "dtype-decimal")]
mod decimal;
#[cfg(feature = "dtype-decimal")]
pub use decimal::*;
#[cfg(feature = "dtype-duration")]
mod duration;
#[cfg(feature = "dtype-duration")]
pub use duration::*;
#[cfg(feature = "dtype-categorical")]
pub mod categorical;
#[cfg(feature = "dtype-time")]
mod time;

use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};

#[cfg(feature = "dtype-categorical")]
pub use categorical::*;
#[cfg(feature = "dtype-time")]
pub use time::*;

use crate::chunked_array::cast::CastOptions;
use crate::prelude::*;

/// Maps a logical type to a chunked array implementation of the physical type.
/// This saves a lot of compiler bloat and allows us to reuse functionality.
pub struct Logical<Logical: PolarsDataType, Physical: PolarsDataType>(
    pub ChunkedArray<Physical>,
    PhantomData<Logical>,
    pub Option<DataType>,
);

impl<K: PolarsDataType, T: PolarsDataType> Clone for Logical<K, T> {
    fn clone(&self) -> Self {
        let mut new = Logical::<K, _>::new_logical(self.0.clone());
        new.2.clone_from(&self.2);
        new
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
    pub fn new_logical<J: PolarsDataType>(ca: ChunkedArray<T>) -> Logical<J, T> {
        Logical(ca, PhantomData, None)
    }
}

pub trait LogicalType {
    /// Get data type of [`ChunkedArray`].
    fn dtype(&self) -> &DataType;

    /// Gets [`AnyValue`] from [`LogicalType`]
    fn get_any_value(&self, _i: usize) -> PolarsResult<AnyValue<'_>> {
        unimplemented!()
    }

    /// # Safety
    /// Does not do any bound checks.
    unsafe fn get_any_value_unchecked(&self, _i: usize) -> AnyValue<'_> {
        unimplemented!()
    }

    fn cast_with_options(&self, dtype: &DataType, options: CastOptions) -> PolarsResult<Series>;

    fn cast(&self, dtype: &DataType) -> PolarsResult<Series> {
        self.cast_with_options(dtype, CastOptions::NonStrict)
    }
}

impl<K: PolarsDataType, T: PolarsDataType> Logical<K, T>
where
    Self: LogicalType,
{
    pub fn physical(&self) -> &ChunkedArray<T> {
        &self.0
    }
    pub fn field(&self) -> Field {
        let name = self.0.ref_field().name();
        Field::new(name, LogicalType::dtype(self).clone())
    }
}
