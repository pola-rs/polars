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

#[cfg(feature = "dtype-categorical")]
pub use categorical::*;
#[cfg(feature = "dtype-time")]
pub use time::*;

use crate::chunked_array::cast::CastOptions;
use crate::prelude::*;

/// Maps a logical type to a chunked array implementation of the physical type.
/// This saves a lot of compiler bloat and allows us to reuse functionality.
pub struct Logical<Logical: PolarsDataType, Physical: PolarsDataType> {
    pub phys: ChunkedArray<Physical>,
    pub dtype: DataType,
    _phantom: PhantomData<Logical>,
}

impl<K: PolarsDataType, T: PolarsDataType> Clone for Logical<K, T> {
    fn clone(&self) -> Self {
        Self {
            phys: self.phys.clone(),
            dtype: self.dtype.clone(),
            _phantom: PhantomData,
        }
    }
}

impl<K: PolarsDataType, T: PolarsDataType> Logical<K, T> {
    /// # Safety
    /// You must uphold the logical types' invariants.
    pub unsafe fn new_logical(phys: ChunkedArray<T>, dtype: DataType) -> Logical<K, T> {
        Logical {
            phys,
            dtype,
            _phantom: PhantomData,
        }
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
    #[inline(always)]
    pub fn name(&self) -> &PlSmallStr {
        self.phys.name()
    }

    #[inline(always)]
    pub fn rename(&mut self, name: PlSmallStr) {
        self.phys.rename(name)
    }

    #[inline(always)]
    pub fn len(&self) -> usize {
        self.phys.len()
    }

    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[inline(always)]
    pub fn null_count(&self) -> usize {
        self.phys.null_count()
    }

    #[inline(always)]
    pub fn has_nulls(&self) -> bool {
        self.phys.has_nulls()
    }

    #[inline(always)]
    pub fn is_null(&self) -> BooleanChunked {
        self.phys.is_null()
    }

    #[inline(always)]
    pub fn is_not_null(&self) -> BooleanChunked {
        self.phys.is_not_null()
    }

    #[inline(always)]
    pub fn split_at(&self, offset: i64) -> (Self, Self) {
        let (left, right) = self.phys.split_at(offset);
        unsafe {
            (
                Self::new_logical(left, self.dtype.clone()),
                Self::new_logical(right, self.dtype.clone()),
            )
        }
    }

    #[inline(always)]
    pub fn slice(&self, offset: i64, length: usize) -> Self {
        unsafe { Self::new_logical(self.phys.slice(offset, length), self.dtype.clone()) }
    }

    #[inline(always)]
    pub fn field(&self) -> Field {
        let name = self.phys.ref_field().name();
        Field::new(name.clone(), LogicalType::dtype(self).clone())
    }

    #[inline(always)]
    pub fn physical(&self) -> &ChunkedArray<T> {
        &self.phys
    }

    #[inline(always)]
    pub fn physical_mut(&mut self) -> &mut ChunkedArray<T> {
        &mut self.phys
    }

    #[inline(always)]
    pub fn into_physical(self) -> ChunkedArray<T> {
        self.phys
    }
}
