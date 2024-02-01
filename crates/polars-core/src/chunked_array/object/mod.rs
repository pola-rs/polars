use std::any::Any;
use std::fmt::{Debug, Display};
use std::hash::Hash;

use arrow::bitmap::utils::{BitmapIter, ZipValidity};
use arrow::bitmap::Bitmap;

use crate::prelude::*;

pub mod builder;
#[cfg(feature = "object")]
pub(crate) mod extension;
mod is_valid;
mod iterator;
pub mod registry;

#[derive(Debug, Clone)]
pub struct ObjectArray<T>
where
    T: PolarsObject,
{
    pub(crate) values: Arc<Vec<T>>,
    pub(crate) null_bitmap: Option<Bitmap>,
    pub(crate) offset: usize,
    pub(crate) len: usize,
}

/// Trimmed down object safe polars object
pub trait PolarsObjectSafe: Any + Debug + Send + Sync + Display {
    fn type_name(&self) -> &'static str;

    fn as_any(&self) -> &dyn Any;

    fn to_boxed(&self) -> Box<dyn PolarsObjectSafe>;
}

/// Values need to implement this so that they can be stored into a Series and DataFrame
pub trait PolarsObject:
    Any + Debug + Clone + Send + Sync + Default + Display + Hash + PartialEq + Eq + TotalEq
{
    /// This should be used as type information. Consider this a part of the type system.
    fn type_name() -> &'static str;
}

impl<T: PolarsObject> PolarsObjectSafe for T {
    fn type_name(&self) -> &'static str {
        T::type_name()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn to_boxed(&self) -> Box<dyn PolarsObjectSafe> {
        Box::new(self.clone())
    }
}

pub type ObjectValueIter<'a, T> = std::slice::Iter<'a, T>;

impl<T> ObjectArray<T>
where
    T: PolarsObject,
{
    /// Get a reference to the underlying data
    pub fn values(&self) -> &Arc<Vec<T>> {
        &self.values
    }

    pub fn values_iter(&self) -> ObjectValueIter<'_, T> {
        self.values.iter()
    }

    /// Returns an iterator of `Option<&T>` over every element of this array.
    pub fn iter(&self) -> ZipValidity<&T, ObjectValueIter<'_, T>, BitmapIter> {
        ZipValidity::new_with_validity(self.values_iter(), self.null_bitmap.as_ref())
    }

    /// Get a value at a certain index location
    pub fn value(&self, index: usize) -> &T {
        &self.values[self.offset + index]
    }

    pub fn get(&self, index: usize) -> Option<&T> {
        if self.is_valid(index) {
            Some(unsafe { self.value_unchecked(index) })
        } else {
            None
        }
    }

    /// Get a value at a certain index location
    ///
    /// # Safety
    ///
    /// This does not any bound checks. The caller needs to ensure the index is within
    /// the size of the array.
    pub unsafe fn value_unchecked(&self, index: usize) -> &T {
        self.values.get_unchecked(index)
    }

    /// Check validity
    ///
    /// # Safety
    /// No bounds checks
    #[inline]
    pub unsafe fn is_valid_unchecked(&self, i: usize) -> bool {
        if let Some(b) = &self.null_bitmap {
            b.get_bit_unchecked(i)
        } else {
            true
        }
    }

    /// Check validity
    ///
    /// # Safety
    /// No bounds checks
    #[inline]
    pub unsafe fn is_null_unchecked(&self, i: usize) -> bool {
        !self.is_valid_unchecked(i)
    }

    #[inline]
    pub(crate) unsafe fn get_unchecked(&self, item: usize) -> Option<&T> {
        if self.is_null_unchecked(item) {
            None
        } else {
            Some(self.value_unchecked(item))
        }
    }

    /// Returns this array with a new validity.
    /// # Panic
    /// Panics iff `validity.len() != self.len()`.
    #[must_use]
    #[inline]
    pub fn with_validity(mut self, validity: Option<Bitmap>) -> Self {
        self.set_validity(validity);
        self
    }

    /// Sets the validity of this array.
    /// # Panics
    /// This function panics iff `values.len() != self.len()`.
    #[inline]
    pub fn set_validity(&mut self, validity: Option<Bitmap>) {
        if matches!(&validity, Some(bitmap) if bitmap.len() != self.len()) {
            panic!("validity must be equal to the array's length")
        }
        self.null_bitmap = validity;
    }
}

impl<T> Array for ObjectArray<T>
where
    T: PolarsObject,
{
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn data_type(&self) -> &ArrowDataType {
        unimplemented!()
    }

    fn slice(&mut self, offset: usize, length: usize) {
        assert!(
            offset + length <= self.len(),
            "the offset of the new Buffer cannot exceed the existing length"
        );
        unsafe { self.slice_unchecked(offset, length) }
    }

    unsafe fn slice_unchecked(&mut self, offset: usize, length: usize) {
        let len = std::cmp::min(self.len - offset, length);

        self.len = len;
        self.offset = offset;
    }

    fn len(&self) -> usize {
        self.len
    }

    fn validity(&self) -> Option<&Bitmap> {
        self.null_bitmap.as_ref()
    }

    fn with_validity(&self, validity: Option<Bitmap>) -> Box<dyn Array> {
        Box::new(self.clone().with_validity(validity))
    }

    fn to_boxed(&self) -> Box<dyn Array> {
        Box::new(self.clone())
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        unimplemented!()
    }

    fn null_count(&self) -> usize {
        match &self.null_bitmap {
            None => 0,
            Some(validity) => validity.unset_bits(),
        }
    }
}

impl<T> ObjectChunked<T>
where
    T: PolarsObject,
{
    /// Get a hold to an object that can be formatted or downcasted via the Any trait.
    ///
    /// # Safety
    ///
    /// No bounds checks
    pub unsafe fn get_object_unchecked(&self, index: usize) -> Option<&dyn PolarsObjectSafe> {
        let (chunk_idx, idx) = self.index_to_chunked_index(index);
        self.get_object_chunked_unchecked(chunk_idx, idx)
    }

    pub(crate) unsafe fn get_object_chunked_unchecked(&self, chunk: usize, index: usize) -> Option<&dyn PolarsObjectSafe> {
        let chunks = self.downcast_chunks();
        let arr = chunks.get_unchecked(chunk);
        if arr.is_valid_unchecked(index) {
            Some(arr.value(index))
        } else {
            None
        }
    }

    /// Get a hold to an object that can be formatted or downcasted via the Any trait.
    pub fn get_object(&self, index: usize) -> Option<&dyn PolarsObjectSafe> {
        if index < self.len() {
            unsafe { self.get_object_unchecked(index) }
        } else {
            None
        }
    }
}
