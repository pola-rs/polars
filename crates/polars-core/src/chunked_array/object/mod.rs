#![allow(unsafe_op_in_unsafe_fn)]
use std::any::Any;
use std::borrow::Cow;
use std::fmt::{Debug, Display};
use std::hash::Hash;

use arrow::bitmap::utils::{BitmapIter, ZipValidity};
use arrow::bitmap::{Bitmap, MutableBitmap};
use polars_array::broadcast::{is_flat_buffer_len, is_valid_buffer_len};
use polars_array::builder::ShareStrategy;
use polars_array::{
    ArrayFromIter, Flat, PlArray, PlArrayType, PlBitmapRef, StaticArray, StaticArrayBuilder,
    ZeroableArrayFromIter,
};
use polars_buffer::Buffer;
use polars_utils::IdxSize;
use polars_utils::total_ord::TotalHash;

use crate::prelude::*;

pub mod builder;
#[cfg(feature = "object")]
pub(crate) mod extension;
pub mod iterator;
pub mod registry;

pub use extension::set_polars_allow_extension;
pub use iterator::ObjectIter;

#[derive(Debug, Clone)]
pub struct ObjectArray<T>
where
    T: PolarsObject,
{
    values: Buffer<T>,
    validity: Option<Bitmap>,
}

/// Trimmed down object safe polars object
pub trait PolarsObjectSafe: Any + Debug + Send + Sync + Display {
    fn type_name(&self) -> &'static str;

    fn as_any(&self) -> &dyn Any;

    fn to_boxed(&self) -> Box<dyn PolarsObjectSafe>;

    fn equal(&self, other: &dyn PolarsObjectSafe) -> bool;
}

impl PartialEq for &dyn PolarsObjectSafe {
    fn eq(&self, other: &Self) -> bool {
        self.equal(*other)
    }
}

/// Values need to implement this so that they can be stored into a Series and DataFrame
pub trait PolarsObject:
    Any + Debug + Clone + Send + Sync + Default + Display + Hash + TotalHash + PartialEq + Eq + TotalEq
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

    fn equal(&self, other: &dyn PolarsObjectSafe) -> bool {
        let Some(other) = other.as_any().downcast_ref::<T>() else {
            return false;
        };
        self == other
    }
}

pub type ObjectValueIter<'a, T> = std::slice::Iter<'a, T>;

impl<T> ObjectArray<T>
where
    T: PolarsObject,
{
    pub fn values_iter(&self) -> ObjectValueIter<'_, T> {
        self.values.iter()
    }

    /// Returns an iterator of `Option<&T>` over every element of this array.
    pub fn iter(&self) -> ZipValidity<&T, ObjectValueIter<'_, T>, BitmapIter<'_>> {
        ZipValidity::new_with_validity(self.values_iter(), self.validity.as_ref())
    }

    /// Get a value at a certain index location
    pub fn value(&self, index: usize) -> &T {
        &self.values[index]
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

    /// Get a value at a certain index location, or `None` if it is null.
    ///
    /// # Safety
    /// This does no bounds check; the index must be within the size of the array.
    #[inline]
    pub unsafe fn get_unchecked_opt(&self, index: usize) -> Option<&T> {
        unsafe {
            self.is_valid_unchecked(index)
                .then(|| self.value_unchecked(index))
        }
    }

    /// Check validity
    ///
    /// # Safety
    /// No bounds checks
    #[inline]
    pub unsafe fn is_valid_unchecked(&self, i: usize) -> bool {
        if let Some(b) = &self.validity {
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

    /// An array of `length` nulls. An object array holds one `T` per element even where the mask
    /// says there is no value, so this is `O(length)` in memory.
    pub fn new_full_null(length: usize) -> Self {
        Self {
            values: vec![T::default(); length].into(),
            validity: Some(Bitmap::new_with_value(false, length)),
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
    /// This function panics iff `validity.len() != self.len()`.
    #[inline]
    pub fn set_validity(&mut self, validity: Option<Bitmap>) {
        if matches!(&validity, Some(bitmap) if bitmap.len() != self.len()) {
            panic!("validity must be equal to the array's length")
        }
        self.validity = validity;
    }
}

impl<T: PolarsObject> Splitable for ObjectArray<T> {
    fn check_bound(&self, offset: usize) -> bool {
        offset <= self.len()
    }

    unsafe fn _split_at_unchecked(&self, offset: usize) -> (Self, Self) {
        let (left_values, right_values) = unsafe { self.values.split_at_unchecked(offset) };
        let (left_validity, right_validity) = unsafe { self.validity.split_at_unchecked(offset) };
        (
            Self {
                values: left_values,
                validity: left_validity,
            },
            Self {
                values: right_values,
                validity: right_validity,
            },
        )
    }
}

/// An object array is always [`flat`](polars_array::broadcast): it holds one `T` per element, so
/// there is no scalar representation for it to be in and no buffer for `to_flat` to write out.
impl<T: PolarsObject> PlArray for ObjectArray<T> {
    #[inline]
    fn as_any(&self) -> &dyn Any {
        self
    }

    #[inline]
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    #[inline]
    fn array_type(&self) -> PlArrayType {
        PlArrayType::Object {
            type_name: T::type_name(),
        }
    }

    #[inline]
    fn len(&self) -> usize {
        self.values.len()
    }

    /// Whether this array is one value repeated over its length, which an object array — holding
    /// one `T` per element — only is when it holds a single element.
    #[inline]
    fn is_scalar(&self) -> bool {
        self.values.len() == 1
    }

    #[inline]
    fn validity(&self) -> Option<PlBitmapRef<'_>> {
        self.validity
            .as_ref()
            .map(|validity| PlBitmapRef::new(validity, self.values.len()))
    }

    #[inline]
    fn null_count(&self) -> usize {
        self.validity.as_ref().map_or(0, |v| v.unset_bits())
    }

    fn slice(&mut self, offset: usize, length: usize) {
        assert!(
            offset + length <= self.len(),
            "the offset of the new Buffer cannot exceed the existing length"
        );
        unsafe { self.slice_unchecked(offset, length) }
    }

    unsafe fn slice_unchecked(&mut self, offset: usize, length: usize) {
        self.validity = self
            .validity
            .take()
            .map(|bitmap| bitmap.sliced_unchecked(offset, length))
            .filter(|bitmap| bitmap.unset_bits() > 0);
        self.values
            .slice_in_place_unchecked(offset..offset + length);
    }

    fn set_validity(&mut self, validity: Option<PlBitmap>) {
        // A mask that repeats a single bit covers every element with it; an object array has
        // nowhere to hold one, so it is written out to the bit per element its length calls for.
        self.validity = validity.map(|validity| {
            assert_eq!(
                validity.len(),
                self.len(),
                "validity mask of {} elements does not cover an array of length {}",
                validity.len(),
                self.len(),
            );
            validity.into_bitmap()
        });
    }

    unsafe fn new_from_index_unchecked(&self, index: usize, length: usize) -> Box<dyn PlArray> {
        debug_assert!(index < self.len());
        let is_valid = unsafe { self.is_valid_unchecked(index) };
        let value = unsafe { self.value_unchecked(index) }.clone();
        Box::new(ObjectArray {
            values: vec![value; length].into(),
            validity: (!is_valid).then(|| Bitmap::new_with_value(false, length)),
        })
    }

    #[inline]
    fn to_boxed(&self) -> Box<dyn PlArray> {
        Box::new(self.clone())
    }

    /// An object array lives outside `polars-array`, which therefore cannot build one; this is
    /// where it is built instead.
    fn full_null_like(&self, length: usize) -> Box<dyn PlArray> {
        Box::new(Self::new_full_null(length))
    }

    fn eq_dyn(&self, other: &dyn PlArray) -> bool {
        let Some(other) = other.as_any().downcast_ref::<Self>() else {
            return false;
        };
        self.len() == other.len()
            && self
                .iter()
                .zip(other.iter())
                .all(|(lhs, rhs)| match (lhs, rhs) {
                    (Some(lhs), Some(rhs)) => lhs == rhs,
                    (None, None) => true,
                    _ => false,
                })
    }
}

impl<T: PolarsObject> StaticArray for ObjectArray<T> {
    type ValueT<'a> = &'a T;
    type ZeroableValueT<'a> = Option<&'a T>;
    type ValueIterT<'a> = ObjectValueIter<'a, T>;
    type IterT<'a> = ObjectIter<'a, T>;
    type Builder = ObjectArrayBuilder<T>;

    #[inline]
    unsafe fn value_unchecked(&self, i: usize) -> &T {
        unsafe { self.value_unchecked(i) }
    }

    #[inline]
    fn values_iter(&self) -> Self::ValueIterT<'_> {
        self.values_iter()
    }

    #[inline]
    fn iter(&self) -> Self::IterT<'_> {
        ObjectIter::new(self)
    }

    #[inline]
    fn broadcast_values_iter(&self, length: usize) -> Self::ValueIterT<'_> {
        assert_eq!(self.len(), length, "an object array never broadcasts");
        self.values_iter()
    }

    #[inline]
    fn with_validity_typed(self, validity: Option<PlBitmap>) -> Self {
        let mut out = self;
        PlArray::set_validity(&mut out, validity);
        out
    }

    #[inline]
    fn new_from_index_typed(&self, index: usize, length: usize) -> Self {
        assert!(index < self.len(), "index out of bounds");
        let is_valid = unsafe { self.is_valid_unchecked(index) };
        let value = unsafe { self.value_unchecked(index) }.clone();
        ObjectArray {
            values: vec![value; length].into(),
            validity: (!is_valid).then(|| Bitmap::new_with_value(false, length)),
        }
    }

    #[inline]
    fn is_flat(&self) -> bool {
        true
    }

    #[inline]
    fn to_flat(&self) -> Cow<'_, Flat<Self>> {
        // SAFETY: an object array holds one `T` per element and never broadcasts.
        Cow::Borrowed(unsafe { Flat::new_ref(self) })
    }

    #[inline]
    fn as_flat(&self) -> Option<&Flat<Self>> {
        // SAFETY: an object array holds one `T` per element and never broadcasts.
        Some(unsafe { Flat::new_ref(self) })
    }
}

/// The builder of an [`ObjectArray`].
pub struct ObjectArrayBuilder<T: PolarsObject> {
    values: Vec<T>,
    validity: MutableBitmap,
}

impl<T: PolarsObject> ObjectArrayBuilder<T> {
    /// A builder with no capacity reserved.
    pub fn new() -> Self {
        Self::with_capacity(0)
    }

    /// A builder with room for `capacity` elements.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            values: Vec::with_capacity(capacity),
            validity: MutableBitmap::with_capacity(capacity),
        }
    }

    /// Appends `value`, or a null if it is [`None`].
    pub fn push(&mut self, value: Option<&T>) {
        self.values.push(value.cloned().unwrap_or_default());
        self.validity.push(value.is_some());
    }
}

impl<T: PolarsObject> Default for ObjectArrayBuilder<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: PolarsObject> StaticArrayBuilder for ObjectArrayBuilder<T> {
    type Array = ObjectArray<T>;

    fn reserve(&mut self, additional: usize) {
        self.values.reserve(additional);
        self.validity.reserve(additional);
    }

    fn len(&self) -> usize {
        self.values.len()
    }

    fn freeze(mut self) -> ObjectArray<T> {
        self.freeze_reset()
    }

    fn freeze_reset(&mut self) -> ObjectArray<T> {
        let values: Buffer<T> = std::mem::take(&mut self.values).into();
        let validity = std::mem::take(&mut self.validity);
        let validity = (validity.unset_bits() > 0).then(|| validity.freeze());
        ObjectArray { values, validity }
    }

    fn extend_nulls(&mut self, length: usize) {
        self.values.resize(self.values.len() + length, T::default());
        self.validity.extend_constant(length, false);
    }

    fn subslice_extend(
        &mut self,
        other: &ObjectArray<T>,
        start: usize,
        length: usize,
        _share: ShareStrategy,
    ) {
        assert!(start + length <= other.len(), "subslice out of bounds");
        self.reserve(length);
        for i in start..start + length {
            self.push(unsafe { other.get_unchecked_opt(i) });
        }
    }

    fn subslice_extend_each_repeated(
        &mut self,
        other: &ObjectArray<T>,
        start: usize,
        length: usize,
        repeats: usize,
        _share: ShareStrategy,
    ) {
        assert!(start + length <= other.len(), "subslice out of bounds");
        self.reserve(length * repeats);
        for i in start..start + length {
            let value = unsafe { other.get_unchecked_opt(i) };
            for _ in 0..repeats {
                self.push(value);
            }
        }
    }

    unsafe fn gather_extend(
        &mut self,
        other: &ObjectArray<T>,
        idxs: &[IdxSize],
        _share: ShareStrategy,
    ) {
        self.reserve(idxs.len());
        for &idx in idxs {
            self.push(unsafe { other.get_unchecked_opt(idx as usize) });
        }
    }

    fn opt_gather_extend(
        &mut self,
        other: &ObjectArray<T>,
        idxs: &[IdxSize],
        _share: ShareStrategy,
    ) {
        self.reserve(idxs.len());
        for &idx in idxs {
            let value = if (idx as usize) < other.len() {
                unsafe { other.get_unchecked_opt(idx as usize) }
            } else {
                None
            };
            self.push(value);
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

    pub(crate) unsafe fn get_object_chunked_unchecked(
        &self,
        chunk: usize,
        index: usize,
    ) -> Option<&dyn PolarsObjectSafe> {
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

impl<T: PolarsObject> From<Vec<T>> for ObjectArray<T> {
    fn from(values: Vec<T>) -> Self {
        Self {
            values: values.into(),
            validity: None,
        }
    }
}

impl<'a, T: PolarsObject> ArrayFromIter<&'a T> for ObjectArray<T> {
    fn arr_from_iter<I: IntoIterator<Item = &'a T>>(iter: I) -> Self {
        iter.into_iter().cloned().collect::<Vec<T>>().into()
    }

    fn try_arr_from_iter<E, I: IntoIterator<Item = Result<&'a T, E>>>(iter: I) -> Result<Self, E> {
        let values = iter
            .into_iter()
            .map(|value| value.cloned())
            .collect::<Result<Vec<T>, E>>()?;
        Ok(values.into())
    }
}

impl<'a, T: PolarsObject> ArrayFromIter<Option<&'a T>> for ObjectArray<T> {
    fn arr_from_iter<I: IntoIterator<Item = Option<&'a T>>>(iter: I) -> Self {
        let mut builder = ObjectArrayBuilder::new();
        for value in iter {
            builder.push(value);
        }
        builder.freeze()
    }

    fn try_arr_from_iter<E, I: IntoIterator<Item = Result<Option<&'a T>, E>>>(
        iter: I,
    ) -> Result<Self, E> {
        let mut builder = ObjectArrayBuilder::new();
        for value in iter {
            builder.push(value?);
        }
        Ok(builder.freeze())
    }
}

// The zeroable stand-in for a `&T` is `Option<&T>`, which is what the collect above takes.
impl<T: PolarsObject> ZeroableArrayFromIter for ObjectArray<T> {}
