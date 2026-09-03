//! The typed counterpart of [`PlArray`].

use arrow::bitmap::Bitmap;
use arrow::trusted_len::TrustedLen;
use arrow::types::NativeType;
use bytemuck::Zeroable;

use crate::array::PlArray;
use crate::binary::{PlBinaryIter, PlBinaryValuesIter};
use crate::binview::{PlBinaryViewIter, PlBinaryViewValuesIter};
use crate::bitmap::{PlBitmapIter, PlBitmapRef, ValidityFold, ValidityIter};
use crate::boolean::PlBooleanIter;
use crate::broadcast::assert_broadcastable;
use crate::builder::StaticArrayBuilder;
use crate::fixed_size_binary::{PlFixedSizeBinaryIter, PlFixedSizeBinaryValuesIter};
use crate::fixed_size_list::{PlFixedSizeListIter, PlFixedSizeListValuesIter};
use crate::flat::Flat;
use crate::list::{PlListIter, PlListValuesIter};
use crate::primitive::{PlPrimitiveIter, PlPrimitiveValuesIter};
use crate::utf8view::{PlUtf8ViewIter, PlUtf8ViewValuesIter};
use crate::{
    PlBinaryArray, PlBinaryArrayBuilder, PlBinaryViewArray, PlBinaryViewArrayBuilder,
    PlBooleanArray, PlBooleanArrayBuilder, PlFixedSizeBinaryArray, PlFixedSizeBinaryArrayBuilder,
    PlFixedSizeListArray, PlFixedSizeListArrayBuilder, PlListArray, PlListArrayBuilder,
    PlNullArray, PlNullArrayBuilder, PlPrimitiveArray, PlPrimitiveArrayBuilder, PlStructArray,
    PlStructArrayBuilder, PlUtf8ViewArray, PlUtf8ViewArrayBuilder,
};

/// An array whose element type is known statically.
pub trait StaticArray: PlArray + Clone {
    /// One element of this array, borrowed from it.
    type ValueT<'a>: Clone
    where
        Self: 'a;

    /// One element of this array, in a type whose all-zero bit pattern is a value of its own.
    type ZeroableValueT<'a>: Zeroable + From<Self::ValueT<'a>>
    where
        Self: 'a;

    /// The iterator [`Self::values_iter`] hands out.
    type ValueIterT<'a>: DoubleEndedIterator<Item = Self::ValueT<'a>>
        + ExactSizeIterator
        + TrustedLen
        + Send
        + Sync
    where
        Self: 'a;

    /// The iterator [`Self::iter`] hands out.
    type IterT<'a>: DoubleEndedIterator<Item = Option<Self::ValueT<'a>>>
        + ExactSizeIterator
        + TrustedLen
        + Send
        + Sync
    where
        Self: 'a;

    /// The builder that builds this array.
    type Builder: StaticArrayBuilder<Array = Self>;

    /// Returns the element at `i`, whether or not it is null.
    ///
    /// # Panics
    /// Panics if `i >= self.len()`.
    #[inline]
    fn value(&self, i: usize) -> Self::ValueT<'_> {
        assert!(i < self.len(), "index out of bounds");
        // SAFETY: `i` was just checked against the length.
        unsafe { self.value_unchecked(i) }
    }

    /// Returns the element at `i`, whether or not it is null.
    ///
    /// # Safety
    /// `i` must be smaller than `self.len()`.
    unsafe fn value_unchecked(&self, i: usize) -> Self::ValueT<'_>;

    /// Returns the element at `i`, or `None` if it is null.
    ///
    /// # Panics
    /// Panics if `i >= self.len()`.
    #[inline]
    fn get(&self, i: usize) -> Option<Self::ValueT<'_>> {
        assert!(i < self.len(), "index out of bounds");
        // SAFETY: `i` was just checked against the length.
        unsafe { self.get_unchecked(i) }
    }

    /// Returns the element at `i`, or `None` if it is null.
    ///
    /// # Safety
    /// `i` must be smaller than `self.len()`.
    #[inline]
    unsafe fn get_unchecked(&self, i: usize) -> Option<Self::ValueT<'_>> {
        // SAFETY: `i` is in bounds of the array, and therefore of its validity mask.
        unsafe { (!self.is_null_unchecked(i)).then(|| self.value_unchecked(i)) }
    }

    /// Returns an iterator over the elements, ignoring validity.
    fn values_iter(&self) -> Self::ValueIterT<'_>;

    /// Returns an iterator over the optional elements.
    fn iter(&self) -> Self::IterT<'_>;

    /// Returns an iterator over `length` elements, repeating the single element of this array if
    /// that is all it holds.
    ///
    /// # Panics
    /// Panics if `self.len()` is neither `length` nor one.
    fn broadcast_values_iter(&self, length: usize) -> Self::ValueIterT<'_>;

    /// Returns this array with its validity mask replaced by a flat one.
    ///
    /// # Panics
    /// Panics if `validity` does not hold one bit per element.
    #[must_use]
    fn with_validity_typed(self, validity: Option<Bitmap>) -> Self;

    /// Returns this array with its validity mask replaced by one that broadcasts over it.
    ///
    /// # Panics
    /// Panics if `validity` is neither flat nor scalar for this array's length.
    #[must_use]
    fn with_validity_broadcast_typed(self, validity: Option<Bitmap>) -> Self;

    /// Returns an array of `length` copies of the element at `index`.
    ///
    /// # Panics
    /// Panics if `index >= self.len()`.
    #[must_use]
    fn new_from_index_typed(&self, index: usize, length: usize) -> Self;

    /// Whether every backing buffer of this array holds one slot per element.
    fn is_flat(&self) -> bool;

    /// The element every element of this array equals, if it is entirely stored in the scalar
    /// representation.
    #[inline]
    fn scalar_value(&self) -> Option<Option<Self::ValueT<'_>>> {
        // SAFETY: the array is not empty, so element 0 is in bounds.
        (PlArray::is_scalar(self) && !self.is_empty()).then(|| unsafe { self.get_unchecked(0) })
    }

    /// Returns this array in the flat representation, writing out every buffer that is scalar.
    #[must_use]
    fn to_flat(&self) -> Flat<Self>;

    /// Borrows this array as a flat one, or `None` if any backing buffer is scalar.
    fn as_flat(&self) -> Option<&Flat<Self>>;

    /// Boxes this array as a [`PlArray`] trait object.
    #[inline]
    fn into_boxed(self) -> Box<dyn PlArray>
    where
        Self: Sized,
    {
        Box::new(self)
    }
}

impl<T: NativeType> StaticArray for PlPrimitiveArray<T> {
    type ValueT<'a> = T;
    type ZeroableValueT<'a> = T;
    type ValueIterT<'a> = PlPrimitiveValuesIter<'a, T>;
    type IterT<'a> = PlPrimitiveIter<'a, T>;
    type Builder = PlPrimitiveArrayBuilder<T>;

    #[inline]
    unsafe fn value_unchecked(&self, i: usize) -> T {
        unsafe { self.value_unchecked(i) }
    }

    #[inline]
    unsafe fn get_unchecked(&self, i: usize) -> Option<T> {
        unsafe { self.get_unchecked(i) }
    }

    #[inline]
    fn values_iter(&self) -> Self::ValueIterT<'_> {
        self.values_iter()
    }

    #[inline]
    fn iter(&self) -> Self::IterT<'_> {
        self.iter()
    }

    #[inline]
    fn broadcast_values_iter(&self, length: usize) -> Self::ValueIterT<'_> {
        self.broadcast_values_iter(length)
    }

    #[inline]
    fn with_validity_typed(self, validity: Option<Bitmap>) -> Self {
        self.with_validity(validity)
    }

    #[inline]
    fn with_validity_broadcast_typed(self, validity: Option<Bitmap>) -> Self {
        self.with_validity_broadcast(validity)
    }

    #[inline]
    fn new_from_index_typed(&self, index: usize, length: usize) -> Self {
        self.new_from_index(index, length)
    }

    #[inline]
    fn is_flat(&self) -> bool {
        self.is_flat()
    }

    #[inline]
    fn to_flat(&self) -> Flat<Self> {
        self.to_flat()
    }

    #[inline]
    fn as_flat(&self) -> Option<&Flat<Self>> {
        self.as_flat()
    }
}

impl StaticArray for PlBooleanArray {
    type ValueT<'a> = bool;
    type ZeroableValueT<'a> = bool;
    type ValueIterT<'a> = PlBitmapIter<'a>;
    type IterT<'a> = PlBooleanIter<'a>;
    type Builder = PlBooleanArrayBuilder;

    #[inline]
    unsafe fn value_unchecked(&self, i: usize) -> bool {
        unsafe { self.value_unchecked(i) }
    }

    #[inline]
    unsafe fn get_unchecked(&self, i: usize) -> Option<bool> {
        unsafe { self.get_unchecked(i) }
    }

    #[inline]
    fn values_iter(&self) -> Self::ValueIterT<'_> {
        self.values_iter()
    }

    #[inline]
    fn iter(&self) -> Self::IterT<'_> {
        self.iter()
    }

    #[inline]
    fn broadcast_values_iter(&self, length: usize) -> Self::ValueIterT<'_> {
        self.broadcast_values_iter(length)
    }

    #[inline]
    fn with_validity_typed(self, validity: Option<Bitmap>) -> Self {
        self.with_validity(validity)
    }

    #[inline]
    fn with_validity_broadcast_typed(self, validity: Option<Bitmap>) -> Self {
        self.with_validity_broadcast(validity)
    }

    #[inline]
    fn new_from_index_typed(&self, index: usize, length: usize) -> Self {
        self.new_from_index(index, length)
    }

    #[inline]
    fn is_flat(&self) -> bool {
        self.is_flat()
    }

    #[inline]
    fn to_flat(&self) -> Flat<Self> {
        self.to_flat()
    }

    #[inline]
    fn as_flat(&self) -> Option<&Flat<Self>> {
        self.as_flat()
    }
}

impl StaticArray for PlBinaryArray {
    type ValueT<'a> = &'a [u8];
    type ZeroableValueT<'a> = Option<&'a [u8]>;
    type ValueIterT<'a> = PlBinaryValuesIter<'a>;
    type IterT<'a> = PlBinaryIter<'a>;
    type Builder = PlBinaryArrayBuilder;

    #[inline]
    unsafe fn value_unchecked(&self, i: usize) -> &[u8] {
        unsafe { self.value_unchecked(i) }
    }

    #[inline]
    unsafe fn get_unchecked(&self, i: usize) -> Option<&[u8]> {
        unsafe { self.get_unchecked(i) }
    }

    #[inline]
    fn values_iter(&self) -> Self::ValueIterT<'_> {
        self.values_iter()
    }

    #[inline]
    fn iter(&self) -> Self::IterT<'_> {
        self.iter()
    }

    #[inline]
    fn broadcast_values_iter(&self, length: usize) -> Self::ValueIterT<'_> {
        self.broadcast_values_iter(length)
    }

    #[inline]
    fn with_validity_typed(self, validity: Option<Bitmap>) -> Self {
        self.with_validity(validity)
    }

    #[inline]
    fn with_validity_broadcast_typed(self, validity: Option<Bitmap>) -> Self {
        self.with_validity_broadcast(validity)
    }

    #[inline]
    fn new_from_index_typed(&self, index: usize, length: usize) -> Self {
        self.new_from_index(index, length)
    }

    #[inline]
    fn is_flat(&self) -> bool {
        self.is_flat()
    }

    #[inline]
    fn to_flat(&self) -> Flat<Self> {
        self.to_flat()
    }

    #[inline]
    fn as_flat(&self) -> Option<&Flat<Self>> {
        self.as_flat()
    }
}

impl StaticArray for PlBinaryViewArray {
    type ValueT<'a> = &'a [u8];
    type ZeroableValueT<'a> = Option<&'a [u8]>;
    type ValueIterT<'a> = PlBinaryViewValuesIter<'a>;
    type IterT<'a> = PlBinaryViewIter<'a>;
    type Builder = PlBinaryViewArrayBuilder;

    #[inline]
    unsafe fn value_unchecked(&self, i: usize) -> &[u8] {
        unsafe { self.value_unchecked(i) }
    }

    #[inline]
    unsafe fn get_unchecked(&self, i: usize) -> Option<&[u8]> {
        unsafe { self.get_unchecked(i) }
    }

    #[inline]
    fn values_iter(&self) -> Self::ValueIterT<'_> {
        self.values_iter()
    }

    #[inline]
    fn iter(&self) -> Self::IterT<'_> {
        self.iter()
    }

    #[inline]
    fn broadcast_values_iter(&self, length: usize) -> Self::ValueIterT<'_> {
        self.broadcast_values_iter(length)
    }

    #[inline]
    fn with_validity_typed(self, validity: Option<Bitmap>) -> Self {
        self.with_validity(validity)
    }

    #[inline]
    fn with_validity_broadcast_typed(self, validity: Option<Bitmap>) -> Self {
        self.with_validity_broadcast(validity)
    }

    #[inline]
    fn new_from_index_typed(&self, index: usize, length: usize) -> Self {
        self.new_from_index(index, length)
    }

    #[inline]
    fn is_flat(&self) -> bool {
        self.is_flat()
    }

    #[inline]
    fn to_flat(&self) -> Flat<Self> {
        self.to_flat()
    }

    #[inline]
    fn as_flat(&self) -> Option<&Flat<Self>> {
        self.as_flat()
    }
}

/// The elements are the strings the wrapper promises they are — see [`crate::utf8view`].
impl StaticArray for PlUtf8ViewArray {
    type ValueT<'a> = &'a str;
    type ZeroableValueT<'a> = Option<&'a str>;
    type ValueIterT<'a> = PlUtf8ViewValuesIter<'a>;
    type IterT<'a> = PlUtf8ViewIter<'a>;
    type Builder = PlUtf8ViewArrayBuilder;

    #[inline]
    unsafe fn value_unchecked(&self, i: usize) -> &str {
        unsafe { self.value_unchecked(i) }
    }

    #[inline]
    unsafe fn get_unchecked(&self, i: usize) -> Option<&str> {
        unsafe { self.get_unchecked(i) }
    }

    #[inline]
    fn values_iter(&self) -> Self::ValueIterT<'_> {
        self.values_iter()
    }

    #[inline]
    fn iter(&self) -> Self::IterT<'_> {
        self.iter()
    }

    #[inline]
    fn broadcast_values_iter(&self, length: usize) -> Self::ValueIterT<'_> {
        self.broadcast_values_iter(length)
    }

    #[inline]
    fn with_validity_typed(self, validity: Option<Bitmap>) -> Self {
        self.with_validity(validity)
    }

    #[inline]
    fn with_validity_broadcast_typed(self, validity: Option<Bitmap>) -> Self {
        self.with_validity_broadcast(validity)
    }

    #[inline]
    fn new_from_index_typed(&self, index: usize, length: usize) -> Self {
        self.new_from_index(index, length)
    }

    #[inline]
    fn is_flat(&self) -> bool {
        self.is_flat()
    }

    #[inline]
    fn to_flat(&self) -> Flat<Self> {
        self.to_flat()
    }

    #[inline]
    fn as_flat(&self) -> Option<&Flat<Self>> {
        self.as_flat()
    }
}

impl StaticArray for PlFixedSizeBinaryArray {
    type ValueT<'a> = &'a [u8];
    type ZeroableValueT<'a> = Option<&'a [u8]>;
    type ValueIterT<'a> = PlFixedSizeBinaryValuesIter<'a>;
    type IterT<'a> = PlFixedSizeBinaryIter<'a>;
    type Builder = PlFixedSizeBinaryArrayBuilder;

    #[inline]
    unsafe fn value_unchecked(&self, i: usize) -> &[u8] {
        unsafe { self.value_unchecked(i) }
    }

    #[inline]
    unsafe fn get_unchecked(&self, i: usize) -> Option<&[u8]> {
        unsafe { self.get_unchecked(i) }
    }

    #[inline]
    fn values_iter(&self) -> Self::ValueIterT<'_> {
        self.values_iter()
    }

    #[inline]
    fn iter(&self) -> Self::IterT<'_> {
        self.iter()
    }

    #[inline]
    fn broadcast_values_iter(&self, length: usize) -> Self::ValueIterT<'_> {
        self.broadcast_values_iter(length)
    }

    #[inline]
    fn with_validity_typed(self, validity: Option<Bitmap>) -> Self {
        self.with_validity(validity)
    }

    #[inline]
    fn with_validity_broadcast_typed(self, validity: Option<Bitmap>) -> Self {
        self.with_validity_broadcast(validity)
    }

    #[inline]
    fn new_from_index_typed(&self, index: usize, length: usize) -> Self {
        self.new_from_index(index, length)
    }

    #[inline]
    fn is_flat(&self) -> bool {
        self.is_flat()
    }

    #[inline]
    fn to_flat(&self) -> Flat<Self> {
        self.to_flat()
    }

    #[inline]
    fn as_flat(&self) -> Option<&Flat<Self>> {
        self.as_flat()
    }
}

impl StaticArray for PlListArray {
    type ValueT<'a> = Box<dyn PlArray>;
    type ZeroableValueT<'a> = Option<Box<dyn PlArray>>;
    type ValueIterT<'a> = PlListValuesIter<'a>;
    type IterT<'a> = PlListIter<'a>;
    type Builder = PlListArrayBuilder;

    #[inline]
    unsafe fn value_unchecked(&self, i: usize) -> Box<dyn PlArray> {
        unsafe { self.value_unchecked(i) }
    }

    #[inline]
    unsafe fn get_unchecked(&self, i: usize) -> Option<Box<dyn PlArray>> {
        unsafe { self.get_unchecked(i) }
    }

    #[inline]
    fn values_iter(&self) -> Self::ValueIterT<'_> {
        self.values_iter()
    }

    #[inline]
    fn iter(&self) -> Self::IterT<'_> {
        self.iter()
    }

    #[inline]
    fn broadcast_values_iter(&self, length: usize) -> Self::ValueIterT<'_> {
        self.broadcast_values_iter(length)
    }

    #[inline]
    fn with_validity_typed(self, validity: Option<Bitmap>) -> Self {
        self.with_validity(validity)
    }

    #[inline]
    fn with_validity_broadcast_typed(self, validity: Option<Bitmap>) -> Self {
        self.with_validity_broadcast(validity)
    }

    #[inline]
    fn new_from_index_typed(&self, index: usize, length: usize) -> Self {
        self.new_from_index(index, length)
    }

    #[inline]
    fn is_flat(&self) -> bool {
        self.is_flat()
    }

    #[inline]
    fn to_flat(&self) -> Flat<Self> {
        self.to_flat()
    }

    #[inline]
    fn as_flat(&self) -> Option<&Flat<Self>> {
        self.as_flat()
    }
}

impl StaticArray for PlFixedSizeListArray {
    type ValueT<'a> = Box<dyn PlArray>;
    type ZeroableValueT<'a> = Option<Box<dyn PlArray>>;
    type ValueIterT<'a> = PlFixedSizeListValuesIter<'a>;
    type IterT<'a> = PlFixedSizeListIter<'a>;
    type Builder = PlFixedSizeListArrayBuilder;

    #[inline]
    unsafe fn value_unchecked(&self, i: usize) -> Box<dyn PlArray> {
        unsafe { self.value_unchecked(i) }
    }

    #[inline]
    unsafe fn get_unchecked(&self, i: usize) -> Option<Box<dyn PlArray>> {
        unsafe { self.get_unchecked(i) }
    }

    #[inline]
    fn values_iter(&self) -> Self::ValueIterT<'_> {
        self.values_iter()
    }

    #[inline]
    fn iter(&self) -> Self::IterT<'_> {
        self.iter()
    }

    #[inline]
    fn broadcast_values_iter(&self, length: usize) -> Self::ValueIterT<'_> {
        self.broadcast_values_iter(length)
    }

    #[inline]
    fn with_validity_typed(self, validity: Option<Bitmap>) -> Self {
        self.with_validity(validity)
    }

    #[inline]
    fn with_validity_broadcast_typed(self, validity: Option<Bitmap>) -> Self {
        self.with_validity_broadcast(validity)
    }

    #[inline]
    fn new_from_index_typed(&self, index: usize, length: usize) -> Self {
        self.new_from_index(index, length)
    }

    #[inline]
    fn is_flat(&self) -> bool {
        self.is_flat()
    }

    #[inline]
    fn to_flat(&self) -> Flat<Self> {
        self.to_flat()
    }

    #[inline]
    fn as_flat(&self) -> Option<&Flat<Self>> {
        self.as_flat()
    }
}

/// A [`PlStructArray`] holds no values of its own: an element is a row across the field arrays,
/// which are reached through [`PlStructArray::fields`] and read as the arrays they are.
impl StaticArray for PlStructArray {
    type ValueT<'a> = ();
    type ZeroableValueT<'a> = ();
    type ValueIterT<'a> = std::iter::RepeatN<()>;
    type IterT<'a> = PlUnitIter<'a>;
    type Builder = PlStructArrayBuilder;

    #[inline]
    unsafe fn value_unchecked(&self, _i: usize) -> Self::ValueT<'_> {}

    #[inline]
    fn values_iter(&self) -> Self::ValueIterT<'_> {
        std::iter::repeat_n((), self.len())
    }

    #[inline]
    fn iter(&self) -> Self::IterT<'_> {
        PlUnitIter::new(self.validity(), self.len())
    }

    #[inline]
    fn broadcast_values_iter(&self, length: usize) -> Self::ValueIterT<'_> {
        assert_broadcastable(self.len(), length);
        std::iter::repeat_n((), length)
    }

    #[inline]
    fn with_validity_typed(self, validity: Option<Bitmap>) -> Self {
        self.with_validity(validity)
    }

    #[inline]
    fn with_validity_broadcast_typed(self, validity: Option<Bitmap>) -> Self {
        self.with_validity_broadcast(validity)
    }

    #[inline]
    fn new_from_index_typed(&self, index: usize, length: usize) -> Self {
        self.new_from_index(index, length)
    }

    #[inline]
    fn is_flat(&self) -> bool {
        self.is_flat()
    }

    #[inline]
    fn to_flat(&self) -> Flat<Self> {
        self.to_flat()
    }

    #[inline]
    fn as_flat(&self) -> Option<&Flat<Self>> {
        self.as_flat()
    }
}

/// A [`PlNullArray`] is nothing but a length: every element is null, and there is no value under
/// the mask, so the value of an element is `()` and [`StaticArray::get`] is always `None`.
impl StaticArray for PlNullArray {
    type ValueT<'a> = ();
    type ZeroableValueT<'a> = ();
    type ValueIterT<'a> = std::iter::RepeatN<()>;
    type IterT<'a> = PlUnitIter<'a>;
    type Builder = PlNullArrayBuilder;

    #[inline]
    unsafe fn value_unchecked(&self, _i: usize) -> Self::ValueT<'_> {}

    #[inline]
    fn values_iter(&self) -> Self::ValueIterT<'_> {
        std::iter::repeat_n((), self.len())
    }

    #[inline]
    fn iter(&self) -> Self::IterT<'_> {
        PlUnitIter::new(Some(self.validity()), self.len())
    }

    #[inline]
    fn broadcast_values_iter(&self, length: usize) -> Self::ValueIterT<'_> {
        assert_broadcastable(self.len(), length);
        std::iter::repeat_n((), length)
    }

    /// Returns this array unchanged: an array of nothing but nulls has no element a mask could make
    /// valid, exactly as [`PlArray::set_validity`] documents.
    #[inline]
    fn with_validity_typed(self, _validity: Option<Bitmap>) -> Self {
        self
    }

    /// Returns this array unchanged, exactly as [`Self::with_validity_typed`] does.
    #[inline]
    fn with_validity_broadcast_typed(self, _validity: Option<Bitmap>) -> Self {
        self
    }

    #[inline]
    fn new_from_index_typed(&self, index: usize, length: usize) -> Self {
        self.new_from_index(index, length)
    }

    #[inline]
    fn is_flat(&self) -> bool {
        self.is_flat()
    }

    #[inline]
    fn to_flat(&self) -> Flat<Self> {
        self.to_flat()
    }

    #[inline]
    fn as_flat(&self) -> Option<&Flat<Self>> {
        self.as_flat()
    }
}

/// Iterator over the optional elements of an array whose elements carry no value of their own.
#[derive(Clone)]
pub struct PlUnitIter<'a> {
    validity: ValidityIter<'a>,
    /// How many elements are left; the mask, when there is one, has one bit per element of them.
    remaining: usize,
}

impl<'a> PlUnitIter<'a> {
    /// # Panics
    /// Panics if `validity` does not have `length` bits.
    #[inline]
    fn new(validity: Option<PlBitmapRef<'a>>, length: usize) -> Self {
        assert!(validity.is_none_or(|validity| validity.len() == length));

        Self {
            validity: ValidityIter::new(validity),
            remaining: length,
        }
    }
}

impl Iterator for PlUnitIter<'_> {
    type Item = Option<()>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.remaining = self.remaining.checked_sub(1)?;
        Some(self.validity.next().then_some(()))
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        // The mask is advanced alongside the elements, whether or not there is one left.
        let is_valid = self.validity.nth(n);
        let Some(remaining) = self.remaining.checked_sub(n + 1) else {
            self.remaining = 0;
            return None;
        };
        self.remaining = remaining;
        Some(is_valid.then_some(()))
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }

    #[inline]
    fn count(self) -> usize {
        self.remaining
    }

    /// Walks to the last element from the back, rather than through every one before it.
    #[inline]
    fn last(mut self) -> Option<Self::Item> {
        self.next_back()
    }

    /// Hoists the validity mask out of the loop: an array without null elements folds over the
    /// count alone, and one with a mask folds over the mask, which is what it is.
    #[inline]
    fn fold<B, F>(self, init: B, mut f: F) -> B
    where
        F: FnMut(B, Self::Item) -> B,
    {
        let remaining = self.remaining;

        match self.validity.into_mask() {
            ValidityFold::Valid => (0..remaining).fold(init, |acc, _| f(acc, Some(()))),
            ValidityFold::Null => (0..remaining).fold(init, |acc, _| f(acc, None)),
            ValidityFold::Bits(mask) => {
                mask.fold(init, |acc, is_valid| f(acc, is_valid.then_some(())))
            },
        }
    }
}

impl DoubleEndedIterator for PlUnitIter<'_> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        self.remaining = self.remaining.checked_sub(1)?;
        Some(self.validity.next_back().then_some(()))
    }
}

impl ExactSizeIterator for PlUnitIter<'_> {
    #[inline]
    fn len(&self) -> usize {
        self.remaining
    }
}

unsafe impl TrustedLen for PlUnitIter<'_> {}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::iterator_tests::assert_iterates;

    /// The iterator of an array whose elements carry no value of their own, in both representations
    /// of its validity mask.
    mod unit_iter {
        use super::*;

        #[test]
        fn a_struct_under_a_flat_mask() {
            let field = Box::new(PlPrimitiveArray::from_vec(vec![1i32, 2, 3]));
            let array =
                PlStructArray::new(vec![field], 3, Some(Bitmap::from_iter([true, false, true])));

            assert_iterates(array.iter(), &[Some(()), None, Some(())]);
        }
    }

    #[test]
    fn scalars_are_read_through_the_broadcast() {
        let array = PlPrimitiveArray::new_scalar(7i32, 1_000_000_000);

        assert_eq!(StaticArray::value(&array, 999_999_999), 7);
        assert_eq!(StaticArray::get(&array, 999_999_999), Some(7));
        assert_eq!(array.iter().nth(999_999_999), Some(Some(7)));

        // An array of a single element iterates as any number of copies of it, in `O(1)`.
        let one = PlPrimitiveArray::from_vec(vec![7i32]);
        assert_eq!(
            one.broadcast_values_iter(1_000_000_000).len(),
            1_000_000_000
        );
        assert_eq!(one.broadcast_values_iter(1_000_000_000).last(), Some(7));

        let one = PlStructArray::from_fields(vec![Box::new(one)]);
        assert_eq!(
            one.broadcast_values_iter(1_000_000_000).len(),
            1_000_000_000
        );

        let nulls = PlNullArray::new(1);
        assert_eq!(
            nulls.broadcast_values_iter(1_000_000_000).len(),
            1_000_000_000
        );
    }

    #[test]
    fn typed_operations_keep_the_concrete_type() {
        let array = PlPrimitiveArray::from_vec(vec![1i32, 2, 3]);

        let nulled: PlPrimitiveArray<i32> = array
            .clone()
            .with_validity_broadcast_typed(Some(Bitmap::new_zeroed(1)));
        assert_eq!(nulled.null_count(), 3);

        let repeated: PlPrimitiveArray<i32> = array.new_from_index_typed(2, 4);
        assert_eq!(repeated, PlPrimitiveArray::new_scalar(3, 4));
    }
}
