use std::sync::LazyLock;

use arrow::bitmap::Bitmap;
use polars_utils::IdxSize;

use crate::array::PlArray;
use crate::array_type::PlArrayType;
use crate::bitmap::PlBitmapRef;
use crate::builder::{ShareStrategy, StaticArrayBuilder, assert_subslice};
use crate::flat::Flat;

/// An immutable, cheaply cloneable sequence of `length` nulls.
#[derive(Clone, Copy)]
pub struct PlNullArray {
    length: usize,
}

impl PlNullArray {
    /// Creates a [`PlNullArray`] of `length` nulls, in `O(1)` memory.
    #[inline]
    pub const fn new(length: usize) -> Self {
        Self { length }
    }

    /// Creates an empty [`PlNullArray`].
    #[inline]
    pub const fn new_empty() -> Self {
        Self { length: 0 }
    }

    /// Creates a [`PlNullArray`] of `length` nulls, in `O(1)` memory.
    #[inline]
    pub const fn new_full_null(length: usize) -> Self {
        Self::new(length)
    }

    /// The number of elements in this array.
    #[inline(always)]
    pub const fn len(&self) -> usize {
        self.length
    }

    /// Whether this array holds no elements.
    #[inline(always)]
    pub const fn is_empty(&self) -> bool {
        self.length == 0
    }

    /// The validity mask, which masks out every element.
    #[inline]
    pub fn validity(&self) -> PlBitmapRef<'static> {
        // An empty array has no element to share the bit, so its mask is empty as well: that is
        // the mask a scalar mask of no bits holds. See [`crate::broadcast`].
        let bitmap = if self.length == 0 { &EMPTY } else { &SCALAR };
        return unsafe { PlBitmapRef::new_broadcast_unchecked(bitmap, self.length) };
        static EMPTY: LazyLock<Bitmap> = LazyLock::new(Bitmap::new);
        static SCALAR: LazyLock<Bitmap> = LazyLock::new(|| Bitmap::new_zeroed(1));
    }

    /// The number of null elements, which is every element.
    #[inline(always)]
    pub const fn null_count(&self) -> usize {
        self.length
    }

    /// Whether this array has at least one null element, which it has unless it is empty.
    #[inline(always)]
    pub const fn has_nulls(&self) -> bool {
        self.length > 0
    }

    /// Returns whether the element at `i` is valid (non-null), which it never is.
    ///
    /// # Panics
    /// Panics if `i >= self.len()`.
    #[inline]
    pub fn is_valid(&self, i: usize) -> bool {
        assert!(i < self.length, "index out of bounds");
        false
    }

    /// Returns whether the element at `i` is valid (non-null), which it never is.
    ///
    /// # Safety
    /// `i` must be smaller than `self.len()`.
    #[inline]
    pub unsafe fn is_valid_unchecked(&self, i: usize) -> bool {
        debug_assert!(i < self.length);
        false
    }

    /// Returns whether the element at `i` is null, which it always is.
    ///
    /// # Panics
    /// Panics if `i >= self.len()`.
    #[inline]
    pub fn is_null(&self, i: usize) -> bool {
        !self.is_valid(i)
    }

    /// Returns whether the element at `i` is null, which it always is.
    ///
    /// # Safety
    /// `i` must be smaller than `self.len()`.
    #[inline]
    pub unsafe fn is_null_unchecked(&self, i: usize) -> bool {
        unsafe { !self.is_valid_unchecked(i) }
    }

    /// Slices this array in place to `length` elements starting at `offset`.
    ///
    /// # Panics
    /// Panics if `offset + length > self.len()`.
    pub fn slice(&mut self, offset: usize, length: usize) {
        assert!(
            offset + length <= self.length,
            "the offset of the new slice must be smaller than the length of the array",
        );
        unsafe { self.slice_unchecked(offset, length) }
    }

    /// Slices this array in place to `length` elements starting at `offset`.
    ///
    /// # Safety
    /// `offset + length` must not exceed `self.len()`.
    pub unsafe fn slice_unchecked(&mut self, offset: usize, length: usize) {
        debug_assert!(offset + length <= self.length);

        // There is nothing to slice: every element is null, so only the length changes.
        self.length = length;
    }

    /// Returns this array sliced to `length` elements starting at `offset`.
    ///
    /// # Panics
    /// Panics if `offset + length > self.len()`.
    #[must_use]
    pub fn sliced(mut self, offset: usize, length: usize) -> Self {
        self.slice(offset, length);
        self
    }

    /// Returns this array sliced to `length` elements starting at `offset`.
    ///
    /// # Safety
    /// `offset + length` must not exceed `self.len()`.
    #[must_use]
    pub unsafe fn sliced_unchecked(mut self, offset: usize, length: usize) -> Self {
        unsafe { self.slice_unchecked(offset, length) };
        self
    }

    /// Creates a [`PlNullArray`] of `length` copies of the element at `index`, which is a null.
    ///
    /// # Panics
    /// Panics if `index >= self.len()`.
    #[inline]
    pub fn new_from_index(&self, index: usize, length: usize) -> Self {
        assert!(index < self.length, "index out of bounds");
        Self::new(length)
    }

    /// Creates a [`PlNullArray`] of `length` copies of the element at `index`, which is a null.
    ///
    /// # Safety
    /// `index` must be smaller than `self.len()`.
    #[inline]
    pub const unsafe fn new_from_index_unchecked(&self, index: usize, length: usize) -> Self {
        debug_assert!(index < self.length);
        Self::new(length)
    }

    /// Whether every backing buffer of this array holds one slot per element, which a
    /// [`PlNullArray`] does vacuously: it has no buffers at all, only a length.
    #[inline]
    pub const fn is_flat(&self) -> bool {
        true
    }

    /// Whether this array is a single element repeated over its length, which a [`PlNullArray`]
    /// always is: every element is the same null, held in `O(1)` memory.
    #[inline]
    pub const fn is_scalar(&self) -> bool {
        true
    }

    /// Returns this array in the flat representation, which is this array — see
    /// [`PlNullArray::is_flat`].
    #[inline]
    pub fn to_flat(&self) -> Flat<Self> {
        // SAFETY: a null array has no backing buffer that could be scalar.
        unsafe { Flat::new(*self) }
    }

    /// Borrows this array as a flat one, which it always is — see [`PlNullArray::is_flat`].
    #[inline]
    pub fn as_flat(&self) -> Option<&Flat<Self>> {
        // SAFETY: a null array has no backing buffer that could be scalar.
        Some(unsafe { Flat::new_ref(self) })
    }
}

impl Default for PlNullArray {
    #[inline]
    fn default() -> Self {
        Self::new_empty()
    }
}

/// Compares two arrays element-wise, which for arrays of nothing but nulls is comparing their
/// lengths.
impl PartialEq for PlNullArray {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.length == other.length
    }
}

impl Eq for PlNullArray {}

impl std::fmt::Debug for PlNullArray {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Never materialize the elements: the length is unbounded by the memory use. The forms
        // match those of the other arrays, which list a single element and abbreviate more.
        match self.length {
            0 => f.write_str("PlNullArray[]"),
            1 => f.write_str("PlNullArray[null]"),
            length => write!(f, "PlNullArray[null; {length}]"),
        }
    }
}

impl PlArray for PlNullArray {
    #[inline]
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    #[inline]
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    #[inline]
    fn array_type(&self) -> PlArrayType {
        PlArrayType::Null
    }

    #[inline]
    fn len(&self) -> usize {
        self.len()
    }

    #[inline]
    fn is_scalar(&self) -> bool {
        self.is_scalar()
    }

    #[inline]
    fn validity(&self) -> Option<PlBitmapRef<'_>> {
        Some(self.validity())
    }

    #[inline]
    fn null_count(&self) -> usize {
        self.null_count()
    }

    #[inline]
    fn slice(&mut self, offset: usize, length: usize) {
        self.slice(offset, length)
    }

    #[inline]
    unsafe fn slice_unchecked(&mut self, offset: usize, length: usize) {
        unsafe { self.slice_unchecked(offset, length) }
    }

    /// Does nothing: an array of nothing but nulls has no element a mask could make valid.
    #[inline]
    fn set_validity(&mut self, _validity: Option<Bitmap>) {}

    /// Does nothing, exactly as [`Self::set_validity`] does.
    #[inline]
    fn set_validity_broadcast(&mut self, _validity: Option<Bitmap>) {}

    #[inline]
    unsafe fn new_from_index_unchecked(&self, index: usize, length: usize) -> Box<dyn PlArray> {
        Box::new(unsafe { self.new_from_index_unchecked(index, length) })
    }

    #[inline]
    fn to_boxed(&self) -> Box<dyn PlArray> {
        Box::new(*self)
    }

    fn eq_dyn(&self, other: &dyn PlArray) -> bool {
        other
            .as_any()
            .downcast_ref::<Self>()
            .is_some_and(|other| self == other)
    }
}

/// A builder of a [`PlNullArray`].
#[derive(Debug, Default, Clone, Copy)]
pub struct PlNullArrayBuilder {
    length: usize,
}

impl PlNullArrayBuilder {
    /// Creates an empty builder.
    #[inline]
    pub const fn new() -> Self {
        Self { length: 0 }
    }
}

impl StaticArrayBuilder for PlNullArrayBuilder {
    type Array = PlNullArray;

    /// Does nothing: there is nothing to hold a null in but the length.
    #[inline]
    fn reserve(&mut self, _additional: usize) {}

    #[inline]
    fn len(&self) -> usize {
        self.length
    }

    #[inline]
    fn freeze(self) -> PlNullArray {
        PlNullArray::new(self.length)
    }

    #[inline]
    fn freeze_reset(&mut self) -> PlNullArray {
        PlNullArray::new(std::mem::take(&mut self.length))
    }

    #[inline]
    fn extend_nulls(&mut self, length: usize) {
        self.length += length;
    }

    #[inline]
    fn subslice_extend(
        &mut self,
        other: &PlNullArray,
        start: usize,
        length: usize,
        _share: ShareStrategy,
    ) {
        assert_subslice(other.len(), start, length);
        self.length += length;
    }

    #[inline]
    fn subslice_extend_repeated(
        &mut self,
        other: &PlNullArray,
        start: usize,
        length: usize,
        repeats: usize,
        _share: ShareStrategy,
    ) {
        assert_subslice(other.len(), start, length);
        self.length += length * repeats;
    }

    #[inline]
    fn subslice_extend_each_repeated(
        &mut self,
        other: &PlNullArray,
        start: usize,
        length: usize,
        repeats: usize,
        _share: ShareStrategy,
    ) {
        assert_subslice(other.len(), start, length);
        self.length += length * repeats;
    }

    #[inline]
    unsafe fn gather_extend(
        &mut self,
        _other: &PlNullArray,
        idxs: &[IdxSize],
        _share: ShareStrategy,
    ) {
        self.length += idxs.len();
    }

    /// Appends one null per index: an out-of-bounds index stands for a null, which is what every
    /// element of a null array is anyway.
    #[inline]
    fn opt_gather_extend(&mut self, _other: &PlNullArray, idxs: &[IdxSize], _share: ShareStrategy) {
        self.length += idxs.len();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_element_is_null() {
        let arr = PlNullArray::new(3);

        assert_eq!(arr.len(), 3);
        assert!(!arr.is_empty());
        assert_eq!(arr.null_count(), 3);
        assert!(arr.has_nulls());
        assert!(arr.is_null(2));
        assert!(!arr.is_valid(2));
        assert_eq!(arr, PlNullArray::new_full_null(3));
    }

    #[test]
    fn a_billion_nulls_cost_nothing() {
        // Nothing here may walk a billion elements: there is no buffer to walk.
        let arr = PlNullArray::new(1_000_000_000);

        assert_eq!(arr.len(), 1_000_000_000);
        assert_eq!(arr.null_count(), 1_000_000_000);
        assert!(arr.is_null(999_999_999));
        assert_eq!(std::mem::size_of_val(&arr), std::mem::size_of::<usize>());
    }

    #[test]
    fn slicing_only_changes_the_length() {
        let arr = PlNullArray::new(1_000_000_000).sliced(500, 2);

        assert_eq!(arr.len(), 2);
        assert_eq!(arr.null_count(), 2);
        assert_eq!(arr.validity().len(), 2);

        let mut arr = PlNullArray::new(3);
        unsafe { arr.slice_unchecked(1, 0) };
        assert!(arr.is_empty());
        assert!(!arr.has_nulls());
    }
}
