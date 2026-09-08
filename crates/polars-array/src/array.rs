use std::any::Any;

use crate::array_type::PlArrayType;
use crate::bitmap::{PlBitmap, PlBitmapRef};

/// A trait object over the arrays in this crate.
pub trait PlArray: std::fmt::Debug + Send + Sync + 'static {
    /// Converts itself to a reference of [`Any`], which enables downcasting to concrete types.
    fn as_any(&self) -> &dyn Any;

    /// Converts itself to a mutable [`Any`] reference, which enables mutable downcasting.
    fn as_any_mut(&mut self) -> &mut dyn Any;

    /// The physical representation of this array.
    fn array_type(&self) -> PlArrayType;

    /// The number of elements in this array.
    fn len(&self) -> usize;

    /// Whether this array holds no elements.
    #[inline]
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Whether this array is scalar throughout: one value repeated [`PlArray::len`] times.
    fn is_scalar(&self) -> bool;

    /// The validity mask, if any element may be null.
    fn validity(&self) -> Option<PlBitmapRef<'_>>;

    /// The number of null elements.
    #[inline]
    fn null_count(&self) -> usize {
        self.validity().map_or(0, |validity| validity.unset_bits())
    }

    /// Whether this array has at least one null element.
    #[inline]
    fn has_nulls(&self) -> bool {
        self.null_count() > 0
    }

    /// Returns whether the element at `i` is valid (non-null).
    #[inline]
    fn is_valid(&self, i: usize) -> bool {
        assert!(i < self.len(), "index out of bounds");
        unsafe { self.is_valid_unchecked(i) }
    }

    /// Returns whether the element at `i` is valid (non-null).
    ///
    /// # Safety
    /// `i` must be smaller than `self.len()`.
    #[inline]
    unsafe fn is_valid_unchecked(&self, i: usize) -> bool {
        debug_assert!(i < self.len());
        // SAFETY: `i` is in bounds of the array, and therefore of its validity mask.
        self.validity()
            .is_none_or(|validity| unsafe { validity.get_unchecked(i) })
    }

    /// Returns whether the element at `i` is null.
    #[inline]
    fn is_null(&self, i: usize) -> bool {
        !self.is_valid(i)
    }

    /// Returns whether the element at `i` is null.
    ///
    /// # Safety
    /// `i` must be smaller than `self.len()`.
    #[inline]
    unsafe fn is_null_unchecked(&self, i: usize) -> bool {
        unsafe { !self.is_valid_unchecked(i) }
    }

    /// Slices this array in place to `length` elements starting at `offset`.
    fn slice(&mut self, offset: usize, length: usize);

    /// Slices this array in place to `length` elements starting at `offset`.
    ///
    /// # Safety
    /// `offset + length` must not exceed `self.len()`.
    unsafe fn slice_unchecked(&mut self, offset: usize, length: usize);

    /// Returns this array sliced to `length` elements starting at `offset`.
    #[must_use]
    fn sliced(&self, offset: usize, length: usize) -> Box<dyn PlArray> {
        let mut sliced = self.to_boxed();
        sliced.slice(offset, length);
        sliced
    }

    /// Returns this array sliced to `length` elements starting at `offset`.
    ///
    /// # Safety
    /// `offset + length` must not exceed `self.len()`.
    #[must_use]
    unsafe fn sliced_unchecked(&self, offset: usize, length: usize) -> Box<dyn PlArray> {
        let mut sliced = self.to_boxed();
        unsafe { sliced.slice_unchecked(offset, length) };
        sliced
    }

    /// Replaces the validity mask, keeping the representation the given mask is in.
    fn set_validity(&mut self, validity: Option<PlBitmap>);

    /// Returns this array with its validity mask replaced, keeping its representation.
    #[must_use]
    fn with_validity(&self, validity: Option<PlBitmap>) -> Box<dyn PlArray> {
        let mut new = self.to_boxed();
        new.set_validity(validity);
        new
    }

    /// Returns this array with its validity mask dropped, making every element valid.
    #[must_use]
    fn without_validity(&self) -> Box<dyn PlArray> {
        self.with_validity(None)
    }

    /// Returns an array of `length` copies of the element at `index`.
    #[must_use]
    fn new_from_index(&self, index: usize, length: usize) -> Box<dyn PlArray> {
        assert!(index < self.len(), "index out of bounds");
        unsafe { self.new_from_index_unchecked(index, length) }
    }

    /// Returns an array of `length` copies of the element at `index`.
    ///
    /// # Safety
    /// `index` must be smaller than `self.len()`.
    #[must_use]
    unsafe fn new_from_index_unchecked(&self, index: usize, length: usize) -> Box<dyn PlArray>;

    /// Clones this array into an owned `Box<dyn PlArray>`.
    fn to_boxed(&self) -> Box<dyn PlArray>;

    /// Returns an array of `length` nulls, shaped like this array, in `O(1)` memory.
    #[must_use]
    fn new_full_null_like_self(&self, length: usize) -> Box<dyn PlArray>;

    /// Compares this array element-wise against `other`, `false` if it is of another type.
    fn eq_dyn(&self, other: &dyn PlArray) -> bool;
}

impl Clone for Box<dyn PlArray> {
    #[inline]
    fn clone(&self) -> Self {
        self.to_boxed()
    }
}

/// Compares two arrays element-wise; arrays of different [`PlArrayType`] never compare equal.
impl PartialEq for dyn PlArray + '_ {
    #[inline]
    fn eq(&self, other: &dyn PlArray) -> bool {
        self.eq_dyn(other)
    }
}

impl Eq for dyn PlArray + '_ {}

/// Compares two arrays element-wise, exactly like [`PartialEq`]: no value is unequal to itself.
impl polars_utils::total_ord::TotalEq for Box<dyn PlArray> {
    #[inline]
    fn tot_eq(&self, other: &Self) -> bool {
        self.eq_dyn(&**other)
    }
}
