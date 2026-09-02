use std::sync::LazyLock;

use arrow::bitmap::Bitmap;
use polars_utils::IdxSize;

use crate::array::PlArray;
use crate::array_type::PlArrayType;
use crate::bitmap::PlBitmapRef;
use crate::builder::{ShareStrategy, StaticArrayBuilder, assert_subslice};
use crate::flat::Flat;

/// An immutable, cheaply cloneable sequence of `length` nulls.
///
/// This is the array of the type that holds nothing but nulls: it has no values, no element type
/// and no buffers, only a length. Every element is null, which is what distinguishes it from the
/// fully null array of any other type — there is no value hiding under the mask, undetermined or
/// otherwise.
///
/// # Example
/// ```
/// use polars_array::PlNullArray;
///
/// let arr = PlNullArray::new(1_000_000_000);
/// assert_eq!(arr.len(), 1_000_000_000);
/// assert_eq!(arr.null_count(), 1_000_000_000);
/// assert!(arr.is_null(999_999_999));
///
/// // The mask covers every element, backed by a single bit.
/// let validity = arr.validity();
/// assert_eq!(validity.len(), 1_000_000_000);
/// assert_eq!(validity.scalar_value(), Some(false));
/// assert!(validity.is_scalar());
/// ```
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
    ///
    /// This is [`Self::new`] under the name the other arrays use, so that a null array can be
    /// built the same way as a fully null array of any other type.
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
        return unsafe { PlBitmapRef::new_unchecked(&SCALAR, self.length) };
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
    /// This function is `O(1)`.
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
    /// This function is `O(1)`.
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
    /// This function is `O(1)`.
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
    /// This function is `O(1)`.
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
    /// This function is `O(1)`. The index is only bounds-checked: every element of this array is
    /// the same null, so there is nothing to read at it.
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
    /// This function is `O(1)`.
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
    ///
    /// Its validity mask reads as scalar — a single unset bit covering every element — but there
    /// is no buffer behind it to write out, so there is no scalar representation to leave.
    #[inline]
    pub const fn is_flat(&self) -> bool {
        true
    }

    /// Returns this array in the flat representation, which is this array — see
    /// [`PlNullArray::is_flat`]. This function is `O(1)`.
    #[inline]
    pub fn to_flat(&self) -> Flat<Self> {
        // SAFETY: a null array has no backing buffer that could be scalar.
        unsafe { Flat::from_array_unchecked(*self) }
    }

    /// Borrows this array as a flat one, which it always is — see [`PlNullArray::is_flat`].
    #[inline]
    pub fn as_flat(&self) -> Option<&Flat<Self>> {
        // SAFETY: a null array has no backing buffer that could be scalar.
        Some(unsafe { Flat::from_ref_unchecked(self) })
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
    ///
    /// This is the one array whose validity cannot be replaced, so unlike the others it accepts
    /// any mask — including one that is neither flat nor scalar — and ignores it. In particular
    /// [`without_validity`](PlArray::without_validity) leaves every element null.
    #[inline]
    fn set_validity(&mut self, _validity: Option<Bitmap>) {}

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
///
/// A null array is nothing but a length, so this builder is nothing but a length either: every
/// element it appends is a null, whatever it was appended from, and the array it freezes costs
/// `O(1)` memory however many elements it holds.
///
/// # Example
/// ```
/// use polars_array::builder::{ShareStrategy, StaticArrayBuilder};
/// use polars_array::{PlNullArray, PlNullArrayBuilder};
///
/// let mut builder = PlNullArrayBuilder::new();
/// builder.extend_nulls(1_000_000_000);
/// builder.extend(&PlNullArray::new(1), ShareStrategy::Always);
///
/// let array = builder.freeze();
/// assert_eq!(array.len(), 1_000_000_001);
/// assert_eq!(array.null_count(), 1_000_000_001);
/// ```
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
    fn validity_is_a_shared_scalar_mask() {
        let validity = PlNullArray::new(1_000_000_000).validity();

        assert_eq!(validity.len(), 1_000_000_000);
        assert!(validity.flat_bitmap().is_none());
        assert!(validity.is_scalar());
        assert_eq!(validity.unset_bits(), 1_000_000_000);
        assert_eq!(validity.set_bits(), 0);
        assert!(!validity.get(999_999_999));
        assert_eq!(validity.scalar_value(), Some(false));

        // Every array reads the same bit, so the mask outlives the array it came from.
        let validity: PlBitmapRef<'static> = PlNullArray::new(2).validity();
        assert!(std::ptr::eq(
            validity.into_inner().0,
            PlNullArray::new(7).validity().into_inner().0,
        ));

        // A mask of one bit over one element is flat and scalar at once, like everywhere else.
        assert!(PlNullArray::new(1).validity().is_flat());
        assert_eq!(PlNullArray::new(0).validity().len(), 0);
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

    #[test]
    #[should_panic(expected = "must be smaller than the length")]
    fn slicing_out_of_bounds_panics() {
        let _ = PlNullArray::new(3).sliced(2, 2);
    }

    #[test]
    #[should_panic(expected = "index out of bounds")]
    fn reading_out_of_bounds_panics() {
        let _ = PlNullArray::new(3).is_null(3);
    }

    #[test]
    fn equality_is_equality_of_lengths() {
        assert_eq!(PlNullArray::new(3), PlNullArray::new(3));
        assert_ne!(PlNullArray::new(3), PlNullArray::new(4));
        assert_eq!(PlNullArray::new_empty(), PlNullArray::default());
    }

    #[test]
    fn debug_does_not_materialize_the_elements() {
        assert_eq!(
            format!("{:?}", PlNullArray::new(1_000_000_000)),
            "PlNullArray[null; 1000000000]",
        );
        assert_eq!(format!("{:?}", PlNullArray::new(1)), "PlNullArray[null]");
        assert_eq!(format!("{:?}", PlNullArray::new_empty()), "PlNullArray[]");
    }

    #[test]
    fn behind_the_trait_object() {
        let arr: Box<dyn PlArray> = Box::new(PlNullArray::new(1_000_000_000));

        assert_eq!(arr.array_type(), PlArrayType::Null);
        assert!(arr.array_type().is_null());
        assert_eq!(arr.len(), 1_000_000_000);
        assert_eq!(arr.null_count(), 1_000_000_000);
        assert!(arr.has_nulls());
        assert!(arr.is_null(999_999_999));
        assert!(arr.validity().unwrap().is_scalar());
        assert_eq!(&arr, &arr.clone());

        let sliced = arr.sliced(500, 2);
        assert_eq!(sliced.len(), 2);
        assert_eq!(sliced.null_count(), 2);

        // A null array is not the fully null array of any other type.
        let other: Box<dyn PlArray> = Box::new(crate::PlBooleanArray::new_full_null(1_000_000_000));
        assert_ne!(&arr, &other);
    }

    #[test]
    fn the_validity_of_a_null_array_cannot_be_replaced() {
        let arr: Box<dyn PlArray> = Box::new(PlNullArray::new(3));

        // Unlike every other array, dropping the mask leaves every element null.
        let valid = arr.without_validity();
        assert_eq!(valid.null_count(), 3);
        assert!(valid.validity().is_some());

        let masked = arr.with_validity(Some(Bitmap::new_with_value(true, 3)));
        assert_eq!(masked.null_count(), 3);
    }

    #[test]
    fn new_from_index_repeats_a_null() {
        let arr = PlNullArray::new(3);

        assert_eq!(
            arr.new_from_index(2, 1_000_000_000),
            PlNullArray::new(1_000_000_000),
        );
        assert_eq!(
            unsafe { arr.new_from_index_unchecked(0, 2) },
            PlNullArray::new(2)
        );
        assert!(arr.new_from_index(0, 0).is_empty());
    }

    #[test]
    #[should_panic(expected = "index out of bounds")]
    fn repeating_an_element_out_of_bounds_panics() {
        let _ = PlNullArray::new(3).new_from_index(3, 1);
    }
}
