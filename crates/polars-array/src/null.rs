use std::sync::LazyLock;

use arrow::bitmap::Bitmap;

use crate::array::PlArray;
use crate::array_type::PlArrayType;
use crate::bitmap::PlBitmapRef;

/// The validity mask every [`PlNullArray`] hands out: a single unset bit, read by every element of
/// every null array.
///
/// A [`Bitmap`] cannot be built in a constant, so this is initialized on first use and then shared
/// for the rest of the program; a [`PlNullArray`] therefore stores no mask of its own.
static ALL_NULL: LazyLock<Bitmap> = LazyLock::new(|| Bitmap::new_zeroed(1));

/// An immutable, cheaply cloneable sequence of `length` nulls.
///
/// This is the array of the type that holds nothing but nulls: it has no values, no element type
/// and no buffers, only a length. Every element is null, which is what distinguishes it from the
/// fully null array of any other type — there is no value hiding under the mask, undetermined or
/// otherwise.
///
/// Because there is nothing to store, an array of any length costs `O(1)` memory, and every
/// operation on it is `O(1)`. Its validity mask is the single shared bit of [`struct@ALL_NULL`],
/// handed out as a scalar [`PlBitmapRef`] over [`Self::len`] elements exactly like the scalar masks
/// of the other arrays — see [`crate::broadcast`] for the rules. Having no buffers, it is neither
/// meaningfully flat nor meaningfully scalar, and so has no `to_flat` counterpart.
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
/// assert_eq!(validity.bitmap().len(), 1);
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
    ///
    /// The returned [`PlBitmapRef`] has [`Self::len`] bits, backed by the single unset bit of
    /// [`struct@ALL_NULL`]. It borrows from that static rather than from this array, so it outlives
    /// the array it came from.
    #[inline]
    pub fn validity(&self) -> PlBitmapRef<'static> {
        let bitmap: &'static Bitmap = &ALL_NULL;
        // SAFETY: a bitmap of one bit is a scalar bitmap for an array of any length.
        unsafe { PlBitmapRef::new_unchecked(bitmap, self.length) }
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
        assert_eq!(validity.bitmap().len(), 1);
        assert!(validity.is_scalar());
        assert_eq!(validity.unset_bits(), 1_000_000_000);
        assert_eq!(validity.set_bits(), 0);
        assert!(!validity.get(999_999_999));
        assert_eq!(validity.scalar_value(), Some(false));

        // Every array reads the same bit, so the mask outlives the array it came from.
        let validity: PlBitmapRef<'static> = PlNullArray::new(2).validity();
        assert!(std::ptr::eq(
            validity.bitmap(),
            PlNullArray::new(7).validity().bitmap(),
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
}
