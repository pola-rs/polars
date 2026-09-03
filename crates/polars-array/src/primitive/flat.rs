//! What a [`PlPrimitiveArray`] gains from being known to be [`Flat`].

use arrow::bitmap::Bitmap;
use arrow::bitmap::utils::{BitmapIter, ZipValidity};
use arrow::types::NativeType;
use polars_buffer::Buffer;

use super::PlPrimitiveArray;
use crate::flat::Flat;

/// The methods a [`PlPrimitiveArray`] gains from having one slot per element in every backing
/// buffer.
///
/// These are the counterparts of the methods on
/// [`PrimitiveArray`](arrow::array::PrimitiveArray), whose values buffer *is* its elements: they
/// hand out the backing buffers as they are and read them without a
/// [`broadcast_index`](crate::broadcast::broadcast_index). Each shadows the broadcast-aware method
/// of the same name on [`PlPrimitiveArray`], which remains reachable through the deref.
impl<T: NativeType> Flat<PlPrimitiveArray<T>> {
    /// The backing values buffer, holding exactly [`len`](PlPrimitiveArray::len) slots.
    ///
    /// Unlike [`PlPrimitiveArray::flat_values`], this needs no [`Option`] to admit a scalar values
    /// buffer: it is guaranteed to hold one slot per element, so slot
    /// `i` is element `i`. The values of null elements are undetermined (they can be anything).
    #[inline(always)]
    pub const fn values(&self) -> &Buffer<T> {
        &self.0.values
    }

    /// The values as a slice of exactly [`len`](PlPrimitiveArray::len) elements.
    ///
    /// The values of null elements are undetermined (they can be anything).
    #[inline(always)]
    pub fn as_slice(&self) -> &[T] {
        self.0.values.as_slice()
    }

    /// The validity mask, if any element may be null, as an ordinary [`Bitmap`] of exactly
    /// [`len`](PlPrimitiveArray::len) bits.
    ///
    /// Unlike [`PlPrimitiveArray::validity`], this needs no [`PlBitmapRef`](crate::PlBitmapRef) to
    /// hide a scalar bit: bit `i` is element `i`.
    #[inline]
    pub fn validity(&self) -> Option<&Bitmap> {
        self.0.validity.as_ref()
    }

    /// Returns the value at `i`.
    ///
    /// The value of a null element is undetermined (it can be anything).
    ///
    /// # Panics
    /// Panics if `i >= self.len()`.
    #[inline]
    pub fn value(&self, i: usize) -> T {
        assert!(i < self.0.length, "index out of bounds");
        unsafe { self.value_unchecked(i) }
    }

    /// Returns the value at `i`.
    ///
    /// The value of a null element is undetermined (it can be anything).
    ///
    /// # Safety
    /// `i` must be smaller than `self.len()`.
    #[inline]
    pub unsafe fn value_unchecked(&self, i: usize) -> T {
        debug_assert!(i < self.0.length);
        unsafe { *self.0.values.get_unchecked(i) }
    }

    /// Returns whether the element at `i` is valid (non-null).
    ///
    /// # Panics
    /// Panics if `i >= self.len()`.
    #[inline]
    pub fn is_valid(&self, i: usize) -> bool {
        assert!(i < self.0.length, "index out of bounds");
        unsafe { self.is_valid_unchecked(i) }
    }

    /// Returns whether the element at `i` is valid (non-null).
    ///
    /// # Safety
    /// `i` must be smaller than `self.len()`.
    #[inline]
    pub unsafe fn is_valid_unchecked(&self, i: usize) -> bool {
        debug_assert!(i < self.0.length);
        // SAFETY: the mask has one bit per element, so `i` is in bounds of it too.
        self.validity()
            .is_none_or(|validity| unsafe { validity.get_bit_unchecked(i) })
    }

    /// Returns whether the element at `i` is null.
    ///
    /// # Panics
    /// Panics if `i >= self.len()`.
    #[inline]
    pub fn is_null(&self, i: usize) -> bool {
        !self.is_valid(i)
    }

    /// Returns whether the element at `i` is null.
    ///
    /// # Safety
    /// `i` must be smaller than `self.len()`.
    #[inline]
    pub unsafe fn is_null_unchecked(&self, i: usize) -> bool {
        unsafe { !self.is_valid_unchecked(i) }
    }

    /// Returns the element at `i`, or `None` if it is null.
    ///
    /// # Panics
    /// Panics if `i >= self.len()`.
    #[inline]
    pub fn get(&self, i: usize) -> Option<T> {
        assert!(i < self.0.length, "index out of bounds");
        unsafe { self.get_unchecked(i) }
    }

    /// Returns the element at `i`, or `None` if it is null.
    ///
    /// # Safety
    /// `i` must be smaller than `self.len()`.
    #[inline]
    pub unsafe fn get_unchecked(&self, i: usize) -> Option<T> {
        unsafe { self.is_valid_unchecked(i).then(|| self.value_unchecked(i)) }
    }

    /// Returns an iterator over the values, ignoring validity.
    ///
    /// This walks the values buffer directly, so — unlike
    /// [`PlPrimitiveArray::values_iter`] — it is an ordinary [`slice::Iter`](std::slice::Iter) and
    /// yields references. The values of null elements are undetermined (they can be anything).
    #[inline]
    pub fn values_iter(&self) -> std::slice::Iter<'_, T> {
        self.as_slice().iter()
    }

    /// Returns an iterator over the optional elements.
    ///
    /// This zips the two backing buffers directly, so — unlike [`PlPrimitiveArray::iter`], which
    /// yields `Option<T>` — it mirrors [`PrimitiveArray::iter`](arrow::array::PrimitiveArray::iter)
    /// and yields `Option<&T>`.
    #[inline]
    pub fn iter(&self) -> ZipValidity<&T, std::slice::Iter<'_, T>, BitmapIter<'_>> {
        ZipValidity::new_with_validity(self.values_iter(), self.validity())
    }

    /// The backing values buffer as a mutable slice, if no other array shares it.
    ///
    /// This is what lets a kernel write its result over its argument. It hands out `None` when
    /// the buffer is shared, which leaves the caller to allocate one of its own.
    #[inline]
    pub fn values_mut(&mut self) -> Option<&mut [T]> {
        self.0.values.get_mut_slice()
    }

    /// Takes the validity mask out, leaving every element valid.
    ///
    /// This is what keeps a kernel from combining the same masks twice: the masks come out before
    /// the values are handed over, and their combination goes onto the result.
    #[inline]
    pub fn take_validity(&mut self) -> Option<Bitmap> {
        self.0.validity.take()
    }

    /// Reinterprets the values buffer as one of `U`, keeping the validity mask.
    ///
    /// This is what a kernel that writes its result over its argument crosses back through, `U`
    /// being the element type it produced. It is `O(1)`.
    ///
    /// # Panics
    /// Panics unless `T` and `U` have the same size and alignment.
    pub fn transmute<U: NativeType>(self) -> Flat<PlPrimitiveArray<U>> {
        let (values, validity) = self.into_inner();
        let values = values
            .try_transmute::<U>()
            .expect("values buffer cannot be reinterpreted");
        let length = values.len();

        // SAFETY: the buffers held one slot per element, and reinterpreting the values buffer as
        // a type of the same size leaves it one slot per element too.
        unsafe { Flat::new(PlPrimitiveArray::new_unchecked(values, length, validity)) }
    }

    /// Fills every element with `value`, leaving the validity mask as it is.
    ///
    /// The result is the single value repeated, so this is `O(1)` — it is not laid out flat, which
    /// is why it comes back as a [`PlPrimitiveArray`].
    pub fn fill_with(self, value: T) -> PlPrimitiveArray<T> {
        let length = self.len();
        let (_, validity) = self.into_inner();

        PlPrimitiveArray::new_scalar(value, length).with_validity(validity)
    }

    /// Consumes this array into its backing buffers, which both hold one slot per element.
    ///
    /// The length is not part of the result: it is the length of the values buffer.
    #[inline]
    pub fn into_inner(self) -> (Buffer<T>, Option<Bitmap>) {
        let PlPrimitiveArray {
            values,
            length: _,
            validity,
        } = self.0;

        (values, validity)
    }
}

impl<'a, T: NativeType> IntoIterator for &'a Flat<PlPrimitiveArray<T>> {
    type Item = Option<&'a T>;
    type IntoIter = ZipValidity<&'a T, std::slice::Iter<'a, T>, BitmapIter<'a>>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

/// Compares an array of unknown representation against a flat one; see
/// [`PartialEq<PlPrimitiveArray<T>> for Flat<PlPrimitiveArray<T>>`](Flat).
impl<T: NativeType> PartialEq<Flat<PlPrimitiveArray<T>>> for PlPrimitiveArray<T> {
    #[inline]
    fn eq(&self, other: &Flat<PlPrimitiveArray<T>>) -> bool {
        *self == other.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn values_are_written_over_when_the_buffer_is_not_shared() {
        let mut flat = PlPrimitiveArray::from_vec(vec![1i32, 2, 3]).to_flat();
        flat.values_mut().expect("the buffer is not shared")[0] = 7;
        assert_eq!(flat.as_slice(), [7, 2, 3]);

        // A second reference to the buffer leaves the caller to allocate one of its own.
        let shared = flat.clone();
        assert!(flat.values_mut().is_none());
        drop(shared);
    }

    #[test]
    fn the_validity_mask_comes_out_flat() {
        let mut flat: Flat<PlPrimitiveArray<i32>> = [Some(1), None, Some(3)]
            .into_iter()
            .collect::<PlPrimitiveArray<i32>>()
            .to_flat();

        let validity = flat.take_validity().expect("the array has a null");
        assert_eq!(validity.len(), 3);
        assert_eq!(flat.null_count(), 0);
        assert!(flat.take_validity().is_none());
    }

    #[test]
    fn values_are_reinterpreted_in_place() {
        let flat = PlPrimitiveArray::from_vec(vec![1u32, 2, 3]).to_flat();
        let values = flat.values().as_ptr();

        let transmuted = flat.transmute::<i32>();

        assert_eq!(transmuted.as_slice(), [1i32, 2, 3]);
        assert_eq!(
            transmuted.values().as_ptr().cast::<u32>(),
            values,
            "the values buffer must be reinterpreted, not copied",
        );
    }

    #[test]
    fn filling_repeats_the_single_value() {
        let flat: Flat<PlPrimitiveArray<i32>> = [Some(1), None, Some(3)]
            .into_iter()
            .collect::<PlPrimitiveArray<i32>>()
            .to_flat();

        let filled = flat.fill_with(7);

        // The value is not written out, and the mask it was filled under is left alone.
        assert!(filled.values_are_scalar());
        assert_eq!(filled.len(), 3);
        assert_eq!(filled.get(0), Some(7));
        assert_eq!(filled.get(1), None);
        assert_eq!(filled.get(2), Some(7));
    }

    #[test]
    fn to_flat_materializes_scalars() {
        let scalar = PlPrimitiveArray::new_scalar(7i32, 3);
        let flat = scalar.to_flat();

        assert!(flat.is_flat());
        assert_eq!(flat.values().as_slice(), [7, 7, 7]);
        assert_eq!(flat, scalar);

        let null_scalar = PlPrimitiveArray::<i32>::new_full_null(3);
        let flat = null_scalar.to_flat();

        assert!(flat.is_flat());
        assert_eq!(flat.validity().unwrap().len(), 3);
        assert_eq!(flat.null_count(), 3);
        assert_eq!(flat, null_scalar);
    }

    #[test]
    fn as_flat_borrows_an_already_flat_array() {
        let arr: PlPrimitiveArray<i32> = [Some(1), None, Some(3)].into_iter().collect();
        let flat = arr.as_flat().expect("the array is flat");

        assert_eq!(flat.as_slice(), [1, 0, 3]);
        assert_eq!(*flat, arr);
        assert!(
            flat.values().is_same_buffer(arr.flat_values().unwrap()),
            "the values buffer must be borrowed, not materialized again",
        );

        // Neither a scalar buffer nor a scalar validity mask can be borrowed as flat.
        assert!(PlPrimitiveArray::new_scalar(7i32, 3).as_flat().is_none());
        assert!(
            PlPrimitiveArray::from_vec(vec![1i32, 2, 3])
                .with_validity_broadcast(Some(Bitmap::new_zeroed(1)))
                .as_flat()
                .is_none()
        );

        // A scalar array of unbounded length is still `O(1)` to reject.
        assert!(
            PlPrimitiveArray::<i32>::new_full_null(1_000_000_000)
                .as_flat()
                .is_none()
        );
    }

    #[test]
    fn elements_are_read_without_a_broadcast() {
        let flat: Flat<PlPrimitiveArray<i32>> = [Some(1), None, Some(3)]
            .into_iter()
            .collect::<PlPrimitiveArray<i32>>()
            .to_flat();

        assert_eq!(flat.value(0), 1);
        assert_eq!(flat.get(0), Some(1));
        assert!(flat.is_valid(0));
        assert!(flat.is_null(1));
        assert_eq!(flat.get(1), None);
        assert_eq!(flat.get(2), Some(3));

        assert_eq!(unsafe { flat.value_unchecked(2) }, 3);
        assert_eq!(unsafe { flat.get_unchecked(1) }, None);
        assert!(unsafe { flat.is_null_unchecked(1) });
    }
}
