use std::any::Any;

use arrow::bitmap::Bitmap;

use crate::array_type::PlArrayType;
use crate::bitmap::PlBitmapRef;

/// A trait object over the arrays in this crate.
///
/// This is the counterpart of [`Array`](arrow::array::Array): a dyn-compatible view of everything
/// the concrete arrays have in common, infallibly downcast to a concrete type according to
/// [`PlArray::array_type`]. It exposes only what does not depend on the element type; reading values
/// means downcasting through [`PlArray::as_any`].
///
/// Like the concrete arrays, an implementor stores its logical length separately from its backing
/// buffers, so each buffer is independently either flat or broadcast. See [`crate::broadcast`]
/// for the rules, and [`PlArray::is_scalar`] to detect the `O(1)`-memory case a `dyn PlArray`
/// must not walk element by element.
///
/// # Example
/// ```
/// use polars_array::{PlArray, PlArrayType, PlPrimitiveArray, PrimitiveType};
///
/// let arr: Box<dyn PlArray> = Box::new(PlPrimitiveArray::<i32>::new_scalar(7, 1_000_000_000));
///
/// assert_eq!(arr.array_type(), PlArrayType::Primitive(PrimitiveType::Int32));
/// assert_eq!(arr.len(), 1_000_000_000);
/// assert!(arr.is_scalar());
/// assert_eq!(arr.null_count(), 0);
///
/// let arr = arr.as_any().downcast_ref::<PlPrimitiveArray<i32>>().unwrap();
/// assert_eq!(arr.value(999_999_999), 7);
/// ```
pub trait PlArray: std::fmt::Debug + Send + Sync + 'static {
    /// Converts itself to a reference of [`Any`], which enables downcasting to concrete types.
    fn as_any(&self) -> &dyn Any;

    /// Converts itself to a mutable reference of [`Any`], which enables mutable downcasting to
    /// concrete types.
    fn as_any_mut(&mut self) -> &mut dyn Any;

    /// The physical representation of this array.
    ///
    /// In combination with [`PlArray::as_any`], this can be used to downcast a `dyn PlArray` to a
    /// concrete array. It is determined by the Rust type of the array rather than stored in it,
    /// which is why it is returned by value and cannot be changed.
    fn array_type(&self) -> PlArrayType;

    /// The number of elements in this array.
    fn len(&self) -> usize;

    /// Whether this array holds no elements.
    #[inline]
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The validity mask, if any element may be null.
    ///
    /// The returned [`PlBitmapRef`] has [`PlArray::len`] bits regardless of whether the backing
    /// bitmap is flat or broadcast.
    fn validity(&self) -> Option<PlBitmapRef<'_>>;

    /// The number of null elements.
    ///
    /// This is `O(1)` for a broadcast validity mask and `O(len)` for a flat one, amortized over
    /// repeated calls on the same [`Bitmap`].
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
    ///
    /// # Panics
    /// Panics if `i >= self.len()`.
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
    ///
    /// # Panics
    /// Panics if `i >= self.len()`.
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

    /// Whether the values buffer holds a single value shared by every element.
    ///
    /// This is `false` for a flat array of length one, where the two representations coincide.
    fn values_are_broadcast(&self) -> bool;

    /// Whether the validity mask holds a single bit shared by every element.
    #[inline]
    fn validity_is_broadcast(&self) -> bool {
        self.validity()
            .is_some_and(|validity| validity.is_broadcast())
    }

    /// Whether every backing buffer has one slot per element.
    #[inline]
    fn is_flat(&self) -> bool {
        !self.values_are_broadcast() && !self.validity_is_broadcast()
    }

    /// Whether this array is entirely stored in the broadcast representation, and therefore is a
    /// single logical value repeated [`PlArray::len`] times in `O(1)` memory.
    #[inline]
    fn is_scalar(&self) -> bool {
        self.values_are_broadcast()
            && self
                .validity()
                .is_none_or(|validity| validity.is_broadcast())
    }

    /// Slices this array in place to `length` elements starting at `offset`.
    ///
    /// This function is `O(1)`.
    ///
    /// # Panics
    /// Panics if `offset + length > self.len()`.
    fn slice(&mut self, offset: usize, length: usize);

    /// Slices this array in place to `length` elements starting at `offset`.
    ///
    /// This function is `O(1)`.
    ///
    /// # Safety
    /// `offset + length` must not exceed `self.len()`.
    unsafe fn slice_unchecked(&mut self, offset: usize, length: usize);

    /// Returns this array sliced to `length` elements starting at `offset`.
    ///
    /// This function is `O(1)`.
    ///
    /// # Panics
    /// Panics if `offset + length > self.len()`.
    #[must_use]
    fn sliced(&self, offset: usize, length: usize) -> Box<dyn PlArray> {
        let mut sliced = self.to_boxed();
        sliced.slice(offset, length);
        sliced
    }

    /// Returns this array sliced to `length` elements starting at `offset`.
    ///
    /// This function is `O(1)`.
    ///
    /// # Safety
    /// `offset + length` must not exceed `self.len()`.
    #[must_use]
    unsafe fn sliced_unchecked(&self, offset: usize, length: usize) -> Box<dyn PlArray> {
        let mut sliced = self.to_boxed();
        unsafe { sliced.slice_unchecked(offset, length) };
        sliced
    }

    /// Replaces the validity mask.
    ///
    /// # Panics
    /// Panics if `validity` is neither flat nor broadcast for this array's length.
    fn set_validity(&mut self, validity: Option<Bitmap>);

    /// Returns this array with its validity mask replaced.
    ///
    /// # Panics
    /// Panics if `validity` is neither flat nor broadcast for this array's length.
    #[must_use]
    fn with_validity(&self, validity: Option<Bitmap>) -> Box<dyn PlArray> {
        let mut new = self.to_boxed();
        new.set_validity(validity);
        new
    }

    /// Returns this array with its validity mask dropped, making every element valid.
    #[must_use]
    fn without_validity(&self) -> Box<dyn PlArray> {
        self.with_validity(None)
    }

    /// Returns an equivalent array whose backing buffers all hold one slot per element.
    ///
    /// This materializes any broadcast buffer and is therefore `O(len)`; it is a no-op clone when
    /// this array [`is_flat`](PlArray::is_flat).
    #[must_use]
    fn to_flat_boxed(&self) -> Box<dyn PlArray>;

    /// Clones this array into an owned `Box<dyn PlArray>`.
    ///
    /// This function is `O(1)`: every backing buffer is cheaply cloneable.
    fn to_boxed(&self) -> Box<dyn PlArray>;

    /// Compares this array element-wise against `other`, returning `false` if `other` is not of
    /// the same concrete type.
    ///
    /// This backs [`PartialEq`] for `dyn PlArray`; call that instead.
    fn eq_dyn(&self, other: &dyn PlArray) -> bool;
}

impl Clone for Box<dyn PlArray> {
    #[inline]
    fn clone(&self) -> Self {
        self.to_boxed()
    }
}

/// Compares two arrays element-wise; the representation (flat or broadcast) is irrelevant, but
/// arrays of different [`PlArrayType`] never compare equal.
///
/// Compare two `Box<dyn PlArray>` through references (`&lhs == &rhs`): `==` on the boxes
/// themselves autoderefs to this impl and consumes the right-hand box.
impl PartialEq for dyn PlArray + '_ {
    #[inline]
    fn eq(&self, other: &dyn PlArray) -> bool {
        self.eq_dyn(other)
    }
}

impl Eq for dyn PlArray + '_ {}

#[cfg(test)]
mod tests {
    use arrow::types::PrimitiveType;

    use super::*;
    use crate::{PlBooleanArray, PlPrimitiveArray, PlStructArray};

    fn arrays() -> Vec<Box<dyn PlArray>> {
        vec![
            Box::new(PlPrimitiveArray::from_vec(vec![1i32, 2, 3])),
            Box::new(PlBooleanArray::from_vec(vec![true, false, true])),
            Box::new(PlStructArray::from_fields(vec![Box::new(
                PlPrimitiveArray::from_vec(vec![1i32, 2, 3]),
            )])),
        ]
    }

    /// A scalar array of each type, all of `length` elements.
    fn scalars(length: usize) -> Vec<Box<dyn PlArray>> {
        vec![
            Box::new(PlPrimitiveArray::<i64>::new_scalar(7, length)),
            Box::new(PlBooleanArray::new_scalar(true, length)),
            Box::new(PlStructArray::from_fields(vec![Box::new(
                PlPrimitiveArray::<i64>::new_scalar(7, length),
            )])),
        ]
    }

    #[test]
    fn array_type_identifies_the_concrete_array() {
        let arr: Box<dyn PlArray> = Box::new(PlPrimitiveArray::from_vec(vec![1u8, 2]));
        assert_eq!(
            arr.array_type(),
            PlArrayType::Primitive(PrimitiveType::UInt8)
        );
        assert!(arr.array_type().is_primitive());
        assert!(arr.array_type().eq_primitive(PrimitiveType::UInt8));
        assert_eq!(
            arr.as_any()
                .downcast_ref::<PlPrimitiveArray<u8>>()
                .unwrap()
                .value(1),
            2,
        );
        // The element type is part of the identity: `u8` is not `i8`.
        assert!(
            arr.as_any()
                .downcast_ref::<PlPrimitiveArray<i8>>()
                .is_none()
        );

        let arr: Box<dyn PlArray> = Box::new(PlBooleanArray::from_vec(vec![true]));
        assert_eq!(arr.array_type(), PlArrayType::Boolean);
        assert!(arr.array_type().is_boolean());
        assert!(!arr.array_type().is_primitive());
        assert!(
            arr.as_any()
                .downcast_ref::<PlBooleanArray>()
                .unwrap()
                .value(0)
        );
    }

    #[test]
    fn as_any_mut_downcasts() {
        let mut arr: Box<dyn PlArray> = Box::new(PlPrimitiveArray::from_vec(vec![1i32, 2, 3]));
        let concrete = arr
            .as_any_mut()
            .downcast_mut::<PlPrimitiveArray<i32>>()
            .unwrap();
        concrete.slice(1, 2);

        assert_eq!(arr.len(), 2);
    }

    #[test]
    fn shared_accessors() {
        for arr in arrays() {
            assert_eq!(arr.len(), 3);
            assert!(!arr.is_empty());
            assert!(arr.validity().is_none());
            assert_eq!(arr.null_count(), 0);
            assert!(!arr.has_nulls());
            assert!(arr.is_valid(2));
            assert!(!arr.is_null(2));
            assert!(arr.is_flat());
            assert!(!arr.is_scalar());
            assert!(!arr.values_are_broadcast());
            assert!(!arr.validity_is_broadcast());
        }
    }

    #[test]
    fn scalars_stay_cheap_behind_the_trait_object() {
        for arr in scalars(1_000_000_000) {
            assert_eq!(arr.len(), 1_000_000_000);
            assert!(arr.is_scalar());
            assert!(!arr.is_flat());
            assert!(arr.values_are_broadcast());
            assert_eq!(arr.null_count(), 0);

            // Slicing a scalar stays `O(1)`, and so does comparing it.
            let sliced = arr.sliced(500, 2);
            assert_eq!(sliced.len(), 2);
            assert!(sliced.is_scalar());
            assert_eq!(&arr, &arr.clone());
        }
    }

    #[test]
    fn nulls_behind_the_trait_object() {
        let arrs: Vec<Box<dyn PlArray>> = vec![
            Box::new(PlPrimitiveArray::<i32>::new_full_null(4)),
            Box::new(PlBooleanArray::new_full_null(4)),
            Box::new(PlStructArray::new_full_null(
                vec![Box::new(PlPrimitiveArray::<i32>::new_scalar(1, 4))],
                4,
            )),
        ];

        for arr in arrs {
            assert_eq!(arr.null_count(), 4);
            assert!(arr.has_nulls());
            assert!(arr.is_null(3));
            assert!(!arr.is_valid(3));
            assert!(arr.validity_is_broadcast());
            assert_eq!(arr.validity().unwrap().len(), 4);
            assert_eq!(arr.validity().unwrap().bitmap().len(), 1);

            let valid = arr.without_validity();
            assert_eq!(valid.null_count(), 0);
            assert!(valid.validity().is_none());
        }
    }

    #[test]
    fn slicing_through_the_trait_object() {
        for arr in arrays() {
            let sliced = arr.sliced(1, 2);
            assert_eq!(sliced.len(), 2);
            assert_eq!(sliced.array_type(), arr.array_type());

            let sliced = unsafe { arr.sliced_unchecked(1, 2) };
            assert_eq!(sliced.len(), 2);

            // The original is untouched: `sliced` clones first.
            assert_eq!(arr.len(), 3);

            let mut arr = arr;
            arr.slice(0, 1);
            assert_eq!(arr.len(), 1);
            unsafe { arr.slice_unchecked(0, 0) };
            assert!(arr.is_empty());
        }
    }

    #[test]
    fn setting_validity_through_the_trait_object() {
        for arr in arrays() {
            // A broadcast mask of one unset bit nulls out every element.
            let nulled = arr.with_validity(Some(Bitmap::new_zeroed(1)));
            assert_eq!(nulled.null_count(), 3);
            assert!(nulled.validity_is_broadcast());
            assert!(!nulled.is_flat());
            assert!(!nulled.is_scalar());
            assert_eq!(arr.null_count(), 0);

            let mut arr = arr;
            arr.set_validity(Some(Bitmap::from_iter([true, false, true])));
            assert_eq!(arr.null_count(), 1);
            assert!(!arr.validity_is_broadcast());
        }
    }

    #[test]
    fn to_flat_boxed_materializes_broadcasts() {
        for arr in scalars(3) {
            let flat = arr.to_flat_boxed();
            assert!(flat.is_flat());
            assert_eq!(flat.len(), 3);
            assert_eq!(flat.array_type(), arr.array_type());
            assert_eq!(&flat, &arr);
        }
    }

    #[test]
    fn boxed_arrays_are_ordinary_values() {
        fn assert_bounds<T: Clone + PartialEq + std::fmt::Debug + Send + Sync>() {}
        assert_bounds::<Box<dyn PlArray>>();
    }

    #[test]
    fn equality_ignores_representation_but_not_type() {
        let scalar: Box<dyn PlArray> = Box::new(PlPrimitiveArray::<i32>::new_scalar(1, 3));
        let flat: Box<dyn PlArray> = Box::new(PlPrimitiveArray::from_vec(vec![1i32, 1, 1]));
        assert_eq!(&scalar, &flat);

        // Same values, different element type.
        let other: Box<dyn PlArray> = Box::new(PlPrimitiveArray::from_vec(vec![1i64, 1, 1]));
        assert_ne!(&scalar, &other);

        // Same values, different array type.
        let struct_: Box<dyn PlArray> = Box::new(PlStructArray::from_fields(vec![Box::new(
            PlPrimitiveArray::from_vec(vec![1i32, 1, 1]),
        )]));
        assert_ne!(&scalar, &struct_);

        let boolean: Box<dyn PlArray> = Box::new(PlBooleanArray::from_vec(vec![true, true, true]));
        assert_ne!(&scalar, &boolean);
        assert_eq!(
            &boolean,
            &(Box::new(PlBooleanArray::new_scalar(true, 3)) as Box<dyn PlArray>),
        );
    }

    #[test]
    fn cloning_a_boxed_array_is_a_deep_clone_of_the_handle() {
        let mut arr: Box<dyn PlArray> = Box::new(PlPrimitiveArray::from_vec(vec![1i32, 2, 3]));
        let clone = arr.clone();
        arr.slice(0, 1);

        assert_eq!(arr.len(), 1);
        assert_eq!(clone.len(), 3);
    }

    #[test]
    fn debug_formats_the_concrete_array() {
        let arr: Box<dyn PlArray> = Box::new(PlPrimitiveArray::<i32>::new_scalar(7, 1_000_000_000));
        assert_eq!(format!("{arr:?}"), "PlPrimitiveArray[7; 1000000000]");

        let arr: Box<dyn PlArray> = Box::new(PlBooleanArray::from_vec(vec![true, false]));
        assert_eq!(format!("{arr:?}"), "PlBooleanArray[true, false]");
    }
}
