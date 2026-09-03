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
/// buffers, so each buffer is independently either flat or scalar — see [`crate::broadcast`] for
/// the rules. Which representation an array is in is a property of the concrete array; this trait
/// does not expose it.
///
/// # Example
/// ```
/// use polars_array::{PlArray, PlArrayType, PlPrimitiveArray, PrimitiveType};
///
/// let arr: Box<dyn PlArray> = Box::new(PlPrimitiveArray::<i32>::new_scalar(7, 1_000_000_000));
///
/// assert_eq!(arr.array_type(), PlArrayType::Primitive(PrimitiveType::Int32));
/// assert_eq!(arr.len(), 1_000_000_000);
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

    /// Whether this array is entirely stored in the scalar representation, and therefore is a
    /// single logical value repeated [`PlArray::len`] times in `O(1)` memory.
    ///
    /// This is what lets an operation hand a repeated value to the kernel that takes one value
    /// rather than write [`PlArray::len`] copies of it out first. An array of one element is
    /// scalar and flat at once: the two representations coincide. See [`crate::broadcast`].
    fn is_scalar(&self) -> bool;

    /// The validity mask, if any element may be null.
    ///
    /// The returned [`PlBitmapRef`] has [`PlArray::len`] bits regardless of whether the backing
    /// bitmap is flat or scalar.
    fn validity(&self) -> Option<PlBitmapRef<'_>>;

    /// The number of null elements.
    ///
    /// This is `O(1)` for a scalar validity mask and `O(len)` for a flat one, amortized over
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

    /// Replaces the validity mask with a flat one.
    ///
    /// # Panics
    /// Panics if `validity` does not hold one bit per element.
    /// [`Self::set_validity_broadcast`] is what installs the single bit every element shares;
    /// this function never infers that from a mask that happens to hold one bit.
    fn set_validity(&mut self, validity: Option<Bitmap>);

    /// Replaces the validity mask with one that broadcasts over this array.
    ///
    /// This is [`Self::set_validity`] widened to the scalar representation: the mask is either
    /// flat — one bit per element — or the single bit every element shares. See
    /// [`crate::broadcast`].
    ///
    /// # Panics
    /// Panics if `validity` is neither flat nor scalar for this array's length.
    fn set_validity_broadcast(&mut self, validity: Option<Bitmap>);

    /// Returns this array with its validity mask replaced by a flat one.
    ///
    /// # Panics
    /// Panics under the conditions [`Self::set_validity`] panics.
    #[must_use]
    fn with_validity(&self, validity: Option<Bitmap>) -> Box<dyn PlArray> {
        let mut new = self.to_boxed();
        new.set_validity(validity);
        new
    }

    /// Returns this array with its validity mask replaced by one that broadcasts over it.
    ///
    /// # Panics
    /// Panics under the conditions [`Self::set_validity_broadcast`] panics.
    #[must_use]
    fn with_validity_broadcast(&self, validity: Option<Bitmap>) -> Box<dyn PlArray> {
        let mut new = self.to_boxed();
        new.set_validity_broadcast(validity);
        new
    }

    /// Returns this array with its validity mask dropped, making every element valid.
    #[must_use]
    fn without_validity(&self) -> Box<dyn PlArray> {
        self.with_validity(None)
    }

    /// Returns an array of `length` copies of the element at `index`.
    ///
    /// The result keeps the scalar representation wherever it admits it, so its length is
    /// unbounded by its memory use: this is `O(1)` for every array but a
    /// [`PlStructArray`](crate::PlStructArray), which repeats one element per field. See the
    /// concrete arrays for the exact cost.
    ///
    /// # Panics
    /// Panics if `index >= self.len()`.
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
    ///
    /// This function is `O(1)`: every backing buffer is cheaply cloneable.
    fn to_boxed(&self) -> Box<dyn PlArray>;

    /// Returns an array of `length` nulls, laid out like this array.
    ///
    /// The arrays of this crate all take the default implementation, which is
    /// [`full_null_like`](crate::builder::full_null_like); this is the dispatch point that array
    /// types outside the crate — a `polars_core::ObjectArray`, which that function cannot
    /// construct — override with their own.
    #[must_use]
    fn full_null_like(&self, length: usize) -> Box<dyn PlArray> {
        // `full_null_like` reads the shape of a `&dyn PlArray`, which `&Self` does not coerce to
        // in a default method. Boxing is `O(1)`, so going through one costs nothing.
        crate::builder::full_null_like(&*self.to_boxed(), length)
    }

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

/// Compares two arrays element-wise; the representation (flat or scalar) is irrelevant, but
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

/// Compares two arrays element-wise, exactly like [`PartialEq`]: an array holds no value that is
/// unequal to itself, so there is nothing for a total comparison to do differently.
impl polars_utils::total_ord::TotalEq for Box<dyn PlArray> {
    #[inline]
    fn tot_eq(&self, other: &Self) -> bool {
        self.eq_dyn(&**other)
    }
}

#[cfg(test)]
mod tests {
    use arrow::types::PrimitiveType;
    use polars_buffer::Buffer;

    use super::*;
    use crate::{
        PlBinaryArray, PlBinaryViewArray, PlBitmap, PlBooleanArray, PlFixedSizeBinaryArray,
        PlFixedSizeListArray, PlListArray, PlNullArray, PlPrimitiveArray, PlStructArray,
        PlUtf8ViewArray, StaticArray,
    };

    fn arrays() -> Vec<Box<dyn PlArray>> {
        vec![
            Box::new(PlPrimitiveArray::from_vec(vec![1i32, 2, 3])),
            Box::new(PlBooleanArray::from_vec(vec![true, false, true])),
            Box::new(PlBinaryArray::from_values_iter([
                b"foo".as_slice(),
                b"",
                b"bar",
            ])),
            Box::new(PlBinaryViewArray::from_values_iter([
                b"foo".as_slice(),
                b"bar",
                b"a value that is too long to inline",
            ])),
            Box::new(PlFixedSizeBinaryArray::from_vec(
                vec![1u8, 2, 3, 4, 5, 6],
                2,
            )),
            Box::new(PlStructArray::from_fields(vec![Box::new(
                PlPrimitiveArray::from_vec(vec![1i32, 2, 3]),
            )])),
            Box::new(PlListArray::from_offsets(
                Box::new(PlPrimitiveArray::from_vec(vec![1i32, 2, 3])),
                Buffer::from(vec![0u64, 1, 2, 3]),
            )),
            Box::new(PlFixedSizeListArray::from_values(
                Box::new(PlPrimitiveArray::from_vec(vec![1i32, 2, 3, 4, 5, 6])),
                2,
            )),
        ]
    }

    /// A scalar array of each type, all of `length` elements.
    fn scalars(length: usize) -> Vec<Box<dyn PlArray>> {
        vec![
            Box::new(PlPrimitiveArray::<i64>::new_scalar(7, length)),
            Box::new(PlBooleanArray::new_scalar(true, length)),
            Box::new(PlBinaryArray::new_scalar(b"ab", length)),
            Box::new(PlBinaryViewArray::new_scalar(
                b"a value that is too long to inline",
                length,
            )),
            Box::new(PlFixedSizeBinaryArray::new_scalar(b"ab", length)),
            Box::new(PlStructArray::from_fields(vec![Box::new(
                PlPrimitiveArray::<i64>::new_scalar(7, length),
            )])),
            Box::new(PlFixedSizeListArray::new_scalar(
                Box::new(PlPrimitiveArray::from_vec(vec![1i64, 2])),
                length,
            )),
        ]
    }

    /// Asserts that `array` holds no elements and no slot in any backing buffer, which is what
    /// makes an empty array flat as well as scalar.
    fn assert_empty_and_flat<A: StaticArray>(array: A) {
        assert!(array.is_empty(), "{array:?}");
        assert!(array.is_flat(), "{array:?}");
    }

    /// The value a scalar array repeats is kept in one slot per backing buffer — except when
    /// there is no element to read it, which is what leaves an empty array with empty buffers.
    #[test]
    fn an_empty_array_keeps_no_scalar_slot() {
        let element = || Box::new(PlPrimitiveArray::from_vec(vec![1i64, 2]));

        // Built empty outright.
        assert_empty_and_flat(PlPrimitiveArray::<i64>::new_scalar(7, 0));
        assert_empty_and_flat(PlPrimitiveArray::<i64>::new_full_null(0));
        assert_empty_and_flat(PlBooleanArray::new_scalar(true, 0));
        assert_empty_and_flat(PlBooleanArray::new_full_null(0));
        assert_empty_and_flat(PlBinaryArray::new_scalar(b"ab", 0));
        assert_empty_and_flat(PlBinaryArray::new_full_null(0));
        assert_empty_and_flat(PlBinaryViewArray::new_scalar(b"ab", 0));
        assert_empty_and_flat(PlBinaryViewArray::new_full_null(0));
        assert_empty_and_flat(PlUtf8ViewArray::new_scalar("ab", 0));
        assert_empty_and_flat(PlUtf8ViewArray::new_full_null(0));
        assert_empty_and_flat(PlFixedSizeBinaryArray::new_scalar(b"ab", 0));
        assert_empty_and_flat(PlFixedSizeBinaryArray::new_full_null(2, 0));
        assert_empty_and_flat(PlListArray::new_scalar(element(), 0));
        assert_empty_and_flat(PlListArray::new_full_null(element(), 0));
        assert_empty_and_flat(PlFixedSizeListArray::new_scalar(element(), 0));
        assert_empty_and_flat(PlFixedSizeListArray::new_full_null(element(), 0));
        assert_empty_and_flat(PlStructArray::new_full_null(
            vec![Box::new(PlPrimitiveArray::<i64>::new_empty())],
            0,
        ));
        assert_empty_and_flat(PlNullArray::new_full_null(0));
        assert!(PlBitmap::new_scalar(true, 0).is_flat());
        // A null array has no buffer of its own; the mask it hands out is the one to check.
        assert!(
            PlNullArray::new_full_null(0)
                .validity()
                .flat_bitmap()
                .is_some()
        );

        // Repeated no times out of an element of an array at hand.
        assert_empty_and_flat(PlPrimitiveArray::from_vec(vec![1i64, 2]).new_from_index(0, 0));
        assert_empty_and_flat(PlBooleanArray::from_vec(vec![true, false]).new_from_index(0, 0));
        assert_empty_and_flat(
            PlBinaryArray::from_values_iter([b"ab".as_slice()]).new_from_index(0, 0),
        );
        assert_empty_and_flat(
            PlBinaryViewArray::from_values_iter([b"a value that is too long to inline".as_slice()])
                .new_from_index(0, 0),
        );
        assert_empty_and_flat(
            PlFixedSizeBinaryArray::from_vec(vec![1u8, 2], 2).new_from_index(0, 0),
        );
        assert_empty_and_flat(PlListArray::new_scalar(element(), 1).new_from_index(0, 0));
        assert_empty_and_flat(PlFixedSizeListArray::new_scalar(element(), 1).new_from_index(0, 0));
        assert_empty_and_flat(
            PlStructArray::new_full_null(
                vec![Box::new(PlPrimitiveArray::<i64>::new_full_null(1))],
                1,
            )
            .new_from_index(0, 0),
        );

        // Sliced down to nothing, which leaves no element to read the slot either.
        assert_empty_and_flat(PlPrimitiveArray::<i64>::new_scalar(7, 5).sliced(0, 0));
        assert_empty_and_flat(PlPrimitiveArray::<i64>::new_full_null(5).sliced(0, 0));
        assert_empty_and_flat(PlBooleanArray::new_full_null(5).sliced(0, 0));
        assert_empty_and_flat(PlBinaryArray::new_full_null(5).sliced(0, 0));
        assert_empty_and_flat(PlBinaryViewArray::new_full_null(5).sliced(0, 0));
        assert_empty_and_flat(PlFixedSizeBinaryArray::new_full_null(2, 5).sliced(0, 0));
        assert_empty_and_flat(PlListArray::new_full_null(element(), 5).sliced(0, 0));
        assert_empty_and_flat(PlFixedSizeListArray::new_full_null(element(), 5).sliced(0, 0));
        assert_empty_and_flat(
            PlStructArray::new_full_null(
                vec![Box::new(PlPrimitiveArray::<i64>::new_full_null(5))],
                5,
            )
            .sliced(0, 0),
        );
        assert!(PlBitmap::new_scalar(true, 5).sliced(0, 0).is_flat());
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
    fn shared_accessors() {
        for arr in arrays() {
            assert_eq!(arr.len(), 3);
            assert!(!arr.is_empty());
            assert!(arr.validity().is_none());
            assert_eq!(arr.null_count(), 0);
            assert!(!arr.has_nulls());
            assert!(arr.is_valid(2));
            assert!(!arr.is_null(2));
        }
    }

    #[test]
    fn is_scalar_behind_the_trait_object() {
        // The arrays of `arrays()` hold three elements each, none of them repeated.
        for arr in arrays() {
            assert!(!arr.is_scalar(), "{arr:?}");
        }

        // A billion elements would not be walked in reasonable time; that this test finishes is
        // what shows the answer is read off the buffers rather than from the elements.
        for arr in scalars(1_000_000_000) {
            assert!(arr.is_scalar(), "{arr:?}");
        }

        // An array of one element is scalar and flat at once, whichever way it was built.
        for arr in scalars(1) {
            assert!(arr.is_scalar(), "{arr:?}");
        }
        for arr in arrays() {
            assert!(arr.sliced(1, 1).is_scalar(), "{arr:?}");
        }

        // An array of no elements repeats nothing, and a null array repeats its null.
        for arr in arrays() {
            assert!(!arr.sliced(0, 0).is_scalar(), "{arr:?}");
        }
        assert!(PlNullArray::new(1_000_000_000).is_scalar());
    }
}
