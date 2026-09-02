use arrow::bitmap::{Bitmap, MutableBitmap};
use arrow::types::NativeType;
use polars_buffer::Buffer;
use polars_error::{PolarsResult, polars_ensure};

use crate::array::PlArray;
use crate::array_type::PlArrayType;
use crate::bitmap::PlBitmapRef;
use crate::broadcast::{
    assert_broadcastable, broadcast_index, is_flat_buffer_len, is_scalar_buffer_len,
    is_valid_buffer_len,
};
use crate::flat::Flat;

mod builder;
mod flat;
mod iterator;

pub use builder::PlPrimitiveArrayBuilder;
pub use iterator::{PlPrimitiveIter, PlPrimitiveValuesIter};

/// An immutable, cheaply cloneable sequence of `length` optional [`NativeType`] values.
///
/// This is the lowest-level array in the Polars vector format. It carries no logical type — only
/// the physical values and their validity.
///
/// The logical length is stored separately from the backing buffers, which lets a *scalar* array —
/// one value repeated `length` times — be represented in `O(1)` memory. Element `i` reads slot
/// [`broadcast_index(i, buf.len())`](crate::broadcast::broadcast_index) of each backing buffer, so
/// both `values` and `validity` are independently either flat (one slot per element) or scalar
/// (a single shared slot). See [`crate::broadcast`] for the full rules.
///
/// # Example
/// ```
/// use polars_array::PlPrimitiveArray;
///
/// let flat = PlPrimitiveArray::from_vec(vec![1i32, 2, 3]);
/// assert_eq!(flat.len(), 3);
/// assert_eq!(flat.flat_values().unwrap().as_slice(), [1, 2, 3]);
/// assert_eq!(flat.iter().collect::<Vec<_>>(), [Some(1), Some(2), Some(3)]);
///
/// // A scalar array of a billion elements costs a single `i32` of memory.
/// let scalar = PlPrimitiveArray::new_scalar(7i32, 1_000_000_000);
/// assert_eq!(scalar.len(), 1_000_000_000);
/// assert_eq!(scalar.scalar_values(), Some(7));
/// assert_eq!(scalar.value(999_999_999), 7);
/// ```
#[derive(Clone)]
pub struct PlPrimitiveArray<T: NativeType> {
    values: Buffer<T>,
    length: usize,
    validity: Option<Bitmap>,
}

impl<T: NativeType> PlPrimitiveArray<T> {
    /// Creates a flat [`PlPrimitiveArray`] out of its internal components.
    ///
    /// Every backing buffer has to hold one slot per element. [`Self::try_new_broadcast`] is what
    /// builds the scalar representation; this function never infers it from a buffer that happens
    /// to hold a single value. This function is `O(1)`.
    ///
    /// # Errors
    /// This function errors if `values` or `validity` does not hold exactly `length` slots.
    pub fn try_new(
        values: Buffer<T>,
        length: usize,
        validity: Option<Bitmap>,
    ) -> PolarsResult<Self> {
        polars_ensure!(
            is_flat_buffer_len(values.len(), length),
            ComputeError:
            "values buffer of length {} is not flat for an array of length {}",
            values.len(), length,
        );

        if let Some(validity) = validity.as_ref() {
            polars_ensure!(
                is_flat_buffer_len(validity.len(), length),
                ComputeError:
                "validity mask of length {} is not flat for an array of length {}",
                validity.len(), length,
            );
        }

        Ok(Self {
            values,
            length,
            validity,
        })
    }

    /// Creates a flat [`PlPrimitiveArray`] out of its internal components.
    ///
    /// # Panics
    /// Panics under the conditions [`Self::try_new`] errors.
    #[inline]
    pub fn new(values: Buffer<T>, length: usize, validity: Option<Bitmap>) -> Self {
        Self::try_new(values, length, validity).unwrap()
    }

    /// Creates a flat [`PlPrimitiveArray`] out of its internal components without validating them.
    ///
    /// # Safety
    /// `values` and `validity` must each hold exactly `length` slots.
    #[inline]
    pub unsafe fn new_unchecked(
        values: Buffer<T>,
        length: usize,
        validity: Option<Bitmap>,
    ) -> Self {
        if cfg!(debug_assertions) {
            assert!(is_flat_buffer_len(values.len(), length));
            assert!(
                validity
                    .as_ref()
                    .is_none_or(|v| is_flat_buffer_len(v.len(), length))
            );
        }

        Self {
            values,
            length,
            validity,
        }
    }

    /// Creates a scalar [`PlPrimitiveArray`] of `length` elements out of its internal components.
    ///
    /// Every backing buffer has to hold the single value every element shares, which makes this
    /// `O(1)` in `length` as well as in time. [`Self::try_new`] is what builds the flat
    /// representation.
    ///
    /// # Errors
    /// This function errors if `values` or `validity` does not hold exactly one slot. An array of
    /// no elements reads no slot at all, so it additionally admits an empty buffer.
    pub fn try_new_broadcast(
        values: Buffer<T>,
        length: usize,
        validity: Option<Bitmap>,
    ) -> PolarsResult<Self> {
        polars_ensure!(
            is_scalar_buffer_len(values.len(), length),
            ComputeError:
            "values buffer of length {} is not the single value the {} elements of a broadcast \
             array share",
            values.len(), length,
        );

        if let Some(validity) = validity.as_ref() {
            polars_ensure!(
                is_scalar_buffer_len(validity.len(), length),
                ComputeError:
                "validity mask of length {} is not the single bit the {} elements of a broadcast \
                 array share",
                validity.len(), length,
            );
        }

        Ok(Self {
            values,
            length,
            validity,
        })
    }

    /// Creates a scalar [`PlPrimitiveArray`] of `length` elements out of its internal components.
    ///
    /// # Panics
    /// Panics under the conditions [`Self::try_new_broadcast`] errors.
    #[inline]
    pub fn new_broadcast(values: Buffer<T>, length: usize, validity: Option<Bitmap>) -> Self {
        Self::try_new_broadcast(values, length, validity).unwrap()
    }

    /// Creates a scalar [`PlPrimitiveArray`] of `length` elements out of its internal components
    /// without validating them.
    ///
    /// # Safety
    /// `values` and `validity` must each hold exactly one slot, or none at all if `length` is
    /// zero.
    #[inline]
    pub unsafe fn new_broadcast_unchecked(
        values: Buffer<T>,
        length: usize,
        validity: Option<Bitmap>,
    ) -> Self {
        if cfg!(debug_assertions) {
            assert!(is_scalar_buffer_len(values.len(), length));
            assert!(
                validity
                    .as_ref()
                    .is_none_or(|v| is_scalar_buffer_len(v.len(), length))
            );
        }

        Self {
            values,
            length,
            validity,
        }
    }

    /// Creates an empty [`PlPrimitiveArray`].
    #[inline]
    pub fn new_empty() -> Self {
        Self {
            values: Buffer::new(),
            length: 0,
            validity: None,
        }
    }

    /// Creates a flat, fully valid [`PlPrimitiveArray`] from `values`.
    #[inline]
    pub fn from_values(values: Buffer<T>) -> Self {
        let length = values.len();
        Self {
            values,
            length,
            validity: None,
        }
    }

    /// Creates a flat, fully valid [`PlPrimitiveArray`] from a [`Vec`].
    #[inline]
    pub fn from_vec(values: Vec<T>) -> Self {
        Self::from_values(Buffer::from(values))
    }

    /// Creates a flat, fully valid [`PlPrimitiveArray`] by copying `values`.
    #[inline]
    pub fn from_slice(values: &[T]) -> Self {
        Self::from_vec(values.to_vec())
    }

    /// Creates a [`PlPrimitiveArray`] of `length` copies of `value`, in `O(1)` memory.
    #[inline]
    pub fn new_scalar(value: T, length: usize) -> Self {
        Self {
            values: Buffer::from_owner([value]),
            length,
            validity: None,
        }
    }

    /// Creates a [`PlPrimitiveArray`] of `length` nulls, in `O(1)` memory.
    #[inline]
    pub fn new_full_null(length: usize) -> Self {
        Self {
            values: Buffer::zeroed(1),
            length,
            validity: Some(Bitmap::new_zeroed(1)),
        }
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

    /// The backing values buffer, if it holds one slot per element.
    ///
    /// Slot `i` is then the value of element `i`, with no [`broadcast_index`] in the way. This is
    /// the `O(1)` counterpart of [`Self::to_flat`]: it materializes nothing, and returns `None`
    /// rather than expanding a scalar buffer. Reach for the value a scalar buffer shares with
    /// [`Self::scalar_values`] instead — between them the two cover every array that has elements
    /// at all, so a `None` from both is an empty array. The values of null elements are
    /// undetermined (they can be anything).
    #[inline]
    pub fn flat_values(&self) -> Option<&Buffer<T>> {
        self.values_are_flat().then_some(&self.values)
    }

    /// The values buffer, if this array holds one slot per element and nothing else shares it.
    ///
    /// This is [`Self::flat_values`] with a mutable borrow, which additionally asks that the
    /// buffer be uniquely held: the buffers of these arrays are cheaply cloneable, so writing over
    /// one that another array shares would change that array too. It returns `None` for a scalar
    /// values buffer and for one that is shared, which is what a caller that means to write in
    /// place has to fall back from.
    #[inline]
    pub fn flat_values_mut(&mut self) -> Option<&mut Buffer<T>> {
        self.values_are_flat().then_some(&mut self.values)
    }

    /// The value every element of this array reads, if the values buffer holds a single slot.
    ///
    /// This is the values half of [`Self::scalar_value`], which additionally asks that the
    /// validity mask be scalar and reports the null the mask makes of this value. Returns `None`
    /// for a values buffer that is flat over more than one element, and for an empty array, which
    /// has no element to share a value. The value of a null element is undetermined (it can be
    /// anything).
    #[inline]
    pub fn scalar_values(&self) -> Option<T> {
        (self.values_are_scalar() && self.length > 0).then(|| self.values[0])
    }

    /// The validity mask, if any element may be null.
    ///
    /// The returned [`PlBitmapRef`] has [`Self::len`] bits regardless of whether the backing
    /// bitmap is flat or scalar, so reading validity through it needs no knowledge of which
    /// representation this array is in. Reach for the backing [`Bitmap`] with
    /// [`PlBitmapRef::flat_bitmap`], or materialize a flat one with [`PlBitmapRef::to_flat`].
    #[inline]
    pub fn validity(&self) -> Option<PlBitmapRef<'_>> {
        // SAFETY: the mask is flat or scalar for `self.length`, upheld by every constructor.
        self.validity
            .as_ref()
            .map(|validity| unsafe { PlBitmapRef::new_unchecked(validity, self.length) })
    }

    /// Whether the values buffer holds a single value shared by every element.
    ///
    /// An array of one element is both scalar and [`flat`](Self::is_flat): the two representations
    /// coincide, and this reports them both.
    #[inline]
    pub fn values_are_scalar(&self) -> bool {
        self.values.len() == 1
    }

    /// Whether the values buffer holds one slot per element.
    ///
    /// An array of one element is both flat and [`scalar`](Self::values_are_scalar).
    #[inline]
    pub fn values_are_flat(&self) -> bool {
        self.values.len() == self.length
    }

    /// Whether the validity mask holds a single value shared by every element.
    #[inline]
    pub fn validity_is_scalar(&self) -> bool {
        self.validity().is_some_and(|v| v.is_scalar())
    }

    /// Whether every backing buffer has one slot per element.
    ///
    /// An array of one element is both flat and [`scalar`](Self::is_scalar).
    #[inline]
    pub fn is_flat(&self) -> bool {
        self.values_are_flat() && self.validity().is_none_or(|validity| validity.is_flat())
    }

    /// Whether this array is entirely stored in the scalar representation, and therefore is a
    /// single logical value repeated [`Self::len`] times in `O(1)` memory.
    #[inline]
    pub fn is_scalar(&self) -> bool {
        self.values_are_scalar() && self.validity().is_none_or(|v| v.is_scalar())
    }

    /// The single element every element of this array equals, if every backing buffer holds one
    /// slot.
    ///
    /// The inner [`Option`] is that element, so an array of nothing but nulls yields
    /// `Some(None)`. Returns `None` for an empty array, and whenever a backing buffer is flat over
    /// more than one element — its elements need not be equal, even if the other buffer is scalar.
    ///
    /// This is what lets equality and formatting avoid walking a scalar array of unbounded length.
    #[inline]
    pub fn scalar_value(&self) -> Option<Option<T>> {
        let is_shared = self.values.len() == 1
            && self
                .validity
                .as_ref()
                .is_none_or(|validity| validity.len() == 1);

        // SAFETY: the array is not empty, so element 0 is in bounds.
        (is_shared && self.length > 0).then(|| unsafe { self.get_unchecked(0) })
    }

    /// Returns the value at `i`.
    ///
    /// The value of a null element is undetermined (it can be anything).
    ///
    /// # Panics
    /// Panics if `i >= self.len()`.
    #[inline]
    pub fn value(&self, i: usize) -> T {
        assert!(i < self.length, "index out of bounds");
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
        debug_assert!(i < self.length);
        unsafe {
            *self
                .values
                .get_unchecked(broadcast_index(i, self.values.len()))
        }
    }

    /// Returns whether the element at `i` is valid (non-null).
    ///
    /// # Panics
    /// Panics if `i >= self.len()`.
    #[inline]
    pub fn is_valid(&self, i: usize) -> bool {
        assert!(i < self.length, "index out of bounds");
        unsafe { self.is_valid_unchecked(i) }
    }

    /// Returns whether the element at `i` is valid (non-null).
    ///
    /// # Safety
    /// `i` must be smaller than `self.len()`.
    #[inline]
    pub unsafe fn is_valid_unchecked(&self, i: usize) -> bool {
        debug_assert!(i < self.length);
        // SAFETY: `i` is in bounds of the array, and therefore of its validity mask.
        self.validity()
            .is_none_or(|validity| unsafe { validity.get_unchecked(i) })
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
        assert!(i < self.length, "index out of bounds");
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

    /// The number of null elements.
    ///
    /// This is `O(1)` for a scalar validity mask and `O(len)` for a flat one, amortized over
    /// repeated calls on the same [`Bitmap`].
    pub fn null_count(&self) -> usize {
        self.validity().map_or(0, |validity| validity.unset_bits())
    }

    /// Whether this array has at least one null element.
    #[inline]
    pub fn has_nulls(&self) -> bool {
        self.null_count() > 0
    }

    /// Returns an iterator over the values, ignoring validity.
    ///
    /// The values of null elements are undetermined (they can be anything).
    #[inline]
    pub fn values_iter(&self) -> PlPrimitiveValuesIter<'_, T> {
        PlPrimitiveValuesIter::new(&self.values, self.length)
    }

    /// Returns an iterator over the optional elements.
    #[inline]
    pub fn iter(&self) -> PlPrimitiveIter<'_, T> {
        PlPrimitiveIter::new(&self.values, self.validity(), self.length)
    }

    /// Returns an iterator over `length` values, repeating the single value of this array if
    /// that is all it holds, and ignoring validity.
    ///
    /// This is [`Self::broadcast_iter`] without the validity check, exactly as
    /// [`Self::values_iter`] is [`Self::iter`] without it. The values of null elements are
    /// undetermined (they can be anything).
    ///
    /// # Panics
    /// Panics if [`self.len()`](Self::len) is neither `length` nor one.
    #[inline]
    pub fn broadcast_values_iter(&self, length: usize) -> PlPrimitiveValuesIter<'_, T> {
        assert_broadcastable(self.length, length);
        // SAFETY: an array of one element holds a single slot, which is scalar for any length;
        // otherwise `length` is the length the values are already valid for.
        PlPrimitiveValuesIter::new(&self.values, length)
    }

    /// Returns an iterator over `length` optional elements, repeating the single element of this
    /// array if that is all it holds.
    ///
    /// This array either has `length` elements — in which case this is [`Self::iter`] — or a
    /// single element, which the `length` elements this yields then all read. Broadcasting is
    /// `O(1)`, and allocates nothing: the element is repeated as it is read, rather than
    /// materialized into an array to iterate the way [`Self::new_from_index`] would have to.
    ///
    /// # Panics
    /// Panics if [`self.len()`](Self::len) is neither `length` nor one.
    #[inline]
    pub fn broadcast_iter(&self, length: usize) -> PlPrimitiveIter<'_, T> {
        assert_broadcastable(self.length, length);
        // SAFETY: an array of one element holds a single slot in every buffer, which is scalar
        // for any length; otherwise `length` is the length they are already valid for.
        PlPrimitiveIter::new(
            &self.values,
            self.validity().map(|validity| validity.broadcast(length)),
            length,
        )
    }

    /// Replaces the validity mask.
    ///
    /// # Panics
    /// Panics if `validity` is neither flat nor scalar for this array's length.
    #[must_use]
    pub fn with_validity(mut self, validity: Option<Bitmap>) -> Self {
        self.set_validity(validity);
        self
    }

    /// Replaces the validity mask.
    ///
    /// # Panics
    /// Panics if `validity` is neither flat nor scalar for this array's length.
    pub fn set_validity(&mut self, validity: Option<Bitmap>) {
        if let Some(validity) = validity.as_ref() {
            assert!(
                is_valid_buffer_len(validity.len(), self.length),
                "validity mask of length {} is neither flat nor scalar for an array of length {}",
                validity.len(),
                self.length,
            );
        }
        self.validity = validity;
    }

    /// Drops the validity mask, making every element valid.
    #[must_use]
    pub fn without_validity(mut self) -> Self {
        self.validity = None;
        self
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

        // Scalar buffers are unaffected by slicing: every element reads the same slot.
        if self.values_are_flat() {
            unsafe {
                self.values
                    .slice_in_place_unchecked(offset..offset + length)
            };
        }
        if let Some(validity) = self.validity.as_mut() {
            if validity.len() == self.length {
                unsafe { validity.slice_unchecked(offset, length) };
            }
        }

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

    /// Creates a [`PlPrimitiveArray`] of `length` copies of the element at `index`.
    ///
    /// This function is `O(1)`: the result is scalar, so it holds a single slot no matter how long
    /// it is. A null element repeats as `length` nulls.
    ///
    /// # Panics
    /// Panics if `index >= self.len()`.
    #[inline]
    pub fn new_from_index(&self, index: usize, length: usize) -> Self {
        assert!(index < self.length, "index out of bounds");
        unsafe { self.new_from_index_unchecked(index, length) }
    }

    /// Creates a [`PlPrimitiveArray`] of `length` copies of the element at `index`.
    ///
    /// This function is `O(1)`.
    ///
    /// # Safety
    /// `index` must be smaller than `self.len()`.
    #[inline]
    pub unsafe fn new_from_index_unchecked(&self, index: usize, length: usize) -> Self {
        debug_assert!(index < self.length);

        if unsafe { self.is_null_unchecked(index) } {
            return Self::new_full_null(length);
        }

        // The value of a null element is undetermined, so it is repeated as it is found: it is the
        // mask that makes every element of the result null.
        let value = unsafe { self.value_unchecked(index) };

        Self {
            values: Buffer::from_owner([value]),
            length,
            validity: None,
        }
    }

    /// Returns an equivalent array whose backing buffers all hold one slot per element.
    ///
    /// This materializes any scalar buffer and is therefore `O(len)`; it is a no-op clone when
    /// this array [`is_flat`](Self::is_flat). The result carries its representation in its type:
    /// see [`Flat`] for what a flat array can do that this one cannot.
    pub fn to_flat(&self) -> Flat<Self> {
        if self.is_flat() {
            return Flat(self.clone());
        }

        let values = if self.values_are_flat() {
            self.values.clone()
        } else if self.length == 0 {
            Buffer::new()
        } else if self.scalar_value() == Some(None) {
            // Every element is null, and the value of a null element is undetermined, so the
            // repeated value need not be written out: a zeroed buffer stands in for it.
            Buffer::zeroed(self.length)
        } else {
            Buffer::from(vec![self.values[0]; self.length])
        };

        let validity = self.validity().map(|validity| validity.to_flat());

        Flat(Self {
            values,
            length: self.length,
            validity,
        })
    }

    /// Borrows this array as a [`Flat`] one, if every backing buffer already holds one slot per
    /// element.
    ///
    /// This is the `O(1)` counterpart of [`Self::to_flat`]: it materializes nothing, and returns
    /// `None` rather than expanding a scalar buffer when this array is not
    /// [`flat`](Self::is_flat).
    ///
    /// # Example
    /// ```
    /// use polars_array::PlPrimitiveArray;
    ///
    /// let arr = PlPrimitiveArray::from_vec(vec![1i32, 2, 3]);
    /// assert_eq!(arr.as_flat().unwrap().as_slice(), [1, 2, 3]);
    ///
    /// // A scalar array holds one slot for all three elements, so it has to be materialized.
    /// let scalar = PlPrimitiveArray::new_scalar(7i32, 3);
    /// assert!(scalar.as_flat().is_none());
    /// assert_eq!(scalar.to_flat().as_slice(), [7, 7, 7]);
    /// ```
    #[inline]
    pub fn as_flat(&self) -> Option<&Flat<Self>> {
        // SAFETY: every backing buffer of a flat array holds one slot per element.
        self.is_flat()
            .then(|| unsafe { Flat::from_ref_unchecked(self) })
    }
}

impl<T: NativeType> Default for PlPrimitiveArray<T> {
    #[inline]
    fn default() -> Self {
        Self::new_empty()
    }
}

impl<T: NativeType> From<Vec<T>> for PlPrimitiveArray<T> {
    #[inline]
    fn from(values: Vec<T>) -> Self {
        Self::from_vec(values)
    }
}

impl<T: NativeType> From<Buffer<T>> for PlPrimitiveArray<T> {
    #[inline]
    fn from(values: Buffer<T>) -> Self {
        Self::from_values(values)
    }
}

impl<T: NativeType> FromIterator<Option<T>> for PlPrimitiveArray<T> {
    fn from_iter<I: IntoIterator<Item = Option<T>>>(iter: I) -> Self {
        let iter = iter.into_iter();
        let (lower, _) = iter.size_hint();

        let mut values = Vec::with_capacity(lower);
        let mut validity = MutableBitmap::with_capacity(lower);

        for item in iter {
            values.push(item.unwrap_or_default());
            validity.push(item.is_some());
        }

        let length = values.len();
        let validity = Bitmap::from(validity);
        let validity = (validity.unset_bits() > 0).then_some(validity);

        Self {
            values: Buffer::from(values),
            length,
            validity,
        }
    }
}

impl<T: NativeType> FromIterator<T> for PlPrimitiveArray<T> {
    #[inline]
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        Self::from_vec(iter.into_iter().collect())
    }
}

impl<'a, T: NativeType> IntoIterator for &'a PlPrimitiveArray<T> {
    type Item = Option<T>;
    type IntoIter = PlPrimitiveIter<'a, T>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

/// Compares two arrays element-wise; the representation (flat or scalar) is irrelevant.
impl<T: NativeType> PartialEq for PlPrimitiveArray<T> {
    fn eq(&self, other: &Self) -> bool {
        if self.length != other.length {
            return false;
        }

        // Never walk two scalar arrays element by element: their length is unbounded by their
        // memory use.
        if let (Some(lhs), Some(rhs)) = (self.scalar_value(), other.scalar_value()) {
            return lhs == rhs;
        }

        self.iter().eq(other.iter())
    }
}

impl<T: NativeType> std::fmt::Debug for PlPrimitiveArray<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        /// Renders nulls as `null` instead of `None`.
        struct Element<T>(Option<T>);

        impl<T: std::fmt::Debug> std::fmt::Debug for Element<T> {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                match &self.0 {
                    Some(value) => value.fmt(f),
                    None => f.write_str("null"),
                }
            }
        }

        f.write_str("PlPrimitiveArray")?;

        // Never materialize a scalar array: its length is unbounded by its memory use.
        if self.length > 1 {
            if let Some(element) = self.scalar_value() {
                return write!(f, "[{:?}; {}]", Element(element), self.length);
            }
        }

        f.debug_list().entries(self.iter().map(Element)).finish()
    }
}

impl<T: NativeType> PlArray for PlPrimitiveArray<T> {
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
        PlArrayType::Primitive(T::PRIMITIVE)
    }

    #[inline]
    fn len(&self) -> usize {
        self.len()
    }

    #[inline]
    fn validity(&self) -> Option<PlBitmapRef<'_>> {
        self.validity()
    }

    #[inline]
    fn slice(&mut self, offset: usize, length: usize) {
        self.slice(offset, length)
    }

    #[inline]
    unsafe fn slice_unchecked(&mut self, offset: usize, length: usize) {
        unsafe { self.slice_unchecked(offset, length) }
    }

    #[inline]
    fn set_validity(&mut self, validity: Option<Bitmap>) {
        self.set_validity(validity)
    }

    #[inline]
    unsafe fn new_from_index_unchecked(&self, index: usize, length: usize) -> Box<dyn PlArray> {
        Box::new(unsafe { self.new_from_index_unchecked(index, length) })
    }

    #[inline]
    fn to_boxed(&self) -> Box<dyn PlArray> {
        Box::new(self.clone())
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
    fn flat() {
        let arr = PlPrimitiveArray::from_vec(vec![1i32, 2, 3]);

        assert_eq!(arr.len(), 3);
        assert!(arr.is_flat());
        assert!(!arr.is_scalar());
        assert_eq!(arr.null_count(), 0);
        assert_eq!(arr.value(1), 2);
        assert_eq!(arr.get(2), Some(3));
        assert_eq!(arr.iter().collect::<Vec<_>>(), [Some(1), Some(2), Some(3)]);
        assert_eq!(arr.values_iter().collect::<Vec<_>>(), [1, 2, 3]);
    }

    #[test]
    fn scalar_scalars_values() {
        let arr = PlPrimitiveArray::new_scalar(7i32, 4);

        assert_eq!(arr.len(), 4);
        assert!(arr.flat_values().is_none());
        assert_eq!(arr.scalar_values(), Some(7));
        assert!(arr.is_scalar());
        assert!(!arr.is_flat());
        assert_eq!(arr.scalar_value(), Some(Some(7)));
        assert_eq!(arr.null_count(), 0);

        for i in 0..arr.len() {
            assert_eq!(arr.get(i), Some(7));
        }
        assert_eq!(arr.iter().collect::<Vec<_>>(), [Some(7); 4]);
        assert_eq!(arr.values_iter().rev().collect::<Vec<_>>(), [7; 4]);
    }

    #[test]
    fn the_values_buffer_is_reached_through_its_representation() {
        let arr = PlPrimitiveArray::from_vec(vec![1i32, 2, 3]);

        assert_eq!(
            arr.flat_values().map(Buffer::as_slice),
            Some([1, 2, 3].as_slice())
        );
        assert_eq!(arr.scalar_values(), None);

        // A scalar array hands out no flat buffer; it is its single value that is reached instead.
        let arr = PlPrimitiveArray::new_scalar(7i32, 1_000_000_000);

        assert_eq!(arr.flat_values(), None);
        assert_eq!(arr.scalar_values(), Some(7));

        // The values are read whether or not the mask makes the elements null.
        let arr = PlPrimitiveArray::new_scalar(7i32, 3).with_validity(Some(Bitmap::new_zeroed(3)));

        assert_eq!(arr.scalar_values(), Some(7));
        assert_eq!(
            arr.scalar_value(),
            None,
            "a flat mask leaves no shared element"
        );

        // An empty array has no value to share, and is flat unless it is backed by a stray slot.
        assert_eq!(
            PlPrimitiveArray::<i32>::new_empty()
                .flat_values()
                .map(Buffer::len),
            Some(0),
        );
        assert_eq!(PlPrimitiveArray::new_scalar(7i32, 0).flat_values(), None);
        assert_eq!(PlPrimitiveArray::new_scalar(7i32, 0).scalar_values(), None);
    }

    #[test]
    fn scalar_value_accounts_for_validity() {
        // An array of nothing but nulls has a shared element, and that element is null.
        assert_eq!(
            PlPrimitiveArray::<i32>::new_full_null(3).scalar_value(),
            Some(None),
        );
        assert_eq!(
            PlPrimitiveArray::new_scalar(7i32, 3)
                .with_validity(Some(Bitmap::new_zeroed(1)))
                .scalar_value(),
            Some(None),
        );

        // A scalar mask of set bits leaves the shared value valid.
        assert_eq!(
            PlPrimitiveArray::new_scalar(7i32, 3)
                .with_validity(Some(Bitmap::new_with_value(true, 1)))
                .scalar_value(),
            Some(Some(7)),
        );

        // A flat mask over scalar values: the elements differ, so there is no shared element.
        assert_eq!(
            PlPrimitiveArray::new_scalar(7i32, 3)
                .with_validity(Some(Bitmap::from_iter([true, false, true])))
                .scalar_value(),
            None,
        );

        // The two representations coincide for a single element, which is therefore shared.
        assert_eq!(
            PlPrimitiveArray::from_vec(vec![1i32]).scalar_value(),
            Some(Some(1)),
        );
        assert_eq!(
            PlPrimitiveArray::from_vec(vec![1i32, 2]).scalar_value(),
            None
        );
    }

    #[test]
    fn null_scalar() {
        let arr = PlPrimitiveArray::<i32>::new_full_null(3);

        assert_eq!(arr.len(), 3);
        assert!(arr.is_scalar());
        assert_eq!(arr.null_count(), 3);
        assert!(arr.has_nulls());
        assert_eq!(arr.iter().collect::<Vec<_>>(), [None, None, None]);
    }

    #[test]
    fn flat_values_with_scalar_validity() {
        let arr =
            PlPrimitiveArray::from_vec(vec![1i32, 2, 3]).with_validity(Some(Bitmap::new_zeroed(1)));

        assert!(arr.validity_is_scalar());
        assert!(!arr.values_are_scalar());
        assert!(!arr.is_flat());
        assert!(!arr.is_scalar());
        assert_eq!(arr.null_count(), 3);
        assert_eq!(arr.iter().collect::<Vec<_>>(), [None, None, None]);
    }

    #[test]
    fn validity_hides_the_representation() {
        let scalar = PlPrimitiveArray::<i32>::new_full_null(1_000);
        let validity = scalar.validity().unwrap();

        // The mask covers every element even though it is backed by a single bit.
        assert_eq!(validity.len(), 1_000);
        assert!(validity.flat_bitmap().is_none());
        assert!(validity.is_scalar());
        assert_eq!(validity.scalar_value(), Some(false));
        assert!(!validity.get(999));
        assert_eq!(validity.unset_bits(), 1_000);
        assert_eq!(validity.set_bits(), 0);

        // Materializing it yields exactly the mask a flat array would carry.
        assert_eq!(validity.to_flat(), *scalar.to_flat().validity().unwrap());

        let flat: PlPrimitiveArray<i32> = [Some(1), None, Some(3)].into_iter().collect();
        let validity = flat.validity().unwrap();

        assert_eq!(validity.len(), 3);
        assert!(validity.is_flat());
        assert_eq!(validity.scalar_value(), None);
        assert!(!validity.get(1));
        assert_eq!(validity.unset_bits(), 1);
        assert_eq!(validity.to_flat(), *validity.flat_bitmap().unwrap());
    }

    #[test]
    fn validity_of_a_fully_valid_array() {
        assert!(
            PlPrimitiveArray::from_vec(vec![1i32, 2])
                .validity()
                .is_none()
        );
        assert!(
            PlPrimitiveArray::new_scalar(7i32, 1_000)
                .validity()
                .is_none()
        );
    }

    #[test]
    fn from_iter_with_nulls() {
        let arr: PlPrimitiveArray<i32> = [Some(1), None, Some(3)].into_iter().collect();

        assert_eq!(arr.len(), 3);
        assert_eq!(arr.null_count(), 1);
        assert!(arr.is_valid(0));
        assert!(arr.is_null(1));
        assert_eq!(arr.get(1), None);
        assert_eq!(arr.iter().collect::<Vec<_>>(), [Some(1), None, Some(3)]);
    }

    #[test]
    fn slicing_a_scalar_is_free() {
        let arr = PlPrimitiveArray::new_scalar(7i32, 1_000).sliced(500, 2);

        assert_eq!(arr.len(), 2);
        assert!(arr.flat_values().is_none());
        assert_eq!(arr.iter().collect::<Vec<_>>(), [Some(7), Some(7)]);
    }

    #[test]
    fn slicing_a_flat_array_slices_its_buffers() {
        let arr: PlPrimitiveArray<i32> = [Some(1), None, Some(3), Some(4)].into_iter().collect();
        let arr = arr.sliced(1, 2);

        assert_eq!(arr.len(), 2);
        assert_eq!(arr.flat_values().unwrap().len(), 2);
        assert_eq!(arr.validity().unwrap().len(), 2);
        assert_eq!(arr.iter().collect::<Vec<_>>(), [None, Some(3)]);
    }

    #[test]
    fn slicing_keeps_scalar_validity() {
        let arr = PlPrimitiveArray::from_vec(vec![1i32, 2, 3])
            .with_validity(Some(Bitmap::new_zeroed(1)))
            .sliced(1, 2);

        assert_eq!(arr.len(), 2);
        assert_eq!(arr.flat_values().unwrap().len(), 2);
        assert_eq!(arr.validity().unwrap().len(), 2);
        assert!(arr.validity().unwrap().is_scalar());
        assert_eq!(arr.iter().collect::<Vec<_>>(), [None, None]);
    }

    #[test]
    fn equality_ignores_representation() {
        let scalar = PlPrimitiveArray::new_scalar(7i32, 3);
        let flat = PlPrimitiveArray::from_vec(vec![7i32, 7, 7]);

        assert_eq!(scalar, flat);
        assert_ne!(scalar, PlPrimitiveArray::new_scalar(7i32, 4));
        assert_ne!(scalar, PlPrimitiveArray::from_vec(vec![7i32, 7, 8]));
        assert_ne!(scalar, PlPrimitiveArray::<i32>::new_full_null(3));
    }

    #[test]
    fn equality_of_scalars_does_not_walk_elements() {
        // Element-by-element comparison of a billion elements would not finish; the fast path must
        // hit.
        let arr = PlPrimitiveArray::new_scalar(7i32, 1_000_000_000);

        assert_eq!(arr, arr.clone());
        assert_ne!(arr, PlPrimitiveArray::new_scalar(8i32, 1_000_000_000));
        assert_ne!(arr, PlPrimitiveArray::<i32>::new_full_null(1_000_000_000));
        assert_eq!(
            PlPrimitiveArray::<i32>::new_full_null(1_000_000_000),
            PlPrimitiveArray::<i32>::new_full_null(1_000_000_000),
        );
    }

    #[test]
    fn empty() {
        let arr = PlPrimitiveArray::<i32>::new_empty();

        assert!(arr.is_empty());
        assert!(arr.is_flat());
        assert_eq!(arr.null_count(), 0);
        assert_eq!(arr.scalar_value(), None);
        assert_eq!(arr.iter().next(), None);
    }

    #[test]
    fn try_new_requires_flat_buffers() {
        let flat = || Buffer::from(vec![1i32, 2, 3]);

        assert!(PlPrimitiveArray::try_new(Buffer::from(vec![1i32, 2]), 3, None).is_err());
        assert!(PlPrimitiveArray::try_new(flat(), 3, None).is_ok());

        // A single value is a scalar array rather than a flat one, and is never inferred to be
        // either.
        assert!(PlPrimitiveArray::try_new(Buffer::from(vec![1i32]), 3, None).is_err());

        // The mask has to be flat as well.
        assert!(PlPrimitiveArray::try_new(flat(), 3, Some(Bitmap::new_zeroed(2))).is_err());
        assert!(PlPrimitiveArray::try_new(flat(), 3, Some(Bitmap::new_zeroed(1))).is_err());
        assert!(PlPrimitiveArray::try_new(flat(), 3, Some(Bitmap::new_zeroed(3))).is_ok());
    }

    #[test]
    fn try_new_broadcast_requires_scalar_buffers() {
        let scalar = || Buffer::from(vec![1i32]);

        assert!(PlPrimitiveArray::try_new_broadcast(scalar(), 1_000_000_000, None).is_ok());
        assert!(PlPrimitiveArray::try_new_broadcast(Buffer::from(vec![1i32, 2]), 2, None).is_err());

        // The mask has to be scalar as well.
        assert!(
            PlPrimitiveArray::try_new_broadcast(scalar(), 3, Some(Bitmap::new_zeroed(3))).is_err()
        );
        assert!(
            PlPrimitiveArray::try_new_broadcast(scalar(), 3, Some(Bitmap::new_zeroed(1))).is_ok()
        );

        // An array of no elements reads no slot at all, so both the empty and the single-slot
        // buffer stand for it.
        assert!(PlPrimitiveArray::<i32>::try_new_broadcast(Buffer::new(), 0, None).is_ok());
        assert!(PlPrimitiveArray::try_new_broadcast(scalar(), 0, None).is_ok());
    }

    #[test]
    fn iterators_are_exact_sized() {
        let arr = PlPrimitiveArray::new_scalar(7i32, 5);

        assert_eq!(arr.iter().len(), 5);
        assert_eq!(arr.values_iter().len(), 5);
        assert_eq!(arr.iter().size_hint(), (5, Some(5)));
    }

    #[test]
    fn debug_does_not_materialize_scalars() {
        let arr = PlPrimitiveArray::new_scalar(7i32, 1_000_000_000);
        assert_eq!(format!("{arr:?}"), "PlPrimitiveArray[7; 1000000000]");

        let arr: PlPrimitiveArray<i32> = [Some(1), None].into_iter().collect();
        assert_eq!(format!("{arr:?}"), "PlPrimitiveArray[1, null]");
    }

    #[test]
    fn new_from_index_repeats_one_element() {
        let arr = PlPrimitiveArray::from_iter([Some(1i32), None, Some(3)]);

        // The result is scalar, so a billion copies of an element cost a single slot.
        let repeated = arr.new_from_index(2, 1_000_000_000);
        assert_eq!(repeated.len(), 1_000_000_000);
        assert!(repeated.is_scalar());
        assert!(repeated.flat_values().is_none());
        assert_eq!(repeated.scalar_value(), Some(Some(3)));

        // A null element repeats as nulls, under a mask of a single bit.
        let repeated = arr.new_from_index(1, 4);
        assert_eq!(repeated.null_count(), 4);
        assert_eq!(repeated.scalar_value(), Some(None));
        assert!(repeated.validity().unwrap().is_scalar());

        // Repeating an element of a scalar array reads the slot every element shares.
        let scalar = PlPrimitiveArray::new_scalar(7i32, 1_000_000_000);
        assert_eq!(
            scalar.new_from_index(999_999_999, 3),
            PlPrimitiveArray::new_scalar(7i32, 3),
        );

        assert!(arr.new_from_index(0, 0).is_empty());
        assert_eq!(
            unsafe { arr.new_from_index_unchecked(0, 2) },
            PlPrimitiveArray::new_scalar(1i32, 2),
        );
    }

    #[test]
    fn broadcasting_one_element() {
        let arr = PlPrimitiveArray::from_iter([Some(1i32)]);

        // A billion copies of the element are iterated without ever being materialized.
        let mut iter = arr.broadcast_iter(1_000_000_000);
        assert_eq!(iter.len(), 1_000_000_000);
        assert_eq!(iter.next(), Some(Some(1)));
        assert_eq!(iter.nth(999_999_997), Some(Some(1)));
        assert_eq!(iter.next_back(), Some(Some(1)));
        assert_eq!(iter.next(), None);
        assert_eq!(
            arr.broadcast_values_iter(1_000_000_000).nth(999_999_999),
            Some(1)
        );

        // A null element broadcasts as nulls.
        let arr = PlPrimitiveArray::from_iter([None::<i32>]);
        assert_eq!(arr.broadcast_iter(3).collect::<Vec<_>>(), [None; 3]);

        // An array of the length asked for iterates as it is, whatever it is backed by.
        let arr = PlPrimitiveArray::from_iter([Some(1i32), None, Some(3)]);
        assert!(arr.broadcast_iter(3).eq(arr.iter()));
        assert!(arr.broadcast_values_iter(3).eq(arr.values_iter()));
        assert!(arr.broadcast_iter(3).rev().eq(arr.iter().rev()));

        let scalar = PlPrimitiveArray::new_scalar(7i32, 3);
        assert!(scalar.broadcast_iter(3).eq(scalar.iter()));

        // Broadcasting to nothing yields nothing, and an empty array broadcasts to nothing
        // else: it has no element to repeat.
        assert_eq!(
            PlPrimitiveArray::new_scalar(1i32, 1)
                .broadcast_iter(0)
                .len(),
            0,
        );
        assert_eq!(
            PlPrimitiveArray::<i32>::new_empty().broadcast_iter(0).len(),
            0,
        );
    }

    #[test]
    #[should_panic(expected = "an array of length 3 does not broadcast to length 4")]
    fn broadcasting_more_than_one_element_panics() {
        let _ = PlPrimitiveArray::from_vec(vec![1i32, 2, 3]).broadcast_iter(4);
    }

    #[test]
    #[should_panic(expected = "index out of bounds")]
    fn repeating_an_element_out_of_bounds_panics() {
        let _ = PlPrimitiveArray::from_vec(vec![1i32, 2, 3]).new_from_index(3, 1);
    }
}
