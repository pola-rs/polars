use arrow::bitmap::{Bitmap, MutableBitmap};
use polars_error::{PolarsResult, polars_ensure};

use crate::array::PlArray;
use crate::array_type::PlArrayType;
use crate::bitmap::{PlBitmap, PlBitmapIter, PlBitmapRef};
use crate::broadcast::{
    is_flat_buffer_len, is_scalar_buffer_len, is_valid_buffer_len, scalar_buffer_len,
};
use crate::flat::Flat;

mod builder;
mod flat;
mod iterator;

pub use builder::PlBooleanArrayBuilder;
pub use iterator::PlBooleanIter;

/// An immutable, cheaply cloneable sequence of `length` optional [`bool`] values.
#[derive(Clone)]
pub struct PlBooleanArray {
    /// Scalar: values.len() == 1
    values: Bitmap,
    length: usize,
    /// Scalar: validity.len() == 1
    validity: Option<Bitmap>,
}

impl PlBooleanArray {
    /// Creates a flat [`PlBooleanArray`] out of its internal components.
    ///
    /// Every backing bitmap has to hold one bit per element. [`Self::try_new_broadcast`] is what
    /// builds the scalar representation; this function never infers it from a bitmap that happens
    /// to hold a single bit. This function is `O(1)`.
    ///
    /// # Errors
    /// This function errors if `values` or `validity` does not hold exactly `length` bits.
    pub fn try_new(values: Bitmap, length: usize, validity: Option<Bitmap>) -> PolarsResult<Self> {
        polars_ensure!(
            is_flat_buffer_len(values.len(), length),
            ComputeError:
            "values bitmap of length {} is not flat for an array of length {}",
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

    /// Creates a flat [`PlBooleanArray`] out of its internal components.
    ///
    /// # Panics
    /// Panics under the conditions [`Self::try_new`] errors.
    #[inline]
    pub fn new(values: Bitmap, length: usize, validity: Option<Bitmap>) -> Self {
        Self::try_new(values, length, validity).unwrap()
    }

    /// Creates a flat [`PlBooleanArray`] out of its internal components without validating them.
    ///
    /// # Safety
    /// `values` and `validity` must each hold exactly `length` bits.
    #[inline]
    pub unsafe fn new_unchecked(values: Bitmap, length: usize, validity: Option<Bitmap>) -> Self {
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

    /// Creates a scalar [`PlBooleanArray`] of `length` elements out of its internal components.
    ///
    /// Every backing bitmap has to hold the single bit every element shares, which makes this
    /// `O(1)` in `length` as well as in time. [`Self::try_new`] is what builds the flat
    /// representation.
    ///
    /// # Errors
    /// This function errors if `values` or `validity` does not hold exactly one bit. An array of
    /// no elements reads no bit at all, so it additionally admits an empty bitmap.
    pub fn try_new_broadcast(
        values: Bitmap,
        length: usize,
        validity: Option<Bitmap>,
    ) -> PolarsResult<Self> {
        polars_ensure!(
            is_scalar_buffer_len(values.len(), length),
            ComputeError:
            "values bitmap of length {} is not the single bit the {} elements of a broadcast \
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

    /// Creates a scalar [`PlBooleanArray`] of `length` elements out of its internal components.
    ///
    /// # Panics
    /// Panics under the conditions [`Self::try_new_broadcast`] errors.
    #[inline]
    pub fn new_broadcast(values: Bitmap, length: usize, validity: Option<Bitmap>) -> Self {
        Self::try_new_broadcast(values, length, validity).unwrap()
    }

    /// Creates a scalar [`PlBooleanArray`] of `length` elements out of its internal components
    /// without validating them.
    ///
    /// # Safety
    /// `values` and `validity` must each hold exactly one bit, or none at all if `length` is zero.
    #[inline]
    pub unsafe fn new_broadcast_unchecked(
        values: Bitmap,
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

    /// Creates an empty [`PlBooleanArray`].
    #[inline]
    pub fn new_empty() -> Self {
        Self {
            values: Bitmap::new(),
            length: 0,
            validity: None,
        }
    }

    /// Creates a flat, fully valid [`PlBooleanArray`] from `values`.
    #[inline]
    pub fn from_values(values: Bitmap) -> Self {
        let length = values.len();
        Self {
            values,
            length,
            validity: None,
        }
    }

    /// Creates a flat, fully valid [`PlBooleanArray`] from a [`Vec`].
    #[inline]
    pub fn from_vec(values: Vec<bool>) -> Self {
        Self::from_values(Bitmap::from(values))
    }

    /// Creates a flat, fully valid [`PlBooleanArray`] by packing `values`.
    #[inline]
    pub fn from_slice(values: &[bool]) -> Self {
        Self::from_values(Bitmap::from(values))
    }

    /// Creates a fully valid [`PlBooleanArray`] whose values are the bits of `values`, in whatever
    /// representation that mask is in.
    ///
    /// A [`PlBitmap`] already knows whether it is flat or scalar for the elements it covers, so
    /// nothing is inferred here: a scalar mask becomes a scalar array of the same length, and a
    /// flat one a flat array. This is what a kernel that computes its result as a mask — a
    /// validity mask inverted, say — hands over without writing a scalar result out. This
    /// function is `O(1)`.
    #[inline]
    pub fn from_pl_bitmap(values: PlBitmap) -> Self {
        let (values, length) = values.into_inner();
        Self {
            values,
            length,
            validity: None,
        }
    }

    /// Creates a [`PlBooleanArray`] of `length` copies of `value`, in `O(1)` memory.
    #[inline]
    pub fn new_scalar(value: bool, length: usize) -> Self {
        Self {
            values: Bitmap::new_with_value(value, scalar_buffer_len(length)),
            length,
            validity: None,
        }
    }

    /// Creates a [`PlBooleanArray`] of `length` nulls, in `O(1)` memory.
    #[inline]
    pub fn new_full_null(length: usize) -> Self {
        Self {
            values: Bitmap::new_zeroed(scalar_buffer_len(length)),
            length,
            validity: Some(Bitmap::new_zeroed(scalar_buffer_len(length))),
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

    /// The values, ignoring validity.
    ///
    /// The returned [`PlBitmapRef`] has [`Self::len`] bits regardless of whether the backing bitmap
    /// is flat or scalar, so reading values through it needs no knowledge of which
    /// representation this array is in. Reach for the backing [`Bitmap`] — which is *not*
    /// guaranteed to have [`Self::len`] bits — with [`Self::flat_values`], or materialize a flat
    /// one with [`PlBitmapRef::to_flat`].
    #[inline]
    pub fn values(&self) -> PlBitmapRef<'_> {
        // SAFETY: the bitmap is flat or scalar for `self.length`, upheld by every constructor.
        unsafe { PlBitmapRef::new_broadcast_unchecked(&self.values, self.length) }
    }

    /// The backing values bitmap, if it holds one bit per element.
    ///
    /// Bit `i` is then the value of element `i`, with no
    /// [`broadcast_index`](crate::broadcast::broadcast_index) in the way. This is the `O(1)`
    /// counterpart of [`Self::to_flat`]: it materializes nothing, and returns `None` rather than
    /// expanding a scalar bitmap. Reach for the bit a scalar bitmap shares with
    /// [`Self::scalar_values`] instead — between them the two cover every array that has elements
    /// at all, so a `None` from both is an empty array. The values of null elements are
    /// undetermined (they can be anything).
    #[inline]
    pub fn flat_values(&self) -> Option<&Bitmap> {
        self.values_are_flat().then_some(&self.values)
    }

    /// The value every element of this array reads, if the values bitmap holds a single bit.
    ///
    /// This is the values half of [`Self::scalar_value`], which additionally asks that the
    /// validity mask be scalar and reports the null the mask makes of this value. Returns `None`
    /// for a values bitmap that is flat over more than one element, and for an empty array, which
    /// has no element to share a value. The value of a null element is undetermined (it can be
    /// anything).
    #[inline]
    pub fn scalar_values(&self) -> Option<bool> {
        self.values().scalar_value()
    }

    /// The validity mask, if any element may be null.
    ///
    /// The returned [`PlBitmapRef`] has [`Self::len`] bits regardless of whether the backing bitmap
    /// is flat or scalar, exactly like [`Self::values`].
    #[inline]
    pub fn validity(&self) -> Option<PlBitmapRef<'_>> {
        // SAFETY: the mask is flat or scalar for `self.length`, upheld by every constructor.
        self.validity
            .as_ref()
            .map(|validity| unsafe { PlBitmapRef::new_broadcast_unchecked(validity, self.length) })
    }

    /// Whether the values bitmap holds a single bit shared by every element.
    ///
    /// An array of one element is both scalar and [`flat`](Self::is_flat): the two representations
    /// coincide, and this reports them both.
    #[inline]
    pub fn values_are_scalar(&self) -> bool {
        self.values.len() == 1
    }

    /// Whether the values bitmap holds one bit per element.
    ///
    /// An array of one element is both flat and [`scalar`](Self::values_are_scalar).
    #[inline]
    pub fn values_are_flat(&self) -> bool {
        self.values.len() == self.length
    }

    /// Whether the validity mask holds a single bit shared by every element.
    #[inline]
    pub fn validity_is_scalar(&self) -> bool {
        self.validity().is_some_and(|v| v.is_scalar())
    }

    /// Whether every backing bitmap has one bit per element.
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

    /// The single element every element of this array equals, if every backing bitmap holds one
    /// bit.
    ///
    /// The inner [`Option`] is that element, so an array of nothing but nulls yields
    /// `Some(None)`. Returns `None` for an empty array, and whenever a backing bitmap is flat over
    /// more than one element — its elements need not be equal, even if the other bitmap is scalar.
    ///
    /// This is what lets equality and formatting avoid walking a scalar array of unbounded length.
    #[inline]
    pub fn scalar_value(&self) -> Option<Option<bool>> {
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
    pub fn value(&self, i: usize) -> bool {
        self.values().get(i)
    }

    /// Returns the value at `i`.
    ///
    /// The value of a null element is undetermined (it can be anything).
    ///
    /// # Safety
    /// `i` must be smaller than `self.len()`.
    #[inline]
    pub unsafe fn value_unchecked(&self, i: usize) -> bool {
        unsafe { self.values().get_unchecked(i) }
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
    pub fn get(&self, i: usize) -> Option<bool> {
        assert!(i < self.length, "index out of bounds");
        unsafe { self.get_unchecked(i) }
    }

    /// Returns the element at `i`, or `None` if it is null.
    ///
    /// # Safety
    /// `i` must be smaller than `self.len()`.
    #[inline]
    pub unsafe fn get_unchecked(&self, i: usize) -> Option<bool> {
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
    pub fn values_iter(&self) -> PlBitmapIter<'_> {
        self.values().iter()
    }

    /// Returns an iterator over the optional elements.
    #[inline]
    pub fn iter(&self) -> PlBooleanIter<'_> {
        PlBooleanIter::new(self.values(), self.validity(), self.length)
    }

    /// Returns an iterator over `length` values, repeating the single value of this array if
    /// that is all it holds, and ignoring validity.
    ///
    /// This array either has `length` elements — in which case this is [`Self::values_iter`] — or
    /// a single element, which the `length` values this yields are then all read from.
    /// Broadcasting is `O(1)`, and allocates nothing: the value is repeated as it is read, rather
    /// than materialized into an array to iterate the way [`Self::new_from_index`] would have to.
    /// The values of null elements are undetermined (they can be anything).
    ///
    /// # Panics
    /// Panics if [`self.len()`](Self::len) is neither `length` nor one.
    #[inline]
    pub fn broadcast_values_iter(&self, length: usize) -> PlBitmapIter<'_> {
        self.values().broadcast(length).iter()
    }

    /// Returns this array with its validity mask replaced by a flat one.
    ///
    /// # Panics
    /// Panics if `validity` does not hold one bit per element.
    /// [`Self::with_validity_broadcast`] is what installs the single bit every element shares;
    /// this function never infers that from a mask that happens to hold one bit.
    #[must_use]
    pub fn with_validity(mut self, validity: Option<Bitmap>) -> Self {
        self.set_validity(validity);
        self
    }

    /// Replaces the validity mask with a flat one.
    ///
    /// # Panics
    /// Panics if `validity` does not hold one bit per element.
    /// [`Self::set_validity_broadcast`] is what installs the single bit every element shares;
    /// this function never infers that from a mask that happens to hold one bit.
    pub fn set_validity(&mut self, validity: Option<Bitmap>) {
        if let Some(validity) = validity.as_ref() {
            assert!(
                is_flat_buffer_len(validity.len(), self.length),
                "validity mask of length {} is not flat for an array of length {}",
                validity.len(),
                self.length,
            );
        }
        self.validity = validity;
    }

    /// Returns this array with its validity mask replaced by one that broadcasts over it.
    ///
    /// # Panics
    /// Panics if `validity` is neither flat nor scalar for this array's length.
    #[must_use]
    pub fn with_validity_broadcast(mut self, validity: Option<Bitmap>) -> Self {
        self.set_validity_broadcast(validity);
        self
    }

    /// Replaces the validity mask with one that broadcasts over this array.
    ///
    /// This is [`Self::set_validity`] widened to the scalar representation: the mask is either
    /// flat — one bit per element — or the single bit every element shares. See
    /// [`crate::broadcast`].
    ///
    /// # Panics
    /// Panics if `validity` is neither flat nor scalar for this array's length.
    pub fn set_validity_broadcast(&mut self, validity: Option<Bitmap>) {
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

        // Scalar bitmaps are unaffected by slicing — every element reads the same bit — with the
        // one exception of an empty slice, which keeps no element to read it.
        if self.values_are_flat() {
            unsafe { self.values.slice_unchecked(offset, length) };
        } else if length == 0 {
            unsafe { self.values.slice_unchecked(0, 0) };
        }
        if let Some(validity) = self.validity.as_mut() {
            if validity.len() == self.length {
                unsafe { validity.slice_unchecked(offset, length) };
            } else if length == 0 {
                unsafe { validity.slice_unchecked(0, 0) };
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

    /// Creates a [`PlBooleanArray`] of `length` copies of the element at `index`.
    ///
    /// This function is `O(1)`: the result is scalar, so it holds a single bit no matter how long
    /// it is. A null element repeats as `length` nulls.
    ///
    /// # Panics
    /// Panics if `index >= self.len()`.
    #[inline]
    pub fn new_from_index(&self, index: usize, length: usize) -> Self {
        assert!(index < self.length, "index out of bounds");
        unsafe { self.new_from_index_unchecked(index, length) }
    }

    /// Creates a [`PlBooleanArray`] of `length` copies of the element at `index`.
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

        Self::new_scalar(value, length)
    }

    /// Returns an equivalent array whose backing bitmaps all hold one bit per element.
    ///
    /// This materializes any scalar bitmap and is therefore `O(len)`; it is a no-op clone when
    /// this array [`is_flat`](Self::is_flat). The result carries its representation in its type:
    /// see [`Flat`] for what a flat array can do that this one cannot.
    pub fn to_flat(&self) -> Flat<Self> {
        if self.is_flat() {
            return Flat(self.clone());
        }

        let values = if self.values_are_scalar() && self.scalar_value() == Some(None) {
            // Every element is null, and the value of a null element is undetermined, so the
            // repeated bit need not be written out: a zeroed bitmap stands in for it.
            Bitmap::new_zeroed(self.length)
        } else {
            self.values().to_flat()
        };

        Flat(Self {
            values,
            length: self.length,
            validity: self.validity().map(|validity| validity.to_flat()),
        })
    }

    /// Borrows this array as a [`Flat`] one, if every backing bitmap already holds one bit per
    /// element.
    ///
    /// This is the `O(1)` counterpart of [`Self::to_flat`]: it materializes nothing, and returns
    /// `None` rather than expanding a scalar bitmap when this array is not
    /// [`flat`](Self::is_flat).
    ///
    /// # Example
    /// ```
    /// use polars_array::PlBooleanArray;
    ///
    /// let arr = PlBooleanArray::from_vec(vec![true, false]);
    /// assert_eq!(arr.as_flat().unwrap().values_iter().collect::<Vec<_>>(), [true, false]);
    ///
    /// // A scalar array holds one bit for both elements, so it has to be materialized.
    /// let scalar = PlBooleanArray::new_scalar(true, 2);
    /// assert!(scalar.as_flat().is_none());
    /// assert_eq!(scalar.to_flat().values().len(), 2);
    /// ```
    #[inline]
    pub fn as_flat(&self) -> Option<&Flat<Self>> {
        // SAFETY: every backing bitmap of a flat array holds one bit per element.
        self.is_flat().then(|| unsafe { Flat::new_ref(self) })
    }
}

impl Default for PlBooleanArray {
    #[inline]
    fn default() -> Self {
        Self::new_empty()
    }
}

impl From<Vec<bool>> for PlBooleanArray {
    #[inline]
    fn from(values: Vec<bool>) -> Self {
        Self::from_vec(values)
    }
}

impl From<Bitmap> for PlBooleanArray {
    #[inline]
    fn from(values: Bitmap) -> Self {
        Self::from_values(values)
    }
}

impl FromIterator<Option<bool>> for PlBooleanArray {
    fn from_iter<I: IntoIterator<Item = Option<bool>>>(iter: I) -> Self {
        let iter = iter.into_iter();
        let (lower, _) = iter.size_hint();

        let mut values = MutableBitmap::with_capacity(lower);
        let mut validity = MutableBitmap::with_capacity(lower);

        for item in iter {
            values.push(item.unwrap_or_default());
            validity.push(item.is_some());
        }

        let length = values.len();
        let validity = Bitmap::from(validity);
        let validity = (validity.unset_bits() > 0).then_some(validity);

        Self {
            values: values.into(),
            length,
            validity,
        }
    }
}

impl FromIterator<bool> for PlBooleanArray {
    #[inline]
    fn from_iter<I: IntoIterator<Item = bool>>(iter: I) -> Self {
        Self::from_values(Bitmap::from_iter(iter))
    }
}

impl<'a> IntoIterator for &'a PlBooleanArray {
    type Item = Option<bool>;
    type IntoIter = PlBooleanIter<'a>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

/// Compares two arrays element-wise; the representation (flat or scalar) is irrelevant.
impl PartialEq for PlBooleanArray {
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

impl Eq for PlBooleanArray {}

impl std::fmt::Debug for PlBooleanArray {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        /// Renders nulls as `null` instead of `None`.
        struct Element(Option<bool>);

        impl std::fmt::Debug for Element {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                match &self.0 {
                    Some(value) => value.fmt(f),
                    None => f.write_str("null"),
                }
            }
        }

        f.write_str("PlBooleanArray")?;

        // Never materialize a scalar array: its length is unbounded by its memory use.
        if self.length > 1 {
            if let Some(element) = self.scalar_value() {
                return write!(f, "[{:?}; {}]", Element(element), self.length);
            }
        }

        f.debug_list().entries(self.iter().map(Element)).finish()
    }
}

impl PlArray for PlBooleanArray {
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
        PlArrayType::Boolean
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
    fn set_validity_broadcast(&mut self, validity: Option<Bitmap>) {
        self.set_validity_broadcast(validity)
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
        let arr = PlBooleanArray::from_vec(vec![true, false, true]);

        assert_eq!(arr.len(), 3);
        assert!(arr.is_flat());
        assert!(!arr.is_scalar());
        assert_eq!(arr.null_count(), 0);
        assert!(!arr.value(1));
        assert_eq!(arr.get(2), Some(true));
        assert_eq!(
            arr.iter().collect::<Vec<_>>(),
            [Some(true), Some(false), Some(true)],
        );
        assert_eq!(arr.values_iter().collect::<Vec<_>>(), [true, false, true]);
    }

    #[test]
    fn scalar_scalars_values() {
        let arr = PlBooleanArray::new_scalar(true, 4);

        assert_eq!(arr.len(), 4);
        assert!(arr.values_are_scalar());
        assert_eq!(arr.values().len(), 4);
        assert!(arr.is_scalar());
        assert!(!arr.is_flat());
        assert_eq!(arr.scalar_value(), Some(Some(true)));
        assert_eq!(arr.null_count(), 0);

        for i in 0..arr.len() {
            assert_eq!(arr.get(i), Some(true));
        }
        assert_eq!(arr.iter().collect::<Vec<_>>(), [Some(true); 4]);
        assert_eq!(arr.values_iter().rev().collect::<Vec<_>>(), [true; 4]);
    }

    #[test]
    fn slicing_a_flat_array_slices_its_bitmaps() {
        let arr: PlBooleanArray = [Some(true), None, Some(false), Some(true)]
            .into_iter()
            .collect();
        let arr = arr.sliced(1, 2);

        assert_eq!(arr.len(), 2);
        assert_eq!(arr.flat_values().unwrap().len(), 2);
        assert_eq!(arr.validity().unwrap().flat_bitmap().unwrap().len(), 2);
        assert_eq!(arr.iter().collect::<Vec<_>>(), [None, Some(false)]);
    }
}
