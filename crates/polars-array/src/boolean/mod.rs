use std::borrow::Cow;

use arrow::bitmap::{Bitmap, BitmapBuilder};
use polars_error::{PolarsResult, polars_ensure};

use crate::array_type::PlArrayType;
use crate::bitmap::{PlBitmap, PlBitmapIter, PlBitmapRef};
use crate::broadcast::{
    is_flat_buffer_len, is_scalar_buffer_len, normalize_bitmap, scalar_buffer_len, slice_bitmap,
    slice_validity, try_validity_covering, validity_covering_unchecked,
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
    /// # Errors
    /// Errors unless `values` holds `length` bits and `validity` covers `length` elements.
    pub fn try_new(
        values: Bitmap,
        length: usize,
        validity: Option<PlBitmap>,
    ) -> PolarsResult<Self> {
        let validity = try_validity_covering(validity, length)?;
        polars_ensure!(
            is_flat_buffer_len(values.len(), length),
            ComputeError:
            "values bitmap of length {} is not flat for an array of length {}",
            values.len(), length,
        );

        Ok(Self {
            values,
            length,
            validity,
        })
    }

    /// Creates a flat [`PlBooleanArray`] out of its internal components.
    #[inline]
    pub fn new(values: Bitmap, length: usize, validity: Option<PlBitmap>) -> Self {
        Self::try_new(values, length, validity).unwrap()
    }

    /// Creates a flat [`PlBooleanArray`] out of its internal components without validating them.
    ///
    /// # Safety
    /// `values` and `validity` must both be flat and valid for `length` elements.
    #[inline]
    pub unsafe fn new_unchecked(values: Bitmap, length: usize, validity: Option<PlBitmap>) -> Self {
        let validity = validity_covering_unchecked(validity, length);
        if cfg!(debug_assertions) {
            assert!(is_flat_buffer_len(values.len(), length));
        }

        Self {
            values,
            length,
            validity,
        }
    }

    /// Creates a scalar [`PlBooleanArray`] of `length` elements out of its internal components.
    ///
    /// # Errors
    /// Errors unless `values` is scalar for `length` and `validity` covers `length` elements.
    pub fn try_new_broadcast(
        values: Bitmap,
        length: usize,
        validity: Option<PlBitmap>,
    ) -> PolarsResult<Self> {
        let validity = try_validity_covering(validity, length)?;
        polars_ensure!(
            is_scalar_buffer_len(values.len(), length),
            ComputeError:
            "values bitmap of length {} is not the single bit the {} elements of a broadcast \
             array share",
            values.len(), length,
        );

        Ok(Self {
            values: normalize_bitmap(values, length),
            length,
            validity,
        })
    }

    /// Creates a scalar [`PlBooleanArray`] of `length` elements out of its internal components.
    #[inline]
    pub fn new_broadcast(values: Bitmap, length: usize, validity: Option<PlBitmap>) -> Self {
        Self::try_new_broadcast(values, length, validity).unwrap()
    }

    /// Creates a scalar [`PlBooleanArray`] of `length` elements without validating them.
    ///
    /// # Safety
    /// `values` and `validity` must both be scalar and valid for `length` elements.
    #[inline]
    pub unsafe fn new_broadcast_unchecked(
        values: Bitmap,
        length: usize,
        validity: Option<PlBitmap>,
    ) -> Self {
        let validity = validity_covering_unchecked(validity, length);
        if cfg!(debug_assertions) {
            assert!(is_scalar_buffer_len(values.len(), length));
        }

        Self {
            values: normalize_bitmap(values, length),
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

    /// Creates a fully valid [`PlBooleanArray`] whose values are the bits of `values`.
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
    #[inline]
    pub fn values(&self) -> PlBitmapRef<'_> {
        // SAFETY: the bitmap is flat or scalar for `self.length`, upheld by every constructor.
        unsafe { PlBitmapRef::new_broadcast_unchecked(&self.values, self.length) }
    }

    /// The elements this array picks out: the ones it holds a set, non-null bit at.
    pub fn true_and_valid(&self) -> PlBitmap {
        let values = self.values();
        let Some(validity) = self.validity() else {
            return values.into();
        };

        match (values.scalar_value(), validity.scalar_value()) {
            (Some(false), _) | (_, Some(false)) => PlBitmap::new_scalar(false, self.length),
            (Some(true), Some(true)) => PlBitmap::new_scalar(true, self.length),
            (Some(true), None) => validity.into(),
            (None, Some(true)) => values.into(),
            (None, None) => PlBitmap::from_bitmap(
                values.flat_bitmap().unwrap() & validity.flat_bitmap().unwrap(),
            ),
        }
    }

    /// The backing values bitmap, if it holds one bit per element.
    #[inline]
    pub fn flat_values(&self) -> Option<&Bitmap> {
        (!self.values_are_scalar()).then_some(&self.values)
    }

    /// The backing values bitmap, if it holds one bit per element.
    #[inline]
    pub fn flat_values_mut(&mut self) -> Option<&mut Bitmap> {
        if self.values_are_scalar() {
            // A single bit stands for every element; there is nothing laid out per element to
            // write into.
            None
        } else {
            Some(&mut self.values)
        }
    }

    /// The value every element of this array reads, if the values bitmap holds a single bit.
    #[inline]
    pub fn scalar_value_ignore_validity(&self) -> Option<bool> {
        // SAFETY: a scalar bitmap holds a single bit, so bit 0 is in bounds.
        self.values_are_scalar()
            .then(|| unsafe { self.values.get_bit_unchecked(0) })
    }

    /// The validity mask, if any element may be null.
    #[inline]
    pub fn validity(&self) -> Option<PlBitmapRef<'_>> {
        // SAFETY: the mask is flat or scalar for `self.length`, upheld by every constructor.
        self.validity
            .as_ref()
            .map(|validity| unsafe { PlBitmapRef::new_broadcast_unchecked(validity, self.length) })
    }

    /// Whether the values bitmap holds a single bit shared by every element.
    #[inline]
    pub fn values_are_scalar(&self) -> bool {
        self.values.len() == 1 && self.length > 0
    }

    /// Whether the values bitmap holds one bit per element.
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
    #[inline]
    pub fn is_flat(&self) -> bool {
        self.values_are_flat() && self.validity().is_none_or(|validity| validity.is_flat())
    }

    /// Whether this array is scalar throughout: one value repeated [`Self::len`] times.
    #[inline]
    pub fn is_scalar(&self) -> bool {
        self.values_are_scalar() && self.validity().is_none_or(|v| v.is_scalar())
    }

    /// The single element every element equals, if every backing bitmap holds one bit.
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
    #[inline]
    pub fn value(&self, i: usize) -> bool {
        self.values().get(i)
    }

    /// Returns the value at `i`.
    ///
    /// # Safety
    /// `i` must be smaller than `self.len()`.
    #[inline]
    pub unsafe fn value_unchecked(&self, i: usize) -> bool {
        unsafe { self.values().get_unchecked(i) }
    }

    /// Returns an iterator over the values, ignoring validity.
    #[inline]
    pub fn values_iter(&self) -> PlBitmapIter<'_> {
        self.values().iter()
    }

    /// Returns an iterator over the optional elements.
    #[inline]
    pub fn iter(&self) -> PlBooleanIter<'_> {
        PlBooleanIter::new(self.values(), self.validity(), self.length)
    }

    /// Iterates `length` values, repeating a scalar array's one value and ignoring validity.
    #[inline]
    pub fn broadcast_values_iter(&self, length: usize) -> PlBitmapIter<'_> {
        self.values().broadcast(length).iter()
    }

    /// Slices this array in place to `length` elements starting at `offset`.
    ///
    /// # Safety
    /// `offset + length` must not exceed `self.len()`.
    pub unsafe fn slice_unchecked(&mut self, offset: usize, length: usize) {
        debug_assert!(offset + length <= self.length);

        unsafe {
            slice_bitmap(&mut self.values, self.length, offset, length);
            slice_validity(&mut self.validity, self.length, offset, length);
        }

        self.length = length;
    }

    /// Creates a [`PlBooleanArray`] of `length` copies of the element at `index`.
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

    /// Returns an equivalent flat array, borrowing this one if it is already flat.
    pub fn to_flat(&self) -> Cow<'_, Flat<Self>> {
        if let Some(flat) = self.as_flat() {
            return Cow::Borrowed(flat);
        }

        let values = if self.values_are_scalar() && self.scalar_value() == Some(None) {
            // Every element is null, and the value of a null element is undetermined, so the
            // repeated bit need not be written out: a zeroed bitmap stands in for it.
            Bitmap::new_zeroed(self.length)
        } else {
            self.values().to_flat().into_owned()
        };

        // SAFETY: both bitmaps hold one bit per element: the values were written out above, and
        // the mask is the flat counterpart of this array's own.
        Cow::Owned(unsafe {
            Flat::new(Self {
                values,
                length: self.length,
                validity: self
                    .validity()
                    .map(|validity| validity.to_flat().into_owned()),
            })
        })
    }

    /// Borrows this array as a [`Flat`] one, if it is already flat.
    #[inline]
    pub fn as_flat(&self) -> Option<&Flat<Self>> {
        // SAFETY: every backing bitmap of a flat array holds one bit per element.
        self.is_flat().then(|| unsafe { Flat::new_ref(self) })
    }
}

crate::impl_array_methods!(PlBooleanArray, bool);

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

        // `BitmapBuilder`, not `MutableBitmap`: it accumulates a word at a time and counts its
        // set bits as it goes, where `MutableBitmap::push` checks its capacity per bit and
        // leaves the count to a scan of the whole mask afterwards.
        let mut values = BitmapBuilder::with_capacity(lower);
        let mut validity = BitmapBuilder::with_capacity(lower);

        for item in iter {
            values.push(item.unwrap_or_default());
            validity.push(item.is_some());
        }

        let length = values.len();

        Self {
            values: values.freeze(),
            length,
            validity: validity.into_opt_validity(),
        }
    }
}

impl FromIterator<bool> for PlBooleanArray {
    #[inline]
    fn from_iter<I: IntoIterator<Item = bool>>(iter: I) -> Self {
        // Not `Bitmap::from_iter`, which goes through `MutableBitmap` and packs a byte at a time
        // behind a capacity check and an exhausted flag.
        <Self as crate::collect::ArrayFromIter<bool>>::arr_from_iter(iter)
    }
}

crate::impl_into_iterator!(PlBooleanArray, PlBooleanIter<'a>);

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

crate::impl_element_debug!(PlBooleanArray, "PlBooleanArray");

crate::impl_pl_array!(PlBooleanArray, PlArrayType::Boolean);

#[cfg(test)]
mod tests {
    use super::*;

    /// Picking out the set, non-null elements cannot depend on the representation.
    #[test]
    fn true_and_valid_names_what_the_written_out_array_does() {
        const LENGTH: usize = 5;
        let flat_values = || PlBitmap::from_bitmap([true, false, true, true, false].into());
        let flat_validity = || PlBitmap::from_bitmap([true, true, false, true, true].into());

        for values in [
            PlBitmap::new_scalar(false, LENGTH),
            PlBitmap::new_scalar(true, LENGTH),
            flat_values(),
        ] {
            for validity in [
                None,
                Some(PlBitmap::new_scalar(false, LENGTH)),
                Some(PlBitmap::new_scalar(true, LENGTH)),
                Some(flat_validity()),
            ] {
                let arr =
                    PlBooleanArray::from_pl_bitmap(values.clone()).with_validity(validity.clone());
                let written_out = arr.to_flat().into_owned().into_array();

                // An element is picked out exactly when it reads back as a non-null `true`,
                // which is what the array's own iterator says independently of either bitmap.
                let expected: Vec<bool> = arr.iter().map(|v| v == Some(true)).collect();

                let picked = arr.true_and_valid();
                assert_eq!(picked.len(), LENGTH, "{arr:?}");
                assert_eq!(picked.iter().collect::<Vec<_>>(), expected, "{arr:?}");
                assert_eq!(
                    written_out.true_and_valid().iter().collect::<Vec<_>>(),
                    expected,
                    "{arr:?}",
                );
            }
        }
    }

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
}
