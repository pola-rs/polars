use std::borrow::Cow;
use std::ops::Range;

use arrow::bitmap::Bitmap;
use polars_error::{PolarsResult, polars_ensure};

use crate::array::PlArray;
use crate::array_type::PlArrayType;
use crate::bitmap::{PlBitmap, PlBitmapRef};
use crate::broadcast::{
    assert_broadcastable, is_flat_fixed_size_values_len, is_scalar_fixed_size_values_len,
    normalize_values, scalar_buffer_len, slice_fixed_size_values, slice_validity,
    try_validity_covering, validity_covering_unchecked,
};
use crate::builder::new_full_null_like;
use crate::concatenate::concatenate_repeated;
use crate::flat::Flat;

mod builder;
mod flat;
mod iterator;

pub use builder::PlFixedSizeListArrayBuilder;
pub use iterator::{PlFixedSizeListIter, PlFixedSizeListValuesIter};

/// An immutable, cheaply cloneable sequence of `length` optional lists of `width` values each.
#[derive(Clone)]
pub struct PlFixedSizeListArray {
    /// Scalar: values.len() == width
    values: Box<dyn PlArray>,
    width: usize,
    length: usize,
    /// Scalar: validity.len() == 1
    validity: Option<Bitmap>,
}

impl PlFixedSizeListArray {
    /// Creates a flat [`PlFixedSizeListArray`] out of its internal components.
    ///
    /// # Errors
    /// Errors if `values` is not `length * width` values, or `validity` not `length` bits.
    pub fn try_new(
        values: Box<dyn PlArray>,
        width: usize,
        length: usize,
        validity: Option<PlBitmap>,
    ) -> PolarsResult<Self> {
        let validity = try_validity_covering(validity, length)?;
        polars_ensure!(
            is_flat_fixed_size_values_len(values.len(), width, length),
            ComputeError:
            "values array of length {} is not flat for a fixed size list array of length {} and \
             width {}: it needs the width of every element laid end to end",
            values.len(), length, width,
        );

        Ok(Self {
            values,
            width,
            length,
            validity,
        })
    }

    /// Creates a flat [`PlFixedSizeListArray`] out of its internal components.
    #[inline]
    pub fn new(
        values: Box<dyn PlArray>,
        width: usize,
        length: usize,
        validity: Option<PlBitmap>,
    ) -> Self {
        Self::try_new(values, width, length, validity).unwrap()
    }

    /// Creates a flat [`PlFixedSizeListArray`] out of its components without validating them.
    ///
    /// # Safety
    /// `values` must hold `length * width` values and `validity` must cover `length` elements.
    #[inline]
    pub unsafe fn new_unchecked(
        values: Box<dyn PlArray>,
        width: usize,
        length: usize,
        validity: Option<PlBitmap>,
    ) -> Self {
        let validity = validity_covering_unchecked(validity, length);
        if cfg!(debug_assertions) {
            assert!(is_flat_fixed_size_values_len(values.len(), width, length));
        }

        Self {
            values,
            width,
            length,
            validity,
        }
    }

    /// Creates a scalar [`PlFixedSizeListArray`] of `length` elements out of its components.
    ///
    /// # Errors
    /// Errors if `values` or `validity` is not scalar for `width` and `length`.
    pub fn try_new_broadcast(
        values: Box<dyn PlArray>,
        width: usize,
        length: usize,
        validity: Option<PlBitmap>,
    ) -> PolarsResult<Self> {
        let validity = try_validity_covering(validity, length)?;
        polars_ensure!(
            is_scalar_fixed_size_values_len(values.len(), width, length),
            ComputeError:
            "values array of length {} is not the one element the {} elements of a broadcast \
             fixed size list array of width {} cover",
            values.len(), length, width,
        );

        Ok(Self {
            values: normalize_values(values, length),
            width,
            length,
            validity,
        })
    }

    /// Creates a scalar [`PlFixedSizeListArray`] of `length` elements out of its components.
    #[inline]
    pub fn new_broadcast(
        values: Box<dyn PlArray>,
        width: usize,
        length: usize,
        validity: Option<PlBitmap>,
    ) -> Self {
        Self::try_new_broadcast(values, width, length, validity).unwrap()
    }

    /// Creates a scalar [`PlFixedSizeListArray`] of `length` elements without validating them.
    ///
    /// # Safety
    /// `values` must be scalar for `width` and `length`, and `validity` scalar for `length`.
    #[inline]
    pub unsafe fn new_broadcast_unchecked(
        values: Box<dyn PlArray>,
        width: usize,
        length: usize,
        validity: Option<PlBitmap>,
    ) -> Self {
        let validity = validity_covering_unchecked(validity, length);
        if cfg!(debug_assertions) {
            assert!(is_scalar_fixed_size_values_len(values.len(), width, length));
        }

        Self {
            values: normalize_values(values, length),
            width,
            length,
            validity,
        }
    }

    /// Creates an empty [`PlFixedSizeListArray`] of lists `width` values wide.
    #[inline]
    pub fn new_empty(values: Box<dyn PlArray>, width: usize) -> Self {
        Self {
            values: values.sliced(0, 0),
            width,
            length: 0,
            validity: None,
        }
    }

    /// Creates a fully valid, flat array by cutting `values` into lists of `width` values.
    pub fn from_values(values: Box<dyn PlArray>, width: usize) -> Self {
        assert!(
            width > 0,
            "the length of a fixed size list array of width zero cannot be taken from its values",
        );
        assert!(
            values.len().is_multiple_of(width),
            "the values of length {} do not divide into lists of width {}",
            values.len(),
            width,
        );

        let length = values.len() / width;
        Self {
            values,
            width,
            length,
            validity: None,
        }
    }

    /// Creates a [`PlFixedSizeListArray`] of `length` copies of `element`, in its own memory.
    #[inline]
    pub fn new_scalar(element: Box<dyn PlArray>, length: usize) -> Self {
        let width = element.len();

        // There is no element for the values to be shared by when there are no elements at all,
        // which is why an empty array is the one that keeps nothing of the list it repeats.
        let values = if length == 0 {
            element.sliced(0, 0)
        } else {
            element
        };

        Self {
            values,
            width,
            length,
            validity: None,
        }
    }

    /// Creates a [`PlFixedSizeListArray`] of `length` nulls whose lists are as wide as `element`.
    #[inline]
    pub fn new_full_null(element: Box<dyn PlArray>, length: usize) -> Self {
        Self {
            validity: Some(Bitmap::new_zeroed(scalar_buffer_len(length))),
            ..Self::new_scalar(element, length)
        }
    }

    /// The number of values in every element.
    #[inline(always)]
    pub const fn width(&self) -> usize {
        self.width
    }

    /// The values array the lists are taken over.
    #[inline]
    pub fn values(&self) -> &dyn PlArray {
        &*self.values
    }

    /// The values array the lists are taken over, if it holds every element's values end to end.
    #[inline]
    pub fn flat_values(&self) -> Option<&dyn PlArray> {
        (!self.values_are_scalar()).then_some(&*self.values)
    }

    /// The list every element of this array reads, if the values hold a single element.
    #[inline]
    pub fn scalar_value_ignore_validity(&self) -> Option<&dyn PlArray> {
        self.values_are_scalar().then_some(&*self.values)
    }

    /// Consumes this array into its internal components.
    #[inline]
    pub fn into_inner(self) -> (Box<dyn PlArray>, usize, usize, Option<Bitmap>) {
        (self.values, self.width, self.length, self.validity)
    }

    /// The validity mask, if any element may be null.
    #[inline]
    pub fn validity(&self) -> Option<PlBitmapRef<'_>> {
        // SAFETY: the mask is flat or scalar for `self.length`, upheld by every constructor.
        self.validity
            .as_ref()
            .map(|validity| unsafe { PlBitmapRef::new_broadcast_unchecked(validity, self.length) })
    }

    /// Whether the values hold one list that every element of this array shares.
    #[inline]
    pub fn values_are_scalar(&self) -> bool {
        self.values.len() == self.width && self.length >= 1
    }

    /// Whether the values hold the values of every element, laid end to end.
    #[inline]
    pub fn values_are_flat(&self) -> bool {
        // A length times a width that overflows a `usize` is longer than any values array can be,
        // so such an array is never flat.
        self.length.checked_mul(self.width) == Some(self.values.len())
    }

    /// Whether the validity mask holds a single bit shared by every element.
    #[inline]
    pub fn validity_is_scalar(&self) -> bool {
        self.validity().is_some_and(|v| v.is_scalar())
    }

    /// Whether the values hold the values of every element and the mask one bit per element.
    #[inline]
    pub fn is_flat(&self) -> bool {
        self.values_are_flat() && self.validity().is_none_or(|validity| validity.is_flat())
    }

    /// Whether this array's own buffers stand for one list repeated [`Self::len`] times.
    #[inline]
    pub fn is_scalar(&self) -> bool {
        self.values_are_scalar() && self.validity().is_none_or(|v| v.is_scalar())
    }

    /// The single element every element equals, if this array's own buffers both hold one slot.
    #[inline]
    pub fn scalar_value(&self) -> Option<Option<Box<dyn PlArray>>> {
        let is_shared = self.values.len() == self.width
            && self
                .validity
                .as_ref()
                .is_none_or(|validity| validity.len() == 1);

        // SAFETY: the array is not empty, so element 0 is in bounds.
        (is_shared && self.length > 0).then(|| unsafe { self.get_unchecked(0) })
    }

    /// The [`Self::width`]-value range of the values array that element `i` covers.
    #[inline]
    pub fn value_range(&self, i: usize) -> Range<usize> {
        assert!(i < self.length, "index out of bounds");
        unsafe { self.value_range_unchecked(i) }
    }

    /// The [`Self::width`]-value range of the values array that element `i` covers.
    ///
    /// # Safety
    /// `i` must be smaller than `self.len()`.
    #[inline]
    pub unsafe fn value_range_unchecked(&self, i: usize) -> Range<usize> {
        debug_assert!(i < self.length);

        // Scalar values hold the one element every element covers, so they are read from the
        // start; flat ones lay the elements end to end, one width apart.
        let start = if self.values_are_scalar() {
            0
        } else {
            i * self.width
        };
        start..start + self.width
    }

    /// Returns the element at `i`: the values array sliced to the range the element covers.
    #[inline]
    pub fn value(&self, i: usize) -> Box<dyn PlArray> {
        assert!(i < self.length, "index out of bounds");
        unsafe { self.value_unchecked(i) }
    }

    /// Returns the element at `i`: the values array sliced to the range the element covers.
    ///
    /// # Safety
    /// `i` must be smaller than `self.len()`.
    #[inline]
    pub unsafe fn value_unchecked(&self, i: usize) -> Box<dyn PlArray> {
        let range = unsafe { self.value_range_unchecked(i) };
        // SAFETY: the values hold the width of every element, or the one they all cover.
        unsafe { self.values.sliced_unchecked(range.start, self.width) }
    }

    /// Returns an iterator over the elements, ignoring validity.
    #[inline]
    pub fn values_iter(&self) -> PlFixedSizeListValuesIter<'_> {
        // SAFETY: the values are flat or scalar for this array's length, upheld by every
        // constructor.
        PlFixedSizeListValuesIter::new(&*self.values, self.width, self.length)
    }

    /// Returns an iterator over the optional elements.
    #[inline]
    pub fn iter(&self) -> PlFixedSizeListIter<'_> {
        // SAFETY: the values are flat or scalar for this array's length, upheld by every
        // constructor.
        PlFixedSizeListIter::new(&*self.values, self.width, self.validity(), self.length)
    }

    /// Iterates `length` elements, repeating a scalar array's one value and ignoring validity.
    #[inline]
    pub fn broadcast_values_iter(&self, length: usize) -> PlFixedSizeListValuesIter<'_> {
        assert_broadcastable(self.length, length);
        // SAFETY: this array broadcasts to `length`, which is what was just asserted, so its
        // values are flat or scalar for it.
        PlFixedSizeListValuesIter::new(&*self.values, self.width, length)
    }

    /// Slices this array in place to `length` elements starting at `offset`.
    ///
    /// # Safety
    /// `offset + length` must not exceed `self.len()`.
    pub unsafe fn slice_unchecked(&mut self, offset: usize, length: usize) {
        debug_assert!(offset + length <= self.length);

        // There are no offsets to leave the values outside the slice behind, so they are sliced
        // along with it; see `slice_fixed_size_values`.
        unsafe {
            slice_fixed_size_values(&mut self.values, self.width, self.length, offset, length);
            slice_validity(&mut self.validity, self.length, offset, length);
        }

        self.length = length;
    }

    /// Creates a [`PlFixedSizeListArray`] of `length` copies of the element at `index`.
    ///
    /// # Safety
    /// `index` must be smaller than `self.len()`.
    pub unsafe fn new_from_index_unchecked(&self, index: usize, length: usize) -> Self {
        debug_assert!(index < self.length);

        // The values of a null element are undetermined, so they are repeated as they are found:
        // it is the mask that makes every element of the result null.
        let element = unsafe { self.value_unchecked(index) };

        if unsafe { self.is_null_unchecked(index) } {
            Self::new_full_null(element, length)
        } else {
            Self::new_scalar(element, length)
        }
    }

    /// Returns an equivalent flat array, borrowing this one if it is already flat.
    pub fn to_flat(&self) -> Cow<'_, Flat<Self>> {
        if let Some(flat) = self.as_flat() {
            return Cow::Borrowed(flat);
        }

        let validity = self
            .validity()
            .map(|validity| PlBitmap::from_bitmap(validity.to_flat().into_owned()));

        let values = if self.values_are_flat() {
            self.values.clone()
        } else if self.length > 0 && self.null_count() == self.length {
            // Every element is null, so every list is undetermined: repeating one value of the
            // element they share is a values array of the right length like any other, and it is
            // `O(1)` for every values array but a struct one.
            let flat_len = self.flat_values_len();
            self.values.new_from_index(0, flat_len)
        } else {
            // The one list every element covers, written out once per element. Concatenating it
            // with copies of itself is what repeats it, and that keeps the values of the result
            // scalar when the list is itself a single repeated value.
            concatenate_repeated(&*self.values, self.length)
                .expect("copies of one array always concatenate")
        };

        // SAFETY: the values are the element every element covers, repeated once per element, and
        // the mask is the flat counterpart of one valid for this array's length, which leaves every
        Cow::Owned(unsafe {
            Flat::new(Self::new_unchecked(
                values,
                self.width,
                self.length,
                validity,
            ))
        })
    }

    /// Borrows this array as a [`Flat`] one, if it is already flat.
    #[inline]
    pub fn as_flat(&self) -> Option<&Flat<Self>> {
        // SAFETY: the values of a flat array hold the width of every element, and its mask one bit
        // per element.
        self.is_flat().then(|| unsafe { Flat::new_ref(self) })
    }

    /// The number of values a flat counterpart of this array holds.
    #[inline]
    fn flat_values_len(&self) -> usize {
        self.length.checked_mul(self.width).expect(
            "the values of the flat counterpart of the fixed size list array overflow a `usize`",
        )
    }
}

crate::impl_array_methods!(PlFixedSizeListArray, Box<dyn PlArray>);

crate::impl_into_iterator!(PlFixedSizeListArray, PlFixedSizeListIter<'a>);

crate::impl_array_eq!(
    PlFixedSizeListArray,
    shape: |lhs, rhs| lhs.width == rhs.width,
    |lhs, rhs| {
        (0..lhs.len()).all(|i| unsafe {
            lhs.is_null_unchecked(i) || lhs.value_unchecked(i).eq_dyn(&*rhs.value_unchecked(i))
        })
    },
);

impl std::fmt::Debug for PlFixedSizeListArray {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // The values array formats its own scalar representation, so this never materializes one:
        // it is listed as it is backed, which is one element's worth for a scalar array.
        let mut s = f.debug_struct("PlFixedSizeListArray");
        s.field("length", &self.length);
        s.field("width", &self.width);
        if let Some(validity) = self.validity() {
            s.field("validity", &validity);
        }
        s.field("values", &self.values).finish()
    }
}

crate::impl_pl_array! {
    PlFixedSizeListArray,
    PlArrayType::FixedSizeList,
    fn new_full_null_like_self(&self, length: usize) -> Box<dyn PlArray> {
        // An element of a null list is as wide as any other, so the one element the values stand
        // for is as many nulls as this array is wide.
        Box::new(Self::new_full_null(
            new_full_null_like(&*self.values, self.width),
            length,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PlPrimitiveArray;

    /// The values array every test builds its lists over.
    fn values() -> Box<dyn PlArray> {
        Box::new(PlPrimitiveArray::from_vec(vec![1i32, 2, 3, 4, 5, 6]))
    }

    /// The three lists `[1, 2]`, `[3, 4]` and `[5, 6]` over [`values`].
    fn arr() -> PlFixedSizeListArray {
        PlFixedSizeListArray::from_values(values(), 2)
    }

    /// The elements of a list, which every test takes over a `PlPrimitiveArray<i32>`.
    fn elements(list: &dyn PlArray) -> Vec<Option<i32>> {
        list.as_any()
            .downcast_ref::<PlPrimitiveArray<i32>>()
            .unwrap()
            .iter()
            .collect()
    }

    #[test]
    fn flat() {
        let arr = arr();

        assert_eq!(arr.len(), 3);
        assert_eq!(arr.width(), 2);
        assert!(!arr.is_empty());
        assert_eq!(arr.null_count(), 0);
        assert!(!arr.has_nulls());
        assert!(arr.is_valid(1));
        assert!(!arr.is_null(1));
        assert!(arr.is_flat());
        assert!(!arr.is_scalar());
        assert!(arr.values_are_flat());
        assert!(!arr.values_are_scalar());
        assert_eq!(arr.flat_values().unwrap().len(), 6);

        assert_eq!(arr.value_range(0), 0..2);
        assert_eq!(arr.value_range(1), 2..4);
        assert_eq!(arr.value_range(2), 4..6);

        assert_eq!(elements(&*arr.value(0)), [Some(1), Some(2)]);
        assert_eq!(elements(&*arr.value(2)), [Some(5), Some(6)]);
        assert_eq!(elements(&*arr.get(1).unwrap()), [Some(3), Some(4)]);
    }

    #[test]
    fn null_scalar() {
        // Every element is null, over the two values they all share: the mask is a single bit, and
        // the values are one list.
        let arr = PlFixedSizeListArray::new_full_null(
            Box::new(PlPrimitiveArray::<i32>::new_full_null(2)),
            1_000_000,
        );

        assert!(arr.is_scalar());
        assert!(arr.validity_is_scalar());
        assert!(arr.values_are_scalar());
        assert!(!arr.values_are_flat());
        assert_eq!(arr.width(), 2);
        assert_eq!(arr.validity().unwrap().len(), 1_000_000);
        assert_eq!(arr.scalar_value_ignore_validity().unwrap().len(), 2);
        assert_eq!(arr.null_count(), 1_000_000);
        assert!(arr.has_nulls());
        assert!(arr.is_null(999_999));
        assert_eq!(arr.get(999_999), None);
        assert_eq!(arr.value_range(999_999), 0..2);

        let valid = arr.without_validity();
        assert_eq!(valid.null_count(), 0);
        assert!(valid.validity().is_none());
    }

    #[test]
    fn to_flat_writes_the_lists_out() {
        let scalar = PlFixedSizeListArray::new_scalar(
            Box::new(PlPrimitiveArray::from_vec(vec![1i32, 2])),
            3,
        );
        let flat = scalar.to_flat();

        assert!(flat.is_flat());
        assert_eq!(flat.values().len(), 6);
        assert_eq!(
            elements(flat.values()),
            [Some(1), Some(2), Some(1), Some(2), Some(1), Some(2)],
        );
        assert_eq!(*flat, scalar);
    }
}
