use std::borrow::Cow;
use std::ops::Range;

use arrow::bitmap::Bitmap;
use polars_buffer::Buffer;
use polars_error::{PolarsResult, polars_ensure};

use crate::array::PlArray;
use crate::array_type::PlArrayType;
use crate::bitmap::{PlBitmap, PlBitmapRef};
use crate::broadcast::{
    assert_broadcastable, broadcast_index, is_flat_offsets_len, is_scalar_offsets_len,
    normalize_offsets, scalar_buffer_len, scalar_offsets_len, slice_offsets, slice_validity,
    try_validity_covering, validity_covering_unchecked,
};
use crate::concatenate::concatenate_repeated;
use crate::flat::Flat;

mod builder;
mod flat;
mod iterator;

pub use builder::PlListArrayBuilder;
pub use iterator::{PlListIter, PlListValuesIter};

use crate::nested::Offsets;

/// An immutable, cheaply cloneable sequence of `length` optional lists over one values array.
#[derive(Clone)]
pub struct PlListArray {
    values: Box<dyn PlArray>,
    /// Scalar: offsets.len() == 2
    offsets: Buffer<u64>,
    length: usize,
    /// Scalar: validity.len() == 1
    validity: Option<Bitmap>,
}

impl PlListArray {
    /// Creates a flat [`PlListArray`] out of its internal components.
    ///
    /// # Errors
    /// Errors unless `offsets` holds `length + 1` non-decreasing offsets within `values`.
    pub fn try_new(
        values: Box<dyn PlArray>,
        offsets: Buffer<u64>,
        length: usize,
        validity: Option<PlBitmap>,
    ) -> PolarsResult<Self> {
        let validity = try_validity_covering(validity, length)?;
        polars_ensure!(
            is_flat_offsets_len(offsets.len(), length),
            ComputeError:
            "offsets buffer of length {} is not flat for a list array of length {}: it needs one \
             offset per element plus the end of the last",
            offsets.len(), length,
        );

        validate_offsets(&*values, &offsets)?;

        Ok(Self {
            values,
            offsets,
            length,
            validity,
        })
    }

    /// Creates a flat [`PlListArray`] out of its internal components.
    #[inline]
    pub fn new(
        values: Box<dyn PlArray>,
        offsets: Buffer<u64>,
        length: usize,
        validity: Option<PlBitmap>,
    ) -> Self {
        Self::try_new(values, offsets, length, validity).unwrap()
    }

    /// Creates a flat [`PlListArray`] out of its internal components without validating them.
    ///
    /// # Safety
    /// `offsets` and `validity` must both be flat and valid for `length` elements.
    #[inline]
    pub unsafe fn new_unchecked(
        values: Box<dyn PlArray>,
        offsets: Buffer<u64>,
        length: usize,
        validity: Option<PlBitmap>,
    ) -> Self {
        let validity = validity_covering_unchecked(validity, length);
        if cfg!(debug_assertions) {
            assert!(is_flat_offsets_len(offsets.len(), length));
            assert!(offsets.windows(2).all(|window| window[0] <= window[1]));
            assert!(offsets[offsets.len() - 1] <= values.len() as u64);
        }

        Self {
            values,
            offsets,
            length,
            validity,
        }
    }

    /// Creates a scalar [`PlListArray`] of `length` elements out of its internal components.
    ///
    /// # Errors
    /// Errors unless `offsets` is scalar for `length`, non-decreasing and within `values`.
    pub fn try_new_broadcast(
        values: Box<dyn PlArray>,
        offsets: Buffer<u64>,
        length: usize,
        validity: Option<PlBitmap>,
    ) -> PolarsResult<Self> {
        let validity = try_validity_covering(validity, length)?;
        polars_ensure!(
            is_scalar_offsets_len(offsets.len(), length),
            ComputeError:
            "offsets buffer of length {} is not the single range the {} elements of a broadcast \
             list array share: it needs the two offsets standing for that range",
            offsets.len(), length,
        );

        validate_offsets(&*values, &offsets)?;

        Ok(Self {
            values,
            offsets: normalize_offsets(offsets, length),
            length,
            validity,
        })
    }

    /// Creates a scalar [`PlListArray`] of `length` elements out of its internal components.
    #[inline]
    pub fn new_broadcast(
        values: Box<dyn PlArray>,
        offsets: Buffer<u64>,
        length: usize,
        validity: Option<PlBitmap>,
    ) -> Self {
        Self::try_new_broadcast(values, offsets, length, validity).unwrap()
    }

    /// Creates a scalar [`PlListArray`] of `length` elements without validating them.
    ///
    /// # Safety
    /// `offsets` and `validity` must both be scalar and valid for `length` elements.
    #[inline]
    pub unsafe fn new_broadcast_unchecked(
        values: Box<dyn PlArray>,
        offsets: Buffer<u64>,
        length: usize,
        validity: Option<PlBitmap>,
    ) -> Self {
        let validity = validity_covering_unchecked(validity, length);
        if cfg!(debug_assertions) {
            assert!(is_scalar_offsets_len(offsets.len(), length));
            assert!(offsets.windows(2).all(|window| window[0] <= window[1]));
            assert!(offsets[offsets.len() - 1] <= values.len() as u64);
        }

        Self {
            values,
            offsets: normalize_offsets(offsets, length),
            length,
            validity,
        }
    }

    /// Creates an empty [`PlListArray`] over `values`.
    #[inline]
    pub fn new_empty(values: Box<dyn PlArray>) -> Self {
        Self {
            values,
            offsets: Buffer::zeroed(1),
            length: 0,
            validity: None,
        }
    }

    /// Creates a fully valid, flat [`PlListArray`] from `values` and `offsets`.
    pub fn from_offsets(values: Box<dyn PlArray>, offsets: Buffer<u64>) -> Self {
        let length = offsets
            .len()
            .checked_sub(1)
            .expect("a list array needs at least one offset");
        Self::new(values, offsets, length, None)
    }

    /// Creates a [`PlListArray`] of `length` copies of `element`, in its own memory.
    #[inline]
    pub fn new_scalar(element: Box<dyn PlArray>, length: usize) -> Self {
        // There is no element for the list to be shared by when there are no elements at all,
        // which is why an empty array is the one that covers no range of the values it repeats.
        if length == 0 {
            return Self::new_empty(element);
        }

        Self {
            offsets: Buffer::from_owner([0, element.len() as u64]),
            values: element,
            length,
            validity: None,
        }
    }

    /// Creates a [`PlListArray`] of `length` nulls over `values`.
    #[inline]
    pub fn new_full_null(values: Box<dyn PlArray>, length: usize) -> Self {
        Self {
            values,
            offsets: Buffer::zeroed(scalar_offsets_len(length)),
            length,
            validity: Some(Bitmap::new_zeroed(scalar_buffer_len(length))),
        }
    }

    /// The values array the lists are taken over.
    #[inline]
    pub fn values(&self) -> &dyn PlArray {
        &*self.values
    }

    /// The backing offsets buffer, if it holds the range of every element, laid end to end.
    #[inline]
    pub fn flat_offsets(&self) -> Option<&Buffer<u64>> {
        (!self.offsets_are_scalar()).then_some(&self.offsets)
    }

    /// The range of [`Self::values`] every element covers, if the offsets hold one range.
    #[inline]
    pub fn scalar_offsets(&self) -> Option<Range<usize>> {
        // SAFETY: a scalar offsets buffer holds two slots, so both are in bounds.
        self.offsets_are_scalar().then(|| unsafe {
            // Every offset of an array that upholds its invariants fits in a `usize`.
            *self.offsets.get_unchecked(0) as usize..*self.offsets.get_unchecked(1) as usize
        })
    }

    /// Consumes this array into its internal components.
    #[inline]
    pub fn into_inner(self) -> (Box<dyn PlArray>, Buffer<u64>, usize, Option<Bitmap>) {
        (self.values, self.offsets, self.length, self.validity)
    }

    /// The validity mask, if any element may be null.
    #[inline]
    pub fn validity(&self) -> Option<PlBitmapRef<'_>> {
        // SAFETY: the mask is flat or scalar for `self.length`, upheld by every constructor.
        self.validity
            .as_ref()
            .map(|validity| unsafe { PlBitmapRef::new_broadcast_unchecked(validity, self.length) })
    }

    /// Whether the offsets hold one range that every element of this array shares.
    #[inline]
    pub fn offsets_are_scalar(&self) -> bool {
        // The offsets hold one slot more than the starts that are flat or scalar for this array's
        // length, so the two of a scalar array are a single start and the end of it. An array of
        // no elements holds the one offset it starts at and no range at all, and is flat.
        self.offsets.len() == 2 && self.length > 0
    }

    /// Whether the offsets hold the range of every element, laid end to end.
    #[inline]
    pub fn offsets_are_flat(&self) -> bool {
        // The offsets are never empty, and hold the start of every element plus the end of the
        // last. This is spelled as the predicate the iterators resolve their own representation
        // with, rather than as the subtraction it comes down to for an array that upholds its
        // invariants: a caller that asserts this ahead of a walk is then asserting the very
        // condition the walk branches on, which folds the branch — and the tag it reads, and the
        // step it computes — out of the loop.
        is_flat_offsets_len(self.offsets.len(), self.length)
    }

    /// Whether the validity mask holds a single bit shared by every element.
    #[inline]
    pub fn validity_is_scalar(&self) -> bool {
        self.validity().is_some_and(|v| v.is_scalar())
    }

    /// Whether both of this array's own backing buffers hold one slot per element.
    #[inline]
    pub fn is_flat(&self) -> bool {
        self.offsets_are_flat() && self.validity().is_none_or(|validity| validity.is_flat())
    }

    /// Whether this array's own buffers stand for one list repeated [`Self::len`] times.
    #[inline]
    pub fn is_scalar(&self) -> bool {
        self.offsets_are_scalar() && self.validity().is_none_or(|v| v.is_scalar())
    }

    /// The single element every element equals, if this array's own buffers both hold one slot.
    #[inline]
    pub fn scalar_value(&self) -> Option<Option<Box<dyn PlArray>>> {
        let is_shared = self.offsets.len() == 2
            && self
                .validity
                .as_ref()
                .is_none_or(|validity| validity.len() == 1);

        // SAFETY: the array is not empty, so element 0 is in bounds.
        (is_shared && self.length > 0).then(|| unsafe { self.get_unchecked(0) })
    }

    /// The range of [`Self::values`] the element at `i` covers.
    #[inline]
    pub fn value_range(&self, i: usize) -> Range<usize> {
        assert!(i < self.length, "index out of bounds");
        unsafe { self.value_range_unchecked(i) }
    }

    /// The range of [`Self::values`] the element at `i` covers.
    ///
    /// # Safety
    /// `i` must be smaller than `self.len()`.
    #[inline]
    pub unsafe fn value_range_unchecked(&self, i: usize) -> Range<usize> {
        debug_assert!(i < self.length);

        // Scalar offsets hold the one range every element covers, so they are read at slot zero.
        let i = broadcast_index(i, self.offsets.len() - 1);

        // SAFETY: the offsets hold one slot more than the starts `broadcast_index` maps onto, so
        // `i + 1` is in bounds, and every offset fits in a `usize`.
        unsafe {
            let start = *self.offsets.get_unchecked(i) as usize;
            let end = *self.offsets.get_unchecked(i + 1) as usize;
            start..end
        }
    }

    /// The number of values in the element at `i`.
    #[inline]
    pub fn value_length(&self, i: usize) -> usize {
        self.value_range(i).len()
    }

    /// The number of values in the element at `i`.
    ///
    /// # Safety
    /// `i` must be smaller than `self.len()`.
    #[inline]
    pub unsafe fn value_length_unchecked(&self, i: usize) -> usize {
        unsafe { self.value_range_unchecked(i) }.len()
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
        // SAFETY: the offsets are ordered and bounded by the length of the values array.
        unsafe { self.values.sliced_unchecked(range.start, range.len()) }
    }

    /// Returns an iterator over the elements, ignoring validity.
    #[inline]
    pub fn values_iter(&self) -> PlListValuesIter<'_> {
        // SAFETY: the offsets are flat or scalar for this array's length, are ordered and are
        // bounded by the length of the values, all upheld by every constructor.
        PlListValuesIter::new(
            &*self.values,
            Offsets::new(&self.offsets, self.length),
            self.length,
        )
    }

    /// Returns an iterator over the optional elements.
    #[inline]
    pub fn iter(&self) -> PlListIter<'_> {
        // SAFETY: the offsets are flat or scalar for this array's length, are ordered and are
        // bounded by the length of the values, all upheld by every constructor.
        PlListIter::new(
            &*self.values,
            Offsets::new(&self.offsets, self.length),
            self.validity(),
            self.length,
        )
    }

    /// Iterates `length` elements, repeating a scalar array's one value and ignoring validity.
    #[inline]
    pub fn broadcast_values_iter(&self, length: usize) -> PlListValuesIter<'_> {
        assert_broadcastable(self.length, length);
        // SAFETY: this array broadcasts to `length`, which is what was just asserted, so its
        // offsets are flat or scalar for it.
        PlListValuesIter::new(&*self.values, Offsets::new(&self.offsets, length), length)
    }

    /// Slices this array in place to `length` elements starting at `offset`.
    ///
    /// # Safety
    /// `offset + length` must not exceed `self.len()`.
    pub unsafe fn slice_unchecked(&mut self, offset: usize, length: usize) {
        debug_assert!(offset + length <= self.length);

        // The values array the offsets point into is left as it is; see `slice_offsets`.
        unsafe {
            slice_offsets(&mut self.offsets, self.length, offset, length);
            slice_validity(&mut self.validity, self.length, offset, length);
        }

        self.length = length;
    }

    /// Creates a [`PlListArray`] of `length` copies of the element at `index`.
    ///
    /// # Safety
    /// `index` must be smaller than `self.len()`.
    pub unsafe fn new_from_index_unchecked(&self, index: usize, length: usize) -> Self {
        debug_assert!(index < self.length);

        if unsafe { self.is_null_unchecked(index) } {
            return Self::new_full_null(self.values.clone(), length);
        }

        if length == 0 {
            return Self::new_empty(self.values.clone());
        }

        // Nothing is repeated: the values array is cloned as it is, and the two offsets every
        // element of the result shares are the ones of the element being repeated.
        let range = unsafe { self.value_range_unchecked(index) };

        Self {
            values: self.values.clone(),
            offsets: Buffer::from_owner([range.start as u64, range.end as u64]),
            length,
            validity: None,
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

        let (values, offsets) = if self.offsets_are_flat() {
            (self.values.clone(), self.offsets.clone())
        } else if self.length == 0
            || self.offsets[0] == self.offsets[1]
            || self.null_count() == self.length
        {
            // Every element is the empty list, or is null and therefore holds an undetermined one:
            // no value is written out, and the offsets all point at the same place. That place is
            // the start of the values array rather than the range every element covered, which is
            // the same empty list and is a buffer that need not be written out either.
            (self.values.clone(), Buffer::zeroed(self.length + 1))
        } else {
            // The one list every element covers, written out once per element. Concatenating it
            // with copies of itself is what repeats it, and that keeps the values of the result
            // scalar when the list is itself a single repeated value.
            let range = unsafe { self.value_range_unchecked(0) };
            let element = self.values.sliced(range.start, range.len());
            let values = concatenate_repeated(&*element, self.length)
                .expect("copies of one array always concatenate");

            let offsets = (0..=self.length as u64)
                .map(|i| i * range.len() as u64)
                .collect::<Vec<_>>();

            (values, Buffer::from(offsets))
        };

        // SAFETY: the offsets are ordered, one per element plus the end of the last, and within the
        // values; the mask is the flat counterpart of one valid for this array's length. That
        Cow::Owned(unsafe {
            Flat::new(Self::new_unchecked(values, offsets, self.length, validity))
        })
    }

    /// Borrows this array as a [`Flat`] one, if it is already flat.
    #[inline]
    pub fn as_flat(&self) -> Option<&Flat<Self>> {
        // SAFETY: both own backing buffers of a flat array hold one slot per element.
        self.is_flat().then(|| unsafe { Flat::new_ref(self) })
    }
}

crate::impl_array_methods!(PlListArray, Box<dyn PlArray>);

crate::impl_into_iterator!(PlListArray, PlListIter<'a>);

crate::impl_array_eq!(PlListArray, |lhs, rhs| {
    (0..lhs.len()).all(|i| unsafe {
        lhs.is_null_unchecked(i) || lhs.value_unchecked(i).eq_dyn(&*rhs.value_unchecked(i))
    })
},);

impl std::fmt::Debug for PlListArray {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // The values array formats its own scalar representation, so this never materializes one,
        // and neither are the offsets: they are listed as they are backed, which is two of them
        // for a scalar array.
        let mut s = f.debug_struct("PlListArray");
        s.field("length", &self.length);
        if let Some(validity) = self.validity() {
            s.field("validity", &validity);
        }
        s.field("offsets", &self.offsets.as_slice());
        s.field("values", &self.values).finish()
    }
}

crate::impl_pl_array! {
    PlListArray,
    PlArrayType::List,
    fn new_full_null_like_self(&self, length: usize) -> Box<dyn PlArray> {
        // Every element is an empty list, so the values are only there to carry their shape.
        Box::new(Self::new_full_null(self.values.sliced(0, 0), length))
    }
}

/// Checks that `offsets` are monotonically non-decreasing and stay within `values`.
fn validate_offsets(values: &dyn PlArray, offsets: &Buffer<u64>) -> PolarsResult<()> {
    // The offsets are ordered, so checking the last one against the values array covers them all —
    // including that every one of them fits in a `usize`.
    for (i, window) in offsets.windows(2).enumerate() {
        polars_ensure!(
            window[0] <= window[1],
            ComputeError:
            "offset {} of the list array is {}, which is smaller than the offset {} before it",
            i + 1, window[1], window[0],
        );
    }

    let last = offsets[offsets.len() - 1];
    polars_ensure!(
        last <= values.len() as u64,
        ComputeError:
        "the last offset of the list array is {}, which exceeds the length {} of its values",
        last, values.len(),
    );

    Ok(())
}
