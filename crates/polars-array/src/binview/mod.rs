use std::borrow::Cow;

use buffers::{copy_only_value, copy_value, own_only_value};
use polars_arrow::array::View;
use polars_arrow::bitmap::{Bitmap, BitmapBuilder};
use polars_buffer::Buffer;
use polars_error::{PolarsResult, polars_bail, polars_ensure, polars_err};
use polars_utils::relaxed_cell::RelaxedCell;

use crate::array_type::PlArrayType;
use crate::bitmap::{PlBitmap, PlBitmapRef};
use crate::broadcast::{
    assert_broadcastable, broadcast_index, is_flat_buffer_len, is_scalar_buffer_len,
    normalize_buffer, scalar_buffer_len, slice_buffer, slice_validity, try_validity_covering,
    validity_covering_unchecked,
};
use crate::flat::Flat;

mod buffers;
mod builder;
mod flat;
mod iterator;

pub use builder::PlBinaryViewArrayBuilder;
pub use iterator::{PlBinaryViewIter, PlBinaryViewValuesIter};

/// The sentinel [`PlBinaryViewArray::total_bytes_len`] holds until someone asks for it.
const UNKNOWN_BYTES_LEN: u64 = u64::MAX;

/// An immutable, cheaply cloneable sequence of `length` optional byte slices.
#[derive(Clone)]
pub struct PlBinaryViewArray {
    views: Buffer<View>,
    buffers: Buffer<Buffer<u8>>,
    length: usize,
    validity: Option<Bitmap>,
    /// What [`Self::total_bytes_len`] last answered, or [`UNKNOWN_BYTES_LEN`] before it is asked
    /// and whenever the views or the mask move underneath it.
    total_bytes_len: RelaxedCell<u64>,
}

impl PlBinaryViewArray {
    /// Creates a flat [`PlBinaryViewArray`] out of its internal components.
    ///
    /// # Errors
    /// Errors unless `views` holds `length` slots and `validity` covers `length` elements.
    pub fn try_new(
        views: Buffer<View>,
        buffers: Buffer<Buffer<u8>>,
        length: usize,
        validity: Option<PlBitmap>,
    ) -> PolarsResult<Self> {
        let validity = try_validity_covering(validity, length)?;
        polars_ensure!(
            is_flat_buffer_len(views.len(), length),
            ComputeError:
            "views buffer of length {} is not flat for an array of length {}",
            views.len(), length,
        );

        validate_views(&views, &buffers)?;

        Ok(Self {
            views,
            buffers,
            length,
            validity,
            total_bytes_len: RelaxedCell::from(UNKNOWN_BYTES_LEN),
        })
    }

    /// Creates a flat [`PlBinaryViewArray`] out of its internal components.
    #[inline]
    pub fn new(
        views: Buffer<View>,
        buffers: Buffer<Buffer<u8>>,
        length: usize,
        validity: Option<PlBitmap>,
    ) -> Self {
        Self::try_new(views, buffers, length, validity).unwrap()
    }

    /// Creates a flat [`PlBinaryViewArray`] out of its internal components without validating them.
    ///
    /// # Safety
    /// `views` and `validity` must both be flat and valid for `length` elements.
    #[inline]
    pub unsafe fn new_unchecked(
        views: Buffer<View>,
        buffers: Buffer<Buffer<u8>>,
        length: usize,
        validity: Option<PlBitmap>,
    ) -> Self {
        let validity = validity_covering_unchecked(validity, length);
        if cfg!(debug_assertions) {
            assert!(is_flat_buffer_len(views.len(), length));
            validate_views(&views, &buffers).unwrap();
        }

        Self {
            views,
            buffers,
            length,
            validity,
            total_bytes_len: RelaxedCell::from(UNKNOWN_BYTES_LEN),
        }
    }

    /// Creates a scalar [`PlBinaryViewArray`] of `length` elements out of its internal components.
    ///
    /// # Errors
    /// Errors unless `views` is scalar for `length` and `validity` covers `length` elements.
    pub fn try_new_broadcast(
        views: Buffer<View>,
        buffers: Buffer<Buffer<u8>>,
        length: usize,
        validity: Option<PlBitmap>,
    ) -> PolarsResult<Self> {
        let validity = try_validity_covering(validity, length)?;
        polars_ensure!(
            is_scalar_buffer_len(views.len(), length),
            ComputeError:
            "views buffer of length {} is not the single view the {} elements of a broadcast \
             array share",
            views.len(), length,
        );

        validate_views(&views, &buffers)?;

        Ok(Self {
            views: normalize_buffer(views, length),
            buffers,
            length,
            validity,
            total_bytes_len: RelaxedCell::from(UNKNOWN_BYTES_LEN),
        })
    }

    /// Creates a scalar [`PlBinaryViewArray`] of `length` elements out of its internal components.
    #[inline]
    pub fn new_broadcast(
        views: Buffer<View>,
        buffers: Buffer<Buffer<u8>>,
        length: usize,
        validity: Option<PlBitmap>,
    ) -> Self {
        Self::try_new_broadcast(views, buffers, length, validity).unwrap()
    }

    /// Creates a scalar [`PlBinaryViewArray`] of `length` elements without validating them.
    ///
    /// # Safety
    /// `views` and `validity` must both be scalar and valid for `length` elements.
    #[inline]
    pub unsafe fn new_broadcast_unchecked(
        views: Buffer<View>,
        buffers: Buffer<Buffer<u8>>,
        length: usize,
        validity: Option<PlBitmap>,
    ) -> Self {
        let validity = validity_covering_unchecked(validity, length);
        if cfg!(debug_assertions) {
            assert!(is_scalar_buffer_len(views.len(), length));
            validate_views(&views, &buffers).unwrap();
        }

        Self {
            views: normalize_buffer(views, length),
            buffers,
            length,
            validity,
            total_bytes_len: RelaxedCell::from(UNKNOWN_BYTES_LEN),
        }
    }

    /// Creates an empty [`PlBinaryViewArray`].
    #[inline]
    pub fn new_empty() -> Self {
        Self {
            views: Buffer::new(),
            buffers: Buffer::new(),
            length: 0,
            validity: None,
            total_bytes_len: RelaxedCell::from(0),
        }
    }

    /// Creates a flat, fully valid [`PlBinaryViewArray`] from `views` over `buffers`.
    #[inline]
    pub fn from_views(views: Buffer<View>, buffers: Buffer<Buffer<u8>>) -> Self {
        let length = views.len();
        Self::new(views, buffers, length, None)
    }

    /// Creates a flat, fully valid [`PlBinaryViewArray`] holding `values`, in order.
    pub fn from_values_iter<V: AsRef<[u8]>, I: IntoIterator<Item = V>>(values: I) -> Self {
        let values = values.into_iter();
        let (lower, _) = values.size_hint();

        let mut views = Vec::with_capacity(lower);
        let mut buffers = Vec::new();
        values.for_each(|value| {
            views.push(copy_value(&mut buffers, 0, value.as_ref()));
        });

        let length = views.len();
        // SAFETY: there is one view per element, each written over the buffers it reads.
        unsafe { Self::new_unchecked(Buffer::from(views), collect_buffers(buffers), length, None) }
    }

    /// Creates a [`PlBinaryViewArray`] of `length` copies of `value`, in `O(value.len())` memory.
    pub fn new_scalar(value: &[u8], length: usize) -> Self {
        if length == 0 {
            return Self::new_empty();
        }

        let (view, buffers) = copy_only_value(value);

        Self {
            views: Buffer::from_owner([view]),
            buffers: collect_buffers(buffers),
            length,
            validity: None,
            total_bytes_len: RelaxedCell::from(scalar_bytes_len(view, length)),
        }
    }

    /// [`Self::new_scalar`], taking over the allocation `value` already holds its bytes in.
    pub fn new_scalar_owned(value: Vec<u8>, length: usize) -> Self {
        if length == 0 {
            return Self::new_empty();
        }

        let (view, buffers) = own_only_value(value);

        Self {
            views: Buffer::from_owner([view]),
            buffers: collect_buffers(buffers),
            length,
            validity: None,
            total_bytes_len: RelaxedCell::from(scalar_bytes_len(view, length)),
        }
    }

    /// Creates a [`PlBinaryViewArray`] of `length` nulls, in `O(1)` memory.
    #[inline]
    pub fn new_full_null(length: usize) -> Self {
        Self {
            views: Buffer::zeroed(scalar_buffer_len(length)),
            buffers: Buffer::new(),
            length,
            validity: Some(Bitmap::new_zeroed(scalar_buffer_len(length))),
            total_bytes_len: RelaxedCell::from(0),
        }
    }

    /// The backing views buffer, if it holds one slot per element.
    #[inline]
    pub fn flat_views(&self) -> Option<&Buffer<View>> {
        (!self.views_are_scalar()).then_some(&self.views)
    }

    /// The backing views buffer, if it holds one slot per element.
    ///
    /// # Safety
    /// Every view left in the buffer must read bytes that [`Self::data_buffers`] holds.
    #[inline]
    pub unsafe fn flat_views_mut(&mut self) -> Option<&mut Buffer<View>> {
        if self.views_are_scalar() {
            None
        } else {
            self.total_bytes_len.store(UNKNOWN_BYTES_LEN);
            Some(&mut self.views)
        }
    }

    /// The view every element of this array reads, if the views buffer holds a single slot.
    #[inline]
    pub fn scalar_views(&self) -> Option<View> {
        self.views_are_scalar().then(|| self.views[0])
    }

    /// The bytes every element of this array reads, if the views buffer holds a single slot.
    #[inline]
    pub fn scalar_value_ignore_validity(&self) -> Option<&[u8]> {
        // SAFETY: a scalar views buffer holds the one view element 0 reads, which is in bounds.
        self.views_are_scalar()
            .then(|| unsafe { self.value_unchecked(0) })
    }

    /// The buffers the views that do not inline their bytes point into.
    #[inline(always)]
    pub const fn data_buffers(&self) -> &Buffer<Buffer<u8>> {
        &self.buffers
    }

    /// The buffers the views that do not inline their bytes point into.
    ///
    /// # Safety
    /// Every view of this array must still read bytes the buffers hold once they are written.
    #[inline]
    pub unsafe fn data_buffers_mut(&mut self) -> &mut Buffer<Buffer<u8>> {
        self.total_bytes_len.store(UNKNOWN_BYTES_LEN);
        &mut self.buffers
    }

    /// The validity mask, if any element may be null.
    #[inline]
    pub fn validity(&self) -> Option<PlBitmapRef<'_>> {
        // SAFETY: the mask is flat or scalar for `self.length`, upheld by every constructor.
        self.validity
            .as_ref()
            .map(|validity| unsafe { PlBitmapRef::new_broadcast_unchecked(validity, self.length) })
    }

    /// Whether the views buffer holds a single view shared by every element.
    #[inline]
    pub fn views_are_scalar(&self) -> bool {
        self.views.len() == 1 && self.length > 0
    }

    /// Whether the views buffer holds one slot per element.
    #[inline]
    pub fn views_are_flat(&self) -> bool {
        self.views.len() == self.length
    }

    /// Whether the validity mask holds a single bit shared by every element.
    #[inline]
    pub fn validity_is_scalar(&self) -> bool {
        self.validity().is_some_and(|v| v.is_scalar())
    }

    /// Whether every backing buffer has one slot per element.
    #[inline]
    pub fn is_flat(&self) -> bool {
        self.views_are_flat() && self.validity().is_none_or(|validity| validity.is_flat())
    }

    /// Whether this array is scalar throughout: one value repeated [`Self::len`] times.
    #[inline]
    pub fn is_scalar(&self) -> bool {
        self.views_are_scalar() && self.validity().is_none_or(|v| v.is_scalar())
    }

    /// The single element every element equals, if every backing buffer holds one slot.
    #[inline]
    pub fn scalar_value(&self) -> Option<Option<&[u8]>> {
        let is_shared = self.views.len() == 1
            && self
                .validity
                .as_ref()
                .is_none_or(|validity| validity.len() == 1);

        // SAFETY: the array is not empty, so element 0 is in bounds.
        (is_shared && self.length > 0).then(|| unsafe { self.get_unchecked(0) })
    }

    /// Returns the view of the element at `i`.
    #[inline]
    pub fn view(&self, i: usize) -> View {
        assert!(i < self.length, "index out of bounds");
        unsafe { self.view_unchecked(i) }
    }

    /// Returns the view of the element at `i`.
    ///
    /// # Safety
    /// `i` must be smaller than `self.len()`.
    #[inline]
    pub unsafe fn view_unchecked(&self, i: usize) -> View {
        debug_assert!(i < self.length);
        unsafe {
            *self
                .views
                .get_unchecked(broadcast_index(i, self.views.len()))
        }
    }

    /// Returns the value at `i`.
    #[inline]
    pub fn value(&self, i: usize) -> &[u8] {
        assert!(i < self.length, "index out of bounds");
        unsafe { self.value_unchecked(i) }
    }

    /// Returns the value at `i`.
    ///
    /// # Safety
    /// `i` must be smaller than `self.len()`.
    #[inline]
    pub unsafe fn value_unchecked(&self, i: usize) -> &[u8] {
        debug_assert!(i < self.length);
        // SAFETY: every view reads bytes the data buffers hold, upheld by every constructor.
        unsafe {
            self.views
                .get_unchecked(broadcast_index(i, self.views.len()))
                .get_slice_unchecked(self.buffers.as_slice())
        }
    }

    /// The number of bytes it would take to lay the values of the valid elements end to end.
    ///
    /// Walking the views to answer this is `O(len)`, so the answer is kept until the views or the
    /// mask move underneath it.
    pub fn total_bytes_len(&self) -> usize {
        let cached = self.total_bytes_len.load();
        if cached != UNKNOWN_BYTES_LEN {
            return cached as usize;
        }

        let total = self.compute_total_bytes_len();
        self.total_bytes_len.store(total as u64);
        total
    }

    /// Walks the views to count the bytes the valid elements hold.
    fn compute_total_bytes_len(&self) -> usize {
        if self.views_are_scalar() {
            let valid = self
                .validity()
                .map_or(self.length, |validity| validity.set_bits());

            return (self.views[0].length as usize)
                .checked_mul(valid)
                .expect("the total length of the values overflows a `usize`");
        }

        match self.validity() {
            None => self.views.iter().map(|view| view.length as usize).sum(),
            Some(validity) => self
                .views
                .iter()
                .zip(validity)
                .filter(|(_, is_valid)| *is_valid)
                .map(|(view, _)| view.length as usize)
                .sum(),
        }
    }

    /// The number of bytes the data buffers hold.
    pub fn total_buffer_len(&self) -> usize {
        self.buffers.iter().map(|buffer| buffer.len()).sum()
    }

    /// Returns an iterator over the values, ignoring validity.
    #[inline]
    pub fn values_iter(&self) -> PlBinaryViewValuesIter<'_> {
        PlBinaryViewValuesIter::new(&self.views, &self.buffers, self.length)
    }

    /// Returns an iterator over the optional elements.
    #[inline]
    pub fn iter(&self) -> PlBinaryViewIter<'_> {
        PlBinaryViewIter::new(&self.views, &self.buffers, self.validity(), self.length)
    }

    /// Iterates `length` values, repeating a scalar array's one value and ignoring validity.
    #[inline]
    pub fn broadcast_values_iter(&self, length: usize) -> PlBinaryViewValuesIter<'_> {
        assert_broadcastable(self.length, length);
        // SAFETY: a single view is scalar for any length, and otherwise the views are already
        // valid for `length`; either way every view reads bytes the buffers hold.
        PlBinaryViewValuesIter::new(&self.views, &self.buffers, length)
    }

    /// Slices this array in place to `length` elements starting at `offset`.
    ///
    /// # Safety
    /// `offset + length` must not exceed `self.len()`.
    pub unsafe fn slice_unchecked(&mut self, offset: usize, length: usize) {
        debug_assert!(offset + length <= self.length);

        unsafe {
            slice_buffer(&mut self.views, self.length, offset, length);
            slice_validity(&mut self.validity, self.length, offset, length);
        }

        self.length = length;
        self.total_bytes_len.store(UNKNOWN_BYTES_LEN);
    }

    /// Creates a [`PlBinaryViewArray`] of `length` copies of the element at `index`.
    ///
    /// # Safety
    /// `index` must be smaller than `self.len()`.
    pub unsafe fn new_from_index_unchecked(&self, index: usize, length: usize) -> Self {
        debug_assert!(index < self.length);

        if unsafe { self.is_null_unchecked(index) } {
            return Self::new_full_null(length);
        }

        if length == 0 {
            return Self::new_empty();
        }

        let view = unsafe { self.view_unchecked(index) };

        let (view, buffers) = if view.is_inline() {
            (view, Buffer::new())
        } else {
            let buffer = self.buffers[view.buffer_idx as usize].clone();
            let view = View {
                buffer_idx: 0,
                ..view
            };
            (view, Buffer::from_owner([buffer]))
        };

        Self {
            views: Buffer::from_owner([view]),
            buffers,
            length,
            validity: None,
            total_bytes_len: RelaxedCell::from(scalar_bytes_len(view, length)),
        }
    }

    /// Returns this array with its elements in the opposite order, keeping the representation.
    #[must_use]
    pub fn reversed(&self) -> Self {
        if self.is_scalar() {
            return self.clone();
        }

        let validity = self
            .validity
            .as_ref()
            .map(|validity| PlBitmap::new_broadcast(validity.clone(), self.length).reversed());

        if self.views_are_scalar() {
            // SAFETY: the views buffer is the one slot it already was, over the same buffers.
            unsafe {
                Self::new_broadcast_unchecked(
                    self.views.clone(),
                    self.buffers.clone(),
                    self.length,
                    validity,
                )
            }
        } else {
            let mut views = Vec::with_capacity(self.length);
            views.extend(self.views.as_slice().iter().rev().copied());

            // SAFETY: one view per element, each already validated against these buffers.
            unsafe {
                Self::new_unchecked(views.into(), self.buffers.clone(), self.length, validity)
            }
        }
    }

    /// Returns an equivalent flat array, borrowing this one if it is already flat.
    pub fn to_flat(&self) -> Cow<'_, Flat<Self>> {
        if let Some(flat) = self.as_flat() {
            return Cow::Borrowed(flat);
        }

        let views = if self.views_are_flat() {
            self.views.clone()
        } else if self.length == 0 {
            Buffer::new()
        } else if self.scalar_value() == Some(None) {
            Buffer::zeroed(self.length)
        } else {
            Buffer::from(vec![self.views[0]; self.length])
        };

        let validity = self
            .validity()
            .map(|validity| validity.to_flat().into_owned());

        // SAFETY: the views hold one slot per element, and the mask is the flat counterpart.
        Cow::Owned(unsafe {
            Flat::new(Self {
                views,
                buffers: self.buffers.clone(),
                length: self.length,
                validity,
                // Flattening repeats the elements it already held, in order.
                total_bytes_len: self.total_bytes_len.clone(),
            })
        })
    }

    /// Borrows this array as a [`Flat`] one, if it is already flat.
    #[inline]
    pub fn as_flat(&self) -> Option<&Flat<Self>> {
        // SAFETY: the views and the mask of a flat array hold one slot per element.
        self.is_flat().then(|| unsafe { Flat::new_ref(self) })
    }
}

crate::impl_array_methods!(PlBinaryViewArray, &[u8]);

/// The bytes `length` copies of the element `view` reads lay end to end.
fn scalar_bytes_len(view: View, length: usize) -> u64 {
    u64::from(view.length)
        .checked_mul(length as u64)
        .filter(|total| *total != UNKNOWN_BYTES_LEN)
        .expect("the total length of the values overflows a `u64`")
}

/// Validates that every view of `views` reads bytes that `buffers` holds.
fn validate_views(views: &[View], buffers: &[Buffer<u8>]) -> PolarsResult<()> {
    for view in views {
        if let Some(inlined) = view.get_inlined_slice() {
            if view.length < View::MAX_INLINE_SIZE && view.as_u128() >> (32 + view.length * 8) != 0
            {
                polars_bail!(
                    ComputeError:
                    "view of {} inlined bytes holds non-zero padding past them", inlined.len(),
                );
            }
            continue;
        }

        let buffer = buffers.get(view.buffer_idx as usize).ok_or_else(|| {
            polars_err!(
                OutOfBounds:
                "view points at data buffer {} of {} buffers", view.buffer_idx, buffers.len(),
            )
        })?;

        let start = view.offset as usize;
        let end = start + view.length as usize;
        let bytes = buffer.as_slice().get(start..end).ok_or_else(|| {
            polars_err!(
                OutOfBounds:
                "view covers bytes {}..{} of a data buffer of {} bytes",
                start, end, buffer.len(),
            )
        })?;

        polars_ensure!(
            bytes.starts_with(&view.prefix.to_le_bytes()),
            ComputeError: "view holds a prefix that the bytes it points at do not start with",
        );
    }

    Ok(())
}

/// The data buffers a view builder filled, as the buffer of buffers an array holds them in.
fn collect_buffers(buffers: Vec<Vec<u8>>) -> Buffer<Buffer<u8>> {
    buffers.into_iter().map(Buffer::from).collect()
}

impl Default for PlBinaryViewArray {
    #[inline]
    fn default() -> Self {
        Self::new_empty()
    }
}

impl<V: AsRef<[u8]>> FromIterator<Option<V>> for PlBinaryViewArray {
    fn from_iter<I: IntoIterator<Item = Option<V>>>(iter: I) -> Self {
        let iter = iter.into_iter();
        let (lower, _) = iter.size_hint();

        let mut views = Vec::with_capacity(lower);
        let mut buffers = Vec::new();
        let mut validity = BitmapBuilder::with_capacity(lower);

        iter.for_each(|value| match value {
            Some(value) => {
                views.push(copy_value(&mut buffers, 0, value.as_ref()));
                validity.push(true);
            },
            None => {
                views.push(View::default());
                validity.push(false);
            },
        });

        let length = views.len();
        // SAFETY: one view and one bit per element, each view over the buffers it reads.
        unsafe {
            Self::new_unchecked(
                Buffer::from(views),
                collect_buffers(buffers),
                length,
                validity.into_opt_validity().map(PlBitmap::from_bitmap),
            )
        }
    }
}

crate::impl_into_iterator!(PlBinaryViewArray, PlBinaryViewIter<'a>);

/// Compares two arrays element-wise, disregarding the representation and how the bytes are reached.
impl PartialEq for PlBinaryViewArray {
    fn eq(&self, other: &Self) -> bool {
        if self.length != other.length {
            return false;
        }

        if let (Some(lhs), Some(rhs)) = (self.scalar_value(), other.scalar_value()) {
            return lhs == rhs;
        }

        self.iter().eq(other.iter())
    }
}

impl Eq for PlBinaryViewArray {}

crate::impl_element_debug!(PlBinaryViewArray, "PlBinaryViewArray");

crate::impl_pl_array!(PlBinaryViewArray, PlArrayType::BinaryView);
