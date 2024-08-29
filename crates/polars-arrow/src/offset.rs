//! Contains the declaration of [`Offset`]
use std::hint::unreachable_unchecked;
use std::ops::Deref;

use polars_error::{polars_bail, polars_err, PolarsError, PolarsResult};
use polars_utils::slice::GetSaferUnchecked;

use crate::array::Splitable;
use crate::buffer::Buffer;
pub use crate::types::Offset;

/// A wrapper type of [`Vec<O>`] representing the invariants of Arrow's offsets.
/// It is guaranteed to (sound to assume that):
/// * every element is `>= 0`
/// * element at position `i` is >= than element at position `i-1`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Offsets<O: Offset>(Vec<O>);

impl<O: Offset> Default for Offsets<O> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl<O: Offset> Deref for Offsets<O> {
    type Target = [O];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl<O: Offset> TryFrom<Vec<O>> for Offsets<O> {
    type Error = PolarsError;

    #[inline]
    fn try_from(offsets: Vec<O>) -> Result<Self, Self::Error> {
        try_check_offsets(&offsets)?;
        Ok(Self(offsets))
    }
}

impl<O: Offset> TryFrom<Buffer<O>> for OffsetsBuffer<O> {
    type Error = PolarsError;

    #[inline]
    fn try_from(offsets: Buffer<O>) -> Result<Self, Self::Error> {
        try_check_offsets(&offsets)?;
        Ok(Self(offsets))
    }
}

impl<O: Offset> TryFrom<Vec<O>> for OffsetsBuffer<O> {
    type Error = PolarsError;

    #[inline]
    fn try_from(offsets: Vec<O>) -> Result<Self, Self::Error> {
        try_check_offsets(&offsets)?;
        Ok(Self(offsets.into()))
    }
}

impl<O: Offset> From<Offsets<O>> for OffsetsBuffer<O> {
    #[inline]
    fn from(offsets: Offsets<O>) -> Self {
        Self(offsets.0.into())
    }
}

impl<O: Offset> Offsets<O> {
    /// Returns an empty [`Offsets`] (i.e. with a single element, the zero)
    #[inline]
    pub fn new() -> Self {
        Self(vec![O::zero()])
    }

    /// Returns an [`Offsets`] whose all lengths are zero.
    #[inline]
    pub fn new_zeroed(length: usize) -> Self {
        Self(vec![O::zero(); length + 1])
    }

    /// Creates a new [`Offsets`] from an iterator of lengths
    #[inline]
    pub fn try_from_iter<I: IntoIterator<Item = usize>>(iter: I) -> PolarsResult<Self> {
        let iterator = iter.into_iter();
        let (lower, _) = iterator.size_hint();
        let mut offsets = Self::with_capacity(lower);
        for item in iterator {
            offsets.try_push(item)?
        }
        Ok(offsets)
    }

    /// Returns a new [`Offsets`] with a capacity, allocating at least `capacity + 1` entries.
    pub fn with_capacity(capacity: usize) -> Self {
        let mut offsets = Vec::with_capacity(capacity + 1);
        offsets.push(O::zero());
        Self(offsets)
    }

    /// Returns the capacity of [`Offsets`].
    pub fn capacity(&self) -> usize {
        self.0.capacity() - 1
    }

    /// Reserves `additional` entries.
    pub fn reserve(&mut self, additional: usize) {
        self.0.reserve(additional);
    }

    /// Shrinks the capacity of self to fit.
    pub fn shrink_to_fit(&mut self) {
        self.0.shrink_to_fit();
    }

    /// Pushes a new element with a given length.
    /// # Error
    /// This function errors iff the new last item is larger than what `O` supports.
    /// # Implementation
    /// This function:
    /// * checks that this length does not overflow
    #[inline]
    pub fn try_push(&mut self, length: usize) -> PolarsResult<()> {
        if O::IS_LARGE {
            let length = O::from_as_usize(length);
            let old_length = self.last();
            let new_length = *old_length + length;
            self.0.push(new_length);
            Ok(())
        } else {
            let length =
                O::from_usize(length).ok_or_else(|| polars_err!(ComputeError: "overflow"))?;

            let old_length = self.last();
            let new_length = old_length
                .checked_add(&length)
                .ok_or_else(|| polars_err!(ComputeError: "overflow"))?;
            self.0.push(new_length);
            Ok(())
        }
    }

    /// Returns [`Offsets`] assuming that `offsets` fulfills its invariants
    ///
    /// # Safety
    /// This is safe iff the invariants of this struct are guaranteed in `offsets`.
    #[inline]
    pub unsafe fn new_unchecked(offsets: Vec<O>) -> Self {
        #[cfg(debug_assertions)]
        {
            let mut prev_offset = O::default();
            let mut is_monotonely_increasing = true;
            for offset in &offsets {
                is_monotonely_increasing &= *offset >= prev_offset;
                prev_offset = *offset;
            }
            assert!(
                is_monotonely_increasing,
                "Unsafe precondition violated. Invariant of offsets broken."
            );
        }

        Self(offsets)
    }

    /// Returns the last offset of this container.
    #[inline]
    pub fn last(&self) -> &O {
        match self.0.last() {
            Some(element) => element,
            None => unsafe { unreachable_unchecked() },
        }
    }

    /// Returns a `length` corresponding to the position `index`
    /// # Panic
    /// This function panics iff `index >= self.len()`
    #[inline]
    pub fn length_at(&self, index: usize) -> usize {
        let (start, end) = self.start_end(index);
        end - start
    }

    /// Returns a range (start, end) corresponding to the position `index`
    /// # Panic
    /// This function panics iff `index >= self.len()`
    #[inline]
    pub fn start_end(&self, index: usize) -> (usize, usize) {
        // soundness: the invariant of the function
        assert!(index < self.len_proxy());
        unsafe { self.start_end_unchecked(index) }
    }

    /// Returns a range (start, end) corresponding to the position `index`
    ///
    /// # Safety
    /// `index` must be `< self.len()`
    #[inline]
    pub unsafe fn start_end_unchecked(&self, index: usize) -> (usize, usize) {
        // soundness: the invariant of the function
        let start = self.0.get_unchecked(index).to_usize();
        let end = self.0.get_unchecked(index + 1).to_usize();
        (start, end)
    }

    /// Returns the length an array with these offsets would be.
    #[inline]
    pub fn len_proxy(&self) -> usize {
        self.0.len() - 1
    }

    #[inline]
    /// Returns the number of offsets in this container.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Returns the byte slice stored in this buffer
    #[inline]
    pub fn as_slice(&self) -> &[O] {
        self.0.as_slice()
    }

    /// Pops the last element
    #[inline]
    pub fn pop(&mut self) -> Option<O> {
        if self.len_proxy() == 0 {
            None
        } else {
            self.0.pop()
        }
    }

    /// Extends itself with `additional` elements equal to the last offset.
    /// This is useful to extend offsets with empty values, e.g. for null slots.
    #[inline]
    pub fn extend_constant(&mut self, additional: usize) {
        let offset = *self.last();
        if additional == 1 {
            self.0.push(offset)
        } else {
            self.0.resize(self.len() + additional, offset)
        }
    }

    /// Try to create a new [`Offsets`] from a sequence of `lengths`
    /// # Errors
    /// This function errors iff this operation overflows for the maximum value of `O`.
    #[inline]
    pub fn try_from_lengths<I: Iterator<Item = usize>>(lengths: I) -> PolarsResult<Self> {
        let mut self_ = Self::with_capacity(lengths.size_hint().0);
        self_.try_extend_from_lengths(lengths)?;
        Ok(self_)
    }

    /// Try extend from an iterator of lengths
    /// # Errors
    /// This function errors iff this operation overflows for the maximum value of `O`.
    #[inline]
    pub fn try_extend_from_lengths<I: Iterator<Item = usize>>(
        &mut self,
        lengths: I,
    ) -> PolarsResult<()> {
        let mut total_length = 0;
        let mut offset = *self.last();
        let original_offset = offset.to_usize();

        let lengths = lengths.map(|length| {
            total_length += length;
            O::from_as_usize(length)
        });

        let offsets = lengths.map(|length| {
            offset += length; // this may overflow, checked below
            offset
        });
        self.0.extend(offsets);

        let last_offset = original_offset
            .checked_add(total_length)
            .ok_or_else(|| polars_err!(ComputeError: "overflow"))?;
        O::from_usize(last_offset).ok_or_else(|| polars_err!(ComputeError: "overflow"))?;
        Ok(())
    }

    /// Extends itself from another [`Offsets`]
    /// # Errors
    /// This function errors iff this operation overflows for the maximum value of `O`.
    pub fn try_extend_from_self(&mut self, other: &Self) -> PolarsResult<()> {
        let mut length = *self.last();
        let other_length = *other.last();
        // check if the operation would overflow
        length
            .checked_add(&other_length)
            .ok_or_else(|| polars_err!(ComputeError: "overflow"))?;

        let lengths = other.as_slice().windows(2).map(|w| w[1] - w[0]);
        let offsets = lengths.map(|new_length| {
            length += new_length;
            length
        });
        self.0.extend(offsets);
        Ok(())
    }

    /// Extends itself from another [`Offsets`] sliced by `start, length`
    /// # Errors
    /// This function errors iff this operation overflows for the maximum value of `O`.
    pub fn try_extend_from_slice(
        &mut self,
        other: &OffsetsBuffer<O>,
        start: usize,
        length: usize,
    ) -> PolarsResult<()> {
        if length == 0 {
            return Ok(());
        }
        let other = &other.0[start..start + length + 1];
        let other_length = other.last().expect("Length to be non-zero");
        let mut length = *self.last();
        // check if the operation would overflow
        length
            .checked_add(other_length)
            .ok_or_else(|| polars_err!(ComputeError: "overflow"))?;

        let lengths = other.windows(2).map(|w| w[1] - w[0]);
        let offsets = lengths.map(|new_length| {
            length += new_length;
            length
        });
        self.0.extend(offsets);
        Ok(())
    }

    /// Returns the inner [`Vec`].
    #[inline]
    pub fn into_inner(self) -> Vec<O> {
        self.0
    }
}

/// Checks that `offsets` is monotonically increasing.
fn try_check_offsets<O: Offset>(offsets: &[O]) -> PolarsResult<()> {
    // this code is carefully constructed to auto-vectorize, don't change naively!
    match offsets.first() {
        None => polars_bail!(ComputeError: "offsets must have at least one element"),
        Some(first) => {
            if *first < O::zero() {
                polars_bail!(ComputeError: "offsets must be larger than 0")
            }
            let mut previous = *first;
            let mut any_invalid = false;

            // This loop will auto-vectorize because there is not any break,
            // an invalid value will be returned once the whole offsets buffer is processed.
            for offset in offsets {
                if previous > *offset {
                    any_invalid = true
                }
                previous = *offset;
            }

            if any_invalid {
                polars_bail!(ComputeError: "offsets must be monotonically increasing")
            } else {
                Ok(())
            }
        },
    }
}

/// A wrapper type of [`Buffer<O>`] that is guaranteed to:
/// * Always contain an element
/// * Every element is `>= 0`
/// * element at position `i` is >= than element at position `i-1`.
#[derive(Clone, PartialEq, Debug)]
pub struct OffsetsBuffer<O: Offset>(Buffer<O>);

impl<O: Offset> Default for OffsetsBuffer<O> {
    #[inline]
    fn default() -> Self {
        Self(vec![O::zero()].into())
    }
}

impl<O: Offset> OffsetsBuffer<O> {
    /// # Safety
    /// This is safe iff the invariants of this struct are guaranteed in `offsets`.
    #[inline]
    pub unsafe fn new_unchecked(offsets: Buffer<O>) -> Self {
        Self(offsets)
    }

    /// Returns an empty [`OffsetsBuffer`] (i.e. with a single element, the zero)
    #[inline]
    pub fn new() -> Self {
        Self(vec![O::zero()].into())
    }

    /// Copy-on-write API to convert [`OffsetsBuffer`] into [`Offsets`].
    #[inline]
    pub fn into_mut(self) -> either::Either<Self, Offsets<O>> {
        self.0
            .into_mut()
            // SAFETY: Offsets and OffsetsBuffer share invariants
            .map_right(|offsets| unsafe { Offsets::new_unchecked(offsets) })
            .map_left(Self)
    }

    /// Returns a reference to its internal [`Buffer`].
    #[inline]
    pub fn buffer(&self) -> &Buffer<O> {
        &self.0
    }

    /// Returns the length an array with these offsets would be.
    #[inline]
    pub fn len_proxy(&self) -> usize {
        self.0.len() - 1
    }

    /// Returns the number of offsets in this container.
    #[inline]
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Returns the byte slice stored in this buffer
    #[inline]
    pub fn as_slice(&self) -> &[O] {
        self.0.as_slice()
    }

    /// Returns the range of the offsets.
    #[inline]
    pub fn range(&self) -> O {
        *self.last() - *self.first()
    }

    /// Returns the first offset.
    #[inline]
    pub fn first(&self) -> &O {
        match self.0.first() {
            Some(element) => element,
            None => unsafe { unreachable_unchecked() },
        }
    }

    /// Returns the last offset.
    #[inline]
    pub fn last(&self) -> &O {
        match self.0.last() {
            Some(element) => element,
            None => unsafe { unreachable_unchecked() },
        }
    }

    /// Returns a `length` corresponding to the position `index`
    /// # Panic
    /// This function panics iff `index >= self.len()`
    #[inline]
    pub fn length_at(&self, index: usize) -> usize {
        let (start, end) = self.start_end(index);
        end - start
    }

    /// Returns a range (start, end) corresponding to the position `index`
    /// # Panic
    /// This function panics iff `index >= self.len()`
    #[inline]
    pub fn start_end(&self, index: usize) -> (usize, usize) {
        // soundness: the invariant of the function
        assert!(index < self.len_proxy());
        unsafe { self.start_end_unchecked(index) }
    }

    /// Returns a range (start, end) corresponding to the position `index`
    ///
    /// # Safety
    /// `index` must be `< self.len()`
    #[inline]
    pub unsafe fn start_end_unchecked(&self, index: usize) -> (usize, usize) {
        // soundness: the invariant of the function
        let start = self.0.get_unchecked_release(index).to_usize();
        let end = self.0.get_unchecked_release(index + 1).to_usize();
        (start, end)
    }

    /// Slices this [`OffsetsBuffer`].
    /// # Panics
    /// Panics if `offset + length` is larger than `len`
    /// or `length == 0`.
    #[inline]
    pub fn slice(&mut self, offset: usize, length: usize) {
        assert!(length > 0);
        self.0.slice(offset, length);
    }

    /// Slices this [`OffsetsBuffer`] starting at `offset`.
    ///
    /// # Safety
    /// The caller must ensure `offset + length <= self.len()`
    #[inline]
    pub unsafe fn slice_unchecked(&mut self, offset: usize, length: usize) {
        self.0.slice_unchecked(offset, length);
    }

    /// Returns an iterator with the lengths of the offsets
    #[inline]
    pub fn lengths(&self) -> impl Iterator<Item = usize> + '_ {
        self.0.windows(2).map(|w| (w[1] - w[0]).to_usize())
    }

    /// Returns the inner [`Buffer`].
    #[inline]
    pub fn into_inner(self) -> Buffer<O> {
        self.0
    }

    /// Returns the offset difference between `start` and `end`.
    #[inline]
    pub fn delta(&self, start: usize, end: usize) -> usize {
        assert!(start <= end);

        (self.0[end + 1] - self.0[start]).to_usize()
    }
}

impl From<&OffsetsBuffer<i32>> for OffsetsBuffer<i64> {
    fn from(offsets: &OffsetsBuffer<i32>) -> Self {
        // this conversion is lossless and uphelds all invariants
        Self(
            offsets
                .buffer()
                .iter()
                .map(|x| *x as i64)
                .collect::<Vec<_>>()
                .into(),
        )
    }
}

impl TryFrom<&OffsetsBuffer<i64>> for OffsetsBuffer<i32> {
    type Error = PolarsError;

    fn try_from(offsets: &OffsetsBuffer<i64>) -> Result<Self, Self::Error> {
        i32::try_from(*offsets.last()).map_err(|_| polars_err!(ComputeError: "overflow"))?;

        // this conversion is lossless and uphelds all invariants
        Ok(Self(
            offsets
                .buffer()
                .iter()
                .map(|x| *x as i32)
                .collect::<Vec<_>>()
                .into(),
        ))
    }
}

impl From<Offsets<i32>> for Offsets<i64> {
    fn from(offsets: Offsets<i32>) -> Self {
        // this conversion is lossless and uphelds all invariants
        Self(
            offsets
                .as_slice()
                .iter()
                .map(|x| *x as i64)
                .collect::<Vec<_>>(),
        )
    }
}

impl TryFrom<Offsets<i64>> for Offsets<i32> {
    type Error = PolarsError;

    fn try_from(offsets: Offsets<i64>) -> Result<Self, Self::Error> {
        i32::try_from(*offsets.last()).map_err(|_| polars_err!(ComputeError: "overflow"))?;

        // this conversion is lossless and uphelds all invariants
        Ok(Self(
            offsets
                .as_slice()
                .iter()
                .map(|x| *x as i32)
                .collect::<Vec<_>>(),
        ))
    }
}

impl<O: Offset> std::ops::Deref for OffsetsBuffer<O> {
    type Target = [O];

    #[inline]
    fn deref(&self) -> &[O] {
        self.0.as_slice()
    }
}

impl<O: Offset> Splitable for OffsetsBuffer<O> {
    fn check_bound(&self, offset: usize) -> bool {
        offset <= self.len_proxy()
    }

    unsafe fn _split_at_unchecked(&self, offset: usize) -> (Self, Self) {
        let mut lhs = self.0.clone();
        let mut rhs = self.0.clone();

        lhs.slice(0, offset + 1);
        rhs.slice(offset, self.0.len() - offset);

        (Self(lhs), Self(rhs))
    }
}
