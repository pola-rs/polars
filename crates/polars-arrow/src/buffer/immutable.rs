use std::iter::FromIterator;
use std::ops::Deref;
use std::sync::Arc;
use std::usize;

use either::Either;
use num_traits::Zero;

use super::{Bytes, IntoIter};
use crate::array::ArrayAccessor;

/// [`Buffer`] is a contiguous memory region that can be shared across
/// thread boundaries.
///
/// The easiest way to think about [`Buffer<T>`] is being equivalent to
/// a `Arc<Vec<T>>`, with the following differences:
/// * slicing and cloning is `O(1)`.
/// * it supports external allocated memory
///
/// The easiest way to create one is to use its implementation of `From<Vec<T>>`.
///
/// # Examples
/// ```
/// use polars_arrow::buffer::Buffer;
///
/// let mut buffer: Buffer<u32> = vec![1, 2, 3].into();
/// assert_eq!(buffer.as_ref(), [1, 2, 3].as_ref());
///
/// // it supports copy-on-write semantics (i.e. back to a `Vec`)
/// let vec: Vec<u32> = buffer.into_mut().right().unwrap();
/// assert_eq!(vec, vec![1, 2, 3]);
///
/// // cloning and slicing is `O(1)` (data is shared)
/// let mut buffer: Buffer<u32> = vec![1, 2, 3].into();
/// let mut sliced = buffer.clone();
/// sliced.slice(1, 1);
/// assert_eq!(sliced.as_ref(), [2].as_ref());
/// // but cloning forbids getting mut since `slice` and `buffer` now share data
/// assert_eq!(buffer.get_mut_slice(), None);
/// ```
#[derive(Clone)]
pub struct Buffer<T> {
    /// The internal byte buffer.
    storage: Arc<Bytes<T>>,

    /// A pointer into the buffer where our data starts.
    ptr: *const T,

    // The length of the buffer.
    length: usize,
}

unsafe impl<T: Sync> Sync for Buffer<T> {}
unsafe impl<T: Send> Send for Buffer<T> {}

impl<T: PartialEq> PartialEq for Buffer<T> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.deref() == other.deref()
    }
}

impl<T: std::fmt::Debug> std::fmt::Debug for Buffer<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Debug::fmt(&**self, f)
    }
}

impl<T> Default for Buffer<T> {
    #[inline]
    fn default() -> Self {
        Vec::new().into()
    }
}

impl<T> Buffer<T> {
    /// Creates an empty [`Buffer`].
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Auxiliary method to create a new Buffer
    pub(crate) fn from_bytes(bytes: Bytes<T>) -> Self {
        let ptr = bytes.as_ptr();
        let length = bytes.len();
        Buffer {
            storage: Arc::new(bytes),
            ptr,
            length,
        }
    }

    /// Returns the number of bytes in the buffer
    #[inline]
    pub fn len(&self) -> usize {
        self.length
    }

    /// Returns whether the buffer is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.length == 0
    }

    /// Returns whether underlying data is sliced.
    /// If sliced the [`Buffer`] is backed by
    /// more data than the length of `Self`.
    pub fn is_sliced(&self) -> bool {
        self.storage.len() != self.length
    }

    /// Returns the byte slice stored in this buffer
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        // Safety:
        // invariant of this struct `offset + length <= data.len()`
        debug_assert!(self.offset() + self.length <= self.storage.len());
        unsafe { std::slice::from_raw_parts(self.ptr, self.length) }
    }

    /// Returns the byte slice stored in this buffer
    /// # Safety
    /// `index` must be smaller than `len`
    #[inline]
    pub(super) unsafe fn get_unchecked(&self, index: usize) -> &T {
        // Safety:
        // invariant of this function
        debug_assert!(index < self.length);
        unsafe { &*self.ptr.add(index) }
    }

    /// Returns a new [`Buffer`] that is a slice of this buffer starting at `offset`.
    /// Doing so allows the same memory region to be shared between buffers.
    /// # Panics
    /// Panics iff `offset + length` is larger than `len`.
    #[inline]
    pub fn sliced(self, offset: usize, length: usize) -> Self {
        assert!(
            offset + length <= self.len(),
            "the offset of the new Buffer cannot exceed the existing length"
        );
        // Safety: we just checked bounds
        unsafe { self.sliced_unchecked(offset, length) }
    }

    /// Slices this buffer starting at `offset`.
    /// # Panics
    /// Panics iff `offset` is larger than `len`.
    #[inline]
    pub fn slice(&mut self, offset: usize, length: usize) {
        assert!(
            offset + length <= self.len(),
            "the offset of the new Buffer cannot exceed the existing length"
        );
        // Safety: we just checked bounds
        unsafe { self.slice_unchecked(offset, length) }
    }

    /// Returns a new [`Buffer`] that is a slice of this buffer starting at `offset`.
    /// Doing so allows the same memory region to be shared between buffers.
    /// # Safety
    /// The caller must ensure `offset + length <= self.len()`
    #[inline]
    #[must_use]
    pub unsafe fn sliced_unchecked(mut self, offset: usize, length: usize) -> Self {
        self.slice_unchecked(offset, length);
        self
    }

    /// Slices this buffer starting at `offset`.
    /// # Safety
    /// The caller must ensure `offset + length <= self.len()`
    #[inline]
    pub unsafe fn slice_unchecked(&mut self, offset: usize, length: usize) {
        self.ptr = self.ptr.add(offset);
        self.length = length;
    }

    /// Returns a pointer to the start of this buffer.
    #[inline]
    pub(crate) fn as_ptr(&self) -> *const T {
        self.ptr
    }

    /// Returns the offset of this buffer.
    #[inline]
    pub fn offset(&self) -> usize {
        unsafe {
            let ret = self.ptr.offset_from(self.storage.as_ptr()) as usize;
            debug_assert!(ret <= self.storage.len());
            ret
        }
    }

    /// # Safety
    /// The caller must ensure that the buffer was properly initialized up to `len`.
    #[inline]
    pub unsafe fn set_len(&mut self, len: usize) {
        self.length = len;
    }

    /// Returns a mutable reference to its underlying [`Vec`], if possible.
    ///
    /// This operation returns [`Either::Right`] iff this [`Buffer`]:
    /// * has not been cloned (i.e. [`Arc`]`::get_mut` yields [`Some`])
    /// * has not been imported from the C data interface (FFI)
    #[inline]
    pub fn into_mut(mut self) -> Either<Self, Vec<T>> {
        // We lose information if the data is sliced.
        if self.is_sliced() {
            return Either::Left(self);
        }
        match Arc::get_mut(&mut self.storage)
            .and_then(|b| b.get_vec())
            .map(std::mem::take)
        {
            Some(inner) => Either::Right(inner),
            None => Either::Left(self),
        }
    }

    /// Returns a mutable reference to its underlying `Vec`, if possible.
    /// Note that only `[self.offset(), self.offset() + self.len()[` in this vector is visible
    /// by this buffer.
    ///
    /// This operation returns [`Some`] iff this [`Buffer`]:
    /// * has not been cloned (i.e. [`Arc`]`::get_mut` yields [`Some`])
    /// * has not been imported from the C data interface (FFI)
    /// # Safety
    /// The caller must ensure that the vector in the mutable reference keeps a length of at least `self.offset() + self.len() - 1`.
    #[inline]
    pub unsafe fn get_mut(&mut self) -> Option<&mut Vec<T>> {
        // Arc::get_mut(&mut self.storage).and_then(|b| b.get_vec())
        unimplemented!()
    }

    /// Returns a mutable reference to its slice, if possible.
    ///
    /// This operation returns [`Some`] iff this [`Buffer`]:
    /// * has not been cloned (i.e. [`Arc`]`::get_mut` yields [`Some`])
    /// * has not been imported from the C data interface (FFI)
    #[inline]
    pub fn get_mut_slice(&mut self) -> Option<&mut [T]> {
        let offset = self.offset();
        let unique = Arc::get_mut(&mut self.storage)?;
        let vec = unique.get_vec()?;
        Some(unsafe { vec.get_unchecked_mut(offset..offset + self.length) })
    }

    /// Get the strong count of underlying `Arc` data buffer.
    pub fn shared_count_strong(&self) -> usize {
        Arc::strong_count(&self.storage)
    }

    /// Get the weak count of underlying `Arc` data buffer.
    pub fn shared_count_weak(&self) -> usize {
        Arc::weak_count(&self.storage)
    }

    /// Returns its internal representation
    #[must_use]
    pub fn into_inner(self) -> (Arc<Bytes<T>>, usize, usize) {
        // let Self {
        //     data,
        //     offset,
        //     length,
        // } = self;
        // (data, offset, length)
        unimplemented!();
    }

    /// Creates a `[Bitmap]` from its internal representation.
    /// This is the inverted from `[Bitmap::into_inner]`
    ///
    /// # Safety
    /// Callers must ensure all invariants of this struct are upheld.
    pub unsafe fn from_inner_unchecked(_data: Arc<Bytes<T>>, _offset: usize, _length: usize) -> Self {
        // Self {
        //     data,
        //     offset,
        //     length,
        // }
        unimplemented!();
    }
}

impl<T: Zero + Copy> Buffer<T> {
    pub fn zeroed(len: usize) -> Self {
        vec![T::zero(); len].into()
    }
}

impl<T> From<Vec<T>> for Buffer<T> {
    #[inline]
    fn from(p: Vec<T>) -> Self {
        let bytes: Bytes<T> = p.into();
        let ptr = bytes.as_ptr();
        let length = bytes.len();
        Self {
            storage: Arc::new(bytes),
            ptr,
            length,
        }
    }
}

impl<T> std::ops::Deref for Buffer<T> {
    type Target = [T];

    #[inline]
    fn deref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<T> FromIterator<T> for Buffer<T> {
    #[inline]
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        Vec::from_iter(iter).into()
    }
}

impl<T: Copy> IntoIterator for Buffer<T> {
    type Item = T;

    type IntoIter = IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        IntoIter::new(self)
    }
}

#[cfg(feature = "arrow_rs")]
impl<T: crate::types::NativeType> From<arrow_buffer::Buffer> for Buffer<T> {
    fn from(value: arrow_buffer::Buffer) -> Self {
        Self::from_bytes(crate::buffer::to_bytes(value))
    }
}

#[cfg(feature = "arrow_rs")]
impl<T: crate::types::NativeType> From<Buffer<T>> for arrow_buffer::Buffer {
    fn from(value: Buffer<T>) -> Self {
        let offset = value.offset();
        crate::buffer::to_buffer(value.storage).slice_with_length(
            offset * std::mem::size_of::<T>(),
            value.length * std::mem::size_of::<T>(),
        )
    }
}

unsafe impl<'a, T: 'a> ArrayAccessor<'a> for Buffer<T> {
    type Item = &'a T;

    unsafe fn value_unchecked(&'a self, index: usize) -> Self::Item {
        unsafe { &*self.ptr.add(index) }
    }

    fn len(&self) -> usize {
        Buffer::len(self)
    }
}
