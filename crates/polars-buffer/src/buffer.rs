use std::ops::{Deref, Range, RangeBounds};
use std::sync::LazyLock;

use bytemuck::{Pod, Zeroable};
use either::Either;

use crate::storage::SharedStorage;

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
/// use polars_buffer::Buffer;
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
pub struct Buffer<T> {
    /// The internal byte buffer.
    storage: SharedStorage<T>,

    /// A pointer into the buffer where our data starts.
    ptr: *const T,

    // The length of the buffer.
    length: usize,
}

impl<T> Clone for Buffer<T> {
    fn clone(&self) -> Self {
        Self {
            storage: self.storage.clone(),
            ptr: self.ptr,
            length: self.length,
        }
    }
}

unsafe impl<T: Send + Sync> Sync for Buffer<T> {}
unsafe impl<T: Send + Sync> Send for Buffer<T> {}

impl<T: PartialEq> PartialEq for Buffer<T> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.deref() == other.deref()
    }
}

impl<T: Eq> Eq for Buffer<T> {}

impl<T: std::hash::Hash> std::hash::Hash for Buffer<T> {
    #[inline]
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.as_slice().hash(state);
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
        Self::new()
    }
}

impl<T> Buffer<T> {
    /// Creates an empty [`Buffer`].
    #[inline]
    pub const fn new() -> Self {
        Self::from_storage(SharedStorage::empty())
    }

    /// Auxiliary method to create a new Buffer.
    pub const fn from_storage(storage: SharedStorage<T>) -> Self {
        let ptr = storage.as_ptr();
        let length = storage.len();
        Buffer {
            storage,
            ptr,
            length,
        }
    }

    /// Creates a [`Buffer`] backed by static data.
    pub fn from_static(data: &'static [T]) -> Self {
        Self::from_storage(SharedStorage::from_static(data))
    }

    /// Creates a [`Buffer`] backed by a vec.
    pub fn from_vec(data: Vec<T>) -> Self {
        Self::from_storage(SharedStorage::from_vec(data))
    }

    /// Creates a [`Buffer`] backed by `owner`.
    pub fn from_owner<O: Send + AsRef<[T]> + 'static>(owner: O) -> Self {
        Self::from_storage(SharedStorage::from_owner(owner))
    }

    /// Calls f with a [`Buffer`] backed by this slice.
    ///
    /// Aborts if any clones of the [`Buffer`] still live when `f` returns.
    pub fn with_slice<R, F: FnOnce(Buffer<T>) -> R>(slice: &[T], f: F) -> R {
        SharedStorage::with_slice(slice, |ss| f(Self::from_storage(ss)))
    }

    /// Calls f with a [`Buffer`] backed by this vec.
    ///
    /// # Panics
    /// Panics if any clones of the [`Buffer`] still live when `f` returns.
    pub fn with_vec<R, F: FnOnce(Buffer<T>) -> R>(vec: &mut Vec<T>, f: F) -> R {
        SharedStorage::with_vec(vec, |ss| f(Self::from_storage(ss)))
    }

    /// Returns the storage backing this [`Buffer`].
    pub fn into_storage(self) -> SharedStorage<T> {
        self.storage
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

    /// Expands this slice to the maximum allowed by the underlying storage.
    /// Only expands towards the end, the offset isn't changed. That is, element
    /// i before and after this operation refer to the same element.
    pub fn expand_end_to_storage(self) -> Self {
        unsafe {
            let offset = self.ptr.offset_from(self.storage.as_ptr()) as usize;
            Self {
                ptr: self.ptr,
                length: self.storage.len() - offset,
                storage: self.storage,
            }
        }
    }

    /// Returns the byte slice stored in this buffer.
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        // SAFETY: invariant of this struct `offset + length <= data.len()`.
        debug_assert!(self.offset() + self.length <= self.storage.len());
        unsafe { std::slice::from_raw_parts(self.ptr, self.length) }
    }

    /// Returns a new [`Buffer`] that is a slice of this buffer.
    /// Doing so allows the same memory region to be shared between buffers.
    ///
    /// # Panics
    /// Panics iff the range is out of bounds.
    #[inline]
    #[must_use]
    pub fn sliced<R: RangeBounds<usize>>(mut self, range: R) -> Self {
        self.slice_in_place(range);
        self
    }

    /// Returns a new [`Buffer`] that is a slice of this buffer starting at `offset`.
    /// Doing so allows the same memory region to be shared between buffers.
    ///
    /// # Safety
    /// The caller must ensure the range is in-bounds.
    #[inline]
    #[must_use]
    pub unsafe fn sliced_unchecked<R: RangeBounds<usize>>(mut self, range: R) -> Self {
        unsafe {
            self.slice_in_place_unchecked(range);
        }
        self
    }

    /// Slices this buffer to the given range.
    ///
    /// # Panics
    /// Panics iff the range is out of bounds.
    #[inline]
    pub fn slice_in_place<R: RangeBounds<usize>>(&mut self, range: R) {
        unsafe {
            let Range { start, end } = crate::check_range(range, ..self.len());
            self.ptr = self.ptr.add(start);
            self.length = end - start;
        }
    }

    /// Slices this buffer to the given range.
    ///
    /// # Safety
    /// The caller must ensure the range is in-bounds.
    #[inline]
    pub unsafe fn slice_in_place_unchecked<R: RangeBounds<usize>>(&mut self, range: R) {
        unsafe {
            let Range { start, end } = crate::decode_range_unchecked(range, ..self.len());
            self.ptr = self.ptr.add(start);
            self.length = end - start;
        }
    }

    /// Divides one buffer into two at an index.
    ///
    /// The first will contain all indices from `[0, mid)` (excluding
    /// the index `mid` itself) and the second will contain all
    /// indices from `[mid, len)` (excluding the index `len` itself).
    ///
    /// # Panics
    /// Panics if `mid > len`.
    #[must_use]
    pub fn split_at(self, mid: usize) -> (Self, Self) {
        (self.clone().sliced(..mid), self.sliced(mid..))
    }

    /// Splits the buffer into two at the given index.
    ///
    /// Returns a buffer containing the elements in the range
    /// `[at, len)`. After the call, self will be left containing
    /// the elements `[0, at)`.
    ///
    /// # Panics
    /// Panics if `at > len`.
    #[must_use]
    pub fn split_off(&mut self, at: usize) -> Self {
        let out = self.clone().sliced(at..);
        self.slice_in_place(..at);
        out
    }

    /// Returns a pointer to the start of the storage underlying this buffer.
    #[inline]
    pub fn storage_ptr(&self) -> *const T {
        self.storage.as_ptr()
    }

    /// Returns the start offset of this buffer within the underlying storage.
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
    /// * has no alive clones
    /// * has not been imported from the C data interface (FFI)
    #[inline]
    pub fn into_mut(mut self) -> Either<Self, Vec<T>> {
        // We lose information if the data is sliced.
        if self.is_sliced() {
            return Either::Left(self);
        }
        match self.storage.try_into_vec() {
            Ok(v) => Either::Right(v),
            Err(slf) => {
                self.storage = slf;
                Either::Left(self)
            },
        }
    }

    /// Returns a mutable reference to its slice, if possible.
    ///
    /// This operation returns [`Some`] iff this [`Buffer`]:
    /// * has no alive clones
    /// * has not been imported from the C data interface (FFI)
    #[inline]
    pub fn get_mut_slice(&mut self) -> Option<&mut [T]> {
        let offset = self.offset();
        let slice = self.storage.try_as_mut_slice()?;
        Some(unsafe { slice.get_unchecked_mut(offset..offset + self.length) })
    }

    /// Since this takes a shared reference to self, beware that others might
    /// increment this after you've checked it's equal to 1.
    pub fn storage_refcount(&self) -> u64 {
        self.storage.refcount()
    }

    /// Whether these two buffers share the exact same data.
    pub fn is_same_buffer(&self, other: &Self) -> bool {
        self.ptr == other.ptr && self.length == other.length
    }
}

impl<T: Pod> Buffer<T> {
    pub fn try_transmute<U: Pod>(mut self) -> Result<Buffer<U>, Self> {
        assert_ne!(size_of::<U>(), 0);
        let ptr = self.ptr as *const U;
        let length = self.length;
        match self.storage.try_transmute() {
            Err(v) => {
                self.storage = v;
                Err(self)
            },
            Ok(storage) => Ok(Buffer {
                storage,
                ptr,
                length: length.checked_mul(size_of::<T>()).expect("overflow") / size_of::<U>(),
            }),
        }
    }
}

impl<T: Clone> Buffer<T> {
    pub fn to_vec(self) -> Vec<T> {
        match self.into_mut() {
            Either::Right(v) => v,
            Either::Left(same) => same.as_slice().to_vec(),
        }
    }
}

#[repr(C, align(4096))]
#[derive(Copy, Clone)]
struct Aligned([u8; 4096]);

// We intentionally leak 8MiB of zeroed memory once so we don't have to
// refcount it.
const GLOBAL_ZERO_SIZE: usize = 8 * 1024 * 1024;
static GLOBAL_ZEROES: LazyLock<SharedStorage<Aligned>> = LazyLock::new(|| {
    assert!(GLOBAL_ZERO_SIZE.is_multiple_of(size_of::<Aligned>()));
    let chunks = GLOBAL_ZERO_SIZE / size_of::<Aligned>();
    let v = vec![Aligned([0; _]); chunks];
    let mut ss = SharedStorage::from_vec(v);
    ss.leak();
    ss
});

impl<T: Zeroable> Buffer<T> {
    pub fn zeroed(length: usize) -> Self {
        let bytes_needed = length * size_of::<T>();
        if align_of::<T>() <= align_of::<Aligned>() && bytes_needed <= GLOBAL_ZERO_SIZE {
            unsafe {
                // SAFETY: we checked the alignment of T, that it fits, and T is zeroable.
                let storage = GLOBAL_ZEROES.clone().transmute_unchecked::<T>();
                let ptr = storage.as_ptr();
                Buffer {
                    storage,
                    ptr,
                    length,
                }
            }
        } else {
            bytemuck::zeroed_vec(length).into()
        }
    }
}

impl<T> From<Vec<T>> for Buffer<T> {
    #[inline]
    fn from(v: Vec<T>) -> Self {
        Self::from_vec(v)
    }
}

impl<T> Deref for Buffer<T> {
    type Target = [T];

    #[inline(always)]
    fn deref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<T> AsRef<[T]> for Buffer<T> {
    #[inline(always)]
    fn as_ref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<T> FromIterator<T> for Buffer<T> {
    #[inline]
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        Vec::from_iter(iter).into()
    }
}

#[cfg(feature = "serde")]
mod _serde_impl {
    use serde::{Deserialize, Serialize};

    use super::Buffer;

    impl<T> Serialize for Buffer<T>
    where
        T: Serialize,
    {
        fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
        where
            S: serde::Serializer,
        {
            <[T] as Serialize>::serialize(self.as_slice(), serializer)
        }
    }

    impl<'de, T> Deserialize<'de> for Buffer<T>
    where
        T: Deserialize<'de>,
    {
        fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
        where
            D: serde::Deserializer<'de>,
        {
            <Vec<T> as Deserialize>::deserialize(deserializer).map(Buffer::from)
        }
    }
}

impl<T: Copy> IntoIterator for Buffer<T> {
    type Item = T;

    type IntoIter = IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        IntoIter::new(self)
    }
}

/// This crates' equivalent of [`std::vec::IntoIter`] for [`Buffer`].
#[derive(Debug, Clone)]
pub struct IntoIter<T: Copy> {
    values: Buffer<T>,
    index: usize,
    end: usize,
}

impl<T: Copy> IntoIter<T> {
    #[inline]
    fn new(values: Buffer<T>) -> Self {
        let end = values.len();
        Self {
            values,
            index: 0,
            end,
        }
    }
}

impl<T: Copy> Iterator for IntoIter<T> {
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.index == self.end {
            return None;
        }
        let old = self.index;
        self.index += 1;
        Some(*unsafe { self.values.get_unchecked(old) })
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.end - self.index, Some(self.end - self.index))
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        let new_index = self.index + n;
        if new_index > self.end {
            self.index = self.end;
            None
        } else {
            self.index = new_index;
            self.next()
        }
    }
}

impl<T: Copy> DoubleEndedIterator for IntoIter<T> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.index == self.end {
            None
        } else {
            self.end -= 1;
            Some(*unsafe { self.values.get_unchecked(self.end) })
        }
    }
}

impl<T: Copy> ExactSizeIterator for IntoIter<T> {}
