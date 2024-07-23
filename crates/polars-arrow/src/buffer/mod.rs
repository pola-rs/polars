//! Contains [`Buffer`], an immutable container for all Arrow physical types (e.g. i32, f64).

mod immutable;
mod iterator;

use std::ops::Deref;

use crate::ffi::InternalArrowArray;

pub(crate) enum BytesAllocator {
    // Dead code lint is a false positive.
    // remove once fixed in rustc
    #[allow(dead_code)]
    InternalArrowArray(InternalArrowArray),

    #[cfg(feature = "arrow_rs")]
    // Dead code lint is a false positive.
    // remove once fixed in rustc
    #[allow(dead_code)]
    Arrow(arrow_buffer::Buffer),
}
pub(crate) type BytesInner<T> = polars_utils::foreign_vec::ForeignVec<BytesAllocator, T>;

/// Bytes representation.
#[repr(transparent)]
pub struct Bytes<T>(BytesInner<T>);

impl<T> Bytes<T> {
    /// Takes ownership of an allocated memory region.
    /// # Panics
    /// This function panics if and only if pointer is not null
    ///
    /// # Safety
    /// This function is safe if and only if `ptr` is valid for `length`
    /// # Implementation
    /// This function leaks if and only if `owner` does not deallocate
    /// the region `[ptr, ptr+length[` when dropped.
    #[inline]
    pub(crate) unsafe fn from_foreign(ptr: *const T, length: usize, owner: BytesAllocator) -> Self {
        Self(BytesInner::from_foreign(ptr, length, owner))
    }

    /// Returns a `Some` mutable reference of [`Vec<T>`] iff this was initialized
    /// from a [`Vec<T>`] and `None` otherwise.
    #[inline]
    pub(crate) fn get_vec(&mut self) -> Option<&mut Vec<T>> {
        self.0.get_vec()
    }
}

impl<T> Deref for Bytes<T> {
    type Target = [T];

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T> From<Vec<T>> for Bytes<T> {
    #[inline]
    fn from(data: Vec<T>) -> Self {
        let inner: BytesInner<T> = data.into();
        Bytes(inner)
    }
}

impl<T> From<BytesInner<T>> for Bytes<T> {
    #[inline]
    fn from(value: BytesInner<T>) -> Self {
        Self(value)
    }
}

#[cfg(feature = "arrow_rs")]
pub(crate) fn to_buffer<T: crate::types::NativeType>(
    value: std::sync::Arc<Bytes<T>>,
) -> arrow_buffer::Buffer {
    // This should never panic as ForeignVec pointer must be non-null
    let ptr = std::ptr::NonNull::new(value.as_ptr() as _).unwrap();
    let len = value.len() * std::mem::size_of::<T>();
    // SAFETY: allocation is guaranteed to be valid for `len` bytes
    unsafe { arrow_buffer::Buffer::from_custom_allocation(ptr, len, value) }
}

#[cfg(feature = "arrow_rs")]
pub(crate) fn to_bytes<T: crate::types::NativeType>(value: arrow_buffer::Buffer) -> Bytes<T> {
    let ptr = value.as_ptr();
    let align = ptr.align_offset(std::mem::align_of::<T>());
    assert_eq!(align, 0, "not aligned");
    let len = value.len() / std::mem::size_of::<T>();

    // Valid as `NativeType: Pod` and checked alignment above
    let ptr = value.as_ptr() as *const T;

    let owner = crate::buffer::BytesAllocator::Arrow(value);

    // SAFETY: slice is valid for len elements of T
    unsafe { Bytes::from_foreign(ptr, len, owner) }
}

pub use immutable::Buffer;
pub(super) use iterator::IntoIter;
