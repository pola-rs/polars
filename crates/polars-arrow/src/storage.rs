use std::mem::ManuallyDrop;
use std::ops::Deref;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::ffi::InternalArrowArray;

enum BackingStorage {
    Vec {
        capacity: usize,
    },
    InternalArrowArray(InternalArrowArray),
    #[cfg(feature = "arrow_rs")]
    ArrowBuffer(arrow_buffer::Buffer),
}

struct SharedStorageInner<T> {
    ref_count: AtomicU64,
    ptr: *mut T,
    length: usize,
    backing: Option<BackingStorage>,
}

impl<T> Drop for SharedStorageInner<T> {
    fn drop(&mut self) {
        match self.backing.take() {
            Some(BackingStorage::InternalArrowArray(a)) => drop(a),
            #[cfg(feature = "arrow_rs")]
            Some(BackingStorage::ArrowBuffer(b)) => drop(b),
            Some(BackingStorage::Vec { capacity }) => unsafe {
                drop(Vec::from_raw_parts(self.ptr, self.length, capacity))
            },
            None => {},
        }
    }
}

pub struct SharedStorage<T> {
    inner: NonNull<SharedStorageInner<T>>,
}

unsafe impl<T: Sync + Send> Send for SharedStorage<T> {}
unsafe impl<T: Sync + Send> Sync for SharedStorage<T> {}

impl<T> SharedStorage<T> {
    pub fn from_static(slice: &'static [T]) -> Self {
        let length = slice.len();
        let ptr = slice.as_ptr().cast_mut();
        let inner = SharedStorageInner {
            ref_count: AtomicU64::new(2), // Never used, but 2 so it won't pass exclusivity tests.
            ptr,
            length,
            backing: None,
        };
        Self {
            inner: NonNull::new(Box::into_raw(Box::new(inner))).unwrap(),
        }
    }

    pub fn from_vec(mut v: Vec<T>) -> Self {
        let length = v.len();
        let capacity = v.capacity();
        let ptr = v.as_mut_ptr();
        core::mem::forget(v);
        let inner = SharedStorageInner {
            ref_count: AtomicU64::new(1),
            ptr,
            length,
            backing: Some(BackingStorage::Vec { capacity }),
        };
        Self {
            inner: NonNull::new(Box::into_raw(Box::new(inner))).unwrap(),
        }
    }

    pub fn from_internal_arrow_array(ptr: *const T, len: usize, arr: InternalArrowArray) -> Self {
        let inner = SharedStorageInner {
            ref_count: AtomicU64::new(1),
            ptr: ptr.cast_mut(),
            length: len,
            backing: Some(BackingStorage::InternalArrowArray(arr)),
        };
        Self {
            inner: NonNull::new(Box::into_raw(Box::new(inner))).unwrap(),
        }
    }
}

#[cfg(feature = "arrow_rs")]
impl<T: crate::types::NativeType> SharedStorage<T> {
    pub fn from_arrow_buffer(buffer: arrow_buffer::Buffer) -> Self {
        let ptr = buffer.as_ptr();
        let align_offset = ptr.align_offset(std::mem::align_of::<T>());
        assert_eq!(align_offset, 0, "arrow_buffer::Buffer misaligned");
        let length = buffer.len() / std::mem::size_of::<T>();

        let inner = SharedStorageInner {
            ref_count: AtomicU64::new(1),
            ptr: ptr as *mut T,
            length,
            backing: Some(BackingStorage::ArrowBuffer(buffer)),
        };
        Self {
            inner: NonNull::new(Box::into_raw(Box::new(inner))).unwrap(),
        }
    }

    pub fn into_arrow_buffer(self) -> arrow_buffer::Buffer {
        let ptr = NonNull::new(self.as_ptr() as *mut u8).unwrap();
        let len = self.len() * std::mem::size_of::<T>();
        let arc = std::sync::Arc::new(self);
        unsafe { arrow_buffer::Buffer::from_custom_allocation(ptr, len, arc) }
    }
}

impl<T> SharedStorage<T> {
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.inner().length
    }

    #[inline(always)]
    pub fn as_ptr(&self) -> *const T {
        self.inner().ptr
    }

    #[inline(always)]
    pub fn is_exclusive(&mut self) -> bool {
        // Ordering semantics copied from Arc<T>.
        self.inner().ref_count.load(Ordering::Acquire) == 1
    }

    /// Gets the reference count of this storage.
    ///
    /// Because this function takes a shared reference this should not be used
    /// in cases where we are checking if the refcount is one for safety,
    /// someone else could increment it in the meantime.
    #[inline(always)]
    pub fn refcount(&self) -> u64 {
        // Ordering semantics copied from Arc<T>.
        self.inner().ref_count.load(Ordering::Acquire)
    }

    pub fn try_as_mut_slice(&mut self) -> Option<&mut [T]> {
        self.is_exclusive().then(|| {
            let inner = self.inner();
            unsafe { core::slice::from_raw_parts_mut(inner.ptr, inner.length) }
        })
    }

    pub fn try_into_vec(mut self) -> Result<Vec<T>, Self> {
        let Some(BackingStorage::Vec { capacity }) = self.inner().backing else {
            return Err(self);
        };
        if self.is_exclusive() {
            let slf = ManuallyDrop::new(self);
            let inner = slf.inner();
            Ok(unsafe { Vec::from_raw_parts(inner.ptr, inner.length, capacity) })
        } else {
            Err(self)
        }
    }

    #[inline(always)]
    fn inner(&self) -> &SharedStorageInner<T> {
        unsafe { &*self.inner.as_ptr() }
    }

    /// # Safety
    /// May only be called once.
    #[cold]
    unsafe fn drop_slow(&mut self) {
        unsafe { drop(Box::from_raw(self.inner.as_ptr())) }
    }
}

impl<T> Deref for SharedStorage<T> {
    type Target = [T];

    #[inline]
    fn deref(&self) -> &Self::Target {
        unsafe {
            let inner = self.inner();
            core::slice::from_raw_parts(inner.ptr, inner.length)
        }
    }
}

impl<T> Clone for SharedStorage<T> {
    fn clone(&self) -> Self {
        let inner = self.inner();
        if inner.backing.is_some() {
            // Ordering semantics copied from Arc<T>.
            inner.ref_count.fetch_add(1, Ordering::Relaxed);
        }
        Self { inner: self.inner }
    }
}

impl<T> Drop for SharedStorage<T> {
    fn drop(&mut self) {
        let inner = self.inner();
        if inner.backing.is_none() {
            return;
        }

        // Ordering semantics copied from Arc<T>.
        if inner.ref_count.fetch_sub(1, Ordering::Release) == 1 {
            std::sync::atomic::fence(Ordering::Acquire);
            unsafe {
                self.drop_slow();
            }
        }
    }
}
