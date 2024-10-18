use std::marker::PhantomData;
use std::mem::ManuallyDrop;
use std::ops::Deref;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicU64, Ordering};

use bytemuck::Pod;

use crate::ffi::InternalArrowArray;
use crate::types::{
    AlignedBytes, Bytes12Alignment4, Bytes16Alignment16, Bytes16Alignment4, Bytes16Alignment8,
    Bytes1Alignment1, Bytes2Alignment2, Bytes32Alignment16, Bytes4Alignment4, Bytes8Alignment4,
    Bytes8Alignment8, NativeSizeAlignment,
};

enum BackingStorage {
    Vec {
        capacity: usize,

        /// Size and alignment of the original vector type.
        ///
        /// We have the following invariants:
        /// - if this is Some(...) then all alignments involved are a power of 2
        /// - align_of(Original) >= align_of(Current)
        /// - size_of(Original) >= size_of(Current)
        /// - size_of(Original) % size_of(Current) == 0
        original_element_size_alignment: Option<NativeSizeAlignment>,
    },
    InternalArrowArray(InternalArrowArray),
}

struct SharedStorageInner<T> {
    ref_count: AtomicU64,
    ptr: *mut T,
    length: usize,
    backing: Option<BackingStorage>,
    // https://github.com/rust-lang/rfcs/blob/master/text/0769-sound-generic-drop.md#phantom-data
    phantom: PhantomData<T>,
}

impl<T> Drop for SharedStorageInner<T> {
    fn drop(&mut self) {
        match self.backing.take() {
            Some(BackingStorage::InternalArrowArray(a)) => drop(a),
            Some(BackingStorage::Vec {
                capacity,
                original_element_size_alignment,
            }) => {
                #[inline]
                unsafe fn drop_vec<T, O>(ptr: *mut T, length: usize, capacity: usize) {
                    let ptr = ptr.cast::<O>();
                    debug_assert!(ptr.is_aligned());

                    debug_assert!(size_of::<O>() >= size_of::<T>());
                    debug_assert_eq!(size_of::<O>() % size_of::<T>(), 0);

                    let scale_factor = size_of::<O>() / size_of::<T>();

                    // If the original element had a different size_of we need to rescale the
                    // length and capacity here.
                    let length = length / scale_factor;
                    let capacity = capacity / scale_factor;

                    // SAFETY:
                    // - The BackingStorage holds an invariants that make this safe
                    drop(unsafe { Vec::from_raw_parts(ptr, length, capacity) });
                }

                let ptr = self.ptr;
                let length = self.length;

                let Some(size_alignment) = original_element_size_alignment else {
                    unsafe { drop_vec::<T, T>(ptr, length, capacity) };
                    return;
                };

                use NativeSizeAlignment as SA;
                unsafe {
                    match size_alignment {
                        SA::S1A1 => drop_vec::<T, Bytes1Alignment1>(ptr, length, capacity),
                        SA::S2A2 => drop_vec::<T, Bytes2Alignment2>(ptr, length, capacity),
                        SA::S4A4 => drop_vec::<T, Bytes4Alignment4>(ptr, length, capacity),
                        SA::S8A4 => drop_vec::<T, Bytes8Alignment4>(ptr, length, capacity),
                        SA::S8A8 => drop_vec::<T, Bytes8Alignment8>(ptr, length, capacity),
                        SA::S12A4 => drop_vec::<T, Bytes12Alignment4>(ptr, length, capacity),
                        SA::S16A4 => drop_vec::<T, Bytes16Alignment4>(ptr, length, capacity),
                        SA::S16A8 => drop_vec::<T, Bytes16Alignment8>(ptr, length, capacity),
                        SA::S16A16 => drop_vec::<T, Bytes16Alignment16>(ptr, length, capacity),
                        SA::S32A16 => drop_vec::<T, Bytes32Alignment16>(ptr, length, capacity),
                    }
                }
            },
            None => {},
        }
    }
}

pub struct SharedStorage<T> {
    inner: NonNull<SharedStorageInner<T>>,
    phantom: PhantomData<SharedStorageInner<T>>,
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
            phantom: PhantomData,
        };
        Self {
            inner: NonNull::new(Box::into_raw(Box::new(inner))).unwrap(),
            phantom: PhantomData,
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
            backing: Some(BackingStorage::Vec {
                capacity,
                original_element_size_alignment: None,
            }),
            phantom: PhantomData,
        };
        Self {
            inner: NonNull::new(Box::into_raw(Box::new(inner))).unwrap(),
            phantom: PhantomData,
        }
    }

    pub fn from_internal_arrow_array(ptr: *const T, len: usize, arr: InternalArrowArray) -> Self {
        let inner = SharedStorageInner {
            ref_count: AtomicU64::new(1),
            ptr: ptr.cast_mut(),
            length: len,
            backing: Some(BackingStorage::InternalArrowArray(arr)),
            phantom: PhantomData,
        };
        Self {
            inner: NonNull::new(Box::into_raw(Box::new(inner))).unwrap(),
            phantom: PhantomData,
        }
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
        let Some(BackingStorage::Vec {
            capacity,
            original_element_size_alignment: None,
        }) = self.inner().backing
        else {
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

impl<T: Pod> SharedStorage<T> {
    /// Create a [`SharedStorage`] from a [`Vec`] of [`AlignedBytes`].
    ///
    /// This will fail if the size and alignment requirements of `T` are stricter than `B`.
    pub fn from_aligned_bytes_vec<B: AlignedBytes>(mut v: Vec<B>) -> Option<Self> {
        if align_of::<B>() < align_of::<T>() {
            return None;
        }

        // @NOTE: This is not a fundamental limitation, but something we impose for now. This makes
        // calculating the capacity a lot easier.
        if size_of::<B>() < size_of::<T>() || size_of::<B>() % size_of::<T>() != 0 {
            return None;
        }

        let scale_factor = size_of::<B>() / size_of::<T>();

        let length = v.len() * scale_factor;
        let capacity = v.capacity() * scale_factor;
        let ptr = v.as_mut_ptr().cast::<T>();
        core::mem::forget(v);

        let inner = SharedStorageInner {
            ref_count: AtomicU64::new(1),
            ptr,
            length,
            backing: Some(BackingStorage::Vec {
                capacity,
                original_element_size_alignment: Some(B::SIZE_ALIGNMENT_PAIR),
            }),
            phantom: PhantomData,
        };

        Some(Self {
            inner: NonNull::new(Box::into_raw(Box::new(inner))).unwrap(),
            phantom: PhantomData,
        })
    }
}

impl SharedStorage<u8> {
    /// Create a [`SharedStorage<u8>`][SharedStorage] from a [`Vec`] of [`AlignedBytes`].
    ///
    /// This will never fail since `u8` has unit size and alignment.
    pub fn bytes_from_aligned_bytes_vec<B: AlignedBytes>(v: Vec<B>) -> Self {
        Self::from_aligned_bytes_vec(v).unwrap()
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
        Self {
            inner: self.inner,
            phantom: PhantomData,
        }
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
