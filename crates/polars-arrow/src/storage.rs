use std::marker::PhantomData;
use std::mem::ManuallyDrop;
use std::ops::{Deref, DerefMut};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicU64, Ordering};

use bytemuck::Pod;

// Allows us to transmute between types while also keeping the original
// stats and drop method of the Vec around.
struct VecVTable {
    size: usize,
    align: usize,
    drop_buffer: unsafe fn(*mut (), usize),
}

impl VecVTable {
    const fn new<T>() -> Self {
        unsafe fn drop_buffer<T>(ptr: *mut (), cap: usize) {
            unsafe { drop(Vec::from_raw_parts(ptr.cast::<T>(), 0, cap)) }
        }

        Self {
            size: size_of::<T>(),
            align: align_of::<T>(),
            drop_buffer: drop_buffer::<T>,
        }
    }

    fn new_static<T>() -> &'static Self {
        const { &Self::new::<T>() }
    }
}

use crate::ffi::InternalArrowArray;

enum BackingStorage {
    Vec {
        original_capacity: usize, // Elements, not bytes.
        vtable: &'static VecVTable,
    },
    InternalArrowArray(InternalArrowArray),
}

struct SharedStorageInner<T> {
    ref_count: AtomicU64,
    ptr: *mut T,
    length_in_bytes: usize,
    backing: Option<BackingStorage>,
    // https://github.com/rust-lang/rfcs/blob/master/text/0769-sound-generic-drop.md#phantom-data
    phantom: PhantomData<T>,
}

impl<T> SharedStorageInner<T> {
    pub fn from_vec(mut v: Vec<T>) -> Self {
        let length_in_bytes = v.len() * size_of::<T>();
        let original_capacity = v.capacity();
        let ptr = v.as_mut_ptr();
        core::mem::forget(v);
        Self {
            ref_count: AtomicU64::new(1),
            ptr,
            length_in_bytes,
            backing: Some(BackingStorage::Vec {
                original_capacity,
                vtable: VecVTable::new_static::<T>(),
            }),
            phantom: PhantomData,
        }
    }
}

impl<T> Drop for SharedStorageInner<T> {
    fn drop(&mut self) {
        match self.backing.take() {
            Some(BackingStorage::InternalArrowArray(a)) => drop(a),
            Some(BackingStorage::Vec {
                original_capacity,
                vtable,
            }) => unsafe {
                // Drop the elements in our slice.
                if std::mem::needs_drop::<T>() {
                    core::ptr::drop_in_place(core::ptr::slice_from_raw_parts_mut(
                        self.ptr,
                        self.length_in_bytes / size_of::<T>(),
                    ));
                }

                // Free the buffer.
                (vtable.drop_buffer)(self.ptr.cast(), original_capacity);
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
        #[expect(clippy::manual_slice_size_calculation)]
        let length_in_bytes = slice.len() * size_of::<T>();
        let ptr = slice.as_ptr().cast_mut();
        let inner = SharedStorageInner {
            ref_count: AtomicU64::new(2), // Never used, but 2 so it won't pass exclusivity tests.
            ptr,
            length_in_bytes,
            backing: None,
            phantom: PhantomData,
        };
        Self {
            inner: NonNull::new(Box::into_raw(Box::new(inner))).unwrap(),
            phantom: PhantomData,
        }
    }

    pub fn from_vec(v: Vec<T>) -> Self {
        Self {
            inner: NonNull::new(Box::into_raw(Box::new(SharedStorageInner::from_vec(v)))).unwrap(),
            phantom: PhantomData,
        }
    }

    pub fn from_internal_arrow_array(ptr: *const T, len: usize, arr: InternalArrowArray) -> Self {
        let inner = SharedStorageInner {
            ref_count: AtomicU64::new(1),
            ptr: ptr.cast_mut(),
            length_in_bytes: len * size_of::<T>(),
            backing: Some(BackingStorage::InternalArrowArray(arr)),
            phantom: PhantomData,
        };
        Self {
            inner: NonNull::new(Box::into_raw(Box::new(inner))).unwrap(),
            phantom: PhantomData,
        }
    }
}

pub struct SharedStorageAsVecMut<'a, T> {
    ss: &'a mut SharedStorage<T>,
    vec: ManuallyDrop<Vec<T>>,
}

impl<'a, T> Deref for SharedStorageAsVecMut<'a, T> {
    type Target = Vec<T>;

    fn deref(&self) -> &Self::Target {
        &self.vec
    }
}

impl<'a, T> DerefMut for SharedStorageAsVecMut<'a, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.vec
    }
}

impl<'a, T> Drop for SharedStorageAsVecMut<'a, T> {
    fn drop(&mut self) {
        unsafe {
            // Restore the SharedStorage.
            let vec = ManuallyDrop::take(&mut self.vec);
            let inner = self.ss.inner.as_ptr();
            inner.write(SharedStorageInner::from_vec(vec));
        }
    }
}

impl<T> SharedStorage<T> {
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.inner().length_in_bytes / size_of::<T>()
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
            let len = inner.length_in_bytes / size_of::<T>();
            unsafe { core::slice::from_raw_parts_mut(inner.ptr, len) }
        })
    }

    /// Try to take the vec backing this SharedStorage, leaving this as an empty slice.
    pub fn try_take_vec(&mut self) -> Option<Vec<T>> {
        // We may only go back to a Vec if we originally came from a Vec
        // where the desired size/align matches the original.
        let Some(BackingStorage::Vec {
            original_capacity,
            vtable,
        }) = self.inner().backing
        else {
            return None;
        };

        if vtable.size != size_of::<T>() || vtable.align != align_of::<T>() {
            return None;
        }

        // If there are other references we can't get an exclusive reference.
        if !self.is_exclusive() {
            return None;
        }

        let ret;
        unsafe {
            let inner = &mut *self.inner.as_ptr();
            let len = inner.length_in_bytes / size_of::<T>();
            ret = Vec::from_raw_parts(inner.ptr, len, original_capacity);
            inner.length_in_bytes = 0;
            inner.backing = None;
        }
        Some(ret)
    }

    /// Attempts to call the given function with this SharedStorage as a
    /// reference to a mutable Vec. If this SharedStorage can't be converted to
    /// a Vec the function is not called and instead returned as an error.
    pub fn try_as_mut_vec(&mut self) -> Option<SharedStorageAsVecMut<'_, T>> {
        Some(SharedStorageAsVecMut {
            vec: ManuallyDrop::new(self.try_take_vec()?),
            ss: self,
        })
    }

    pub fn try_into_vec(mut self) -> Result<Vec<T>, Self> {
        self.try_take_vec().ok_or(self)
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
    fn try_transmute<U: Pod>(self) -> Result<SharedStorage<U>, Self> {
        let inner = self.inner();

        // The length of the array in bytes must be a multiple of the target size.
        // We can skip this check if the size of U divides the size of T.
        if size_of::<T>() % size_of::<U>() != 0 && inner.length_in_bytes % size_of::<U>() != 0 {
            return Err(self);
        }

        // The pointer must be properly aligned for U.
        // We can skip this check if the alignment of U divides the alignment of T.
        if align_of::<T>() % align_of::<U>() != 0 && !inner.ptr.cast::<U>().is_aligned() {
            return Err(self);
        }

        Ok(SharedStorage {
            inner: self.inner.cast(),
            phantom: PhantomData,
        })
    }
}

impl SharedStorage<u8> {
    /// Create a [`SharedStorage<u8>`][SharedStorage] from a [`Vec`] of [`Pod`].
    pub fn bytes_from_pod_vec<T: Pod>(v: Vec<T>) -> Self {
        // This can't fail, bytes is compatible with everything.
        SharedStorage::from_vec(v)
            .try_transmute::<u8>()
            .unwrap_or_else(|_| unreachable!())
    }
}

impl<T> Deref for SharedStorage<T> {
    type Target = [T];

    #[inline]
    fn deref(&self) -> &Self::Target {
        unsafe {
            let inner = self.inner();
            let len = inner.length_in_bytes / size_of::<T>();
            core::slice::from_raw_parts(inner.ptr, len)
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
