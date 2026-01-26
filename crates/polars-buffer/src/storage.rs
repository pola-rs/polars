use std::any::Any;
use std::marker::PhantomData;
use std::mem::ManuallyDrop;
use std::ops::{Deref, DerefMut};
use std::process::abort;
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

enum BackingStorage {
    Vec {
        original_capacity: usize, // Elements, not bytes.
        vtable: &'static VecVTable,
    },
    ForeignOwner(Box<dyn Any + Send + 'static>),

    /// Backed by some external method which we do not need to take care of,
    /// but we still should refcount and drop the SharedStorageInner.
    External,

    /// Both the backing storage and the SharedStorageInner are leaked, no
    /// refcounting is done. This technically should be a flag on
    /// SharedStorageInner instead of being here, but that would add 8 more
    /// bytes to SharedStorageInner, so here it is.
    Leaked,
}

struct SharedStorageInner<T> {
    ref_count: AtomicU64,
    ptr: *mut T,
    length_in_bytes: usize,
    backing: BackingStorage,
    // https://github.com/rust-lang/rfcs/blob/master/text/0769-sound-generic-drop.md#phantom-data
    phantom: PhantomData<T>,
}

unsafe impl<T: Sync + Send> Sync for SharedStorageInner<T> {}

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
            backing: BackingStorage::Vec {
                original_capacity,
                vtable: VecVTable::new_static::<T>(),
            },
            phantom: PhantomData,
        }
    }
}

impl<T> Drop for SharedStorageInner<T> {
    fn drop(&mut self) {
        match core::mem::replace(&mut self.backing, BackingStorage::External) {
            BackingStorage::ForeignOwner(o) => drop(o),
            BackingStorage::Vec {
                original_capacity,
                vtable,
            } => unsafe {
                // Drop the elements in our slice.
                if std::mem::needs_drop::<T>() {
                    core::ptr::drop_in_place(core::ptr::slice_from_raw_parts_mut(
                        self.ptr,
                        self.length_in_bytes / size_of::<T>(),
                    ));
                }

                // Free the buffer.
                if original_capacity > 0 {
                    (vtable.drop_buffer)(self.ptr.cast(), original_capacity);
                }
            },
            BackingStorage::External | BackingStorage::Leaked => {},
        }
    }
}

#[repr(transparent)]
pub struct SharedStorage<T> {
    inner: NonNull<SharedStorageInner<T>>,
    phantom: PhantomData<SharedStorageInner<T>>,
}

unsafe impl<T: Sync + Send> Send for SharedStorage<T> {}
unsafe impl<T: Sync + Send> Sync for SharedStorage<T> {}

impl<T> Default for SharedStorage<T> {
    fn default() -> Self {
        Self::empty()
    }
}

impl<T> SharedStorage<T> {
    /// Creates an empty SharedStorage.
    pub const fn empty() -> Self {
        assert!(align_of::<T>() <= 1 << 30);
        static INNER: SharedStorageInner<()> = SharedStorageInner {
            ref_count: AtomicU64::new(1),
            ptr: core::ptr::without_provenance_mut(1 << 30), // Very overaligned for any T.
            length_in_bytes: 0,
            backing: BackingStorage::Leaked,
            phantom: PhantomData,
        };

        Self {
            inner: NonNull::new(&raw const INNER as *mut SharedStorageInner<T>).unwrap(),
            phantom: PhantomData,
        }
    }

    /// Creates a SharedStorage backed by this static slice.
    pub fn from_static(slice: &'static [T]) -> Self {
        // SAFETY: the slice has a static lifetime.
        unsafe { Self::from_slice_unchecked(slice) }
    }

    /// Creates a SharedStorage backed by this slice.
    ///
    /// # Safety
    /// You must ensure this SharedStorage or any of its clones does not outlive
    /// this slice.
    pub unsafe fn from_slice_unchecked(slice: &[T]) -> Self {
        #[expect(clippy::manual_slice_size_calculation)]
        let length_in_bytes = slice.len() * size_of::<T>();
        let ptr = slice.as_ptr().cast_mut();
        let inner = SharedStorageInner {
            ref_count: AtomicU64::new(1),
            ptr,
            length_in_bytes,
            backing: BackingStorage::External,
            phantom: PhantomData,
        };
        Self {
            inner: NonNull::new(Box::into_raw(Box::new(inner))).unwrap(),
            phantom: PhantomData,
        }
    }

    /// Calls f with a `SharedStorage` backed by this slice.
    ///
    /// Aborts if any clones of the SharedStorage still live when `f` returns.
    pub fn with_slice<R, F: FnOnce(SharedStorage<T>) -> R>(slice: &[T], f: F) -> R {
        struct AbortIfNotExclusive<T>(SharedStorage<T>);
        impl<T> Drop for AbortIfNotExclusive<T> {
            fn drop(&mut self) {
                if !self.0.is_exclusive() {
                    abort()
                }
            }
        }

        unsafe {
            let ss = AbortIfNotExclusive(Self::from_slice_unchecked(slice));
            f(ss.0.clone())
        }
    }

    /// Calls f with a `SharedStorage` backed by this vec.
    ///
    /// # Panics
    /// Panics if any clones of the SharedStorage still live when `f` returns.
    pub fn with_vec<R, F: FnOnce(SharedStorage<T>) -> R>(vec: &mut Vec<T>, f: F) -> R {
        // TODO: this function is intended to allow exclusive conversion back to
        // a vec, but we need some kind of weak reference for this (that is, two
        // tiers of 'is_exclusive', one for access and one for keeping the inner
        // state alive).
        struct RestoreVec<'a, T>(&'a mut Vec<T>, SharedStorage<T>);
        impl<'a, T> Drop for RestoreVec<'a, T> {
            fn drop(&mut self) {
                *self.0 = self.1.try_take_vec().unwrap();
            }
        }

        let tmp = core::mem::take(vec);
        let ss = RestoreVec(vec, Self::from_vec(tmp));
        f(ss.1.clone())
    }

    /// # Safety
    /// The slice must be valid as long as owner lives.
    pub unsafe fn from_slice_with_owner<O: Send + 'static>(slice: &[T], owner: O) -> Self {
        #[expect(clippy::manual_slice_size_calculation)]
        let length_in_bytes = slice.len() * size_of::<T>();
        let ptr = slice.as_ptr().cast_mut();
        let inner = SharedStorageInner {
            ref_count: AtomicU64::new(1),
            ptr,
            length_in_bytes,
            backing: BackingStorage::ForeignOwner(Box::new(owner)),
            phantom: PhantomData,
        };
        Self {
            inner: NonNull::new(Box::into_raw(Box::new(inner))).unwrap(),
            phantom: PhantomData,
        }
    }

    pub fn from_owner<O: Send + AsRef<[T]> + 'static>(owner: O) -> Self {
        let owner = Box::new(owner);
        let slice: &[T] = (*owner).as_ref();
        #[expect(clippy::manual_slice_size_calculation)]
        let length_in_bytes = slice.len() * size_of::<T>();
        let ptr = slice.as_ptr().cast_mut();
        let inner = SharedStorageInner {
            ref_count: AtomicU64::new(1),
            ptr,
            length_in_bytes,
            backing: BackingStorage::ForeignOwner(owner),
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

    /// Leaks this SharedStorage such that it and its inner value is never
    /// dropped. In return no refcounting needs to be performed.
    ///
    /// The SharedStorage must be exclusive.
    pub fn leak(&mut self) {
        assert!(self.is_exclusive());
        unsafe {
            let inner = &mut *self.inner.as_ptr();
            core::mem::forget(core::mem::replace(
                &mut inner.backing,
                BackingStorage::Leaked,
            ));
        }
    }

    /// # Safety
    /// The caller is responsible for ensuring the resulting slice is valid and aligned for U.
    pub unsafe fn transmute_unchecked<U>(self) -> SharedStorage<U> {
        let storage = SharedStorage {
            inner: self.inner.cast(),
            phantom: PhantomData,
        };
        std::mem::forget(self);
        storage
    }
}

pub struct SharedStorageAsVecMut<'a, T> {
    ss: &'a mut SharedStorage<T>,
    vec: ManuallyDrop<Vec<T>>,
}

impl<T> Deref for SharedStorageAsVecMut<'_, T> {
    type Target = Vec<T>;

    fn deref(&self) -> &Self::Target {
        &self.vec
    }
}

impl<T> DerefMut for SharedStorageAsVecMut<'_, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.vec
    }
}

impl<T> Drop for SharedStorageAsVecMut<'_, T> {
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
    pub const fn len(&self) -> usize {
        self.inner().length_in_bytes / size_of::<T>()
    }

    #[inline(always)]
    pub const fn is_empty(&self) -> bool {
        self.inner().length_in_bytes == 0
    }

    #[inline(always)]
    pub const fn as_ptr(&self) -> *const T {
        self.inner().ptr
    }

    #[inline(always)]
    pub fn is_exclusive(&self) -> bool {
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
        // We don't know if what we're created from may be mutated unless we're
        // backed by an exclusive Vec. Perhaps in the future we can add a
        // mutability bit?
        let inner = self.inner();
        let may_mut = inner.ref_count.load(Ordering::Acquire) == 1
            && matches!(inner.backing, BackingStorage::Vec { .. });
        may_mut.then(|| {
            let inner = self.inner();
            let len = inner.length_in_bytes / size_of::<T>();
            unsafe { core::slice::from_raw_parts_mut(inner.ptr, len) }
        })
    }

    /// Try to take the vec backing this SharedStorage, leaving this as an empty slice.
    pub fn try_take_vec(&mut self) -> Option<Vec<T>> {
        // If there are other references we can't get an exclusive reference.
        if !self.is_exclusive() {
            return None;
        }

        let ret;
        unsafe {
            let inner = &mut *self.inner.as_ptr();

            // We may only go back to a Vec if we originally came from a Vec
            // where the desired size/align matches the original.
            let BackingStorage::Vec {
                original_capacity,
                vtable,
            } = &mut inner.backing
            else {
                return None;
            };

            if vtable.size != size_of::<T>() || vtable.align != align_of::<T>() {
                return None;
            }

            // Steal vec from inner.
            let len = inner.length_in_bytes / size_of::<T>();
            ret = Vec::from_raw_parts(inner.ptr, len, *original_capacity);
            *original_capacity = 0;
            inner.length_in_bytes = 0;
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
    const fn inner(&self) -> &SharedStorageInner<T> {
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
    pub fn try_transmute<U: Pod>(self) -> Result<SharedStorage<U>, Self> {
        let inner = self.inner();

        // The length of the array in bytes must be a multiple of the target size.
        // We can skip this check if the size of U divides the size of T.
        if !size_of::<T>().is_multiple_of(size_of::<U>())
            && !inner.length_in_bytes.is_multiple_of(size_of::<U>())
        {
            return Err(self);
        }

        // The pointer must be properly aligned for U.
        // We can skip this check if the alignment of U divides the alignment of T.
        if !align_of::<T>().is_multiple_of(align_of::<U>()) && !inner.ptr.cast::<U>().is_aligned() {
            return Err(self);
        }

        Ok(unsafe { self.transmute_unchecked::<U>() })
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
        if !matches!(inner.backing, BackingStorage::Leaked) {
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
        if matches!(inner.backing, BackingStorage::Leaked) {
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
