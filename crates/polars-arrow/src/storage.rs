use std::hint::unreachable_unchecked;
use std::marker::PhantomData;
use std::mem::ManuallyDrop;
use std::ops::{Deref, DerefMut};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

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
        initialized_bytes: AtomicUsize,
        /// Elements, not bytes.
        original_capacity: usize,
        vtable: &'static VecVTable,
    },
    InternalArrowArray(InternalArrowArray),

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
            backing: BackingStorage::Vec {
                initialized_bytes: AtomicUsize::new(length_in_bytes),
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
            BackingStorage::InternalArrowArray(a) => drop(a),
            BackingStorage::Vec {
                initialized_bytes,
                original_capacity,
                vtable,
            } => unsafe {
                // Drop the elements in our slice.
                if std::mem::needs_drop::<T>() {
                    core::ptr::drop_in_place(core::ptr::slice_from_raw_parts_mut(
                        self.ptr,
                        initialized_bytes.load(Ordering::Acquire) / size_of::<T>(),
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

pub struct SharedStorage<T> {
    inner: NonNull<SharedStorageInner<T>>,
    length_in_bytes: usize,
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
    const fn empty() -> Self {
        assert!(align_of::<T>() <= 1 << 30);
        static INNER: SharedStorageInner<()> = SharedStorageInner {
            ref_count: AtomicU64::new(1),
            ptr: core::ptr::without_provenance_mut(1 << 30), // Very overaligned for any T.
            backing: BackingStorage::Leaked,
            phantom: PhantomData,
        };

        Self {
            inner: NonNull::new(&raw const INNER as *mut SharedStorageInner<T>).unwrap(),
            length_in_bytes: 0,
            phantom: PhantomData,
        }
    }

    pub fn from_static(slice: &'static [T]) -> Self {
        #[expect(clippy::manual_slice_size_calculation)]
        let length_in_bytes = slice.len() * size_of::<T>();
        let ptr = slice.as_ptr().cast_mut();
        let inner = SharedStorageInner {
            ref_count: AtomicU64::new(1),
            ptr,
            backing: BackingStorage::External,
            phantom: PhantomData,
        };
        Self {
            inner: NonNull::new(Box::into_raw(Box::new(inner))).unwrap(),
            length_in_bytes,
            phantom: PhantomData,
        }
    }

    pub fn from_vec(v: Vec<T>) -> Self {
        let length_in_bytes = v.len() * size_of::<T>();
        Self {
            inner: NonNull::new(Box::into_raw(Box::new(SharedStorageInner::from_vec(v)))).unwrap(),
            length_in_bytes,
            phantom: PhantomData,
        }
    }

    /// # Safety
    /// The range [ptr, ptr+len) needs to be valid and aligned for T.
    /// ptr may not be null.
    pub unsafe fn from_internal_arrow_array(
        ptr: *const T,
        len: usize,
        arr: InternalArrowArray,
    ) -> Self {
        assert!(!ptr.is_null() && ptr.is_aligned());
        let inner = SharedStorageInner {
            ref_count: AtomicU64::new(1),
            ptr: ptr.cast_mut(),
            backing: BackingStorage::InternalArrowArray(arr),
            phantom: PhantomData,
        };
        Self {
            inner: NonNull::new(Box::into_raw(Box::new(inner))).unwrap(),
            length_in_bytes: len * size_of::<T>(),
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
    pub fn len(&self) -> usize {
        self.length_in_bytes / size_of::<T>()
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
            let len = self.length_in_bytes / size_of::<T>();
            unsafe { core::slice::from_raw_parts_mut(inner.ptr, len) }
        })
    }

    /// Try to take the vec backing this SharedStorage, leaving this as an empty slice.
    pub fn try_take_vec(&mut self) -> Option<Vec<T>> {
        // If there are other references we can't get an exclusive reference.
        if !self.is_exclusive() {
            return None;
        }

        let mut ret: Vec<T>;
        unsafe {
            let inner = &mut *self.inner.as_ptr();

            // We may only go back to a Vec if we originally came from a Vec
            // where the desired size/align matches the original.
            let BackingStorage::Vec {
                initialized_bytes,
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
            let initialized_len = initialized_bytes.load(Ordering::Acquire) / size_of::<T>();
            let visible_len = self.length_in_bytes / size_of::<T>();
            ret = Vec::from_raw_parts(inner.ptr, initialized_len, *original_capacity);
            ret.truncate(visible_len);
            *original_capacity = 0;
            self.length_in_bytes = 0;
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

impl<T> SharedStorage<T>
where
    T: Clone,
{
    pub fn try_into_extendable(self) -> Option<ExtendableSharedStorage<T>> {
        ExtendableSharedStorage::try_new(self)
    }
}

impl<T: Pod> SharedStorage<T> {
    pub fn try_transmute<U: Pod>(self) -> Result<SharedStorage<U>, Self> {
        let inner = self.inner();

        // The length of the array in bytes must be a multiple of the target size.
        // We can skip this check if the size of U divides the size of T.
        if !size_of::<T>().is_multiple_of(size_of::<U>())
            && !self.length_in_bytes.is_multiple_of(size_of::<U>())
        {
            return Err(self);
        }

        // The pointer must be properly aligned for U.
        // We can skip this check if the alignment of U divides the alignment of T.
        if !align_of::<T>().is_multiple_of(align_of::<U>()) && !inner.ptr.cast::<U>().is_aligned() {
            return Err(self);
        }

        let storage = SharedStorage {
            inner: self.inner.cast(),
            length_in_bytes: self.length_in_bytes,
            phantom: PhantomData,
        };
        std::mem::forget(self);
        Ok(storage)
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
            let len = self.length_in_bytes / size_of::<T>();
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
            length_in_bytes: self.length_in_bytes,
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

/// Pushes to the excess capacity of a `SharedStorage` that may have shared references (potentially
/// across multiple threads).
pub struct ExtendableSharedStorage<T> {
    storage: SharedStorage<T>,
    data_ptr: *mut T,
    /// Elements, not bytes
    capacity: usize,
}

impl<T> ExtendableSharedStorage<T>
where
    T: Clone,
{
    fn try_new(mut storage: SharedStorage<T>) -> Option<Self> {
        if !storage.is_exclusive() {
            return None;
        }

        let BackingStorage::Vec {
            initialized_bytes: _,
            original_capacity,
            vtable: _,
        } = &storage.inner().backing
        else {
            return None;
        };

        let data_ptr = storage.inner().ptr;
        let capacity = *original_capacity;

        Some(Self {
            storage,
            data_ptr,
            capacity,
        })
    }

    pub fn make_storage(&self) -> SharedStorage<T> {
        assert_eq!(
            self.storage.length_in_bytes,
            self.initialized_bytes().load(Ordering::Acquire)
        );

        self.storage.clone()
    }

    fn initialized_bytes(&self) -> &AtomicUsize {
        let BackingStorage::Vec {
            initialized_bytes, ..
        } = &self.storage.inner().backing
        else {
            unsafe { unreachable_unchecked() }
        };

        initialized_bytes
    }

    #[inline]
    fn initialized_len(&self) -> usize {
        self.initialized_bytes().load(Ordering::Acquire) / size_of::<T>()
    }

    /// # Safety
    /// `self.initialized_len() < self.capacity`
    #[inline]
    unsafe fn increment_len(&mut self) {
        debug_assert!(self.initialized_len() < self.capacity);

        // Release: Written value must be observable on other threads.
        self.initialized_bytes()
            .fetch_add(size_of::<T>(), Ordering::Release);
        self.storage.length_in_bytes += size_of::<T>();

        debug_assert_eq!(
            self.initialized_bytes().load(Ordering::Acquire),
            self.storage.length_in_bytes
        );
    }

    #[inline(always)]
    pub fn push(&mut self, idx: T) {
        if self.initialized_len() == self.capacity {
            self.reserve(1);
        }

        unsafe { self.push_unchecked(idx) }
    }

    /// # Safety
    /// `self.initialized_len() < self.capacity`
    #[inline(always)]
    unsafe fn push_unchecked(&mut self, value: T) {
        unsafe {
            self.data_ptr.add(self.initialized_len()).write(value);
            self.increment_len();
        }
    }

    #[cold]
    #[inline(never)]
    pub fn reserve(&mut self, additional: usize) {
        let initialized_len = self.initialized_len();

        let new_len = initialized_len.checked_add(additional).unwrap();

        if new_len > self.capacity {
            let double = self.capacity * 2;
            self.realloc(double.max(new_len).max(8));
        }
    }

    fn realloc(&mut self, new_cap: usize) {
        assert!(new_cap >= self.initialized_len());
        let mut out: Vec<T> = Vec::with_capacity(new_cap);

        assert_eq!(
            self.storage.length_in_bytes,
            self.initialized_bytes().load(Ordering::Acquire)
        );

        if let Some(v) = self.storage.try_take_vec() {
            out.extend(v)
        } else {
            let slice: &[T] = &*self.storage;
            assert_eq!(slice.len(), self.initialized_len());

            out.extend_from_slice(slice);
        }

        *self = Self::try_new(SharedStorage::from_vec(out)).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use polars_utils::relaxed_cell::RelaxedCell;

    use crate::storage::SharedStorage;

    #[test]
    fn test_extendable_shared_storage() {
        static DROP_COUNT: RelaxedCell<usize> = RelaxedCell::new_usize(0);

        #[derive(Clone)]
        struct TrackedDrop(#[expect(unused)] i64);

        impl Drop for TrackedDrop {
            fn drop(&mut self) {
                DROP_COUNT.fetch_add(1);
            }
        }

        use TrackedDrop as V;

        let mut v: Vec<TrackedDrop> = Vec::with_capacity(5);
        let capacity = v.capacity();
        v.extend([V(1), V(1), V(1)]);

        let mut extendable = SharedStorage::from_vec(v).try_into_extendable().unwrap();

        let mut storage_3 = extendable.make_storage();

        for _ in 0..capacity - 3 {
            extendable.push(V(1))
        }

        assert!(capacity > 3);

        // Should not affect length of existing `SharedStorage`s
        assert_eq!(storage_3.len(), 3);
        assert!(!storage_3.is_exclusive());

        assert_eq!(extendable.make_storage().len(), capacity);
        // This should cause realloc
        extendable.push(V(1));

        // `storage_3` is now exclusive
        assert!(storage_3.is_exclusive());

        assert_eq!(DROP_COUNT.load(), 0);

        drop(extendable);

        assert_eq!(DROP_COUNT.load(), capacity + 1);

        drop(storage_3);

        assert_eq!(DROP_COUNT.load(), 2 * capacity + 1);
    }
}
