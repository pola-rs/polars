use arrow::array::{ArrayData, PrimitiveArray};
use arrow::buffer::Buffer;
use arrow::datatypes::*;
use arrow::memory;
use std::iter::FromIterator;
use std::mem;
use std::mem::ManuallyDrop;

/// A `Vec` wrapper with a memory alignment equal to Arrow's primitive arrays.
/// Can be useful in creating a new ChunkedArray or Arrow Primitive array without copying.
#[derive(Debug)]
pub struct AlignedVec<T> {
    pub inner: Vec<T>,
    // if into_inner is called, this will be true and we can use the default Vec's destructor
    taken: bool,
}

impl<T> Drop for AlignedVec<T> {
    fn drop(&mut self) {
        if !self.taken {
            let inner = mem::take(&mut self.inner);
            let mut me = mem::ManuallyDrop::new(inner);
            let ptr: *mut T = me.as_mut_ptr();
            let ptr = ptr as *mut u8;
            let ptr = std::ptr::NonNull::new(ptr).unwrap();
            unsafe { memory::free_aligned(ptr, self.capacity()) }
        }
    }
}

impl<T> FromIterator<T> for AlignedVec<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let iter = iter.into_iter();
        let sh = iter.size_hint();
        let size = sh.1.unwrap_or(sh.0);

        let mut av = Self::with_capacity_aligned(size);

        for v in iter {
            av.push(v)
        }

        // Iterator size hint wasn't correct and reallocation has occurred
        assert!(av.len() <= size);
        av
    }
}

impl<T: Clone> AlignedVec<T> {
    pub fn resize(&mut self, new_len: usize, value: T) {
        self.inner.resize(new_len, value)
    }

    pub fn extend_from_slice(&mut self, other: &[T]) {
        let remaining_cap = self.capacity() - self.len();
        let needed_cap = other.len();
        if needed_cap > remaining_cap {
            self.reserve(needed_cap - remaining_cap);
        }
        self.inner.extend_from_slice(other)
    }
}

impl<T> AlignedVec<T> {
    /// Create a new Vec where first bytes memory address has an alignment of 64 bytes, as described
    /// by arrow spec.
    /// Read more:
    /// <https://github.com/rust-ndarray/ndarray/issues/771>
    pub fn with_capacity_aligned(size: usize) -> Self {
        // Can only have a zero copy to arrow memory if address of first byte % 64 == 0
        let t_size = std::mem::size_of::<T>();
        let capacity = size * t_size;
        let ptr = memory::allocate_aligned(capacity).as_ptr() as *mut T;
        let v = unsafe { Vec::from_raw_parts(ptr, 0, size) };
        AlignedVec {
            inner: v,
            taken: false,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    pub fn reserve(&mut self, additional: usize) {
        let mut me = ManuallyDrop::new(mem::take(&mut self.inner));
        let ptr = me.as_mut_ptr() as *mut u8;
        let ptr = std::ptr::NonNull::new(ptr).unwrap();
        let t_size = mem::size_of::<T>();
        let cap = me.capacity();
        let old_capacity = t_size * cap;
        let new_capacity = old_capacity + t_size * additional;
        let ptr = unsafe { memory::reallocate(ptr, old_capacity, new_capacity) };
        let ptr = ptr.as_ptr() as *mut T;
        let v = unsafe { Vec::from_raw_parts(ptr, me.len(), cap + additional) };
        self.inner = v;
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Create a new aligned vec from a ptr.
    ///
    /// # Safety
    /// The ptr should be 64 byte aligned and `len` and `capacity` should be correct otherwise it is UB.
    pub unsafe fn from_ptr(ptr: usize, len: usize, capacity: usize) -> Self {
        assert_eq!((ptr as usize) % memory::ALIGNMENT, 0);
        let ptr = ptr as *mut T;
        let v = Vec::from_raw_parts(ptr, len, capacity);
        Self {
            inner: v,
            taken: false,
        }
    }

    /// Take ownership of the Vec. This is UB because the destructor of Vec<T> probably has a different
    /// alignment than what we allocated.
    unsafe fn into_inner(mut self) -> Vec<T> {
        self.shrink_to_fit();
        self.taken = true;
        mem::take(&mut self.inner)
    }

    /// Push at the end of the Vec. This is unsafe because a push when the capacity of the
    /// inner Vec is reached will reallocate the Vec without the alignment, leaving this destructor's
    /// alignment incorrect
    pub fn push(&mut self, value: T) {
        if self.inner.len() == self.capacity() {
            self.reserve(1);
        }
        self.inner.push(value)
    }

    /// Set the length of the underlying `Vec`.
    ///
    /// # Safety
    ///
    /// - `new_len` must be less than or equal to `capacity`.
    /// - The elements at `old_len..new_len` must be initialized.
    pub unsafe fn set_len(&mut self, new_len: usize) {
        self.inner.set_len(new_len);
    }

    pub fn as_ptr(&self) -> *const T {
        self.inner.as_ptr()
    }

    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.inner.as_mut_ptr()
    }

    pub fn as_mut_slice(&mut self) -> &mut [T] {
        self.inner.as_mut_slice()
    }

    pub fn capacity(&self) -> usize {
        self.inner.capacity()
    }

    pub fn into_raw_parts(self) -> (*mut T, usize, usize) {
        let mut me = ManuallyDrop::new(self);
        (me.as_mut_ptr(), me.len(), me.capacity())
    }

    pub fn shrink_to_fit(&mut self) {
        if self.capacity() > self.len() {
            let mut me = ManuallyDrop::new(mem::take(&mut self.inner));
            let ptr = me.as_mut_ptr() as *mut u8;
            let ptr = std::ptr::NonNull::new(ptr).unwrap();

            let t_size = mem::size_of::<T>();
            let new_size = t_size * me.len();
            let old_size = t_size * me.capacity();
            let v = unsafe {
                let ptr = memory::reallocate(ptr, old_size, new_size).as_ptr() as *mut T;
                Vec::from_raw_parts(ptr, me.len(), me.len())
            };
            self.inner = v;
        }
    }

    /// Transform this array to an Arrow Buffer.
    pub fn into_arrow_buffer(self) -> Buffer {
        let values = unsafe { self.into_inner() };

        let me = mem::ManuallyDrop::new(values);
        let ptr = me.as_ptr() as *mut u8;
        let len = me.len() * std::mem::size_of::<T>();
        let capacity = me.capacity() * std::mem::size_of::<T>();
        debug_assert_eq!((ptr as usize) % 64, 0);
        let ptr = std::ptr::NonNull::new(ptr).unwrap();

        unsafe { Buffer::from_raw_parts(ptr, len, capacity) }
    }

    pub fn into_primitive_array<A: ArrowPrimitiveType>(
        self,
        null_buf: Option<Buffer>,
    ) -> PrimitiveArray<A> {
        debug_assert_eq!(mem::size_of::<A::Native>(), mem::size_of::<T>());

        let vec_len = self.len();
        let buffer = self.into_arrow_buffer();

        let mut builder = ArrayData::builder(A::DATA_TYPE)
            .len(vec_len)
            .add_buffer(buffer);

        if let Some(buf) = null_buf {
            builder = builder.null_bit_buffer(buf);
        }
        let data = builder.build();

        PrimitiveArray::<A>::from(data)
    }
}
