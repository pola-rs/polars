use crate::trusted_len::TrustedLen;
use crate::utils::FromTrustedLenIterator;
use arrow::alloc;
use arrow::array::{ArrayData, PrimitiveArray};
use arrow::buffer::{Buffer, MutableBuffer};
use arrow::datatypes::*;
use std::iter::FromIterator;
use std::mem;
use std::mem::ManuallyDrop;

/// A `Vec` wrapper with a memory alignment equal to Arrow's primitive arrays.
/// Can be useful in creating a new ChunkedArray or Arrow Primitive array without copying.
#[derive(Debug)]
pub struct AlignedVec<T: ArrowNativeType> {
    pub inner: Vec<T>,
    // if into_inner is called, this will be true and we can use the default Vec's destructor
    taken: bool,
}

impl<T: ArrowNativeType> Drop for AlignedVec<T> {
    fn drop(&mut self) {
        if !self.taken {
            let inner = mem::take(&mut self.inner);
            let mut me = mem::ManuallyDrop::new(inner);
            let ptr: *mut T = me.as_mut_ptr();
            let ptr = ptr as *mut u8;
            let ptr = std::ptr::NonNull::new(ptr).unwrap();
            unsafe { alloc::free_aligned::<u8>(ptr, me.capacity() * mem::size_of::<T>()) }
        }
    }
}

impl<T: ArrowNativeType> FromIterator<T> for AlignedVec<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let iter = iter.into_iter();
        let sh = iter.size_hint();
        let size = sh.1.unwrap_or(sh.0);

        let mut av = Self::with_capacity(size);
        av.extend(iter);

        // Iterator size hint wasn't correct and reallocation has occurred
        assert!(av.len() <= size);
        av
    }
}

impl<T: Copy + ArrowNativeType> AlignedVec<T> {
    /// Uses a memcpy to initialize this AlignedVec
    pub fn new_from_slice(other: &[T]) -> Self {
        let len = other.len();
        let mut av = Self::with_capacity(len);
        unsafe {
            // Safety:
            // we set initiate the memory after this with a memcpy.
            av.set_len(len);
        }
        av.inner.copy_from_slice(other);
        av
    }
}

impl<T: Clone + ArrowNativeType> AlignedVec<T> {
    pub fn resize(&mut self, new_len: usize, value: T) {
        self.inner.resize(new_len, value)
    }

    pub fn extend_from_slice(&mut self, other: &[T]) {
        let remaining_cap = self.capacity() - self.len();
        let needed_cap = other.len();
        // exponential allocation
        if needed_cap > remaining_cap {
            self.reserve(std::cmp::max(needed_cap, self.capacity()));
        }
        self.inner.extend_from_slice(other)
    }
}

impl<T: ArrowNativeType> AlignedVec<T> {
    /// Create a new Vec where first bytes memory address has an alignment of 64 bytes, as described
    /// by arrow spec.
    /// Read more:
    /// <https://github.com/rust-ndarray/ndarray/issues/771>
    pub fn with_capacity(size: usize) -> Self {
        // Can only have a zero copy to arrow memory if address of first byte % 64 == 0
        let t_size = std::mem::size_of::<T>();
        let capacity = size * t_size;
        let ptr = alloc::allocate_aligned::<u8>(capacity).as_ptr() as *mut T;
        let v = unsafe { Vec::from_raw_parts(ptr, 0, size) };
        AlignedVec {
            inner: v,
            taken: false,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    #[inline]
    pub fn reserve(&mut self, additional: usize) {
        let mut me = ManuallyDrop::new(mem::take(&mut self.inner));
        let ptr = me.as_mut_ptr() as *mut u8;
        let ptr = std::ptr::NonNull::new(ptr).unwrap();
        let t_size = mem::size_of::<T>();
        let cap = me.capacity();
        let old_capacity = t_size * cap;
        let new_capacity = old_capacity + t_size * additional;
        let ptr = unsafe { alloc::reallocate::<u8>(ptr, old_capacity, new_capacity) };
        let ptr = ptr.as_ptr() as *mut T;
        let v = unsafe { Vec::from_raw_parts(ptr, me.len(), cap + additional) };
        self.inner = v;
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Create a new aligned vec from a ptr.
    ///
    /// # Safety
    /// The ptr should be 64 byte aligned and `len` and `capacity` should be correct otherwise it is UB.
    pub unsafe fn from_ptr(ptr: usize, len: usize, capacity: usize) -> Self {
        assert_eq!((ptr as usize) % alloc::ALIGNMENT, 0);
        let ptr = ptr as *mut T;
        let v = Vec::from_raw_parts(ptr, len, capacity);
        Self {
            inner: v,
            taken: false,
        }
    }

    /// Take ownership of the Vec. This is UB because the destructor of Vec<T> probably has a different
    /// alignment than what we allocated.
    ///
    /// Only used for inner workings
    unsafe fn into_inner(mut self) -> Vec<T> {
        if self.taken {
            eprintln!("inner vec was already taken: UB");
            std::process::abort()
        }
        self.taken = true;
        mem::take(&mut self.inner)
    }

    /// Push at the end of the Vec. This is unsafe because a push when the capacity of the
    /// inner Vec is reached will reallocate the Vec without the alignment, leaving this destructor's
    /// alignment incorrect
    #[inline]
    pub fn push(&mut self, value: T) {
        if self.inner.len() == self.capacity() {
            // exponential allocation
            self.reserve(std::cmp::max(self.capacity(), 5));
        }
        self.inner.push(value)
    }

    /// Set the length of the underlying `Vec`.
    ///
    /// # Safety
    ///
    /// - `new_len` must be less than or equal to `capacity`.
    /// - The elements at `old_len..new_len` must be initialized.
    #[inline]
    pub unsafe fn set_len(&mut self, new_len: usize) {
        self.inner.set_len(new_len);
    }

    #[inline]
    pub fn as_ptr(&self) -> *const T {
        self.inner.as_ptr()
    }

    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.inner.as_mut_ptr()
    }

    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        self.inner.as_mut_slice()
    }

    #[inline]
    pub fn as_slice(&self) -> &[T] {
        self.inner.as_slice()
    }

    #[inline]
    pub fn capacity(&self) -> usize {
        self.inner.capacity()
    }

    pub fn shrink_to_fit(&mut self) {
        if self.capacity() > self.len() && !self.is_empty() {
            let mut me = ManuallyDrop::new(mem::take(&mut self.inner));
            let ptr = me.as_mut_ptr() as *mut u8;
            let ptr = std::ptr::NonNull::new(ptr).unwrap();

            let t_size = mem::size_of::<T>();
            let new_size = t_size * me.len();
            let old_size = t_size * me.capacity();
            let v = unsafe {
                let ptr = alloc::reallocate::<u8>(ptr, old_size, new_size).as_ptr() as *mut T;
                Vec::from_raw_parts(ptr, me.len(), me.len())
            };

            self.inner = v;
        }
    }

    /// Transform this array to an Arrow Buffer.
    pub fn into_arrow_buffer(self) -> Buffer {
        if self.is_empty() && self.capacity() == 0 {
            MutableBuffer::new(0).into()
        } else {
            let values = unsafe { self.into_inner() };

            let me = mem::ManuallyDrop::new(values);
            let ptr = me.as_ptr() as *mut u8;
            let len = me.len() * std::mem::size_of::<T>();
            let capacity = me.capacity() * std::mem::size_of::<T>();
            debug_assert_eq!((ptr as usize) % 64, 0);
            let ptr = std::ptr::NonNull::new(ptr).unwrap();

            unsafe { Buffer::from_raw_parts(ptr, len, capacity) }
        }
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

    fn reserve_from_size_hint(&mut self, size: usize) {
        let (extra_cap, overflow) = size.overflowing_sub(self.capacity());
        if extra_cap > 0 && !overflow {
            self.reserve(extra_cap);
        }
    }

    /// # Panic
    /// Must be a trusted len iterator or else it will panic
    pub fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        let iter = iter.into_iter();
        let cap = iter.size_hint().1.expect("a trusted length iterator");
        self.reserve_from_size_hint(cap);
        let len_before = self.len();
        self.inner.extend(iter);
        let added = self.len() - len_before;
        if added != cap {
            eprintln!("size hint was incorrect, this is UB. aborting");
            std::process::abort()
        }
    }

    /// Extend this Vector with an iterator whose length can be trusted.
    ///
    /// # Safety
    /// - iterator must be TrustedLen
    pub unsafe fn extend_trusted_len_unchecked<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        let iter = iter.into_iter();
        let iter_len = iter.size_hint().1.expect("a trusted length iterator");
        self.reserve_from_size_hint(iter_len);

        let mut dst = self.inner.as_ptr() as *mut T;
        dst = dst.add(self.len());
        let start = dst;
        for item in iter {
            // note how there is no reserve here
            std::ptr::write(dst, item);
            dst = dst.add(1);
        }
        assert_eq!(
            dst.offset_from(start) as usize,
            iter_len,
            "Trusted iterator length was not accurately reported"
        );
        self.inner.set_len(self.len() + iter_len)
    }

    pub fn extend_trusted_len<I: IntoIterator<Item = T> + TrustedLen>(&mut self, iter: I) {
        unsafe { self.extend_trusted_len_unchecked(iter) }
    }
}

impl<T: ArrowNativeType> Default for AlignedVec<T> {
    fn default() -> Self {
        // Be careful here. Don't initialize with a normal Vec as this will cause the wrong deallocator
        // to run and SIGSEGV
        Self::with_capacity(0)
    }
}

impl<T: ArrowNativeType> FromTrustedLenIterator<T> for AlignedVec<T> {
    fn from_iter_trusted_length<I: IntoIterator<Item = T>>(iter: I) -> Self
    where
        I::IntoIter: TrustedLen,
    {
        let iter = iter.into_iter();
        let len = iter.size_hint().0;
        // Safety:
        // trait trustedlen
        let buf = unsafe { MutableBuffer::from_trusted_len_iter(iter) };
        let ptr = buf.as_ptr() as usize;
        let capacity = buf.capacity() / std::mem::size_of::<T>();
        std::mem::forget(buf);
        unsafe { AlignedVec::from_ptr(ptr, len, capacity) }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use arrow::alloc;

    #[test]
    fn test_aligned_vec_allocations() {
        // Can only have a zero copy to arrow memory if address of first byte % 64 == 0
        // check if we can increase above initial capacity and keep the Arrow alignment
        let mut v = AlignedVec::with_capacity(2);
        v.push(1);
        v.push(2);
        v.push(3);
        v.push(4);

        let ptr = v.as_ptr();
        assert_eq!((ptr as usize) % alloc::ALIGNMENT, 0);

        // check if we can shrink to fit
        let mut v = AlignedVec::with_capacity(10);
        v.push(1);
        v.push(2);
        v.shrink_to_fit();
        assert_eq!(v.len(), 2);
        assert_eq!(v.capacity(), 2);
        let ptr = v.as_ptr();
        assert_eq!((ptr as usize) % alloc::ALIGNMENT, 0);

        let a = v.into_primitive_array::<Int32Type>(None);
        assert_eq!(&a.values()[..2], &[1, 2])
    }

    #[test]
    fn test_trusted_len() {
        let av = AlignedVec::from_iter_trusted_length([1, 2, 3, 4, 5].iter().copied());
        let v: Vec<_> = av.inner.iter().copied().collect();
        assert_eq!(v, &[1, 2, 3, 4, 5]);
    }
}
