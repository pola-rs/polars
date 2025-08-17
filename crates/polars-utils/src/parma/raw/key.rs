/// A low-level trait for keys.
///
/// This is used to allow unsized types such as [`str`] for keys.
pub trait Key {
    /// The alignment necessary for the key. Must return a power of two.
    fn align() -> usize;

    /// The size of the key in bytes.
    fn size(&self) -> usize;

    /// Initialize the key in the given memory location.
    ///
    /// # Safety
    /// The memory location must satisfy the specified size and alignment.
    unsafe fn init(&self, ptr: *mut u8);

    /// Get a reference to the key from the given memory location.
    ///
    /// # Safety
    /// The pointer must be valid and initialized with [`Key::init`].
    unsafe fn get<'a>(ptr: *const u8) -> &'a Self;

    /// Drop the key in place.
    ///
    /// # Safety
    /// The pointer must be valid and initialized with [`Key::init`].
    unsafe fn drop_in_place(ptr: *mut u8);
}

impl<T: Copy> Key for [T] {
    #[inline(always)]
    fn align() -> usize {
        align_of::<usize>().max(align_of::<T>())
    }

    #[inline(always)]
    fn size(&self) -> usize {
        size_of::<usize>().next_multiple_of(align_of::<T>()) + self.len()
    }

    #[inline(always)]
    unsafe fn init(&self, ptr: *mut u8) {
        unsafe {
            let p_len = ptr.cast::<usize>();
            let p_data = ptr.add(size_of::<usize>().next_multiple_of(align_of::<T>()));
            let len = self.len();
            p_len.write(len);
            std::ptr::copy_nonoverlapping(self.as_ptr(), p_data.cast(), len)
        }
    }

    #[inline(always)]
    unsafe fn get<'a>(ptr: *const u8) -> &'a Self {
        unsafe {
            let p_len = ptr.cast::<usize>();
            let p_data = ptr.add(size_of::<usize>().next_multiple_of(align_of::<T>()));
            let len = p_len.read();
            core::slice::from_raw_parts(p_data.cast(), len)
        }
    }

    #[inline(always)]
    unsafe fn drop_in_place(_ptr: *mut u8) {}
}

impl Key for str {
    #[inline(always)]
    fn align() -> usize {
        <[u8] as Key>::align()
    }

    #[inline(always)]
    fn size(&self) -> usize {
        <[u8] as Key>::size(self.as_bytes())
    }

    #[inline(always)]
    unsafe fn init(&self, ptr: *mut u8) {
        unsafe { <[u8] as Key>::init(self.as_bytes(), ptr) }
    }

    #[inline(always)]
    unsafe fn get<'a>(ptr: *const u8) -> &'a Self {
        unsafe { core::str::from_utf8_unchecked(<[u8] as Key>::get(ptr)) }
    }

    #[inline(always)]
    unsafe fn drop_in_place(_ptr: *mut u8) {}
}

impl<T: Clone + Sized> Key for T {
    fn align() -> usize {
        align_of::<T>()
    }

    fn size(&self) -> usize {
        size_of::<T>()
    }

    #[inline(always)]
    unsafe fn init(&self, ptr: *mut u8) {
        unsafe {
            ptr.cast::<T>().write(self.clone());
        }
    }

    #[inline(always)]
    unsafe fn get<'a>(ptr: *const u8) -> &'a Self {
        unsafe { &*ptr.cast::<T>() }
    }

    #[inline(always)]
    unsafe fn drop_in_place(ptr: *mut u8) {
        unsafe { ptr.cast::<T>().drop_in_place() }
    }
}
