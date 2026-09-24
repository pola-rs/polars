/// Bytes that are owned by the side that created them.
/// Dropping it calls `release`, so the owner frees its own allocation.
#[repr(C)]
pub struct BytesExport {
    ptr: *const u8,
    len: usize,
    release: Option<unsafe extern "C" fn(arg1: *mut BytesExport)>,
    private_data: *mut std::os::raw::c_void,
}

impl BytesExport {
    pub fn empty() -> Self {
        Self {
            ptr: std::ptr::null(),
            len: 0,
            release: None,
            private_data: std::ptr::null_mut(),
        }
    }

    pub fn is_null(&self) -> bool {
        self.private_data.is_null()
    }

    /// # Safety
    /// `self` must be valid and not released.
    pub unsafe fn as_slice(&self) -> &[u8] {
        if self.len == 0 {
            return &[];
        }
        std::slice::from_raw_parts(self.ptr, self.len)
    }
}

impl From<Vec<u8>> for BytesExport {
    fn from(bytes: Vec<u8>) -> Self {
        let bytes = Box::new(bytes);
        Self {
            ptr: bytes.as_ptr(),
            len: bytes.len(),
            release: Some(c_release_bytes_export),
            private_data: Box::into_raw(bytes) as *mut std::os::raw::c_void,
        }
    }
}

impl Drop for BytesExport {
    fn drop(&mut self) {
        if let Some(release) = self.release {
            unsafe { release(self) }
        }
    }
}

// callback used to drop [BytesExport] when it is exported.
unsafe extern "C" fn c_release_bytes_export(e: *mut BytesExport) {
    if e.is_null() {
        return;
    }
    let e = &mut *e;
    drop(Box::from_raw(e.private_data as *mut Vec<u8>));
    e.private_data = std::ptr::null_mut();
    e.release = None;
}
