//! `#[repr(C)]` value types shared by all plugin contracts.
//!
//! Changing anything in this module changes every contract that uses it, so it requires new IDs
//! for all of them (see the crate docs).
use std::ffi::{c_char, c_int, c_void};

pub use async_ffi::{FfiFuture, FutureExt};

/// The layout of [`FfiFuture`] is part of every contract. A dependency update that changes it
/// fails here and requires new IDs for all contracts.
const _: () = assert!(async_ffi::ABI_VERSION == 2);
use std::marker::PhantomData;

/// Borrowed UTF-8 string. Only valid for the duration of the call it is passed to.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct FfiStr<'a> {
    ptr: *const u8,
    len: usize,
    _lifetime: PhantomData<&'a str>,
}

unsafe impl Send for FfiStr<'_> {}
unsafe impl Sync for FfiStr<'_> {}

impl<'a> FfiStr<'a> {
    pub const fn new(s: &'a str) -> Self {
        Self {
            ptr: s.as_ptr(),
            len: s.len(),
            _lifetime: PhantomData,
        }
    }

    pub fn as_str(&self) -> Result<&'a str, std::str::Utf8Error> {
        let bytes = if self.len == 0 {
            &[]
        } else {
            unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
        };
        std::str::from_utf8(bytes)
    }
}

impl std::fmt::Debug for FfiStr<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.as_str() {
            Ok(s) => std::fmt::Debug::fmt(s, f),
            Err(_) => f.write_str("<non-UTF-8>"),
        }
    }
}

impl<'a> From<&'a str> for FfiStr<'a> {
    fn from(s: &'a str) -> Self {
        Self::new(s)
    }
}

type ReleaseFn = unsafe extern "C" fn(*mut c_void);

unsafe extern "C" fn release_box<T>(owner: *mut c_void) {
    drop(unsafe { Box::from_raw(owner as *mut T) });
}

/// Owned bytes. Dropping it calls the release callback of the side that created it, so the
/// memory is always freed by the allocator that allocated it.
#[repr(C)]
pub struct FfiBuf {
    ptr: *const u8,
    len: usize,
    owner: *mut c_void,
    release: Option<ReleaseFn>,
}

unsafe impl Send for FfiBuf {}
unsafe impl Sync for FfiBuf {}

impl FfiBuf {
    pub fn empty() -> Self {
        Self {
            ptr: std::ptr::null(),
            len: 0,
            owner: std::ptr::null_mut(),
            release: None,
        }
    }

    /// Wrap any owned byte container without copying.
    pub fn from_owner<T: AsRef<[u8]> + Send + Sync + 'static>(owner: T) -> Self {
        let owner = Box::new(owner);
        let bytes: &[u8] = (*owner).as_ref();
        let (ptr, len) = (bytes.as_ptr(), bytes.len());

        Self {
            ptr,
            len,
            owner: Box::into_raw(owner) as *mut c_void,
            release: Some(release_box::<T>),
        }
    }

    pub fn from_vec(v: Vec<u8>) -> Self {
        Self::from_owner(v)
    }

    pub fn from_string(s: String) -> Self {
        Self::from_owner(s)
    }

    pub fn as_slice(&self) -> &[u8] {
        if self.len == 0 {
            &[]
        } else {
            unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn to_string_lossy(&self) -> String {
        String::from_utf8_lossy(self.as_slice()).into_owned()
    }
}

impl std::ops::Deref for FfiBuf {
    type Target = [u8];

    fn deref(&self) -> &[u8] {
        self.as_slice()
    }
}

impl AsRef<[u8]> for FfiBuf {
    fn as_ref(&self) -> &[u8] {
        self.as_slice()
    }
}

impl Drop for FfiBuf {
    fn drop(&mut self) {
        if let Some(release) = self.release.take() {
            unsafe { release(self.owner) }
        }
    }
}

/// Optional value.
#[repr(C, u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FfiOption<T> {
    None,
    Some(T),
}

impl<T> FfiOption<T> {
    pub fn into_option(self) -> Option<T> {
        match self {
            Self::None => None,
            Self::Some(v) => Some(v),
        }
    }
}

impl<T> From<Option<T>> for FfiOption<T> {
    fn from(value: Option<T>) -> Self {
        match value {
            None => Self::None,
            Some(v) => Self::Some(v),
        }
    }
}

/// Borrowed slice. Only valid for the duration of the call it is passed to.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct FfiSlice<'a, T> {
    ptr: *const T,
    len: usize,
    _lifetime: PhantomData<&'a [T]>,
}

unsafe impl<T: Sync> Send for FfiSlice<'_, T> {}
unsafe impl<T: Sync> Sync for FfiSlice<'_, T> {}

impl<'a, T> FfiSlice<'a, T> {
    pub const fn new(s: &'a [T]) -> Self {
        Self {
            ptr: s.as_ptr(),
            len: s.len(),
            _lifetime: PhantomData,
        }
    }

    pub fn as_slice(&self) -> &'a [T] {
        if self.len == 0 {
            &[]
        } else {
            unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
        }
    }
}

/// Owned array of `#[repr(C)]` values. Like [`FfiBuf`], dropping it calls the release callback
/// of the side that created it, which also drops the elements.
#[repr(C)]
pub struct FfiVec<T> {
    ptr: *const T,
    len: usize,
    owner: *mut c_void,
    release: Option<ReleaseFn>,
}

unsafe impl<T: Send> Send for FfiVec<T> {}
unsafe impl<T: Sync> Sync for FfiVec<T> {}

impl<T: Send + Sync + 'static> FfiVec<T> {
    pub fn from_vec(v: Vec<T>) -> Self {
        let owner = Box::new(v);
        let (ptr, len) = (owner.as_ptr(), owner.len());
        Self {
            ptr,
            len,
            owner: Box::into_raw(owner) as *mut c_void,
            release: Some(release_box::<Vec<T>>),
        }
    }
}

impl<T> FfiVec<T> {
    pub fn as_slice(&self) -> &[T] {
        if self.len == 0 {
            &[]
        } else {
            unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
        }
    }
}

impl<T> Drop for FfiVec<T> {
    fn drop(&mut self) {
        if let Some(release) = self.release.take() {
            unsafe { release(self.owner) }
        }
    }
}

/// Error categories. Stored as `u32` so that unknown future kinds remain representable.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FfiErrorKind(pub u32);

impl FfiErrorKind {
    pub const OTHER: Self = Self(0);
    pub const IO: Self = Self(1);
    pub const NOT_FOUND: Self = Self(2);
    pub const NOT_IMPLEMENTED: Self = Self(3);
    pub const PANIC: Self = Self(4);
    pub const INVALID_ARGUMENT: Self = Self(5);
    /// The user's input to the dataset is invalid (e.g. an unknown snapshot ID). Hosts report it
    /// like other invalid parameter values (Python `ValueError`) and pass the message unchanged.
    pub const INVALID_INPUT: Self = Self(7);

    pub fn name(self) -> &'static str {
        match self {
            Self::IO => "io",
            Self::NOT_FOUND => "not found",
            Self::NOT_IMPLEMENTED => "not implemented",
            Self::PANIC => "panic",
            Self::INVALID_ARGUMENT => "invalid argument",
            Self::INVALID_INPUT => "invalid input",
            _ => "other",
        }
    }
}

#[repr(C)]
pub struct FfiError {
    pub kind: u32,
    /// UTF-8 message.
    pub msg: FfiBuf,
}

impl FfiError {
    pub fn new(kind: FfiErrorKind, msg: impl Into<String>) -> Self {
        Self {
            kind: kind.0,
            msg: FfiBuf::from_string(msg.into()),
        }
    }

    pub fn kind(&self) -> FfiErrorKind {
        FfiErrorKind(self.kind)
    }

    pub fn message(&self) -> String {
        self.msg.to_string_lossy()
    }
}

impl std::fmt::Debug for FfiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "FfiError({}: {})", self.kind().name(), self.message())
    }
}

#[repr(C, u8)]
pub enum FfiResult<T> {
    Ok(T),
    Err(FfiError),
}

impl<T> FfiResult<T> {
    pub fn into_result(self) -> Result<T, FfiError> {
        match self {
            Self::Ok(v) => Ok(v),
            Self::Err(e) => Err(e),
        }
    }
}

impl<T> From<Result<T, FfiError>> for FfiResult<T> {
    fn from(value: Result<T, FfiError>) -> Self {
        match value {
            Ok(v) => Self::Ok(v),
            Err(e) => Self::Err(e),
        }
    }
}

// Arrow C Data Interface structs, as defined by
// https://arrow.apache.org/docs/format/CDataInterface.html. Other Arrow implementations define
// structs with the identical layout, so values are moved across with `transmute_*`.

#[repr(C)]
pub struct ArrowSchema {
    pub format: *const c_char,
    pub name: *const c_char,
    pub metadata: *const c_char,
    pub flags: i64,
    pub n_children: i64,
    pub children: *mut *mut ArrowSchema,
    pub dictionary: *mut ArrowSchema,
    pub release: Option<unsafe extern "C" fn(arg1: *mut ArrowSchema)>,
    pub private_data: *mut c_void,
}

#[repr(C)]
pub struct ArrowArray {
    pub length: i64,
    pub null_count: i64,
    pub offset: i64,
    pub n_buffers: i64,
    pub n_children: i64,
    pub buffers: *mut *const c_void,
    pub children: *mut *mut ArrowArray,
    pub dictionary: *mut ArrowArray,
    pub release: Option<unsafe extern "C" fn(arg1: *mut ArrowArray)>,
    pub private_data: *mut c_void,
}

#[repr(C)]
pub struct ArrowArrayStream {
    pub get_schema:
        Option<unsafe extern "C" fn(arg1: *mut ArrowArrayStream, out: *mut ArrowSchema) -> c_int>,
    pub get_next:
        Option<unsafe extern "C" fn(arg1: *mut ArrowArrayStream, out: *mut ArrowArray) -> c_int>,
    pub get_last_error: Option<unsafe extern "C" fn(arg1: *mut ArrowArrayStream) -> *const c_char>,
    pub release: Option<unsafe extern "C" fn(arg1: *mut ArrowArrayStream)>,
    pub private_data: *mut c_void,
}

// The C Data Interface allows moving these structs between threads.
unsafe impl Send for ArrowSchema {}
unsafe impl Send for ArrowArrayStream {}

macro_rules! impl_arrow_struct {
    ($t:ty) => {
        impl $t {
            /// Move a layout-identical struct from another Arrow implementation into this type.
            ///
            /// # Safety
            /// `T` must be a `#[repr(C)]` definition of the same Arrow C struct.
            pub unsafe fn transmute_from<T>(value: T) -> Self {
                assert_eq!(std::mem::size_of::<T>(), std::mem::size_of::<Self>());
                let value = std::mem::ManuallyDrop::new(value);
                unsafe { std::ptr::read(&*value as *const T as *const Self) }
            }

            /// Move this struct into a layout-identical struct of another Arrow implementation.
            ///
            /// # Safety
            /// `T` must be a `#[repr(C)]` definition of the same Arrow C struct.
            pub unsafe fn transmute_into<T>(self) -> T {
                assert_eq!(std::mem::size_of::<T>(), std::mem::size_of::<Self>());
                let value = std::mem::ManuallyDrop::new(self);
                unsafe { std::ptr::read(&*value as *const Self as *const T) }
            }
        }

        impl Drop for $t {
            fn drop(&mut self) {
                if let Some(release) = self.release {
                    unsafe { release(self) }
                }
            }
        }
    };
}

impl ArrowSchema {
    /// A released (empty) schema.
    pub fn empty() -> Self {
        Self {
            format: std::ptr::null(),
            name: std::ptr::null(),
            metadata: std::ptr::null(),
            flags: 0,
            n_children: 0,
            children: std::ptr::null_mut(),
            dictionary: std::ptr::null_mut(),
            release: None,
            private_data: std::ptr::null_mut(),
        }
    }

    pub fn is_released(&self) -> bool {
        self.release.is_none()
    }
}

impl_arrow_struct!(ArrowSchema);
impl_arrow_struct!(ArrowArray);
impl_arrow_struct!(ArrowArrayStream);
