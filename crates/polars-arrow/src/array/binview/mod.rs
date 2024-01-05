//! See thread: https://lists.apache.org/thread/w88tpz76ox8h3rxkjl4so6rg3f1rv7wt
mod ffi;
pub(super) mod fmt;
mod iterator;
mod mutable;
mod view;

use std::any::Any;
use std::fmt::Debug;
use std::marker::PhantomData;
use std::sync::Arc;

use polars_error::*;

use crate::array::Array;
use crate::bitmap::Bitmap;
use crate::buffer::Buffer;
use crate::datatypes::ArrowDataType;

mod private {
    pub trait Sealed: Send + Sync {}

    impl Sealed for str {}
    impl Sealed for [u8] {}
}
use private::Sealed;

use crate::array::binview::iterator::BinaryViewValueIter;
use crate::array::binview::view::{validate_binary_view, validate_utf8_view};
use crate::array::iterator::NonNullValuesIter;
use crate::bitmap::utils::{BitmapIter, ZipValidity};

pub type BinaryViewArray = BinaryViewArrayGeneric<[u8]>;
pub type Utf8ViewArray = BinaryViewArrayGeneric<str>;

pub trait ViewType: Sealed + 'static + PartialEq {
    const IS_UTF8: bool;
    const DATA_TYPE: ArrowDataType;
    type Owned: Debug + Clone + Sync + Send + AsRef<Self>;

    /// # Safety
    /// The caller must ensure `index < self.len()`.
    unsafe fn from_bytes_unchecked(slice: &[u8]) -> &Self;

    fn to_bytes(&self) -> &[u8];

    #[allow(clippy::wrong_self_convention)]
    fn into_owned(&self) -> Self::Owned;
}

impl ViewType for str {
    const IS_UTF8: bool = true;
    const DATA_TYPE: ArrowDataType = ArrowDataType::Utf8View;
    type Owned = String;

    #[inline(always)]
    unsafe fn from_bytes_unchecked(slice: &[u8]) -> &Self {
        std::str::from_utf8_unchecked(slice)
    }

    #[inline(always)]
    fn to_bytes(&self) -> &[u8] {
        self.as_bytes()
    }

    fn into_owned(&self) -> Self::Owned {
        self.to_string()
    }
}

impl ViewType for [u8] {
    const IS_UTF8: bool = false;
    const DATA_TYPE: ArrowDataType = ArrowDataType::BinaryView;
    type Owned = Vec<u8>;

    #[inline(always)]
    unsafe fn from_bytes_unchecked(slice: &[u8]) -> &Self {
        slice
    }

    #[inline(always)]
    fn to_bytes(&self) -> &[u8] {
        self
    }

    fn into_owned(&self) -> Self::Owned {
        self.to_vec()
    }
}

pub struct BinaryViewArrayGeneric<T: ViewType + ?Sized> {
    data_type: ArrowDataType,
    views: Buffer<u128>,
    buffers: Arc<[Buffer<u8>]>,
    // Raw buffer access. (pointer, len).
    raw_buffers: Arc<[(*const u8, usize)]>,
    validity: Option<Bitmap>,
    phantom: PhantomData<T>,
}

impl<T: ViewType + ?Sized> Clone for BinaryViewArrayGeneric<T> {
    fn clone(&self) -> Self {
        Self {
            data_type: self.data_type.clone(),
            views: self.views.clone(),
            buffers: self.buffers.clone(),
            raw_buffers: self.raw_buffers.clone(),
            validity: self.validity.clone(),
            phantom: Default::default(),
        }
    }
}

unsafe impl<T: ViewType + ?Sized> Send for BinaryViewArrayGeneric<T> {}
unsafe impl<T: ViewType + ?Sized> Sync for BinaryViewArrayGeneric<T> {}

fn buffers_into_raw<T>(buffers: &[Buffer<T>]) -> Arc<[(*const T, usize)]> {
    buffers
        .iter()
        .map(|buf| (buf.as_ptr(), buf.len()))
        .collect()
}

impl<T: ViewType + ?Sized> BinaryViewArrayGeneric<T> {
    /// # Safety
    /// The caller must ensure
    /// - the data is valid utf8 (if required)
    /// - The offsets match the buffers.
    pub unsafe fn new_unchecked(
        data_type: ArrowDataType,
        views: Buffer<u128>,
        buffers: Arc<[Buffer<u8>]>,
        validity: Option<Bitmap>,
    ) -> Self {
        let raw_buffers = buffers_into_raw(&buffers);
        Self {
            data_type,
            views,
            buffers,
            raw_buffers,
            validity,
            phantom: Default::default(),
        }
    }

    pub fn data_buffers(&self) -> &[Buffer<u8>] {
        self.buffers.as_ref()
    }

    pub fn views(&self) -> &Buffer<u128> {
        &self.views
    }

    pub fn try_new(
        data_type: ArrowDataType,
        views: Buffer<u128>,
        buffers: Arc<[Buffer<u8>]>,
        validity: Option<Bitmap>,
    ) -> PolarsResult<Self> {
        if T::IS_UTF8 {
            validate_utf8_view(views.as_ref(), buffers.as_ref())?;
        } else {
            validate_binary_view(views.as_ref(), buffers.as_ref())?;
        }

        if let Some(validity) = &validity {
            polars_ensure!(validity.len()== views.len(), ComputeError: "validity mask length must match the number of values" )
        }

        let raw_buffers = buffers_into_raw(&buffers);
        Ok(Self {
            data_type,
            views,
            buffers,
            raw_buffers,
            validity,
            phantom: Default::default(),
        })
    }

    /// Creates an empty [`BinaryViewArrayGeneric`], i.e. whose `.len` is zero.
    #[inline]
    pub fn new_empty(data_type: ArrowDataType) -> Self {
        unsafe { Self::new_unchecked(data_type, Buffer::new(), Arc::from([]), None) }
    }

    /// Returns a new null [`BinaryViewArrayGeneric`] of `length`.
    #[inline]
    pub fn new_null(data_type: ArrowDataType, length: usize) -> Self {
        let validity = Some(Bitmap::new_zeroed(length));
        unsafe { Self::new_unchecked(data_type, Buffer::zeroed(length), Arc::from([]), validity) }
    }

    /// Returns the element at index `i`
    /// # Panics
    /// iff `i >= self.len()`
    #[inline]
    pub fn value(&self, i: usize) -> &T {
        assert!(i < self.len());
        unsafe { self.value_unchecked(i) }
    }

    /// Returns the element at index `i`
    /// # Safety
    /// Assumes that the `i < self.len`.
    #[inline]
    pub unsafe fn value_unchecked(&self, i: usize) -> &T {
        let v = *self.views.get_unchecked(i);
        let len = v as u32;

        // view layout:
        // length: 4 bytes
        // prefix: 4 bytes
        // buffer_index: 4 bytes
        // offset: 4 bytes

        // inlined layout:
        // length: 4 bytes
        // data: 12 bytes

        let bytes = if len <= 12 {
            let ptr = self.views.as_ptr() as *const u8;
            std::slice::from_raw_parts(ptr.add(i * 16 + 4), len as usize)
        } else {
            let buffer_idx = (v >> 64) as u32;
            let offset = (v >> 96) as u32;
            let (data_ptr, data_len) = *self.raw_buffers.get_unchecked(buffer_idx as usize);
            let data = std::slice::from_raw_parts(data_ptr, data_len);
            let offset = offset as usize;
            data.get_unchecked(offset..offset + len as usize)
        };
        T::from_bytes_unchecked(bytes)
    }

    /// Returns an iterator of `Option<&T>` over every element of this array.
    pub fn iter(&self) -> ZipValidity<&T, BinaryViewValueIter<T>, BitmapIter> {
        ZipValidity::new_with_validity(self.values_iter(), self.validity.as_ref())
    }

    /// Returns an iterator of `&[u8]` over every element of this array, ignoring the validity
    pub fn values_iter(&self) -> BinaryViewValueIter<T> {
        BinaryViewValueIter::new(self)
    }

    /// Returns an iterator of the non-null values.
    #[inline]
    pub fn non_null_values_iter(&self) -> NonNullValuesIter<'_, BinaryViewArrayGeneric<T>> {
        NonNullValuesIter::new(self, self.validity())
    }

    impl_sliced!();
    impl_mut_validity!();
    impl_into_array!();
}

impl<T: ViewType + ?Sized> Array for BinaryViewArrayGeneric<T> {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn len(&self) -> usize {
        self.views.len()
    }

    fn data_type(&self) -> &ArrowDataType {
        &self.data_type
    }

    fn validity(&self) -> Option<&Bitmap> {
        self.validity.as_ref()
    }

    fn slice(&mut self, offset: usize, length: usize) {
        assert!(
            offset + length <= self.len(),
            "the offset of the new Buffer cannot exceed the existing length"
        );
        unsafe { self.slice_unchecked(offset, length) }
        todo!()
    }

    unsafe fn slice_unchecked(&mut self, offset: usize, length: usize) {
        self.validity = self
            .validity
            .take()
            .map(|bitmap| bitmap.sliced_unchecked(offset, length))
            .filter(|bitmap| bitmap.unset_bits() > 0);
        self.views.slice_unchecked(offset, length);
    }

    fn with_validity(&self, validity: Option<Bitmap>) -> Box<dyn Array> {
        let mut new = self.clone();
        new.validity = validity;
        Box::new(new)
    }

    fn to_boxed(&self) -> Box<dyn Array> {
        Box::new(self.clone())
    }
}
