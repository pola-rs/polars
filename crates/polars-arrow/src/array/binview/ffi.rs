use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use polars_error::PolarsResult;

use super::BinaryViewArrayGeneric;
use crate::array::binview::ViewType;
use crate::array::{FromFfi, ToFfi};
use crate::bitmap::align;
use crate::ffi;

unsafe impl<T: ViewType + ?Sized> ToFfi for BinaryViewArrayGeneric<T> {
    fn buffers(&self) -> Vec<Option<*const u8>> {
        let mut buffers = Vec::with_capacity(self.buffers.len() + 2);
        buffers.push(self.validity.as_ref().map(|x| x.as_ptr()));
        buffers.push(Some(self.views.storage_ptr().cast::<u8>()));
        buffers.extend(self.buffers.iter().map(|b| Some(b.storage_ptr())));
        buffers
    }

    fn offset(&self) -> Option<usize> {
        let offset = self.views.offset();
        if let Some(bitmap) = self.validity.as_ref() {
            if bitmap.offset() == offset {
                Some(offset)
            } else {
                None
            }
        } else {
            Some(offset)
        }
    }

    fn to_ffi_aligned(&self) -> Self {
        let offset = self.views.offset();

        let validity = self.validity.as_ref().map(|bitmap| {
            if bitmap.offset() == offset {
                bitmap.clone()
            } else {
                align(bitmap, offset)
            }
        });

        Self {
            data_type: self.data_type.clone(),
            validity,
            views: self.views.clone(),
            buffers: self.buffers.clone(),
            raw_buffers: self.raw_buffers.clone(),
            phantom: Default::default(),
            total_bytes_len: AtomicU64::new(self.total_bytes_len.load(Ordering::Relaxed)),
            total_buffer_len: self.total_buffer_len,
        }
    }
}

impl<T: ViewType + ?Sized, A: ffi::ArrowArrayRef> FromFfi<A> for BinaryViewArrayGeneric<T> {
    unsafe fn try_from_ffi(array: A) -> PolarsResult<Self> {
        let data_type = array.data_type().clone();

        let validity = unsafe { array.validity() }?;
        let views = unsafe { array.buffer::<u128>(1) }?;

        let n = array.n_buffers() - 2;
        let mut buffers = Vec::with_capacity(n);

        for i in 2..n + 2 {
            let values = unsafe { array.buffer::<u8>(i) }?;
            buffers.push(values);
        }

        Ok(Self::new_unchecked_unknown_md(
            data_type,
            views,
            Arc::from(buffers),
            validity,
        ))
    }
}
