//! Functionality to mmap in-memory data regions.
use std::sync::Arc;

use polars_error::{polars_bail, PolarsResult};

use super::{ArrowArray, InternalArrowArray};
use crate::array::{BooleanArray, FromFfi, PrimitiveArray};
use crate::datatypes::ArrowDataType;
use crate::types::NativeType;

#[allow(dead_code)]
struct PrivateData<T> {
    // the owner of the pointers' regions
    data: T,
    buffers_ptr: Box<[*const std::os::raw::c_void]>,
    children_ptr: Box<[*mut ArrowArray]>,
    dictionary_ptr: Option<*mut ArrowArray>,
}

pub(crate) unsafe fn create_array<
    T,
    I: Iterator<Item = Option<*const u8>>,
    II: Iterator<Item = ArrowArray>,
>(
    data: Arc<T>,
    num_rows: usize,
    null_count: usize,
    buffers: I,
    children: II,
    dictionary: Option<ArrowArray>,
    offset: Option<usize>,
) -> ArrowArray {
    let buffers_ptr = buffers
        .map(|maybe_buffer| match maybe_buffer {
            Some(b) => b as *const std::os::raw::c_void,
            None => std::ptr::null(),
        })
        .collect::<Box<[_]>>();
    let n_buffers = buffers_ptr.len() as i64;

    let children_ptr = children
        .map(|child| Box::into_raw(Box::new(child)))
        .collect::<Box<_>>();
    let n_children = children_ptr.len() as i64;

    let dictionary_ptr = dictionary.map(|array| Box::into_raw(Box::new(array)));

    let mut private_data = Box::new(PrivateData::<Arc<T>> {
        data,
        buffers_ptr,
        children_ptr,
        dictionary_ptr,
    });

    ArrowArray {
        length: num_rows as i64,
        null_count: null_count as i64,
        offset: offset.unwrap_or(0) as i64, // Unwrap: IPC files are by definition not offset
        n_buffers,
        n_children,
        buffers: private_data.buffers_ptr.as_mut_ptr(),
        children: private_data.children_ptr.as_mut_ptr(),
        dictionary: private_data.dictionary_ptr.unwrap_or(std::ptr::null_mut()),
        release: Some(release::<Arc<T>>),
        private_data: Box::into_raw(private_data) as *mut ::std::os::raw::c_void,
    }
}

/// callback used to drop [`ArrowArray`] when it is exported specified for [`PrivateData`].
unsafe extern "C" fn release<T>(array: *mut ArrowArray) {
    if array.is_null() {
        return;
    }
    let array = &mut *array;

    // take ownership of `private_data`, therefore dropping it
    let private = Box::from_raw(array.private_data as *mut PrivateData<T>);
    for child in private.children_ptr.iter() {
        let _ = Box::from_raw(*child);
    }

    if let Some(ptr) = private.dictionary_ptr {
        let _ = Box::from_raw(ptr);
    }

    array.release = None;
}

/// Creates a (non-null) [`PrimitiveArray`] from a slice of values.
/// This does not have memcopy and is the fastest way to create a [`PrimitiveArray`].
///
/// This can be useful if you want to apply arrow kernels on slices without incurring
/// a memcopy cost.
///
/// # Safety
///
/// Using this function is not unsafe, but the returned PrimitiveArray's lifetime is bound to the lifetime
/// of the slice. The returned [`PrimitiveArray`] _must not_ outlive the passed slice.
pub unsafe fn slice<T: NativeType>(slice: &[T]) -> PrimitiveArray<T> {
    slice_and_owner(slice, ())
}

/// Creates a (non-null) [`PrimitiveArray`] from a slice of values.
/// This does not have memcopy and is the fastest way to create a [`PrimitiveArray`].
///
/// This can be useful if you want to apply arrow kernels on slices without incurring
/// a memcopy cost.
///
/// # Safety
///
/// The caller must ensure the passed `owner` ensures the data remains alive.
pub unsafe fn slice_and_owner<T: NativeType, O>(slice: &[T], owner: O) -> PrimitiveArray<T> {
    let num_rows = slice.len();
    let null_count = 0;
    let validity = None;

    let data: &[u8] = bytemuck::cast_slice(slice);
    let ptr = data.as_ptr();
    let data = Arc::new(owner);

    // SAFETY: the underlying assumption of this function: the array will not be used
    // beyond the
    let array = create_array(
        data,
        num_rows,
        null_count,
        [validity, Some(ptr)].into_iter(),
        [].into_iter(),
        None,
        None,
    );
    let array = InternalArrowArray::new(array, T::PRIMITIVE.into());

    // SAFETY: we just created a valid array
    unsafe { PrimitiveArray::<T>::try_from_ffi(array) }.unwrap()
}

/// Creates a (non-null) [`BooleanArray`] from a slice of bits.
/// This does not have memcopy and is the fastest way to create a [`BooleanArray`].
///
/// This can be useful if you want to apply arrow kernels on slices without
/// incurring a memcopy cost.
///
/// The `offset` indicates where the first bit starts in the first byte.
///
/// # Safety
///
/// Using this function is not unsafe, but the returned BooleanArrays's lifetime
/// is bound to the lifetime of the slice. The returned [`BooleanArray`] _must
/// not_ outlive the passed slice.
pub unsafe fn bitmap(data: &[u8], offset: usize, length: usize) -> PolarsResult<BooleanArray> {
    bitmap_and_owner(data, offset, length, ())
}

/// Creates a (non-null) [`BooleanArray`] from a slice of bits.
/// This does not have memcopy and is the fastest way to create a [`BooleanArray`].
///
/// This can be useful if you want to apply arrow kernels on slices without
/// incurring a memcopy cost.
///
/// The `offset` indicates where the first bit starts in the first byte.
///
/// # Safety
///
/// The caller must ensure the passed `owner` ensures the data remains alive.
pub unsafe fn bitmap_and_owner<O>(
    data: &[u8],
    offset: usize,
    length: usize,
    owner: O,
) -> PolarsResult<BooleanArray> {
    if offset >= 8 {
        polars_bail!(InvalidOperation: "offset should be < 8")
    };
    if length > data.len() * 8 - offset {
        polars_bail!(InvalidOperation: "given length is oob")
    }
    let null_count = 0;
    let validity = None;

    let ptr = data.as_ptr();
    let data = Arc::new(owner);

    // SAFETY: the underlying assumption of this function: the array will not be used
    // beyond the
    let array = create_array(
        data,
        length,
        null_count,
        [validity, Some(ptr)].into_iter(),
        [].into_iter(),
        None,
        Some(offset),
    );
    let array = InternalArrowArray::new(array, ArrowDataType::Boolean);

    // SAFETY: we just created a valid array
    Ok(unsafe { BooleanArray::try_from_ffi(array) }.unwrap())
}
