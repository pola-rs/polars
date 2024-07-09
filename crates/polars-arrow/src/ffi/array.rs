//! Contains functionality to load an ArrayData from the C Data Interface
use std::sync::Arc;

use polars_error::{polars_bail, PolarsResult};

use super::ArrowArray;
use crate::array::*;
use crate::bitmap::utils::bytes_for;
use crate::bitmap::Bitmap;
use crate::buffer::{Buffer, Bytes, BytesAllocator};
use crate::datatypes::{ArrowDataType, PhysicalType};
use crate::ffi::schema::get_child;
use crate::types::NativeType;
use crate::{match_integer_type, with_match_primitive_type_full};

/// Reads a valid `ffi` interface into a `Box<dyn Array>`
/// # Errors
/// If and only if:
/// * the interface is not valid (e.g. a null pointer)
pub unsafe fn try_from<A: ArrowArrayRef>(array: A) -> PolarsResult<Box<dyn Array>> {
    use PhysicalType::*;
    Ok(match array.data_type().to_physical_type() {
        Null => Box::new(NullArray::try_from_ffi(array)?),
        Boolean => Box::new(BooleanArray::try_from_ffi(array)?),
        Primitive(primitive) => with_match_primitive_type_full!(primitive, |$T| {
            Box::new(PrimitiveArray::<$T>::try_from_ffi(array)?)
        }),
        Utf8 => Box::new(Utf8Array::<i32>::try_from_ffi(array)?),
        LargeUtf8 => Box::new(Utf8Array::<i64>::try_from_ffi(array)?),
        Binary => Box::new(BinaryArray::<i32>::try_from_ffi(array)?),
        LargeBinary => Box::new(BinaryArray::<i64>::try_from_ffi(array)?),
        FixedSizeBinary => Box::new(FixedSizeBinaryArray::try_from_ffi(array)?),
        List => Box::new(ListArray::<i32>::try_from_ffi(array)?),
        LargeList => Box::new(ListArray::<i64>::try_from_ffi(array)?),
        FixedSizeList => Box::new(FixedSizeListArray::try_from_ffi(array)?),
        Struct => Box::new(StructArray::try_from_ffi(array)?),
        Dictionary(key_type) => {
            match_integer_type!(key_type, |$T| {
                Box::new(DictionaryArray::<$T>::try_from_ffi(array)?)
            })
        },
        Union => Box::new(UnionArray::try_from_ffi(array)?),
        Map => Box::new(MapArray::try_from_ffi(array)?),
        BinaryView => Box::new(BinaryViewArray::try_from_ffi(array)?),
        Utf8View => Box::new(Utf8ViewArray::try_from_ffi(array)?),
    })
}

// Sound because the arrow specification does not allow multiple implementations
// to change this struct
// This is intrinsically impossible to prove because the implementations agree
// on this as part of the Arrow specification
unsafe impl Send for ArrowArray {}
unsafe impl Sync for ArrowArray {}

impl Drop for ArrowArray {
    fn drop(&mut self) {
        match self.release {
            None => (),
            Some(release) => unsafe { release(self) },
        };
    }
}

// callback used to drop [ArrowArray] when it is exported
unsafe extern "C" fn c_release_array(array: *mut ArrowArray) {
    if array.is_null() {
        return;
    }
    let array = &mut *array;

    // take ownership of `private_data`, therefore dropping it
    let private = Box::from_raw(array.private_data as *mut PrivateData);
    for child in private.children_ptr.iter() {
        let _ = Box::from_raw(*child);
    }

    if let Some(ptr) = private.dictionary_ptr {
        let _ = Box::from_raw(ptr);
    }

    array.release = None;
}

#[allow(dead_code)]
struct PrivateData {
    array: Box<dyn Array>,
    buffers_ptr: Box<[*const std::os::raw::c_void]>,
    children_ptr: Box<[*mut ArrowArray]>,
    dictionary_ptr: Option<*mut ArrowArray>,
    variadic_buffer_sizes: Box<[i64]>,
}

impl ArrowArray {
    /// creates a new `ArrowArray` from existing data.
    ///
    /// # Safety
    /// This method releases `buffers`. Consumers of this struct *must* call `release` before
    /// releasing this struct, or contents in `buffers` leak.
    pub(crate) fn new(array: Box<dyn Array>) -> Self {
        let needs_variadic_buffer_sizes = matches!(
            array.data_type(),
            ArrowDataType::BinaryView | ArrowDataType::Utf8View
        );

        let (offset, mut buffers, children, dictionary) =
            offset_buffers_children_dictionary(array.as_ref());

        let variadic_buffer_sizes = if needs_variadic_buffer_sizes {
            #[cfg(feature = "compute_cast")]
            {
                let arr = crate::compute::cast::cast_unchecked(
                    array.as_ref(),
                    &ArrowDataType::BinaryView,
                )
                .unwrap();
                let arr = arr.as_any().downcast_ref::<BinaryViewArray>().unwrap();
                let boxed = arr.variadic_buffer_lengths().into_boxed_slice();
                let ptr = boxed.as_ptr().cast::<u8>();
                buffers.push(Some(ptr));
                boxed
            }
            #[cfg(not(feature = "compute_cast"))]
            {
                panic!("activate 'compute_cast' feature")
            }
        } else {
            Box::from([])
        };

        let buffers_ptr = buffers
            .iter()
            .map(|maybe_buffer| match maybe_buffer {
                Some(b) => *b as *const std::os::raw::c_void,
                None => std::ptr::null(),
            })
            .collect::<Box<[_]>>();
        let n_buffers = buffers.len() as i64;

        let children_ptr = children
            .into_iter()
            .map(|child| Box::into_raw(Box::new(ArrowArray::new(child))))
            .collect::<Box<_>>();
        let n_children = children_ptr.len() as i64;

        let dictionary_ptr =
            dictionary.map(|array| Box::into_raw(Box::new(ArrowArray::new(array))));

        let length = array.len() as i64;
        let null_count = array.null_count() as i64;

        let mut private_data = Box::new(PrivateData {
            array,
            buffers_ptr,
            children_ptr,
            dictionary_ptr,
            variadic_buffer_sizes,
        });

        Self {
            length,
            null_count,
            offset: offset as i64,
            n_buffers,
            n_children,
            buffers: private_data.buffers_ptr.as_mut_ptr(),
            children: private_data.children_ptr.as_mut_ptr(),
            dictionary: private_data.dictionary_ptr.unwrap_or(std::ptr::null_mut()),
            release: Some(c_release_array),
            private_data: Box::into_raw(private_data) as *mut ::std::os::raw::c_void,
        }
    }

    /// creates an empty [`ArrowArray`], which can be used to import data into
    pub fn empty() -> Self {
        Self {
            length: 0,
            null_count: 0,
            offset: 0,
            n_buffers: 0,
            n_children: 0,
            buffers: std::ptr::null_mut(),
            children: std::ptr::null_mut(),
            dictionary: std::ptr::null_mut(),
            release: None,
            private_data: std::ptr::null_mut(),
        }
    }

    /// the length of the array
    pub(crate) fn len(&self) -> usize {
        self.length as usize
    }

    /// the offset of the array
    pub(crate) fn offset(&self) -> usize {
        self.offset as usize
    }

    /// the null count of the array
    pub(crate) fn null_count(&self) -> usize {
        self.null_count as usize
    }
}

/// # Safety
/// The caller must ensure that the buffer at index `i` is not mutably shared.
unsafe fn get_buffer_ptr<T: NativeType>(
    array: &ArrowArray,
    data_type: &ArrowDataType,
    index: usize,
) -> PolarsResult<*mut T> {
    if array.buffers.is_null() {
        polars_bail!( ComputeError:
            "an ArrowArray of type {data_type:?} must have non-null buffers"
        );
    }

    if array
        .buffers
        .align_offset(std::mem::align_of::<*mut *const u8>())
        != 0
    {
        polars_bail!( ComputeError:
            "an ArrowArray of type {data_type:?}
            must have buffer {index} aligned to type {}",
            std::any::type_name::<*mut *const u8>()
        );
    }
    let buffers = array.buffers as *mut *const u8;

    if index >= array.n_buffers as usize {
        polars_bail!(ComputeError:
            "An ArrowArray of type {data_type:?}
             must have buffer {index}."
        )
    }

    let ptr = *buffers.add(index);
    if ptr.is_null() {
        polars_bail!(ComputeError:
            "An array of type {data_type:?}
            must have a non-null buffer {index}"
        )
    }

    // note: we can't prove that this pointer is not mutably shared - part of the safety invariant
    Ok(ptr as *mut T)
}

unsafe fn create_buffer_known_len<T: NativeType>(
    array: &ArrowArray,
    data_type: &ArrowDataType,
    owner: InternalArrowArray,
    len: usize,
    index: usize,
) -> PolarsResult<Buffer<T>> {
    if len == 0 {
        return Ok(Buffer::new());
    }
    let ptr: *mut T = get_buffer_ptr(array, data_type, index)?;
    let bytes = Bytes::from_foreign(ptr, len, BytesAllocator::InternalArrowArray(owner));
    Ok(Buffer::from_bytes(bytes))
}

/// returns the buffer `i` of `array` interpreted as a [`Buffer`].
/// # Safety
/// This function is safe iff:
/// * the buffers up to position `index` are valid for the declared length
/// * the buffers' pointers are not mutably shared for the lifetime of `owner`
unsafe fn create_buffer<T: NativeType>(
    array: &ArrowArray,
    data_type: &ArrowDataType,
    owner: InternalArrowArray,
    index: usize,
) -> PolarsResult<Buffer<T>> {
    let len = buffer_len(array, data_type, index)?;

    if len == 0 {
        return Ok(Buffer::new());
    }

    let offset = buffer_offset(array, data_type, index);
    let ptr: *mut T = get_buffer_ptr(array, data_type, index)?;

    // We have to check alignment.
    // This is the zero-copy path.
    if ptr.align_offset(std::mem::align_of::<T>()) == 0 {
        let bytes = Bytes::from_foreign(ptr, len, BytesAllocator::InternalArrowArray(owner));
        Ok(Buffer::from_bytes(bytes).sliced(offset, len - offset))
    }
    // This is the path where alignment isn't correct.
    // We copy the data to a new vec
    else {
        let buf = std::slice::from_raw_parts(ptr, len - offset).to_vec();
        Ok(Buffer::from(buf))
    }
}

/// returns the buffer `i` of `array` interpreted as a [`Bitmap`].
/// # Safety
/// This function is safe iff:
/// * the buffer at position `index` is valid for the declared length
/// * the buffers' pointer is not mutable for the lifetime of `owner`
unsafe fn create_bitmap(
    array: &ArrowArray,
    data_type: &ArrowDataType,
    owner: InternalArrowArray,
    index: usize,
    // if this is the validity bitmap
    // we can use the null count directly
    is_validity: bool,
) -> PolarsResult<Bitmap> {
    let len: usize = array.length.try_into().expect("length to fit in `usize`");
    if len == 0 {
        return Ok(Bitmap::new());
    }
    let ptr = get_buffer_ptr(array, data_type, index)?;

    // Pointer of u8 has alignment 1, so we don't have to check alignment.

    let offset: usize = array.offset.try_into().expect("offset to fit in `usize`");
    let bytes_len = bytes_for(offset + len);
    let bytes = Bytes::from_foreign(ptr, bytes_len, BytesAllocator::InternalArrowArray(owner));

    let null_count = if is_validity {
        Some(array.null_count())
    } else {
        None
    };
    Ok(Bitmap::from_inner_unchecked(
        Arc::new(bytes),
        offset,
        len,
        null_count,
    ))
}

fn buffer_offset(array: &ArrowArray, data_type: &ArrowDataType, i: usize) -> usize {
    use PhysicalType::*;
    match (data_type.to_physical_type(), i) {
        (LargeUtf8, 2) | (LargeBinary, 2) | (Utf8, 2) | (Binary, 2) => 0,
        (FixedSizeBinary, 1) => {
            if let ArrowDataType::FixedSizeBinary(size) = data_type.to_logical_type() {
                let offset: usize = array.offset.try_into().expect("Offset to fit in `usize`");
                offset * *size
            } else {
                unreachable!()
            }
        },
        _ => array.offset.try_into().expect("Offset to fit in `usize`"),
    }
}

/// Returns the length, in slots, of the buffer `i` (indexed according to the C data interface)
unsafe fn buffer_len(
    array: &ArrowArray,
    data_type: &ArrowDataType,
    i: usize,
) -> PolarsResult<usize> {
    Ok(match (data_type.to_physical_type(), i) {
        (PhysicalType::FixedSizeBinary, 1) => {
            if let ArrowDataType::FixedSizeBinary(size) = data_type.to_logical_type() {
                *size * (array.offset as usize + array.length as usize)
            } else {
                unreachable!()
            }
        },
        (PhysicalType::FixedSizeList, 1) => {
            if let ArrowDataType::FixedSizeList(_, size) = data_type.to_logical_type() {
                *size * (array.offset as usize + array.length as usize)
            } else {
                unreachable!()
            }
        },
        (PhysicalType::Utf8, 1)
        | (PhysicalType::LargeUtf8, 1)
        | (PhysicalType::Binary, 1)
        | (PhysicalType::LargeBinary, 1)
        | (PhysicalType::List, 1)
        | (PhysicalType::LargeList, 1)
        | (PhysicalType::Map, 1) => {
            // the len of the offset buffer (buffer 1) equals length + 1
            array.offset as usize + array.length as usize + 1
        },
        (PhysicalType::BinaryView, 1) | (PhysicalType::Utf8View, 1) => {
            array.offset as usize + array.length as usize
        },
        (PhysicalType::Utf8, 2) | (PhysicalType::Binary, 2) => {
            // the len of the data buffer (buffer 2) equals the last value of the offset buffer (buffer 1)
            let len = buffer_len(array, data_type, 1)?;
            // first buffer is the null buffer => add(1)
            let offset_buffer = unsafe { *(array.buffers as *mut *const u8).add(1) };
            // interpret as i32
            let offset_buffer = offset_buffer as *const i32;
            // get last offset

            (unsafe { *offset_buffer.add(len - 1) }) as usize
        },
        (PhysicalType::LargeUtf8, 2) | (PhysicalType::LargeBinary, 2) => {
            // the len of the data buffer (buffer 2) equals the last value of the offset buffer (buffer 1)
            let len = buffer_len(array, data_type, 1)?;
            // first buffer is the null buffer => add(1)
            let offset_buffer = unsafe { *(array.buffers as *mut *const u8).add(1) };
            // interpret as i64
            let offset_buffer = offset_buffer as *const i64;
            // get last offset
            (unsafe { *offset_buffer.add(len - 1) }) as usize
        },
        // buffer len of primitive types
        _ => array.offset as usize + array.length as usize,
    })
}

/// # Safety
///
/// This function is safe iff:
/// * `array.children` at `index` is valid
/// * `array.children` is not mutably shared for the lifetime of `parent`
/// * the pointer of `array.children` at `index` is valid
/// * the pointer of `array.children` at `index` is not mutably shared for the lifetime of `parent`
unsafe fn create_child(
    array: &ArrowArray,
    data_type: &ArrowDataType,
    parent: InternalArrowArray,
    index: usize,
) -> PolarsResult<ArrowArrayChild<'static>> {
    let data_type = get_child(data_type, index)?;

    // catch what we can
    if array.children.is_null() {
        polars_bail!(ComputeError: "an ArrowArray of type {data_type:?} must have non-null children");
    }

    if index >= array.n_children as usize {
        polars_bail!(ComputeError:
            "an ArrowArray of type {data_type:?}
             must have child {index}."
        );
    }

    // SAFETY: part of the invariant
    let arr_ptr = unsafe { *array.children.add(index) };

    // catch what we can
    if arr_ptr.is_null() {
        polars_bail!(ComputeError:
            "an array of type {data_type:?}
            must have a non-null child {index}"
        )
    }

    // SAFETY: invariant of this function
    let arr_ptr = unsafe { &*arr_ptr };
    Ok(ArrowArrayChild::new(arr_ptr, data_type, parent))
}

/// # Safety
///
/// This function is safe iff:
/// * `array.dictionary` is valid
/// * `array.dictionary` is not mutably shared for the lifetime of `parent`
unsafe fn create_dictionary(
    array: &ArrowArray,
    data_type: &ArrowDataType,
    parent: InternalArrowArray,
) -> PolarsResult<Option<ArrowArrayChild<'static>>> {
    if let ArrowDataType::Dictionary(_, values, _) = data_type {
        let data_type = values.as_ref().clone();
        // catch what we can
        if array.dictionary.is_null() {
            polars_bail!(ComputeError:
                "an array of type {data_type:?}
                must have a non-null dictionary"
            )
        }

        // SAFETY: part of the invariant
        let array = unsafe { &*array.dictionary };
        Ok(Some(ArrowArrayChild::new(array, data_type, parent)))
    } else {
        Ok(None)
    }
}

pub trait ArrowArrayRef: std::fmt::Debug {
    fn owner(&self) -> InternalArrowArray {
        (*self.parent()).clone()
    }

    /// returns the null bit buffer.
    /// Rust implementation uses a buffer that is not part of the array of buffers.
    /// The C Data interface's null buffer is part of the array of buffers.
    ///
    /// # Safety
    /// The caller must guarantee that the buffer `index` corresponds to a bitmap.
    /// This function assumes that the bitmap created from FFI is valid; this is impossible to prove.
    unsafe fn validity(&self) -> PolarsResult<Option<Bitmap>> {
        if self.array().null_count() == 0 {
            Ok(None)
        } else {
            create_bitmap(self.array(), self.data_type(), self.owner(), 0, true).map(Some)
        }
    }

    /// # Safety
    /// The caller must guarantee that the buffer `index` corresponds to a buffer.
    /// This function assumes that the buffer created from FFI is valid; this is impossible to prove.
    unsafe fn buffer<T: NativeType>(&self, index: usize) -> PolarsResult<Buffer<T>> {
        create_buffer::<T>(self.array(), self.data_type(), self.owner(), index)
    }

    /// # Safety
    /// The caller must guarantee that the buffer `index` corresponds to a buffer.
    /// This function assumes that the buffer created from FFI is valid; this is impossible to prove.
    unsafe fn buffer_known_len<T: NativeType>(
        &self,
        index: usize,
        len: usize,
    ) -> PolarsResult<Buffer<T>> {
        create_buffer_known_len::<T>(self.array(), self.data_type(), self.owner(), len, index)
    }

    /// # Safety
    /// This function is safe iff:
    /// * the buffer at position `index` is valid for the declared length
    /// * the buffers' pointer is not mutable for the lifetime of `owner`
    unsafe fn bitmap(&self, index: usize) -> PolarsResult<Bitmap> {
        create_bitmap(self.array(), self.data_type(), self.owner(), index, false)
    }

    /// # Safety
    /// * `array.children` at `index` is valid
    /// * `array.children` is not mutably shared for the lifetime of `parent`
    /// * the pointer of `array.children` at `index` is valid
    /// * the pointer of `array.children` at `index` is not mutably shared for the lifetime of `parent`
    unsafe fn child(&self, index: usize) -> PolarsResult<ArrowArrayChild> {
        create_child(self.array(), self.data_type(), self.parent().clone(), index)
    }

    unsafe fn dictionary(&self) -> PolarsResult<Option<ArrowArrayChild>> {
        create_dictionary(self.array(), self.data_type(), self.parent().clone())
    }

    fn n_buffers(&self) -> usize;

    fn offset(&self) -> usize;
    fn length(&self) -> usize;

    fn parent(&self) -> &InternalArrowArray;
    fn array(&self) -> &ArrowArray;
    fn data_type(&self) -> &ArrowDataType;
}

/// Struct used to move an Array from and to the C Data Interface.
/// Its main responsibility is to expose functionality that requires
/// both [ArrowArray] and [ArrowSchema].
///
/// This struct has two main paths:
///
/// ## Import from the C Data Interface
/// * [InternalArrowArray::empty] to allocate memory to be filled by an external call
/// * [InternalArrowArray::try_from_raw] to consume two non-null allocated pointers
/// ## Export to the C Data Interface
/// * [InternalArrowArray::try_new] to create a new [InternalArrowArray] from Rust-specific information
/// * [InternalArrowArray::into_raw] to expose two pointers for [ArrowArray] and [ArrowSchema].
///
/// # Safety
/// Whoever creates this struct is responsible for releasing their resources. Specifically,
/// consumers *must* call [InternalArrowArray::into_raw] and take ownership of the individual pointers,
/// calling [ArrowArray::release] and [ArrowSchema::release] accordingly.
///
/// Furthermore, this struct assumes that the incoming data agrees with the C data interface.
#[derive(Debug, Clone)]
pub struct InternalArrowArray {
    // Arc is used for sharability since this is immutable
    array: Arc<ArrowArray>,
    // Arced to reduce cost of cloning
    data_type: Arc<ArrowDataType>,
}

impl InternalArrowArray {
    pub fn new(array: ArrowArray, data_type: ArrowDataType) -> Self {
        Self {
            array: Arc::new(array),
            data_type: Arc::new(data_type),
        }
    }
}

impl ArrowArrayRef for InternalArrowArray {
    /// the data_type as declared in the schema
    fn data_type(&self) -> &ArrowDataType {
        &self.data_type
    }

    fn parent(&self) -> &InternalArrowArray {
        self
    }

    fn array(&self) -> &ArrowArray {
        self.array.as_ref()
    }

    fn n_buffers(&self) -> usize {
        self.array.n_buffers as usize
    }

    fn offset(&self) -> usize {
        self.array.offset as usize
    }

    fn length(&self) -> usize {
        self.array.length as usize
    }
}

#[derive(Debug)]
pub struct ArrowArrayChild<'a> {
    array: &'a ArrowArray,
    data_type: ArrowDataType,
    parent: InternalArrowArray,
}

impl<'a> ArrowArrayRef for ArrowArrayChild<'a> {
    /// the data_type as declared in the schema
    fn data_type(&self) -> &ArrowDataType {
        &self.data_type
    }

    fn parent(&self) -> &InternalArrowArray {
        &self.parent
    }

    fn array(&self) -> &ArrowArray {
        self.array
    }

    fn n_buffers(&self) -> usize {
        self.array.n_buffers as usize
    }

    fn offset(&self) -> usize {
        self.array.offset as usize
    }

    fn length(&self) -> usize {
        self.array.length as usize
    }
}

impl<'a> ArrowArrayChild<'a> {
    fn new(array: &'a ArrowArray, data_type: ArrowDataType, parent: InternalArrowArray) -> Self {
        Self {
            array,
            data_type,
            parent,
        }
    }
}
