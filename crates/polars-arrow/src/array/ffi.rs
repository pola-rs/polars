use crate::array::*;
use crate::datatypes::PhysicalType;
use crate::ffi;

/// Trait describing how a struct presents itself to the
/// [C data interface](https://arrow.apache.org/docs/format/CDataInterface.html) (FFI).
/// # Safety
/// Implementing this trait incorrect will lead to UB
pub(crate) unsafe trait ToFfi {
    /// The pointers to the buffers.
    fn buffers(&self) -> Vec<Option<*const u8>>;

    /// The children
    fn children(&self) -> Vec<Box<dyn Array>> {
        vec![]
    }

    /// The offset
    fn offset(&self) -> Option<usize>;

    /// return a partial clone of self with an offset.
    fn to_ffi_aligned(&self) -> Self;
}

/// Trait describing how a struct imports into itself from the
/// [C data interface](https://arrow.apache.org/docs/format/CDataInterface.html) (FFI).
pub(crate) trait FromFfi<T: ffi::ArrowArrayRef>: Sized {
    /// Convert itself from FFI.
    ///
    /// # Safety
    /// This function is intrinsically `unsafe` as it requires the FFI to be made according
    /// to the [C data interface](https://arrow.apache.org/docs/format/CDataInterface.html)
    unsafe fn try_from_ffi(array: T) -> PolarsResult<Self>;
}

macro_rules! ffi_dyn {
    ($array:expr, $ty:ty) => {{
        let array = $array.as_any().downcast_ref::<$ty>().unwrap();
        (
            array.offset().unwrap(),
            array.buffers(),
            array.children(),
            None,
        )
    }};
}

type BuffersChildren = (
    usize,
    Vec<Option<*const u8>>,
    Vec<Box<dyn Array>>,
    Option<Box<dyn Array>>,
);

pub fn offset_buffers_children_dictionary(array: &dyn Array) -> BuffersChildren {
    use PhysicalType::*;
    match array.data_type().to_physical_type() {
        Null => ffi_dyn!(array, NullArray),
        Boolean => ffi_dyn!(array, BooleanArray),
        Primitive(primitive) => with_match_primitive_type_full!(primitive, |$T| {
            ffi_dyn!(array, PrimitiveArray<$T>)
        }),
        Binary => ffi_dyn!(array, BinaryArray<i32>),
        LargeBinary => ffi_dyn!(array, BinaryArray<i64>),
        FixedSizeBinary => ffi_dyn!(array, FixedSizeBinaryArray),
        Utf8 => ffi_dyn!(array, Utf8Array::<i32>),
        LargeUtf8 => ffi_dyn!(array, Utf8Array::<i64>),
        List => ffi_dyn!(array, ListArray::<i32>),
        LargeList => ffi_dyn!(array, ListArray::<i64>),
        FixedSizeList => ffi_dyn!(array, FixedSizeListArray),
        Struct => ffi_dyn!(array, StructArray),
        Union => ffi_dyn!(array, UnionArray),
        Map => ffi_dyn!(array, MapArray),
        BinaryView => ffi_dyn!(array, BinaryViewArray),
        Utf8View => ffi_dyn!(array, Utf8ViewArray),
        Dictionary(key_type) => {
            match_integer_type!(key_type, |$T| {
                let array = array.as_any().downcast_ref::<DictionaryArray<$T>>().unwrap();
                (
                    array.offset().unwrap(),
                    array.buffers(),
                    array.children(),
                    Some(array.values().clone()),
                )
            })
        },
    }
}
