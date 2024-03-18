use crate::array::*;
use crate::bitmap::Bitmap;
use crate::datatypes::PhysicalType;
pub use crate::types::PrimitiveType;
use crate::{match_integer_type, with_match_primitive_type_full};
fn validity_size(validity: Option<&Bitmap>) -> usize {
    validity.as_ref().map(|b| b.as_slice().0.len()).unwrap_or(0)
}

macro_rules! dyn_binary {
    ($array:expr, $ty:ty, $o:ty) => {{
        let array = $array.as_any().downcast_ref::<$ty>().unwrap();
        let offsets = array.offsets().buffer();

        // in case of Binary/Utf8/List the offsets are sliced,
        // not the values buffer
        let values_start = offsets[0] as usize;
        let values_end = offsets[offsets.len() - 1] as usize;

        values_end - values_start
            + offsets.len() * std::mem::size_of::<$o>()
            + validity_size(array.validity())
    }};
}

fn binview_size<T: ViewType + ?Sized>(array: &BinaryViewArrayGeneric<T>) -> usize {
    // We choose the optimal usage as data can be shared across buffers.
    // If we would sum all buffers we overestimate memory usage and trigger OOC when not needed.
    array.total_bytes_len()
}

/// Returns the total (heap) allocated size of the array in bytes.
/// # Implementation
/// This estimation is the sum of the size of its buffers, validity, including nested arrays.
/// Multiple arrays may share buffers and bitmaps. Therefore, the size of 2 arrays is not the
/// sum of the sizes computed from this function. In particular, [`StructArray`]'s size is an upper bound.
///
/// When an array is sliced, its allocated size remains constant because the buffer unchanged.
/// However, this function will yield a smaller number. This is because this function returns
/// the visible size of the buffer, not its total capacity.
///
/// FFI buffers are included in this estimation.
pub fn estimated_bytes_size(array: &dyn Array) -> usize {
    use PhysicalType::*;
    match array.data_type().to_physical_type() {
        Null => 0,
        Boolean => {
            let array = array.as_any().downcast_ref::<BooleanArray>().unwrap();
            array.values().as_slice().0.len() + validity_size(array.validity())
        },
        Primitive(PrimitiveType::DaysMs) => {
            let array = array.as_any().downcast_ref::<DaysMsArray>().unwrap();
            array.values().len() * std::mem::size_of::<i32>() * 2 + validity_size(array.validity())
        },
        Primitive(primitive) => with_match_primitive_type_full!(primitive, |$T| {
            let array = array
                .as_any()
                .downcast_ref::<PrimitiveArray<$T>>()
                .unwrap();

            array.values().len() * std::mem::size_of::<$T>() + validity_size(array.validity())
        }),
        Binary => dyn_binary!(array, BinaryArray<i32>, i32),
        FixedSizeBinary => {
            let array = array
                .as_any()
                .downcast_ref::<FixedSizeBinaryArray>()
                .unwrap();
            array.values().len() + validity_size(array.validity())
        },
        LargeBinary => dyn_binary!(array, BinaryArray<i64>, i64),
        Utf8 => dyn_binary!(array, Utf8Array<i32>, i32),
        LargeUtf8 => dyn_binary!(array, Utf8Array<i64>, i64),
        List => {
            let array = array.as_any().downcast_ref::<ListArray<i32>>().unwrap();
            estimated_bytes_size(array.values().as_ref())
                + array.offsets().len_proxy() * std::mem::size_of::<i32>()
                + validity_size(array.validity())
        },
        FixedSizeList => {
            let array = array.as_any().downcast_ref::<FixedSizeListArray>().unwrap();
            estimated_bytes_size(array.values().as_ref()) + validity_size(array.validity())
        },
        LargeList => {
            let array = array.as_any().downcast_ref::<ListArray<i64>>().unwrap();
            estimated_bytes_size(array.values().as_ref())
                + array.offsets().len_proxy() * std::mem::size_of::<i64>()
                + validity_size(array.validity())
        },
        Struct => {
            let array = array.as_any().downcast_ref::<StructArray>().unwrap();
            array
                .values()
                .iter()
                .map(|x| x.as_ref())
                .map(estimated_bytes_size)
                .sum::<usize>()
                + validity_size(array.validity())
        },
        Union => {
            let array = array.as_any().downcast_ref::<UnionArray>().unwrap();
            let types = array.types().len() * std::mem::size_of::<i8>();
            let offsets = array
                .offsets()
                .as_ref()
                .map(|x| x.len() * std::mem::size_of::<i32>())
                .unwrap_or_default();
            let fields = array
                .fields()
                .iter()
                .map(|x| x.as_ref())
                .map(estimated_bytes_size)
                .sum::<usize>();
            types + offsets + fields
        },
        Dictionary(key_type) => match_integer_type!(key_type, |$T| {
            let array = array
                .as_any()
                .downcast_ref::<DictionaryArray<$T>>()
                .unwrap();
            estimated_bytes_size(array.keys()) + estimated_bytes_size(array.values().as_ref())
        }),
        Utf8View => binview_size::<str>(array.as_any().downcast_ref().unwrap()),
        BinaryView => binview_size::<[u8]>(array.as_any().downcast_ref().unwrap()),
        Map => {
            let array = array.as_any().downcast_ref::<MapArray>().unwrap();
            let offsets = array.offsets().len_proxy() * std::mem::size_of::<i32>();
            offsets + estimated_bytes_size(array.field().as_ref()) + validity_size(array.validity())
        },
    }
}
