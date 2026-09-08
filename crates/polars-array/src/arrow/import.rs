//! Importing the Arrow arrays of `polars-arrow` as the arrays of this crate.

use std::any::Any;

use arrow::array::{
    Array, BinaryArray, BinaryViewArrayGeneric, BooleanArray, FixedSizeBinaryArray,
    FixedSizeListArray, ListArray, NullArray, PrimitiveArray, StructArray, Utf8Array, View,
    ViewType,
};
use arrow::datatypes::{PhysicalType, PrimitiveType};
use arrow::offset::OffsetsBuffer;
use arrow::types::{NativeType, Offset, days_ms, i256, months_days_ns};
use polars_buffer::Buffer;
use polars_utils::float16::pf16;

use crate::bitmap::PlBitmap;
use crate::{
    PlArray, PlBinaryArray, PlBinaryViewArray, PlBooleanArray, PlFixedSizeBinaryArray,
    PlFixedSizeListArray, PlListArray, PlNullArray, PlPrimitiveArray, PlStructArray,
    PlUtf8ViewArray,
};

/// Imports an Arrow array as the array of this crate that holds the same elements.
pub fn from_arrow(array: &dyn Array) -> Box<dyn PlArray> {
    match array.dtype().to_physical_type() {
        PhysicalType::Null => Box::new(null_from_arrow(downcast(array))),
        PhysicalType::Boolean => Box::new(boolean_from_arrow(downcast(array))),

        PhysicalType::Primitive(primitive) => primitive_from_arrow_dyn(array, primitive),

        PhysicalType::Binary => Box::new(binary_from_arrow(downcast::<BinaryArray<i32>>(array))),
        PhysicalType::LargeBinary => {
            Box::new(binary_from_arrow(downcast::<BinaryArray<i64>>(array)))
        },
        PhysicalType::Utf8 => Box::new(utf8_from_arrow(downcast::<Utf8Array<i32>>(array))),
        PhysicalType::LargeUtf8 => Box::new(utf8_from_arrow(downcast::<Utf8Array<i64>>(array))),

        PhysicalType::BinaryView => Box::new(binary_view_from_arrow(downcast::<
            BinaryViewArrayGeneric<[u8]>,
        >(array))),
        PhysicalType::Utf8View => Box::new(utf8_view_from_arrow(downcast::<
            BinaryViewArrayGeneric<str>,
        >(array))),

        PhysicalType::FixedSizeBinary => Box::new(fixed_size_binary_from_arrow(downcast(array))),

        PhysicalType::List => Box::new(list_from_arrow(downcast::<ListArray<i32>>(array))),
        PhysicalType::LargeList => Box::new(list_from_arrow(downcast::<ListArray<i64>>(array))),
        PhysicalType::FixedSizeList => Box::new(fixed_size_list_from_arrow(downcast(array))),

        PhysicalType::Struct => Box::new(struct_from_arrow(downcast(array))),

        physical @ (PhysicalType::Dictionary(_) | PhysicalType::Union | PhysicalType::Map) => {
            unimplemented!(
                "cannot import an arrow array of physical type {physical:?}: no array of \
                 polars-array holds its elements",
            )
        },
    }
}

/// Imports an Arrow [`NullArray`] as a [`PlNullArray`], which is `O(1)`.
pub fn null_from_arrow(array: &NullArray) -> PlNullArray {
    PlNullArray::new(array.len())
}

/// Imports an Arrow [`BooleanArray`] as a [`PlBooleanArray`], which is `O(1)`.
pub fn boolean_from_arrow(array: &BooleanArray) -> PlBooleanArray {
    // SAFETY: the values of an Arrow boolean array hold one bit per element, as does its validity
    // mask, which is what makes them flat here.
    unsafe {
        PlBooleanArray::new_unchecked(
            array.values().clone(),
            array.len(),
            array.validity().cloned().map(PlBitmap::from_bitmap),
        )
    }
}

/// Imports an Arrow [`PrimitiveArray`] as a [`PlPrimitiveArray`], which is `O(1)`.
pub fn primitive_from_arrow<T: NativeType>(array: &PrimitiveArray<T>) -> PlPrimitiveArray<T> {
    // SAFETY: the values of an Arrow primitive array hold one slot per element, as does its
    // validity mask, which is what makes them flat here.
    unsafe {
        PlPrimitiveArray::new_unchecked(
            array.values().clone(),
            array.len(),
            array.validity().cloned().map(PlBitmap::from_bitmap),
        )
    }
}

/// Imports an Arrow [`BinaryArray`] as a [`PlBinaryArray`].
pub fn binary_from_arrow<O: Offset>(array: &BinaryArray<O>) -> PlBinaryArray {
    // SAFETY: an Arrow array's offsets are ordered, one per element plus the end of the last, and
    // end within the values; widening them preserves that, as does its flat validity mask.
    unsafe {
        PlBinaryArray::new_unchecked(
            array.values().clone(),
            offsets_from_arrow(array.offsets()),
            array.len(),
            array.validity().cloned().map(PlBitmap::from_bitmap),
        )
    }
}

/// Imports an Arrow [`Utf8Array`] as a [`PlBinaryArray`] of its bytes.
pub fn utf8_from_arrow<O: Offset>(array: &Utf8Array<O>) -> PlBinaryArray {
    // SAFETY: an Arrow array's offsets are ordered, one per element plus the end of the last, and
    // end within the values; widening them preserves that, as does its flat validity mask.
    unsafe {
        PlBinaryArray::new_unchecked(
            array.values().clone(),
            offsets_from_arrow(array.offsets()),
            array.len(),
            array.validity().cloned().map(PlBitmap::from_bitmap),
        )
    }
}

/// Imports an Arrow binary or UTF-8 view array as a [`PlBinaryViewArray`] of its bytes, in `O(1)`.
pub fn binary_view_from_arrow<T: ViewType + ?Sized>(
    array: &BinaryViewArrayGeneric<T>,
) -> PlBinaryViewArray {
    // SAFETY: the views of an Arrow view array read bytes its buffers hold, and there is one view
    // per element, as there is one validity bit per element.
    unsafe {
        PlBinaryViewArray::new_unchecked(
            array.views().clone(),
            array.data_buffers().clone(),
            array.len(),
            array.validity().cloned().map(PlBitmap::from_bitmap),
        )
    }
}

/// Imports an Arrow `Utf8ViewArray` as a [`PlUtf8ViewArray`], which is `O(1)`.
pub fn utf8_view_from_arrow(array: &BinaryViewArrayGeneric<str>) -> PlUtf8ViewArray {
    // SAFETY: the elements of an Arrow `Utf8ViewArray` are valid UTF-8.
    unsafe { PlUtf8ViewArray::from_binview_unchecked(binary_view_from_arrow(array)) }
}

/// Imports an Arrow [`FixedSizeBinaryArray`] as a [`PlFixedSizeBinaryArray`], which is `O(1)`.
pub fn fixed_size_binary_from_arrow(array: &FixedSizeBinaryArray) -> PlFixedSizeBinaryArray {
    // SAFETY: the values of an Arrow fixed size binary array hold `size` bytes per element, and
    // its validity mask one bit per element, which is what makes them flat here.
    unsafe {
        PlFixedSizeBinaryArray::new_unchecked(
            array.values().clone(),
            array.size(),
            array.len(),
            array.validity().cloned().map(PlBitmap::from_bitmap),
        )
    }
}

/// Imports an Arrow [`ListArray`] as a [`PlListArray`], importing its values along with it.
pub fn list_from_arrow<O: Offset>(array: &ListArray<O>) -> PlListArray {
    let values = from_arrow(&**array.values());

    // SAFETY: an Arrow array's offsets are ordered, one per element plus the end of the last, and
    // end within the values; widening them and importing the values preserves that.
    unsafe {
        PlListArray::new_unchecked(
            values,
            offsets_from_arrow(array.offsets()),
            array.len(),
            array.validity().cloned().map(PlBitmap::from_bitmap),
        )
    }
}

/// Imports an Arrow [`FixedSizeListArray`] as a [`PlFixedSizeListArray`], values and all.
pub fn fixed_size_list_from_arrow(array: &FixedSizeListArray) -> PlFixedSizeListArray {
    let values = from_arrow(&**array.values());

    // SAFETY: an Arrow fixed size list array holds `size` values per element and one validity bit
    // per element, and importing the values preserves how many there are.
    unsafe {
        PlFixedSizeListArray::new_unchecked(
            values,
            array.size(),
            array.len(),
            array.validity().cloned().map(PlBitmap::from_bitmap),
        )
    }
}

/// Imports an Arrow [`StructArray`] as a [`PlStructArray`], fields and all, in `O(fields)`.
pub fn struct_from_arrow(array: &StructArray) -> PlStructArray {
    let fields = array
        .values()
        .iter()
        .map(|field| from_arrow(&**field))
        .collect();

    // SAFETY: every field of an Arrow struct array has as many elements as the array, and its
    // validity mask one bit per element; importing a field preserves that.
    unsafe {
        PlStructArray::new_unchecked(
            fields,
            array.len(),
            array.validity().cloned().map(PlBitmap::from_bitmap),
        )
    }
}

/// Imports Arrow offsets as the 64-bit offsets a [`PlBinaryArray`] and a [`PlListArray`] hold.
pub fn offsets_from_arrow<O: Offset>(offsets: &OffsetsBuffer<O>) -> Buffer<u64> {
    // The dispatch is on the concrete type rather than on `O::IS_LARGE` so that a buffer of
    // another width is never reinterpreted: only an `i64` buffer is handed to `try_transmute`.
    if let Some(offsets) = (offsets.buffer() as &dyn Any).downcast_ref::<Buffer<i64>>() {
        return offsets
            .clone()
            .try_transmute::<u64>()
            .expect("`i64` and `u64` have the same size and alignment");
    }

    Buffer::from(
        offsets
            .buffer()
            .iter()
            .map(|offset| offset.to_usize() as u64)
            .collect::<Vec<_>>(),
    )
}

/// Downcasts an Arrow array whose physical type has already been matched on.
#[inline]
fn downcast<A: Array + 'static>(array: &dyn Array) -> &A {
    array
        .as_any()
        .downcast_ref()
        .expect("the physical type of an arrow array determines the array it downcasts to")
}

/// Imports an Arrow primitive array of `primitive` elements as a [`PlPrimitiveArray`].
fn primitive_from_arrow_dyn(array: &dyn Array, primitive: PrimitiveType) -> Box<dyn PlArray> {
    macro_rules! import {
        ($T:ty) => {
            Box::new(primitive_from_arrow(downcast::<PrimitiveArray<$T>>(array)))
                as Box<dyn PlArray>
        };
    }

    match primitive {
        PrimitiveType::Int8 => import!(i8),
        PrimitiveType::Int16 => import!(i16),
        PrimitiveType::Int32 => import!(i32),
        PrimitiveType::Int64 => import!(i64),
        PrimitiveType::Int128 => import!(i128),
        PrimitiveType::Int256 => import!(i256),
        PrimitiveType::UInt8 => import!(u8),
        PrimitiveType::UInt16 => import!(u16),
        PrimitiveType::UInt32 => import!(u32),
        PrimitiveType::UInt64 => import!(u64),
        // A `View` and a `u128` are both `PrimitiveType::UInt128`, so the data type does not pin
        // the element type down and the array itself has to say which of the two it is over.
        PrimitiveType::UInt128 => match array.as_any().downcast_ref::<PrimitiveArray<View>>() {
            Some(array) => Box::new(primitive_from_arrow(array)),
            None => import!(u128),
        },
        PrimitiveType::Float16 => import!(pf16),
        PrimitiveType::Float32 => import!(f32),
        PrimitiveType::Float64 => import!(f64),
        PrimitiveType::DaysMs => import!(days_ms),
        PrimitiveType::MonthDayNano => import!(months_days_ns),
        PrimitiveType::MonthDayMillis => unimplemented!(
            "cannot import an arrow array of months_days_ms elements: they are of no rust type an \
             array of polars-array is taken over",
        ),
    }
}
