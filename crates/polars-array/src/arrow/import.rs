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

use crate::{
    PlArray, PlBinaryArray, PlBinaryViewArray, PlBooleanArray, PlFixedSizeBinaryArray,
    PlFixedSizeListArray, PlListArray, PlNullArray, PlPrimitiveArray, PlStructArray,
    PlUtf8ViewArray,
};

/// Imports an Arrow array as the array of this crate that holds the same elements.
///
/// # Panics
/// Panics if `array` is a dictionary, union or map array, or if its elements are of a type no array
/// of this crate is taken over.
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
            array.validity().cloned(),
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
            array.validity().cloned(),
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
            array.validity().cloned(),
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
            array.validity().cloned(),
        )
    }
}

/// Imports an Arrow binary or UTF-8 view array as a [`PlBinaryViewArray`] of its bytes, in
/// `O(1)`.
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
            array.validity().cloned(),
        )
    }
}

/// Imports an Arrow [`Utf8ViewArray`](arrow::array::Utf8ViewArray) as a [`PlUtf8ViewArray`], which
/// is `O(1)`.
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
            array.validity().cloned(),
        )
    }
}

/// Imports an Arrow [`ListArray`] as a [`PlListArray`], importing its values along with it.
///
/// # Panics
/// Panics if the values of `array` have no counterpart in this crate — see the [module docs](self).
pub fn list_from_arrow<O: Offset>(array: &ListArray<O>) -> PlListArray {
    let values = from_arrow(&**array.values());

    // SAFETY: an Arrow array's offsets are ordered, one per element plus the end of the last, and
    // end within the values; widening them and importing the values preserves that.
    unsafe {
        PlListArray::new_unchecked(
            values,
            offsets_from_arrow(array.offsets()),
            array.len(),
            array.validity().cloned(),
        )
    }
}

/// Imports an Arrow [`FixedSizeListArray`] as a [`PlFixedSizeListArray`], importing its values
/// along with it, which is `O(1)`.
///
/// # Panics
/// Panics if the values of `array` have no counterpart in this crate — see the [module docs](self).
pub fn fixed_size_list_from_arrow(array: &FixedSizeListArray) -> PlFixedSizeListArray {
    let values = from_arrow(&**array.values());

    // SAFETY: an Arrow fixed size list array holds `size` values per element and one validity bit
    // per element, and importing the values preserves how many there are.
    unsafe {
        PlFixedSizeListArray::new_unchecked(
            values,
            array.size(),
            array.len(),
            array.validity().cloned(),
        )
    }
}

/// Imports an Arrow [`StructArray`] as a [`PlStructArray`], importing its fields along with it,
/// which is `O(fields)`.
///
/// # Panics
/// Panics if a field of `array` has no counterpart in this crate — see the [module docs](self).
pub fn struct_from_arrow(array: &StructArray) -> PlStructArray {
    let fields = array
        .values()
        .iter()
        .map(|field| from_arrow(&**field))
        .collect();

    // SAFETY: every field of an Arrow struct array has as many elements as the array, and its
    // validity mask one bit per element; importing a field preserves that.
    unsafe { PlStructArray::new_unchecked(fields, array.len(), array.validity().cloned()) }
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
///
/// # Panics
/// Panics if `array` is not an `A`, which the physical type of its data type rules out.
#[inline]
fn downcast<A: Array + 'static>(array: &dyn Array) -> &A {
    array
        .as_any()
        .downcast_ref()
        .expect("the physical type of an arrow array determines the array it downcasts to")
}

/// Imports an Arrow primitive array of `primitive` elements as a [`PlPrimitiveArray`] of the Rust
/// type they are of.
///
/// # Panics
/// Panics if the elements are of no Rust type an array can be taken over, which is what
/// [`PrimitiveType::MonthDayMillis`] is.
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

#[cfg(test)]
mod tests {
    use arrow::array::{Array, BinaryArray, FixedSizeListArray, Int32Array, PrimitiveArray};
    use arrow::datatypes::ArrowDataType;

    use super::*;

    /// Downcasts an imported array to the array of this crate it is expected to be.
    fn imported<A: PlArray + Clone>(array: &dyn Array) -> Box<A> {
        let array = from_arrow(array);
        let array_type = array.array_type();
        match array.as_any().downcast_ref::<A>() {
            Some(array) => Box::new(array.clone()),
            None => panic!("an arrow array imported as {array_type:?}, which is another array"),
        }
    }

    #[test]
    fn every_primitive_type_of_a_rust_type_is_imported() {
        fn imports<T: NativeType>() {
            let arrow = PrimitiveArray::<T>::new_null(T::PRIMITIVE.into(), 2);
            let array = imported::<PlPrimitiveArray<T>>(&arrow);
            assert_eq!(array.len(), 2);
        }

        imports::<i8>();
        imports::<i16>();
        imports::<i32>();
        imports::<i64>();
        imports::<i128>();
        imports::<i256>();
        imports::<u8>();
        imports::<u16>();
        imports::<u32>();
        imports::<u64>();
        imports::<u128>();
        imports::<pf16>();
        imports::<f32>();
        imports::<f64>();
        imports::<days_ms>();
        imports::<months_days_ns>();
    }

    #[test]
    fn a_sliced_arrow_array_is_imported_as_the_elements_it_holds() {
        let arrow = Int32Array::from_slice([1, 2, 3, 4]).sliced(1, 2);
        let array = imported::<PlPrimitiveArray<i32>>(&arrow);
        assert_eq!(array.iter().collect::<Vec<_>>(), [Some(2), Some(3)]);

        let arrow = BinaryArray::<i64>::from_slice([b"foo", b"bar", b"baz"]).sliced(1, 2);
        let array = imported::<PlBinaryArray>(&arrow);
        assert_eq!(array.value(0), b"bar");
        assert_eq!(array.value(1), b"baz");

        let values = Int32Array::from_slice([1, 2, 3, 4]);
        let arrow = FixedSizeListArray::new(
            FixedSizeListArray::default_datatype(ArrowDataType::Int32, 2),
            2,
            values.boxed(),
            None,
        )
        .sliced(1, 1);
        let array = imported::<PlFixedSizeListArray>(&arrow);
        assert_eq!(array.len(), 1);
        let element = array.value(0);
        let element = element
            .as_any()
            .downcast_ref::<PlPrimitiveArray<i32>>()
            .unwrap();
        assert_eq!(element.iter().collect::<Vec<_>>(), [Some(3), Some(4)]);
    }

    #[test]
    fn an_imported_array_is_flat_rather_than_scalar() {
        let arrow = Int32Array::from_slice([1, 1, 1]);
        let array = imported::<PlPrimitiveArray<i32>>(&arrow);
        assert!(array.values_are_flat());

        let arrow = BinaryArray::<i32>::from_slice([b"a", b"a"]);
        let array = imported::<PlBinaryArray>(&arrow);
        assert!(array.offsets_are_flat());
    }
}
