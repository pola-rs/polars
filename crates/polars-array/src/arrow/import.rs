//! Importing the Arrow arrays of `polars-arrow` as the arrays of this crate.
//!
//! An Arrow array lays its elements out the way the array of this crate that holds them does, and
//! is built on the same [`Buffer`] and [`Bitmap`], so importing hands the backing buffers over
//! rather than copying the elements: [`from_arrow`] is `O(1)` for every Arrow array but the ones
//! whose offsets are 32 bits wide, which are widened into the 64-bit offsets a [`PlBinaryArray`]
//! and a [`PlListArray`] hold. See [`offsets_from_arrow`].
//!
//! # The logical type is dropped
//!
//! The arrays of this crate are purely a physical representation, so what an Arrow array imports
//! as is the physical array underneath its [`ArrowDataType`](arrow::datatypes::ArrowDataType),
//! which is the same for every logical type over one physical representation: a `Date32` array
//! imports as a [`PlPrimitiveArray<i32>`], and a [`Utf8Array`] and a [`BinaryArray`] both import
//! as a [`PlBinaryArray`]. Nothing in the imported array says the bytes of that
//! [`PlBinaryArray`] are a string, and nothing in this module validates that they are — it is the
//! caller that remembers which logical type the physical array stands for.
//!
//! # Importing never produces a scalar array
//!
//! An Arrow array holds one slot per element in every one of its buffers, which is what makes an
//! imported array [`flat`](crate::broadcast) rather than scalar. The scalar representation has no
//! Arrow counterpart to be imported from.
//!
//! # Arrays with no counterpart
//!
//! A dictionary, union or map array has no counterpart in this crate, and neither has an Arrow
//! array whose elements are [`months_days_ms`](arrow::types::months_days_ms), which is of no Rust
//! type an array can be taken over. Importing one panics with [`unimplemented!`]: decoding it into
//! an array that does have a counterpart is a decision for the caller, and there is nothing here
//! to encode it as in the meantime.
//!
//! # Example
//! ```
//! use arrow::array::{Int32Array, Utf8ViewArray};
//! use arrow::datatypes::ArrowDataType;
//! use polars_array::arrow::import::from_arrow;
//! use polars_array::{PlBinaryViewArray, PlPrimitiveArray};
//!
//! // The logical type is dropped: a `Date32` array is the `i32` array underneath it.
//! let arrow = Int32Array::from_slice([1, 2, 3]).to(ArrowDataType::Date32);
//! let array = from_arrow(&arrow);
//! let array = array.as_any().downcast_ref::<PlPrimitiveArray<i32>>().unwrap();
//! assert_eq!(array.value(2), 3);
//!
//! // So is the promise that the bytes of a string array are a string.
//! let arrow = Utf8ViewArray::from_slice_values(["foo", "bar"]);
//! let array = from_arrow(&arrow);
//! let array = array.as_any().downcast_ref::<PlBinaryViewArray>().unwrap();
//! assert_eq!(array.value(0), b"foo");
//! ```

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
};

/// Imports an Arrow array as the array of this crate that holds the same elements.
///
/// The logical type of `array` is dropped and what comes back is the physical array underneath it,
/// which shares its backing buffers with `array` — see the [module docs](self).
///
/// # Panics
/// Panics if `array` is a dictionary, union or map array, or if its elements are of a type no
/// array of this crate is taken over.
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
        PhysicalType::Utf8View => Box::new(binary_view_from_arrow(downcast::<
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
///
/// This is `O(1)` for `BinaryArray<i64>` and `O(len)` for `BinaryArray<i32>`, whose offsets are
/// widened — see [`offsets_from_arrow`].
pub fn binary_from_arrow<O: Offset>(array: &BinaryArray<O>) -> PlBinaryArray {
    // SAFETY: the offsets of an Arrow array are monotonically non-decreasing, hold one per element
    // plus the end of the last, and end within the values; its validity mask holds one bit per
    // element. Widening the offsets preserves all of that.
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
///
/// The promise that those bytes are valid UTF-8 is a logical type, which the imported array does
/// not carry. This is `O(1)` for `Utf8Array<i64>` and `O(len)` for `Utf8Array<i32>`, whose offsets
/// are widened — see [`offsets_from_arrow`].
pub fn utf8_from_arrow<O: Offset>(array: &Utf8Array<O>) -> PlBinaryArray {
    // SAFETY: the offsets of an Arrow array are monotonically non-decreasing, hold one per element
    // plus the end of the last, and end within the values; its validity mask holds one bit per
    // element. Widening the offsets preserves all of that.
    unsafe {
        PlBinaryArray::new_unchecked(
            array.values().clone(),
            offsets_from_arrow(array.offsets()),
            array.len(),
            array.validity().cloned(),
        )
    }
}

/// Imports an Arrow [`BinaryViewArray`](arrow::array::BinaryViewArray) or
/// [`Utf8ViewArray`](arrow::array::Utf8ViewArray) as a [`PlBinaryViewArray`] of its bytes, which
/// is `O(1)`.
///
/// The promise that the bytes of a [`Utf8ViewArray`](arrow::array::Utf8ViewArray) are valid UTF-8
/// is a logical type, which the imported array does not carry.
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
/// This is `O(1)` for `ListArray<i64>` and `O(len)` for `ListArray<i32>`, whose offsets are
/// widened — see [`offsets_from_arrow`].
///
/// # Panics
/// Panics if the values of `array` have no counterpart in this crate — see the [module
/// docs](self).
pub fn list_from_arrow<O: Offset>(array: &ListArray<O>) -> PlListArray {
    let values = from_arrow(&**array.values());

    // SAFETY: the offsets of an Arrow array are monotonically non-decreasing, hold one per element
    // plus the end of the last, and end within the values; its validity mask holds one bit per
    // element. Widening the offsets preserves all of that, and importing the values preserves how
    // many of them there are.
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
/// Panics if the values of `array` have no counterpart in this crate — see the [module
/// docs](self).
pub fn fixed_size_list_from_arrow(array: &FixedSizeListArray) -> PlFixedSizeListArray {
    let values = from_arrow(&**array.values());

    // SAFETY: the values of an Arrow fixed size list array hold `size` values per element, and its
    // validity mask one bit per element, which is what makes them flat here; importing the values
    // preserves how many of them there are.
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
/// The names of the fields are part of the data type of `array` rather than of its values, so they
/// are dropped along with it: what carries over is the field arrays, in order.
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
    // validity mask holds one bit per element, which is what makes it flat here; importing a field
    // preserves how many elements it has.
    unsafe { PlStructArray::new_unchecked(fields, array.len(), array.validity().cloned()) }
}

/// Imports Arrow offsets as the 64-bit offsets a [`PlBinaryArray`] and a [`PlListArray`] hold.
///
/// The 64-bit Arrow offsets have the layout the ones here do and are non-negative, so they are
/// reinterpreted in `O(1)`. The 32-bit ones are of another width, so they are widened into a
/// buffer of their own, which is `O(len)` in both time and memory.
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
    use arrow::array::{
        Array, BinaryArray, BinaryViewArray, BooleanArray, FixedSizeBinaryArray,
        FixedSizeListArray, Int32Array, Int64Array, ListArray, NullArray, PrimitiveArray,
        StructArray, Utf8Array, Utf8ViewArray,
    };
    use arrow::bitmap::Bitmap;
    use arrow::datatypes::{ArrowDataType, Field};
    use arrow::offset::OffsetsBuffer;
    use polars_buffer::Buffer;

    use super::*;
    use crate::PlArrayType;

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
    fn null_is_imported() {
        let arrow = NullArray::new(ArrowDataType::Null, 7);
        let array = imported::<PlNullArray>(&arrow);

        assert_eq!(array.len(), 7);
        assert_eq!(array.array_type(), PlArrayType::Null);
    }

    #[test]
    fn boolean_is_imported() {
        let arrow = BooleanArray::from([Some(true), None, Some(false)]);
        let array = imported::<PlBooleanArray>(&arrow);

        assert_eq!(array.len(), 3);
        assert_eq!(array.get(0), Some(true));
        assert_eq!(array.get(1), None);
        assert_eq!(array.get(2), Some(false));
    }

    #[test]
    fn primitive_is_imported_of_the_element_type_of_the_arrow_array() {
        let arrow = Int32Array::from([Some(1), None, Some(3)]);
        let array = imported::<PlPrimitiveArray<i32>>(&arrow);

        assert_eq!(array.len(), 3);
        assert_eq!(array.get(0), Some(1));
        assert_eq!(array.get(1), None);
        assert_eq!(array.get(2), Some(3));
        assert_eq!(
            array.array_type(),
            PlArrayType::Primitive(PrimitiveType::Int32)
        );
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
    fn a_view_is_imported_as_a_view_rather_than_as_the_u128_of_its_primitive_type() {
        let arrow = PrimitiveArray::<View>::new_null(ArrowDataType::UInt128, 2);
        assert!(
            arrow
                .dtype()
                .to_physical_type()
                .eq_primitive(PrimitiveType::UInt128)
        );

        let array = imported::<PlPrimitiveArray<View>>(&arrow);
        assert_eq!(array.len(), 2);
    }

    #[test]
    fn the_logical_type_of_an_arrow_array_is_dropped() {
        let arrow = Int32Array::from_slice([1, 2, 3]).to(ArrowDataType::Date32);
        let array = imported::<PlPrimitiveArray<i32>>(&arrow);

        assert_eq!(array.value(1), 2);
    }

    #[test]
    fn binary_is_imported_behind_widened_offsets() {
        let arrow = BinaryArray::<i32>::from([Some(b"foo".as_slice()), None, Some(b"barbar")]);
        let array = imported::<PlBinaryArray>(&arrow);

        assert_eq!(array.len(), 3);
        assert_eq!(array.get(0), Some(b"foo".as_slice()));
        assert_eq!(array.get(1), None);
        assert_eq!(array.get(2), Some(b"barbar".as_slice()));
        assert_eq!(array.flat_offsets().unwrap().as_slice(), [0, 3, 3, 9]);
    }

    #[test]
    fn large_binary_is_imported_behind_the_offsets_it_already_has() {
        let arrow = BinaryArray::<i64>::from([Some(b"foo".as_slice()), None, Some(b"barbar")]);
        let offsets = arrow.offsets().buffer().as_ptr();

        let array = imported::<PlBinaryArray>(&arrow);

        assert_eq!(array.get(2), Some(b"barbar".as_slice()));
        // The offsets were reinterpreted rather than copied, so they are the same allocation.
        assert_eq!(
            array.flat_offsets().unwrap().as_ptr(),
            offsets.cast::<u64>()
        );
    }

    #[test]
    fn utf8_is_imported_as_the_bytes_of_its_elements() {
        let arrow = Utf8Array::<i32>::from([Some("foo"), None, Some("bar")]);
        let array = imported::<PlBinaryArray>(&arrow);

        assert_eq!(array.get(0), Some(b"foo".as_slice()));
        assert_eq!(array.get(1), None);
        assert_eq!(array.array_type(), PlArrayType::Binary);
    }

    #[test]
    fn large_utf8_is_imported_as_the_bytes_of_its_elements() {
        let arrow = Utf8Array::<i64>::from([Some("foo"), None, Some("bar")]);
        let array = imported::<PlBinaryArray>(&arrow);

        assert_eq!(array.get(2), Some(b"bar".as_slice()));
        assert_eq!(array.flat_offsets().unwrap().as_slice(), [0, 3, 3, 6]);
    }

    #[test]
    fn binary_view_is_imported() {
        let arrow =
            BinaryViewArray::from_slice([Some(b"foo".as_slice()), None, Some(b"bar".as_slice())]);
        let array = imported::<PlBinaryViewArray>(&arrow);

        assert_eq!(array.len(), 3);
        assert_eq!(array.get(0), Some(b"foo".as_slice()));
        assert_eq!(array.get(1), None);
    }

    #[test]
    fn utf8_view_is_imported_as_the_bytes_of_its_elements() {
        let arrow =
            Utf8ViewArray::from_slice([Some("foo"), None, Some("a rather long string value")]);
        let array = imported::<PlBinaryViewArray>(&arrow);

        assert_eq!(array.get(0), Some(b"foo".as_slice()));
        assert_eq!(array.get(2), Some(b"a rather long string value".as_slice()));
        assert_eq!(array.array_type(), PlArrayType::BinaryView);
    }

    #[test]
    fn fixed_size_binary_is_imported_with_its_width() {
        let arrow = FixedSizeBinaryArray::new(
            ArrowDataType::FixedSizeBinary(2),
            Buffer::from(vec![1, 2, 3, 4, 5, 6]),
            Some(Bitmap::from([true, false, true])),
        );
        let array = imported::<PlFixedSizeBinaryArray>(&arrow);

        assert_eq!(array.len(), 3);
        assert_eq!(array.width(), 2);
        assert_eq!(array.get(0), Some([1, 2].as_slice()));
        assert_eq!(array.get(1), None);
        assert_eq!(array.get(2), Some([5, 6].as_slice()));
    }

    #[test]
    fn list_is_imported_behind_widened_offsets() {
        let values = Int32Array::from_slice([1, 2, 3, 4]);
        let arrow = ListArray::<i32>::new(
            ListArray::<i32>::default_datatype(ArrowDataType::Int32),
            OffsetsBuffer::try_from(vec![0i32, 2, 2, 4]).unwrap(),
            values.boxed(),
            Some(Bitmap::from([true, false, true])),
        );
        let array = imported::<PlListArray>(&arrow);

        assert_eq!(array.len(), 3);
        assert_eq!(array.flat_offsets().unwrap().as_slice(), [0, 2, 2, 4]);
        assert_eq!(
            array.values().array_type(),
            PlArrayType::Primitive(PrimitiveType::Int32)
        );
        assert!(array.get(1).is_none());

        let element = array.get(0).unwrap();
        let element = element
            .as_any()
            .downcast_ref::<PlPrimitiveArray<i32>>()
            .unwrap();
        assert_eq!(element.iter().collect::<Vec<_>>(), [Some(1), Some(2)]);
    }

    #[test]
    fn large_list_is_imported_behind_the_offsets_it_already_has() {
        let values = Int64Array::from_slice([1, 2, 3, 4]);
        let arrow = ListArray::<i64>::new(
            ListArray::<i64>::default_datatype(ArrowDataType::Int64),
            OffsetsBuffer::try_from(vec![0i64, 2, 4]).unwrap(),
            values.boxed(),
            None,
        );
        let offsets = arrow.offsets().buffer().as_ptr();

        let array = imported::<PlListArray>(&arrow);

        assert_eq!(array.len(), 2);
        assert_eq!(
            array.flat_offsets().unwrap().as_ptr(),
            offsets.cast::<u64>()
        );
    }

    #[test]
    fn fixed_size_list_is_imported_with_its_width() {
        let values = Int32Array::from_slice([1, 2, 3, 4]);
        let arrow = FixedSizeListArray::new(
            FixedSizeListArray::default_datatype(ArrowDataType::Int32, 2),
            2,
            values.boxed(),
            Some(Bitmap::from([true, false])),
        );
        let array = imported::<PlFixedSizeListArray>(&arrow);

        assert_eq!(array.len(), 2);
        assert_eq!(array.width(), 2);
        assert!(array.get(1).is_none());

        let element = array.get(0).unwrap();
        let element = element
            .as_any()
            .downcast_ref::<PlPrimitiveArray<i32>>()
            .unwrap();
        assert_eq!(element.iter().collect::<Vec<_>>(), [Some(1), Some(2)]);
    }

    #[test]
    fn struct_is_imported_as_its_fields_in_order() {
        let arrow = StructArray::new(
            ArrowDataType::Struct(vec![
                Field::new("a".into(), ArrowDataType::Int32, true),
                Field::new("b".into(), ArrowDataType::Boolean, true),
            ]),
            2,
            vec![
                Int32Array::from_slice([1, 2]).boxed(),
                BooleanArray::from_slice([true, false]).boxed(),
            ],
            Some(Bitmap::from([true, false])),
        );
        let array = imported::<PlStructArray>(&arrow);

        assert_eq!(array.len(), 2);
        assert_eq!(array.fields().len(), 2);
        assert_eq!(
            array.fields()[0].array_type(),
            PlArrayType::Primitive(PrimitiveType::Int32)
        );
        assert_eq!(array.fields()[1].array_type(), PlArrayType::Boolean);
    }

    #[test]
    fn nested_values_are_imported_along_with_the_array_over_them() {
        let values = Utf8Array::<i32>::from_slice(["foo", "bar"]);
        let arrow = ListArray::<i32>::new(
            ListArray::<i32>::default_datatype(ArrowDataType::Utf8),
            OffsetsBuffer::try_from(vec![0i32, 1, 2]).unwrap(),
            values.boxed(),
            None,
        );
        let array = imported::<PlListArray>(&arrow);

        assert_eq!(array.values().array_type(), PlArrayType::Binary);
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

    #[test]
    #[should_panic(expected = "no array of polars-array holds its elements")]
    fn a_dictionary_array_has_no_counterpart_to_import_as() {
        use arrow::array::DictionaryArray;
        use arrow::datatypes::IntegerType;

        let keys = PrimitiveArray::<u32>::from_slice([0, 1]);
        let values = Utf8Array::<i32>::from_slice(["foo", "bar"]);
        let arrow = DictionaryArray::try_new(
            ArrowDataType::Dictionary(IntegerType::UInt32, Box::new(ArrowDataType::Utf8), false),
            keys,
            values.boxed(),
        )
        .unwrap();

        from_arrow(&arrow);
    }

    #[test]
    #[should_panic(expected = "they are of no rust type")]
    fn months_days_ms_elements_have_no_rust_type_to_import_as() {
        let arrow = NullArray::new(ArrowDataType::Null, 1);
        primitive_from_arrow_dyn(&arrow, PrimitiveType::MonthDayMillis);
    }
}
