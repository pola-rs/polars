//! Exporting the arrays of this crate as the Arrow arrays of `polars-arrow`.

use arrow::array::{
    Array, BinaryArray, BinaryViewArray, BooleanArray, FixedSizeBinaryArray, FixedSizeListArray,
    ListArray, NullArray, PrimitiveArray, StructArray, Utf8Array, Utf8ViewArray,
};
use arrow::datatypes::{ArrowDataType, Field};
use arrow::offset::OffsetsBuffer;
use arrow::types::NativeType;
use polars_buffer::Buffer;
use polars_utils::format_pl_smallstr;

use crate::{
    PlArray, PlArrayType, PlBinaryArray, PlBinaryViewArray, PlBooleanArray, PlFixedSizeBinaryArray,
    PlFixedSizeListArray, PlListArray, PlNullArray, PlPrimitiveArray, PlStructArray,
    PlUtf8ViewArray, with_match_pl_primitive_array_type,
};

/// Exports an array of this crate as the Arrow array that holds the same elements.
///
/// # Panics
/// Panics if `array` is an object array, which has no Arrow counterpart.
pub fn to_arrow(array: &dyn PlArray) -> Box<dyn Array> {
    match array.array_type() {
        PlArrayType::Null => Box::new(null_to_arrow_null(downcast(array))),
        PlArrayType::Boolean => Box::new(boolean_to_arrow_boolean(downcast(array))),

        // The element type is taken from the array rather than from the `PrimitiveType`, which
        // does not pin it down: a `View` and a `u128` are both `PrimitiveType::UInt128`.
        PlArrayType::Primitive(_) => with_match_pl_primitive_array_type!(array, |T| {
            Box::new(primitive_to_arrow_primitive(
                downcast::<PlPrimitiveArray<T>>(array),
            )) as Box<dyn Array>
        })
        .expect("a primitive array is taken over one of the element types dispatched on"),

        PlArrayType::Binary => Box::new(binary_to_arrow_large_binary(downcast(array))),
        PlArrayType::BinaryView => Box::new(binview_to_arrow_binview(downcast(array))),
        PlArrayType::Utf8View => Box::new(utf8view_to_arrow_utf8view(downcast(array))),
        PlArrayType::FixedSizeBinary => Box::new(fixed_size_binary_to_arrow_fixed_size_binary(
            downcast(array),
        )),

        PlArrayType::List => Box::new(list_to_arrow_large_list(downcast(array))),
        PlArrayType::FixedSizeList => {
            Box::new(fixed_size_list_to_arrow_fixed_size_list(downcast(array)))
        },

        PlArrayType::Struct => Box::new(struct_to_arrow_struct(downcast(array))),

        array_type @ PlArrayType::Object { .. } => {
            unimplemented!("polars-array: cannot export {array_type:?} typed array")
        },
    }
}

/// Exports a [`PlNullArray`] as an Arrow [`NullArray`] of [`Null`](ArrowDataType::Null), which is
/// `O(1)`.
pub fn null_to_arrow_null(array: &PlNullArray) -> NullArray {
    NullArray::new(ArrowDataType::Null, array.len())
}

/// Exports a [`PlBooleanArray`] as an Arrow [`BooleanArray`] of
/// [`Boolean`](ArrowDataType::Boolean).
pub fn boolean_to_arrow_boolean(array: &PlBooleanArray) -> BooleanArray {
    let (values, validity) = array.to_flat().into_owned().into_inner();
    BooleanArray::new(ArrowDataType::Boolean, values, validity)
}

/// Exports a [`PlPrimitiveArray`] as an Arrow [`PrimitiveArray`] of the data type `T` is the
/// storage of.
pub fn primitive_to_arrow_primitive<T: NativeType>(
    array: &PlPrimitiveArray<T>,
) -> PrimitiveArray<T> {
    let (values, validity) = array.to_flat().into_owned().into_inner();
    PrimitiveArray::new(T::PRIMITIVE.into(), values, validity)
}

/// Exports a [`PlBinaryArray`] as an Arrow [`BinaryArray`] of
/// [`LargeBinary`](ArrowDataType::LargeBinary).
pub fn binary_to_arrow_large_binary(array: &PlBinaryArray) -> BinaryArray<i64> {
    let (values, offsets, validity) = array.to_flat().into_owned().into_inner();
    BinaryArray::new(
        ArrowDataType::LargeBinary,
        offsets_to_arrow(offsets),
        values,
        validity,
    )
}

/// Exports a [`PlBinaryArray`] as an Arrow [`Utf8Array`] of
/// [`LargeUtf8`](ArrowDataType::LargeUtf8), without checking that its bytes are valid UTF-8.
///
/// # Safety
/// Every element of `array` — including the ones under a null — must be valid UTF-8.
pub unsafe fn binary_to_arrow_large_utf8(array: &PlBinaryArray) -> Utf8Array<i64> {
    let (values, offsets, validity) = array.to_flat().into_owned().into_inner();

    // SAFETY: the caller guarantees the elements are valid UTF-8, and the offsets came out of a
    // flat `PlBinaryArray`, which lays them end to end within the values.
    unsafe {
        Utf8Array::new_unchecked(
            ArrowDataType::LargeUtf8,
            offsets_to_arrow(offsets),
            values,
            validity,
        )
    }
}

/// Exports a [`PlBinaryViewArray`] as an Arrow [`BinaryViewArray`] of
/// [`BinaryView`](ArrowDataType::BinaryView).
pub fn binview_to_arrow_binview(array: &PlBinaryViewArray) -> BinaryViewArray {
    let (views, buffers, validity) = array.to_flat().into_owned().into_inner();

    // SAFETY: the views came out of a `PlBinaryViewArray`, which validates every one of them
    // against the buffers it reads.
    unsafe {
        BinaryViewArray::new_unchecked_unknown_md(
            ArrowDataType::BinaryView,
            views,
            buffers,
            validity,
            None,
        )
    }
}

/// Exports a [`PlUtf8ViewArray`] as an Arrow [`Utf8ViewArray`] of
/// [`Utf8View`](ArrowDataType::Utf8View).
pub fn utf8view_to_arrow_utf8view(array: &PlUtf8ViewArray) -> Utf8ViewArray {
    let (views, buffers, validity) = array.as_binview().to_flat().into_owned().into_inner();

    // SAFETY: every element of a `PlUtf8ViewArray` is valid UTF-8, and the views came out of a
    // `PlBinaryViewArray`, which validates every one of them against the buffers it reads.
    unsafe {
        Utf8ViewArray::new_unchecked_unknown_md(
            ArrowDataType::Utf8View,
            views,
            buffers,
            validity,
            None,
        )
    }
}

/// Exports a [`PlFixedSizeBinaryArray`] as an Arrow [`FixedSizeBinaryArray`] of
/// [`FixedSizeBinary`](ArrowDataType::FixedSizeBinary) of the width its elements have.
///
/// # Panics
/// Panics if the elements of `array` are zero bytes wide: an Arrow fixed size binary array derives
/// its length from the length of its values, which leaves it none to derive.
pub fn fixed_size_binary_to_arrow_fixed_size_binary(
    array: &PlFixedSizeBinaryArray,
) -> FixedSizeBinaryArray {
    assert!(
        array.width() > 0,
        "cannot export a fixed size binary array of zero-width elements: an arrow array of them \
         has no length",
    );

    let flat = array.to_flat();
    FixedSizeBinaryArray::new(
        ArrowDataType::FixedSizeBinary(array.width()),
        flat.values().clone(),
        flat.validity().cloned(),
    )
}

/// Exports a [`PlListArray`] as an Arrow [`ListArray`] of [`LargeList`](ArrowDataType::LargeList),
/// exporting its values along with it.
///
/// # Panics
/// Panics if the values of `array` have no Arrow counterpart — see the [module docs](self).
pub fn list_to_arrow_large_list(array: &PlListArray) -> ListArray<i64> {
    let (values, offsets, validity) = array.to_flat().into_owned().into_inner();
    let values = to_arrow(&*values);

    let dtype = ListArray::<i64>::default_datatype(values.dtype().clone());
    ListArray::new(dtype, offsets_to_arrow(offsets), values, validity)
}

/// Exports a [`PlFixedSizeListArray`] as an Arrow [`FixedSizeListArray`] of the width its elements
/// have, exporting its values along with it.
///
/// # Panics
/// Panics if the values of `array` have no Arrow counterpart — see the [module docs](self).
pub fn fixed_size_list_to_arrow_fixed_size_list(
    array: &PlFixedSizeListArray,
) -> FixedSizeListArray {
    let length = array.len();
    let (values, width, validity) = array.to_flat().into_owned().into_inner();
    let values = to_arrow(&*values);

    let dtype = FixedSizeListArray::default_datatype(values.dtype().clone(), width);
    FixedSizeListArray::new(dtype, length, values, validity)
}

/// Exports a [`PlStructArray`] as an Arrow [`StructArray`] of [`Struct`](ArrowDataType::Struct),
/// exporting its fields along with it.
///
/// # Panics
/// Panics if a field of `array` has no Arrow counterpart — see the [module docs](self).
pub fn struct_to_arrow_struct(array: &PlStructArray) -> StructArray {
    let values = array
        .fields()
        .iter()
        .map(|field| to_arrow(&**field))
        .collect::<Vec<_>>();

    let fields = values
        .iter()
        .enumerate()
        .map(|(i, value)| Field::new(format_pl_smallstr!("{i}"), value.dtype().clone(), true))
        .collect();

    let validity = array.validity().map(|validity| validity.to_flat());

    StructArray::new(ArrowDataType::Struct(fields), array.len(), values, validity)
}

/// Exports the 64-bit offsets a [`PlBinaryArray`] and a [`PlListArray`] hold as the 64-bit Arrow
/// offsets, which is `O(1)`.
pub fn offsets_to_arrow(offsets: Buffer<u64>) -> OffsetsBuffer<i64> {
    debug_assert!(offsets.last().is_none_or(|&last| last <= i64::MAX as u64));

    let offsets = offsets
        .try_transmute::<i64>()
        .expect("`u64` and `i64` have the same size and alignment");

    // SAFETY: the offsets came out of an array of this crate, so they are monotonically
    // non-decreasing and hold the end of the last element, and reinterpreting preserves both.
    unsafe { OffsetsBuffer::new_unchecked(offsets) }
}

/// Downcasts an array of this crate whose array type has already been matched on.
///
/// # Panics
/// Panics if `array` is not an `A`, which its array type rules out.
#[inline]
fn downcast<A: PlArray>(array: &dyn PlArray) -> &A {
    array
        .as_any()
        .downcast_ref()
        .expect("the array type of an array determines the array it downcasts to")
}

#[cfg(test)]
mod tests {
    use arrow::array::{Int32Array, View};
    use arrow::types::{days_ms, i256, months_days_ns};
    use polars_utils::float16::pf16;

    use super::*;
    use crate::arrow::import::from_arrow;

    /// Downcasts an exported array to the Arrow array it is expected to be.
    fn exported<A: Array + Clone>(array: &dyn PlArray) -> Box<A> {
        let arrow = to_arrow(array);
        let dtype = arrow.dtype().clone();
        match arrow.as_any().downcast_ref::<A>() {
            Some(arrow) => Box::new(arrow.clone()),
            None => panic!("an array exported as {dtype:?}, which is another arrow array"),
        }
    }

    #[test]
    fn every_primitive_type_of_a_rust_type_is_exported() {
        fn exports<T: NativeType>() {
            let array = PlPrimitiveArray::<T>::new_full_null(2);
            let arrow = exported::<PrimitiveArray<T>>(&array);

            assert_eq!(arrow.len(), 2);
            assert_eq!(arrow.dtype(), &ArrowDataType::from(T::PRIMITIVE));
        }

        exports::<i8>();
        exports::<i16>();
        exports::<i32>();
        exports::<i64>();
        exports::<i128>();
        exports::<i256>();
        exports::<u8>();
        exports::<u16>();
        exports::<u32>();
        exports::<u64>();
        exports::<u128>();
        exports::<pf16>();
        exports::<f32>();
        exports::<f64>();
        exports::<days_ms>();
        exports::<months_days_ns>();
        exports::<View>();
    }

    #[test]
    fn a_scalar_array_is_written_out() {
        let array = PlPrimitiveArray::new_scalar(7i32, 3);
        assert!(array.flat_values().is_none());

        let arrow = exported::<Int32Array>(&array);
        assert_eq!(arrow.values().as_slice(), [7, 7, 7]);

        // A scalar validity mask is written out along with the values it covers.
        let array = PlBooleanArray::new_full_null(3);
        let arrow = exported::<BooleanArray>(&array);
        assert_eq!(arrow.len(), 3);
        assert_eq!(arrow.null_count(), 3);

        // And so is a scalar mask over a struct, whose fields are never scalar.
        let array =
            PlStructArray::new_full_null(vec![Box::new(PlPrimitiveArray::new_scalar(1i32, 3))], 3);
        let arrow = exported::<StructArray>(&array);
        assert_eq!(arrow.len(), 3);
        assert_eq!(arrow.null_count(), 3);
        assert_eq!(arrow.values()[0].len(), 3);

        // The lists of a scalar list array are laid end to end.
        let array = PlListArray::new_scalar(Box::new(PlPrimitiveArray::from_vec(vec![1i32, 2])), 3);
        let arrow = exported::<ListArray<i64>>(&array);
        assert_eq!(arrow.offsets().buffer().as_slice(), [0, 2, 4, 6]);

        let array = PlFixedSizeListArray::new_scalar(
            Box::new(PlPrimitiveArray::from_vec(vec![1i32, 2])),
            3,
        );
        let arrow = exported::<FixedSizeListArray>(&array);
        assert_eq!(arrow.len(), 3);
        assert_eq!(arrow.values().len(), 6);
    }

    #[test]
    fn an_exported_array_imports_back_as_the_array_it_was() {
        let arrays: [Box<dyn PlArray>; 8] = [
            Box::new(PlNullArray::new(3)),
            Box::new(PlBooleanArray::from_iter([Some(true), None, Some(false)])),
            Box::new(PlPrimitiveArray::from_iter([Some(1i32), None])),
            Box::new(PlBinaryArray::from_iter([Some(b"foo".as_slice()), None])),
            Box::new(PlBinaryViewArray::from_iter([
                Some(b"foo".as_slice()),
                None,
            ])),
            Box::new(PlFixedSizeBinaryArray::from_values(
                Buffer::from(b"abcd".to_vec()),
                2,
            )),
            Box::new(PlListArray::from_offsets(
                Box::new(PlPrimitiveArray::from_vec(vec![1i32, 2, 3])),
                Buffer::from(vec![0u64, 2, 3]),
            )),
            Box::new(PlStructArray::from_fields(vec![Box::new(
                PlPrimitiveArray::from_vec(vec![1i32, 2]),
            )])),
        ];

        for array in &arrays {
            let imported = from_arrow(&*to_arrow(&**array));
            assert!(
                <dyn PlArray>::eq(&*imported, &**array),
                "{:?} did not survive a round trip: {imported:?}",
                array.array_type(),
            );
        }
    }
}
