//! Exporting the arrays of this crate as the Arrow arrays of `polars-arrow`.

use arrow::array::{
    Array, BinaryArray, BinaryViewArray, BooleanArray, FixedSizeBinaryArray, FixedSizeListArray,
    ListArray, NullArray, PrimitiveArray, StructArray, Utf8ViewArray,
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

/// Exports a [`PlNullArray`] as an Arrow [`NullArray`], which is `O(1)`.
pub fn null_to_arrow_null(array: &PlNullArray) -> NullArray {
    NullArray::new(ArrowDataType::Null, array.len())
}

/// Exports a [`PlBooleanArray`] as an Arrow [`BooleanArray`].
pub fn boolean_to_arrow_boolean(array: &PlBooleanArray) -> BooleanArray {
    let (values, validity) = array.to_flat().into_owned().into_inner();
    BooleanArray::new(ArrowDataType::Boolean, values, validity)
}

/// Exports a [`PlPrimitiveArray`] as an Arrow [`PrimitiveArray`] of the type `T` is the storage of.
pub fn primitive_to_arrow_primitive<T: NativeType>(
    array: &PlPrimitiveArray<T>,
) -> PrimitiveArray<T> {
    let (values, validity) = array.to_flat().into_owned().into_inner();
    PrimitiveArray::new(T::PRIMITIVE.into(), values, validity)
}

/// Exports a [`PlBinaryArray`] as an Arrow [`BinaryArray`] of [`LargeBinary`](ArrowDataType).
pub fn binary_to_arrow_large_binary(array: &PlBinaryArray) -> BinaryArray<i64> {
    let (values, offsets, validity) = array.to_flat().into_owned().into_inner();
    BinaryArray::new(
        ArrowDataType::LargeBinary,
        offsets_to_arrow(offsets),
        values,
        validity,
    )
}

/// Exports a [`PlBinaryViewArray`] as an Arrow [`BinaryViewArray`].
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

/// Exports a [`PlUtf8ViewArray`] as an Arrow [`Utf8ViewArray`].
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

/// Exports a [`PlFixedSizeBinaryArray`] as an Arrow [`FixedSizeBinaryArray`] of its own width.
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

/// Exports a [`PlListArray`] as an Arrow [`ListArray`], exporting its values along with it.
pub fn list_to_arrow_large_list(array: &PlListArray) -> ListArray<i64> {
    let (values, offsets, validity) = array.to_flat().into_owned().into_inner();
    let values = to_arrow(&*values);

    let dtype = ListArray::<i64>::default_datatype(values.dtype().clone());
    ListArray::new(dtype, offsets_to_arrow(offsets), values, validity)
}

/// Exports a [`PlFixedSizeListArray`] as an Arrow [`FixedSizeListArray`] of its own width.
pub fn fixed_size_list_to_arrow_fixed_size_list(
    array: &PlFixedSizeListArray,
) -> FixedSizeListArray {
    let length = array.len();
    let (values, width, validity) = array.to_flat().into_owned().into_inner();
    let values = to_arrow(&*values);

    let dtype = FixedSizeListArray::default_datatype(values.dtype().clone(), width);
    FixedSizeListArray::new(dtype, length, values, validity)
}

/// Exports a [`PlStructArray`] as an Arrow [`StructArray`], exporting its fields along with it.
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

    let validity = array
        .validity()
        .map(|validity| validity.to_flat().into_owned());

    StructArray::new(ArrowDataType::Struct(fields), array.len(), values, validity)
}

/// Exports the 64-bit offsets of a binary or list array as the Arrow ones, which is `O(1)`.
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
#[inline]
fn downcast<A: PlArray>(array: &dyn PlArray) -> &A {
    array
        .as_any()
        .downcast_ref()
        .expect("the array type of an array determines the array it downcasts to")
}
