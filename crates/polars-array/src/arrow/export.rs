//! Exporting the arrays of this crate as the Arrow arrays of `polars-arrow`.
//!
//! An array of this crate lays its elements out the way the Arrow array that holds them does, and
//! is built on the same [`Buffer`] and [`Bitmap`](arrow::bitmap::Bitmap), so exporting hands the
//! backing buffers over rather than copying the elements: [`to_arrow`] is `O(1)` for every array
//! that is already [`flat`](crate::broadcast). See [`offsets_to_arrow`] for the one buffer that is
//! reinterpreted rather than handed over as it is.
//!
//! # The logical type is inlined
//!
//! Apart from [`PlUtf8ViewArray`], the arrays of this crate carry no logical type, so there is
//! nothing in one to derive an [`ArrowDataType`] from and nothing that says which of several Arrow
//! arrays over one physical representation it stands for. Rather than take a data type it would
//! have to validate the array against, this module gives every Arrow array a function of its own
//! that inlines the data type it exports as: a [`PlBinaryViewArray`] exports as a
//! [`BinaryViewArray`] through [`binview_to_arrow_binview`], and a [`PlBinaryArray`] as a
//! [`BinaryArray`] through [`binary_to_arrow_large_binary`]. Each promises the least about the
//! elements that their physical layout allows, and it is the caller that replaces the data type
//! afterwards when it remembers a logical type the array stood for.
//!
//! A [`PlUtf8ViewArray`] is the exception, because the UTF-8 promise is part of the array rather
//! than something the caller remembers about it: it exports as a [`Utf8ViewArray`] through
//! [`utf8view_to_arrow_utf8view`], which needs no `unsafe` and is what [`to_arrow`] picks for it.
//!
//! Nothing here is inferred from the values: the data type of a nested array is built from the
//! data type its exported values came back with, so the fields of an exported [`PlStructArray`]
//! are named after their index and the values of an exported [`PlListArray`] are named the way
//! [`ListArray::default_datatype`] names them.
//!
//! # UTF-8
//!
//! [`PlUtf8ViewArray`] is the one array here whose elements are known to be a string, which is why
//! exporting one through [`utf8view_to_arrow_utf8view`] is safe.
//!
//! Nothing says that the bytes of a [`PlBinaryArray`] or a [`PlBinaryViewArray`] are a string, and
//! nothing here ever validates that they are, so neither has a safe export as an Arrow array that
//! claims they are one. A [`PlBinaryViewArray`] whose bytes *are* UTF-8 is wrapped as a
//! [`PlUtf8ViewArray`] first — see [`crate::utf8view`] — rather than exported as a string
//! directly. [`binary_to_arrow_large_utf8`] is the one function that still hands bytes over behind
//! a string data type, for the offset-based [`PlBinaryArray`] that has no string counterpart in
//! this crate to be wrapped as; it is `unsafe` because it is the caller that knows the bytes are
//! valid UTF-8.
//!
//! # The scalar representation is written out
//!
//! An Arrow array holds one slot per element in every one of its buffers, so it has no counterpart
//! of the [`scalar`](crate::broadcast) representation and there is nothing to export a scalar
//! array *as*. Every function here writes one out with `to_flat`, which is `O(len)` in both time
//! and memory: an array whose length is unbounded by its memory use does not stay that way across
//! the export.
//!
//! # Arrays with no counterpart
//!
//! The Arrow arrays whose offsets are 32 bits wide are not export targets. Narrowing the 64-bit
//! offsets of a [`PlBinaryArray`] or a [`PlListArray`] is a lossy `O(len)` conversion rather than
//! a reinterpretation of the buffer they already hold, so it belongs to the caller that wants it.
//! An object array has no Arrow counterpart at all; exporting one panics with [`unimplemented!`].
//!
//! # Example
//! ```
//! use arrow::array::{Array, BinaryArray, BinaryViewArray, Utf8ViewArray};
//! use arrow::datatypes::ArrowDataType;
//! use polars_array::arrow::export::to_arrow;
//! use polars_array::{PlBinaryArray, PlBinaryViewArray, PlUtf8ViewArray};
//!
//! // The bytes of a `PlBinaryArray` are not a string, so it exports as a binary array.
//! let array = PlBinaryArray::from_values_iter([b"foo".as_slice(), b"bar"]);
//! let arrow = to_arrow(&array);
//! assert_eq!(arrow.dtype(), &ArrowDataType::LargeBinary);
//! assert_eq!(arrow.as_any().downcast_ref::<BinaryArray<i64>>().unwrap().value(0), b"foo");
//!
//! // Nor are the bytes of a `PlBinaryViewArray`, whatever they happen to hold.
//! let array = PlBinaryViewArray::from_values_iter([b"foo".as_slice(), b"bar"]);
//! let arrow = to_arrow(&array);
//! assert_eq!(arrow.dtype(), &ArrowDataType::BinaryView);
//! assert_eq!(arrow.as_any().downcast_ref::<BinaryViewArray>().unwrap().value(0), b"foo");
//!
//! // A `PlUtf8ViewArray` carries the promise that they are, so it exports as a string array with
//! // no `unsafe` and no data type to remember.
//! let array: PlUtf8ViewArray = [Some("foo"), Some("bar")].into_iter().collect();
//! let arrow = to_arrow(&array);
//! assert_eq!(arrow.dtype(), &ArrowDataType::Utf8View);
//! assert_eq!(arrow.as_any().downcast_ref::<Utf8ViewArray>().unwrap().value(0), "foo");
//!
//! // A scalar array is written out: an arrow array holds one slot per element.
//! let array = PlBinaryViewArray::new_scalar(b"foo", 3);
//! let arrow = to_arrow(&array);
//! assert_eq!(arrow.len(), 3);
//! ```

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
/// The array carries no logical type, so the Arrow array it comes back as is the one that promises
/// the least about its elements: a [`PlBinaryArray`] exports as a [`BinaryArray`] rather than as a
/// [`Utf8Array`], and a [`PlBinaryViewArray`] as a [`BinaryViewArray`] rather than as a
/// [`Utf8ViewArray`]. The functions this dispatches to are what export the others — see the
/// [module docs](self).
///
/// This shares the backing buffers with `array`, so it is `O(1)` for a
/// [`flat`](crate::broadcast) array and `O(len)` for a scalar one, which is written out.
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
///
/// This is `O(1)` for a [`flat`](crate::broadcast) array and `O(len)` for a scalar one, which is
/// written out.
pub fn boolean_to_arrow_boolean(array: &PlBooleanArray) -> BooleanArray {
    let (values, validity) = array.to_flat().into_inner();
    BooleanArray::new(ArrowDataType::Boolean, values, validity)
}

/// Exports a [`PlPrimitiveArray`] as an Arrow [`PrimitiveArray`] of the data type its element type
/// is the storage of — [`Int32`](ArrowDataType::Int32) for an `i32`, and so on for every
/// [`PrimitiveType`](arrow::datatypes::PrimitiveType).
///
/// This is `O(1)` for a [`flat`](crate::broadcast) array and `O(len)` for a scalar one, which is
/// written out.
pub fn primitive_to_arrow_primitive<T: NativeType>(
    array: &PlPrimitiveArray<T>,
) -> PrimitiveArray<T> {
    let (values, validity) = array.to_flat().into_inner();
    PrimitiveArray::new(T::PRIMITIVE.into(), values, validity)
}

/// Exports a [`PlBinaryArray`] as an Arrow [`BinaryArray`] of
/// [`LargeBinary`](ArrowDataType::LargeBinary).
///
/// The 32-bit-offset [`Binary`](ArrowDataType::Binary) is not an export target — see the [module
/// docs](self). This is `O(1)` for a [`flat`](crate::broadcast) array and `O(len)` for a scalar
/// one, which is written out.
pub fn binary_to_arrow_large_binary(array: &PlBinaryArray) -> BinaryArray<i64> {
    let (values, offsets, validity) = array.to_flat().into_inner();
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
/// This is [`binary_to_arrow_large_binary`] behind a data type that promises the bytes are a
/// string, which nothing in this crate establishes — see the [module docs](self). The
/// 32-bit-offset [`Utf8`](ArrowDataType::Utf8) is not an export target.
///
/// This is `O(1)` for a [`flat`](crate::broadcast) array and `O(len)` for a scalar one, which is
/// written out.
///
/// # Safety
/// Every element of `array` — including the ones under a null — must be valid UTF-8.
pub unsafe fn binary_to_arrow_large_utf8(array: &PlBinaryArray) -> Utf8Array<i64> {
    let (values, offsets, validity) = array.to_flat().into_inner();

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
///
/// This is `O(1)` for a [`flat`](crate::broadcast) array and `O(len)` for a scalar one, which is
/// written out.
pub fn binview_to_arrow_binview(array: &PlBinaryViewArray) -> BinaryViewArray {
    let (views, buffers, validity) = array.to_flat().into_inner();

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
///
/// This needs no `unsafe`: the UTF-8 the Arrow data type promises is exactly the invariant
/// [`PlUtf8ViewArray`] carries — see [`crate::utf8view`]. A [`PlBinaryViewArray`], whose bytes are
/// not known to be a string, exports through [`binview_to_arrow_binview`] instead.
///
/// This is `O(1)` for a [`flat`](crate::broadcast) array and `O(len)` for a scalar one, which is
/// written out.
pub fn utf8view_to_arrow_utf8view(array: &PlUtf8ViewArray) -> Utf8ViewArray {
    let (views, buffers, validity) = array.as_binview().to_flat().into_inner();

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
/// This is `O(1)` for a [`flat`](crate::broadcast) array and `O(len)` for a scalar one, which is
/// written out.
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

/// Exports a [`PlListArray`] as an Arrow [`ListArray`] of
/// [`LargeList`](ArrowDataType::LargeList), exporting its values along with it.
///
/// The data type of the values is the one they export with, named the way
/// [`ListArray::default_datatype`] names it. The 32-bit-offset [`List`](ArrowDataType::List) is not
/// an export target — see the [module docs](self).
///
/// This is `O(1)` for a [`flat`](crate::broadcast) array over flat values and `O(len)` for a
/// scalar one, which is written out.
///
/// # Panics
/// Panics if the values of `array` have no Arrow counterpart — see the [module docs](self).
pub fn list_to_arrow_large_list(array: &PlListArray) -> ListArray<i64> {
    let (values, offsets, validity) = array.to_flat().into_inner();
    let values = to_arrow(&*values);

    let dtype = ListArray::<i64>::default_datatype(values.dtype().clone());
    ListArray::new(dtype, offsets_to_arrow(offsets), values, validity)
}

/// Exports a [`PlFixedSizeListArray`] as an Arrow [`FixedSizeListArray`] of
/// [`FixedSizeList`](ArrowDataType::FixedSizeList) of the width its elements have, exporting its
/// values along with it.
///
/// The data type of the values is the one they export with, named the way
/// [`FixedSizeListArray::default_datatype`] names it.
///
/// This is `O(1)` for a [`flat`](crate::broadcast) array over flat values and `O(len)` for a
/// scalar one, which is written out.
///
/// # Panics
/// Panics if the values of `array` have no Arrow counterpart — see the [module docs](self).
pub fn fixed_size_list_to_arrow_fixed_size_list(
    array: &PlFixedSizeListArray,
) -> FixedSizeListArray {
    let length = array.len();
    let (values, width, validity) = array.to_flat().into_inner();
    let values = to_arrow(&*values);

    let dtype = FixedSizeListArray::default_datatype(values.dtype().clone(), width);
    FixedSizeListArray::new(dtype, length, values, validity)
}

/// Exports a [`PlStructArray`] as an Arrow [`StructArray`] of
/// [`Struct`](ArrowDataType::Struct), exporting its fields along with it.
///
/// A field array carries no name, so the exported fields are named after their index — `"0"`,
/// `"1"`, and so on — and are of the data type they export with.
///
/// This is `O(1)` for a [`flat`](crate::broadcast) array over flat fields and `O(len)` for one
/// whose validity mask or fields are written out.
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
///
/// The two have the same layout, so the buffer is reinterpreted rather than copied. That
/// reinterpretation is unchecked: an offset past [`i64::MAX`] comes back negative, which no offset
/// into a buffer that fits in memory is.
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
    use arrow::array::{Int32Array, StaticArray, View};
    use arrow::bitmap::Bitmap;
    use arrow::datatypes::PrimitiveType;
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
    fn null_is_exported() {
        let array = PlNullArray::new(7);
        let arrow = exported::<NullArray>(&array);

        assert_eq!(arrow.len(), 7);
        assert_eq!(arrow.dtype(), &ArrowDataType::Null);
    }

    #[test]
    fn boolean_is_exported() {
        let array = PlBooleanArray::from_iter([Some(true), None, Some(false)]);
        let arrow = exported::<BooleanArray>(&array);

        assert_eq!(arrow.dtype(), &ArrowDataType::Boolean);
        assert_eq!(arrow.get(0), Some(true));
        assert_eq!(arrow.get(1), None);
        assert_eq!(arrow.get(2), Some(false));
    }

    #[test]
    fn primitive_is_exported_of_the_data_type_of_its_element_type() {
        let array = PlPrimitiveArray::from_iter([Some(1i32), None, Some(3)]);
        let arrow = exported::<Int32Array>(&array);

        assert_eq!(arrow.dtype(), &ArrowDataType::Int32);
        assert_eq!(arrow.get(0), Some(1));
        assert_eq!(arrow.get(1), None);
        assert_eq!(arrow.get(2), Some(3));
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
    fn a_view_is_exported_as_a_view_rather_than_as_the_u128_of_its_primitive_type() {
        let array = PlPrimitiveArray::<View>::new_full_null(2);
        let arrow = exported::<PrimitiveArray<View>>(&array);

        assert!(
            arrow
                .dtype()
                .to_physical_type()
                .eq_primitive(PrimitiveType::UInt128)
        );
    }

    #[test]
    fn binary_is_exported_behind_the_offsets_it_already_has() {
        let array = PlBinaryArray::from_iter([Some(b"foo".as_slice()), None, Some(b"barbar")]);
        let offsets = array.flat_offsets().unwrap().as_ptr();

        let arrow = exported::<BinaryArray<i64>>(&array);

        assert_eq!(arrow.dtype(), &ArrowDataType::LargeBinary);
        assert_eq!(arrow.get(0), Some(b"foo".as_slice()));
        assert_eq!(arrow.get(1), None);
        assert_eq!(arrow.get(2), Some(b"barbar".as_slice()));
        // The offsets were reinterpreted rather than copied, so they are the same allocation.
        assert_eq!(arrow.offsets().buffer().as_ptr(), offsets.cast::<i64>());
    }

    #[test]
    fn binary_is_exported_as_a_string_only_when_the_caller_promises_it_is_one() {
        let array = PlBinaryArray::from_values_iter([b"foo".as_slice(), b"bar"]);

        // SAFETY: both elements are valid UTF-8.
        let arrow = unsafe { binary_to_arrow_large_utf8(&array) };

        assert_eq!(arrow.dtype(), &ArrowDataType::LargeUtf8);
        assert_eq!(arrow.value(0), "foo");
        assert_eq!(arrow.value(1), "bar");
    }

    #[test]
    fn binary_view_is_exported() {
        let array =
            PlBinaryViewArray::from_iter([Some(b"foo".as_slice()), None, Some(b"a long value")]);
        let arrow = exported::<BinaryViewArray>(&array);

        assert_eq!(arrow.dtype(), &ArrowDataType::BinaryView);
        assert_eq!(arrow.get(0), Some(b"foo".as_slice()));
        assert_eq!(arrow.get(1), None);
        assert_eq!(arrow.get(2), Some(b"a long value".as_slice()));
    }

    #[test]
    fn utf8_view_is_exported_as_a_string() {
        let array: PlUtf8ViewArray = [Some("foo"), None, Some("a rather long string value")]
            .into_iter()
            .collect();

        let arrow = utf8view_to_arrow_utf8view(&array);

        assert_eq!(arrow.dtype(), &ArrowDataType::Utf8View);
        assert_eq!(arrow.value(0), "foo");
        assert_eq!(arrow.get(1), None);
        assert_eq!(arrow.value(2), "a rather long string value");
    }

    #[test]
    fn fixed_size_binary_is_exported_with_its_width() {
        let array = PlFixedSizeBinaryArray::new(
            Buffer::from(vec![1u8, 2, 3, 4, 5, 6]),
            2,
            3,
            Some(Bitmap::from([true, false, true])),
        );
        let arrow = exported::<FixedSizeBinaryArray>(&array);

        assert_eq!(arrow.dtype(), &ArrowDataType::FixedSizeBinary(2));
        assert_eq!(arrow.len(), 3);
        assert_eq!(arrow.get(0), Some([1, 2].as_slice()));
        assert_eq!(arrow.get(1), None);
        assert_eq!(arrow.get(2), Some([5, 6].as_slice()));
    }

    #[test]
    #[should_panic(expected = "zero-width elements")]
    fn a_zero_width_fixed_size_binary_array_has_no_arrow_counterpart() {
        to_arrow(&PlFixedSizeBinaryArray::new(Buffer::new(), 0, 3, None));
    }

    #[test]
    fn list_is_exported_behind_the_offsets_it_already_has() {
        let array = PlListArray::new(
            Box::new(PlPrimitiveArray::from_vec(vec![1i32, 2, 3, 4])),
            Buffer::from(vec![0u64, 2, 2, 4]),
            3,
            Some(Bitmap::from([true, false, true])),
        );
        let offsets = array.flat_offsets().unwrap().as_ptr();

        let arrow = exported::<ListArray<i64>>(&array);

        assert_eq!(
            arrow.dtype(),
            &ListArray::<i64>::default_datatype(ArrowDataType::Int32)
        );
        assert_eq!(arrow.offsets().buffer().as_ptr(), offsets.cast::<i64>());
        assert!(arrow.get(1).is_none());

        let element = arrow.value(0);
        let element = element.as_any().downcast_ref::<Int32Array>().unwrap();
        assert_eq!(element.values().as_slice(), [1, 2]);
    }

    #[test]
    fn fixed_size_list_is_exported_with_its_width() {
        let array = PlFixedSizeListArray::new(
            Box::new(PlPrimitiveArray::from_vec(vec![1i32, 2, 3, 4])),
            2,
            2,
            Some(Bitmap::from([true, false])),
        );
        let arrow = exported::<FixedSizeListArray>(&array);

        assert_eq!(
            arrow.dtype(),
            &FixedSizeListArray::default_datatype(ArrowDataType::Int32, 2)
        );
        assert_eq!(arrow.len(), 2);
        assert!(arrow.get(1).is_none());

        let element = arrow.value(0);
        let element = element.as_any().downcast_ref::<Int32Array>().unwrap();
        assert_eq!(element.values().as_slice(), [1, 2]);
    }

    #[test]
    fn struct_is_exported_with_its_fields_named_after_their_index() {
        let array = PlStructArray::new(
            vec![
                Box::new(PlPrimitiveArray::from_vec(vec![1i32, 2])),
                Box::new(PlBooleanArray::from_vec(vec![true, false])),
            ],
            2,
            Some(Bitmap::from([true, false])),
        );
        let arrow = exported::<StructArray>(&array);

        assert_eq!(
            arrow.dtype(),
            &ArrowDataType::Struct(vec![
                Field::new("0".into(), ArrowDataType::Int32, true),
                Field::new("1".into(), ArrowDataType::Boolean, true),
            ])
        );
        assert_eq!(arrow.len(), 2);
        assert_eq!(arrow.values().len(), 2);
        assert!(arrow.is_null(1));
    }

    #[test]
    fn nested_values_are_exported_along_with_the_array_over_them() {
        let array = PlListArray::from_offsets(
            Box::new(PlBinaryViewArray::from_values_iter([
                b"foo".as_slice(),
                b"bar",
            ])),
            Buffer::from(vec![0u64, 1, 2]),
        );
        let arrow = exported::<ListArray<i64>>(&array);

        // The values are exported as the array that promises the least about their elements.
        assert_eq!(
            arrow.dtype(),
            &ListArray::<i64>::default_datatype(ArrowDataType::BinaryView)
        );
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
    fn a_sliced_array_is_exported_as_the_elements_it_holds() {
        let array = PlPrimitiveArray::from_vec(vec![1i32, 2, 3, 4]).sliced(1, 2);
        let arrow = exported::<Int32Array>(&array);
        assert_eq!(arrow.values().as_slice(), [2, 3]);

        let array =
            PlBinaryArray::from_values_iter([b"foo".as_slice(), b"bar", b"baz"]).sliced(1, 2);
        let arrow = exported::<BinaryArray<i64>>(&array);
        assert_eq!(arrow.value(0), b"bar");
        assert_eq!(arrow.value(1), b"baz");
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
