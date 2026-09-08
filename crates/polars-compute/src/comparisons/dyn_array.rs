//! The equality kernels a nested array recurses into its children through, over a `&dyn PlArray`.

use arrow::with_match_primitive_type;
#[cfg(feature = "dtype-array")]
use polars_array::PlFixedSizeListArray;
use polars_array::{
    PlArray, PlArrayType, PlBinaryArray, PlBinaryViewArray, PlBitmap, PlBooleanArray,
    PlFixedSizeBinaryArray, PlListArray, PlNullArray, PlPrimitiveArray, PlStructArray,
    PlUtf8ViewArray,
};

use super::PlTotalEqKernel;

fn downcast<A: PlArray + 'static>(array: &dyn PlArray) -> &A {
    array
        .as_any()
        .downcast_ref()
        .expect("the array type dispatched on names the array")
}

/// Dispatches a nested comparison on the array type both sides share.
macro_rules! compare {
    ($lhs:expr, $rhs:expr, $op:path $(,)?) => {{
        let (lhs, rhs) = ($lhs, $rhs);
        assert_eq!(
            lhs.array_type(),
            rhs.array_type(),
            "a nested comparison reached children of different array types",
        );

        macro_rules! call_binary {
            ($A:ty) => {{ $op(downcast::<$A>(lhs), downcast::<$A>(rhs)) }};
        }

        use PlArrayType as A;
        match lhs.array_type() {
            A::Null => call_binary!(PlNullArray),
            A::Boolean => call_binary!(PlBooleanArray),
            // Dispatched on the element type the array type names, not on the concrete array,
            // so that the arms are exactly the primitives a `PlArrayType::Primitive` can hold.
            A::Primitive(primitive) => with_match_primitive_type!(primitive, |$T| $op(
                downcast::<PlPrimitiveArray<$T>>(lhs),
                downcast::<PlPrimitiveArray<$T>>(rhs),
            )),
            A::Binary => call_binary!(PlBinaryArray),
            A::BinaryView => call_binary!(PlBinaryViewArray),
            A::Utf8View => call_binary!(PlUtf8ViewArray),
            A::FixedSizeBinary => call_binary!(PlFixedSizeBinaryArray),
            A::Struct => call_binary!(PlStructArray),
            A::List => call_binary!(PlListArray),
            #[cfg(feature = "dtype-array")]
            A::FixedSizeList => call_binary!(PlFixedSizeListArray),
            #[cfg(not(feature = "dtype-array"))]
            A::FixedSizeList => todo!(
                "comparison of a fixed-size-list array is not supported without the dtype-array \
                 feature"
            ),
            array_type @ A::Object { .. } => {
                unimplemented!("polars-compute: comparison of a nested {array_type:?}")
            },
        }
    }};
}

/// Whether both sides hold the same element, reading a null as a value equal only to itself.
pub fn pl_array_tot_eq_missing_kernel(lhs: &dyn PlArray, rhs: &dyn PlArray) -> PlBitmap {
    compare!(lhs, rhs, PlTotalEqKernel::tot_eq_missing_kernel)
}

/// Whether the two sides differ, reading a null as a value equal only to itself.
pub fn pl_array_tot_ne_missing_kernel(lhs: &dyn PlArray, rhs: &dyn PlArray) -> PlBitmap {
    compare!(lhs, rhs, PlTotalEqKernel::tot_ne_missing_kernel)
}
