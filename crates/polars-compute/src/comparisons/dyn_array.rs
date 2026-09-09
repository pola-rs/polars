//! The equality kernels a nested array recurses into its children through, over a `&dyn PlArray`.

use polars_array::{PlArray, PlBitmap};

use super::PlTotalEqKernel;

pub(super) fn downcast<A: PlArray + 'static>(array: &dyn PlArray) -> &A {
    array
        .as_any()
        .downcast_ref()
        .expect("the array type dispatched on names the array")
}

/// Resolves the array type both sides share, once, and runs `$body` over them downcast to it.
///
/// The dispatch is what a `&dyn PlArray` costs, so a caller that reads many elements out of one
/// pair of arrays runs its whole walk inside the body rather than coming back through here for
/// every element.
macro_rules! with_array_pair {
    ($lhs_array:expr, $rhs_array:expr, |$lhs:ident, $rhs:ident| $body:expr $(,)?) => {{
        let (lhs, rhs) = ($lhs_array, $rhs_array);
        assert_eq!(
            lhs.array_type(),
            rhs.array_type(),
            "a nested comparison reached children of different array types",
        );

        macro_rules! call_binary {
            ($A:ty) => {{
                let $lhs = $crate::comparisons::dyn_array::downcast::<$A>(lhs);
                let $rhs = $crate::comparisons::dyn_array::downcast::<$A>(rhs);
                $body
            }};
        }

        use ::polars_array::PlArrayType as A;
        match lhs.array_type() {
            A::Null => call_binary!(::polars_array::PlNullArray),
            A::Boolean => call_binary!(::polars_array::PlBooleanArray),
            // Dispatched on the element type the array type names, not on the concrete array,
            // so that the arms are exactly the primitives a `PlArrayType::Primitive` can hold.
            A::Primitive(primitive) => ::arrow::with_match_primitive_type!(primitive, |$T| {
                let $lhs = $crate::comparisons::dyn_array::downcast::<
                    ::polars_array::PlPrimitiveArray<$T>,
                >(lhs);
                let $rhs = $crate::comparisons::dyn_array::downcast::<
                    ::polars_array::PlPrimitiveArray<$T>,
                >(rhs);
                $body
            }),
            A::Binary => call_binary!(::polars_array::PlBinaryArray),
            A::BinaryView => call_binary!(::polars_array::PlBinaryViewArray),
            A::Utf8View => call_binary!(::polars_array::PlUtf8ViewArray),
            A::FixedSizeBinary => call_binary!(::polars_array::PlFixedSizeBinaryArray),
            A::Struct => call_binary!(::polars_array::PlStructArray),
            A::List => call_binary!(::polars_array::PlListArray),
            #[cfg(feature = "dtype-array")]
            A::FixedSizeList => call_binary!(::polars_array::PlFixedSizeListArray),
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

pub(super) use with_array_pair;

/// Whether both sides hold the same element, reading a null as a value equal only to itself.
pub fn pl_array_tot_eq_missing_kernel(lhs: &dyn PlArray, rhs: &dyn PlArray) -> PlBitmap {
    with_array_pair!(lhs, rhs, |lhs, rhs| PlTotalEqKernel::tot_eq_missing_kernel(
        lhs, rhs
    ))
}

/// Whether the two sides differ, reading a null as a value equal only to itself.
pub fn pl_array_tot_ne_missing_kernel(lhs: &dyn PlArray, rhs: &dyn PlArray) -> PlBitmap {
    with_array_pair!(lhs, rhs, |lhs, rhs| PlTotalEqKernel::tot_ne_missing_kernel(
        lhs, rhs
    ))
}
