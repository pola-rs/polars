//! Macros over the array types of this crate.

/// Runs a body with `T` bound to the element type of a
/// [`PlPrimitiveArray`](crate::PlPrimitiveArray).
#[macro_export]
macro_rules! with_match_pl_primitive_array_type {
    ($array:expr, |$T:ident| $body:expr $(,)?) => {{
        use ::arrow::array::View;
        use ::arrow::types::{days_ms, i256, months_days_ns};
        use ::polars_utils::float16::pf16;

        // `NativeType` is a sealed trait, so this list of element types is exhaustive.
        $crate::__with_match_pl_primitive_array_type__! {
            $array,
            [
                i8, i16, i32, i64, i128, i256,
                u8, u16, u32, u64, u128,
                pf16, f32, f64,
                days_ms, months_days_ns, View,
            ],
            $T,
            $body
        }
    }};
}

/// The body of [`with_match_pl_primitive_array_type`], which binds `$T` to the element type in
/// `$element` the array is taken over.
#[doc(hidden)]
#[macro_export]
macro_rules! __with_match_pl_primitive_array_type__ {(
    $array:expr, [$($element:ty),* $(,)?], $T:ident, $body:expr
) => ({
    let array: &dyn $crate::PlArray = $array;
    $(if array.as_any().is::<$crate::PlPrimitiveArray<$element>>() {
        Some({
            #[allow(dead_code)]
            type $T = $element;
            $body
        })
    } else)* {
        None
    }
})}

#[cfg(test)]
mod tests {
    use arrow::array::View;
    use arrow::types::{days_ms, i256, months_days_ns};
    use polars_utils::float16::pf16;

    use crate::{
        PlArray, PlBinaryArray, PlFixedSizeBinaryArray, PlFixedSizeListArray, PlListArray,
        PlNullArray, PlPrimitiveArray, PlStructArray,
    };

    /// Whether the body runs with `T` bound to the element type of an array of `T`.
    fn dispatches<T: arrow::types::NativeType>() -> bool {
        let array: Box<dyn PlArray> = Box::new(PlPrimitiveArray::<T>::new_empty());
        with_match_pl_primitive_array_type!(&*array, |E| {
            std::any::TypeId::of::<E>() == std::any::TypeId::of::<T>()
        })
        .unwrap()
    }

    #[test]
    fn every_element_type_is_dispatched_to() {
        assert!(dispatches::<i8>());
        assert!(dispatches::<i16>());
        assert!(dispatches::<i32>());
        assert!(dispatches::<i64>());
        assert!(dispatches::<i128>());
        assert!(dispatches::<i256>());
        assert!(dispatches::<u8>());
        assert!(dispatches::<u16>());
        assert!(dispatches::<u32>());
        assert!(dispatches::<u64>());
        assert!(dispatches::<u128>());
        assert!(dispatches::<pf16>());
        assert!(dispatches::<f32>());
        assert!(dispatches::<f64>());
        assert!(dispatches::<days_ms>());
        assert!(dispatches::<months_days_ns>());
        assert!(dispatches::<View>());
    }

    #[test]
    fn arrays_that_are_not_primitive_have_no_element_type() {
        let arrays: [Box<dyn PlArray>; 6] = [
            Box::new(PlNullArray::new(1)),
            Box::new(PlBinaryArray::new_empty()),
            Box::new(PlFixedSizeBinaryArray::new_empty(2)),
            Box::new(PlListArray::new_empty(Box::new(
                PlPrimitiveArray::<i32>::new_empty(),
            ))),
            Box::new(PlFixedSizeListArray::new_empty(
                Box::new(PlPrimitiveArray::<i32>::new_empty()),
                2,
            )),
            Box::new(PlStructArray::new_empty()),
        ];
        for array in &arrays {
            assert_eq!(
                with_match_pl_primitive_array_type!(&**array, |T| size_of::<T>()),
                None,
            );
        }
    }
}
