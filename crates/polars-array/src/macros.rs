//! Macros over the array types of this crate.

/// Runs a body with `T` bound to the element type of a
/// [`PlPrimitiveArray`](crate::PlPrimitiveArray).
///
/// The array to dispatch on comes first and the body second, written as a closure over the element
/// type: `with_match_pl_primitive_array_type!(array, |T| { ... })`. The body is expanded once per
/// element type a primitive array can be taken over, and the one whose element type the array
/// actually has is the one that runs. This evaluates to `Some(body)` for that element type, and to
/// `None` when the array is not a primitive array at all — so every expansion of the body has to
/// have the same type.
///
/// The dispatch is on the concrete type the array downcasts to rather than on its
/// [`PlArrayType`](crate::PlArrayType), since a [`PrimitiveType`](crate::PrimitiveType) does not
/// pin the element type down: [`View`](arrow::array::View) and `u128` are both
/// [`PrimitiveType::UInt128`](crate::PrimitiveType::UInt128).
///
/// # Example
/// ```
/// use polars_array::{PlArray, PlBooleanArray, PlPrimitiveArray, with_match_pl_primitive_array_type};
///
/// let array: Box<dyn PlArray> = Box::new(PlPrimitiveArray::from_vec(vec![1i32, 2, 3]));
/// let size = with_match_pl_primitive_array_type!(&*array, |T| size_of::<T>());
/// assert_eq!(size, Some(4));
///
/// // There is no element type to run the body with if the array is not a primitive array.
/// let array: Box<dyn PlArray> = Box::new(PlBooleanArray::from_vec(vec![true]));
/// let size = with_match_pl_primitive_array_type!(&*array, |T| size_of::<T>());
/// assert_eq!(size, None);
/// ```
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
///
/// The binding is a type alias, so `$T` is a type and nothing else: it cannot be used as the name
/// of a value, a trait or a lifetime.
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

    use crate::{PlArray, PlListArray, PlNullArray, PlPrimitiveArray, PlStructArray};

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
        let arrays: [Box<dyn PlArray>; 3] = [
            Box::new(PlNullArray::new(1)),
            Box::new(PlListArray::new_empty(Box::new(
                PlPrimitiveArray::<i32>::new_empty(),
            ))),
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
