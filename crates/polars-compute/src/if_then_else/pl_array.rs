//! The if-then-else kernels over the arrays of `polars-array`.

use arrow::array::{Array, LIST_VALUES_NAME};
use arrow::bitmap::Bitmap;
use arrow::datatypes::{ArrowDataType, Field};
use arrow::types::NativeType;
#[cfg(feature = "dtype-array")]
use polars_array::PlFixedSizeListArray;
use polars_array::arrow::bridge::{chunk_from_arrow, flat_to_arrow};
use polars_array::arrow::export;
use polars_array::{
    Flat, PlArray, PlBinaryViewArray, PlBitmapRef, PlBooleanArray, PlListArray, PlPrimitiveArray,
    PlUtf8ViewArray, StaticArray,
};

use super::IfThenElseArrowKernel;

/// The if-then-else kernel over the arrays of `polars-array`.
pub trait IfThenElseKernel: StaticArray {
    /// The elements of `if_true` where `mask` is set, and of `if_false` where it is not.
    fn if_then_else_flat(mask: &Bitmap, if_true: &Flat<Self>, if_false: &Flat<Self>) -> Self;

    /// As [`Self::if_then_else_flat`], with one value standing for every element of `if_true`.
    fn if_then_else_flat_broadcast_true(
        mask: &Bitmap,
        if_true: Self::ValueT<'_>,
        if_false: &Flat<Self>,
    ) -> Self;

    /// As [`Self::if_then_else_flat`], with one value standing for every element of `if_false`.
    fn if_then_else_flat_broadcast_false(
        mask: &Bitmap,
        if_true: &Flat<Self>,
        if_false: Self::ValueT<'_>,
    ) -> Self;

    /// As [`Self::if_then_else_flat`], with one value standing for every element of either side.
    fn if_then_else_flat_broadcast_both(
        mask: &Bitmap,
        if_true: Self::ValueT<'_>,
        if_false: Self::ValueT<'_>,
    ) -> Self;

    /// The elements of `if_true` where `mask` is set, and of `if_false` where it is not.
    fn if_then_else(mask: PlBitmapRef<'_>, if_true: &Self, if_false: &Self) -> Self {
        assert_eq!(mask.len(), if_true.len(), "{LENGTH_MISMATCH}");
        assert_eq!(mask.len(), if_false.len(), "{LENGTH_MISMATCH}");

        // One bit picks the same side at every element, which is therefore that side itself.
        match mask.scalar_value() {
            Some(true) => if_true.clone(),
            Some(false) => if_false.clone(),
            None => {
                Self::if_then_else_flat(&mask.to_flat(), &if_true.to_flat(), &if_false.to_flat())
            },
        }
    }

    /// As [`Self::if_then_else`], with a single value standing for every element of `if_true`.
    fn if_then_else_broadcast_true(
        mask: PlBitmapRef<'_>,
        if_true: Self::ValueT<'_>,
        if_false: &Self,
    ) -> Self {
        assert_eq!(mask.len(), if_false.len(), "{LENGTH_MISMATCH}");

        // A mask that is unset everywhere is `if_false` itself. One that is set everywhere has no
        // array to hand back, only the one value, which the kernel below writes out.
        if mask.scalar_value() == Some(false) {
            return if_false.clone();
        }

        Self::if_then_else_flat_broadcast_true(&mask.to_flat(), if_true, &if_false.to_flat())
    }

    /// As [`Self::if_then_else`], with a single value standing for every element of `if_false`.
    fn if_then_else_broadcast_false(
        mask: PlBitmapRef<'_>,
        if_true: &Self,
        if_false: Self::ValueT<'_>,
    ) -> Self {
        assert_eq!(mask.len(), if_true.len(), "{LENGTH_MISMATCH}");

        // As above, with the sides the other way around.
        if mask.scalar_value() == Some(true) {
            return if_true.clone();
        }

        Self::if_then_else_flat_broadcast_false(&mask.to_flat(), &if_true.to_flat(), if_false)
    }

    /// As [`Self::if_then_else`], with a single value standing for either side.
    fn if_then_else_broadcast_both(
        mask: PlBitmapRef<'_>,
        if_true: Self::ValueT<'_>,
        if_false: Self::ValueT<'_>,
    ) -> Self {
        // Neither side is an array here, so there is nothing for a repeated bit to hand back: the
        // kernel writes the chosen value out either way.
        Self::if_then_else_flat_broadcast_both(&mask.to_flat(), if_true, if_false)
    }
}

const LENGTH_MISMATCH: &str =
    "an if-then-else mask covers a different number of elements than the sides it picks between";

/// The element of a nested array, exported as the Arrow array its kernel takes.
#[inline]
fn to_arrow_element(element: Box<dyn PlArray>) -> Box<dyn Array> {
    export::to_arrow(&*element)
}

/// The body of an [`IfThenElseKernel`] whose chunks cross over to [`IfThenElseArrowKernel`].
macro_rules! arrow_if_then_else_kernel {
    ($to_arrow_scalar:expr, |$t:ident, $f:ident| $dtype:expr) => {
        #[inline]
        fn if_then_else_flat(mask: &Bitmap, if_true: &Flat<Self>, if_false: &Flat<Self>) -> Self {
            chunk_from_arrow(&IfThenElseArrowKernel::if_then_else(
                mask,
                &flat_to_arrow(if_true),
                &flat_to_arrow(if_false),
            ))
        }

        #[inline]
        fn if_then_else_flat_broadcast_true(
            mask: &Bitmap,
            if_true: Self::ValueT<'_>,
            if_false: &Flat<Self>,
        ) -> Self {
            chunk_from_arrow(&IfThenElseArrowKernel::if_then_else_broadcast_true(
                mask,
                $to_arrow_scalar(if_true),
                &flat_to_arrow(if_false),
            ))
        }

        #[inline]
        fn if_then_else_flat_broadcast_false(
            mask: &Bitmap,
            if_true: &Flat<Self>,
            if_false: Self::ValueT<'_>,
        ) -> Self {
            chunk_from_arrow(&IfThenElseArrowKernel::if_then_else_broadcast_false(
                mask,
                &flat_to_arrow(if_true),
                $to_arrow_scalar(if_false),
            ))
        }

        #[inline]
        fn if_then_else_flat_broadcast_both(
            mask: &Bitmap,
            if_true: Self::ValueT<'_>,
            if_false: Self::ValueT<'_>,
        ) -> Self {
            let $t = $to_arrow_scalar(if_true);
            let $f = $to_arrow_scalar(if_false);
            let dtype = $dtype;
            chunk_from_arrow(&IfThenElseArrowKernel::if_then_else_broadcast_both(
                dtype, mask, $t, $f,
            ))
        }
    };
}

impl<T: NativeType> IfThenElseKernel for PlPrimitiveArray<T>
where
    arrow::array::PrimitiveArray<T>: for<'a> IfThenElseArrowKernel<Scalar<'a> = T>,
{
    arrow_if_then_else_kernel!(std::convert::identity, |_t, _f| T::PRIMITIVE.into());
}

impl IfThenElseKernel for PlBooleanArray {
    arrow_if_then_else_kernel!(std::convert::identity, |_t, _f| ArrowDataType::Boolean);
}

impl IfThenElseKernel for PlUtf8ViewArray {
    arrow_if_then_else_kernel!(std::convert::identity, |_t, _f| ArrowDataType::Utf8View);
}

impl IfThenElseKernel for PlBinaryViewArray {
    arrow_if_then_else_kernel!(std::convert::identity, |_t, _f| ArrowDataType::BinaryView);
}

impl IfThenElseKernel for PlListArray {
    // The elements of a list array are arrays of their own, whose data type the result carries.
    arrow_if_then_else_kernel!(to_arrow_element, |t, _f| ArrowDataType::LargeList(
        Box::new(Field::new(LIST_VALUES_NAME, t.dtype().clone(), true))
    ));
}

#[cfg(feature = "dtype-array")]
impl IfThenElseKernel for PlFixedSizeListArray {
    arrow_if_then_else_kernel!(to_arrow_element, |t, _f| ArrowDataType::FixedSizeList(
        Box::new(Field::new(LIST_VALUES_NAME, t.dtype().clone(), true)),
        t.len(),
    ));
}
