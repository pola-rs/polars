//! The if-then-else kernels over the arrays of `polars-array`.
//!
//! A mask that repeats one bit says the same of every element, so it picks the same side
//! throughout: that side is handed back exactly as it is, in `O(1)` and in whatever representation
//! it is already in, and the other side is never read at all. Anything else crosses over to the
//! Arrow kernel, which reads one slot per element.

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
///
/// An array implements the four `_flat` kernels, which read one slot per element. The four entry
/// points above them read the representation the mask is in first, and only reach a kernel for the
/// elements it does not already answer for.
pub trait IfThenElseKernel: StaticArray {
    /// The elements of `if_true` where `mask` is set, and of `if_false` where it is not.
    ///
    /// # Panics
    /// Panics unless the mask and both sides cover the same number of elements.
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
    ///
    /// # Panics
    /// Panics unless the mask and both sides cover the same number of elements.
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
    ///
    /// # Panics
    /// Panics unless the mask and `if_false` cover the same number of elements.
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
    ///
    /// # Panics
    /// Panics unless the mask and `if_true` cover the same number of elements.
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

    /// As [`Self::if_then_else`], with a single value standing for either side. The result covers
    /// as many elements as the mask.
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
///
/// `$to_arrow_scalar` maps one element to the value the Arrow kernel takes, and `$dtype` names the
/// data type of a result that is built from nothing but two such values, which is the one case in
/// which no array is around to read it off.
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

#[cfg(test)]
mod tests {
    use polars_array::PlBitmap;

    use super::*;

    /// A mask that repeats one bit picks the same side at every element, which it hands back as it
    /// is: a chunk that repeats one value stays that way rather than being written out.
    #[test]
    fn a_repeated_bit_picks_one_side_whole() {
        let if_true = PlPrimitiveArray::new_scalar(7i32, 4);
        let if_false = PlPrimitiveArray::from_vec(vec![1i32, 2, 3, 4]);

        let picked = IfThenElseKernel::if_then_else(
            PlBitmap::new_scalar(true, 4).as_ref(),
            &if_true,
            &if_false,
        );
        assert!(picked.is_scalar());
        assert_eq!(picked, if_true);

        let picked = IfThenElseKernel::if_then_else(
            PlBitmap::new_scalar(false, 4).as_ref(),
            &if_true,
            &if_false,
        );
        assert_eq!(picked, if_false);
    }

    /// The same holds where one side is a single value: the side that is an array comes back
    /// whole, and the value is never written out.
    #[test]
    fn a_repeated_bit_picks_the_side_that_is_an_array() {
        let arr = PlPrimitiveArray::new_scalar(7i32, 4);

        let picked = IfThenElseKernel::if_then_else_broadcast_true(
            PlBitmap::new_scalar(false, 4).as_ref(),
            9i32,
            &arr,
        );
        assert!(picked.is_scalar());
        assert_eq!(picked, arr);

        let picked = IfThenElseKernel::if_then_else_broadcast_false(
            PlBitmap::new_scalar(true, 4).as_ref(),
            &arr,
            9i32,
        );
        assert!(picked.is_scalar());
        assert_eq!(picked, arr);
    }

    /// A mask that is set everywhere still has to write out the value standing in for `if_true`,
    /// which is the one side that is not an array to hand back.
    #[test]
    fn a_broadcast_value_is_written_out() {
        let arr = PlPrimitiveArray::from_vec(vec![1i32, 2, 3]);

        let picked = IfThenElseKernel::if_then_else_broadcast_true(
            PlBitmap::new_scalar(true, 3).as_ref(),
            9i32,
            &arr,
        );
        assert_eq!(picked, PlPrimitiveArray::from_vec(vec![9i32; 3]));
    }

    /// A mask laid out one bit per element picks side by side, keeping the nulls of whichever side
    /// each element comes from.
    #[test]
    fn every_element_picks_its_own_side() {
        let if_true = PlPrimitiveArray::from_iter([Some(1i32), None, Some(3)]);
        let if_false = PlPrimitiveArray::from_iter([Some(4i32), Some(5), None]);
        let mask = PlBitmap::from_iter([true, true, false]);

        assert_eq!(
            IfThenElseKernel::if_then_else(mask.as_ref(), &if_true, &if_false),
            PlPrimitiveArray::from_iter([Some(1i32), None, None]),
        );
    }

    /// The kernels agree with each other whichever representation the mask reaches them in.
    #[test]
    fn the_representation_of_the_mask_does_not_change_the_answer() {
        let if_true = PlUtf8ViewArray::from_iter([Some("a"), None, Some("c")]);
        let if_false = PlUtf8ViewArray::new_scalar("z", 3);

        for bit in [false, true] {
            let scalar = PlBitmap::new_scalar(bit, 3);
            let flat = PlBitmap::from_iter([bit; 3]);

            assert_eq!(
                IfThenElseKernel::if_then_else(scalar.as_ref(), &if_true, &if_false),
                IfThenElseKernel::if_then_else(flat.as_ref(), &if_true, &if_false),
            );
            assert_eq!(
                IfThenElseKernel::if_then_else_broadcast_true(scalar.as_ref(), "y", &if_false),
                IfThenElseKernel::if_then_else_broadcast_true(flat.as_ref(), "y", &if_false),
            );
            assert_eq!(
                IfThenElseKernel::if_then_else_broadcast_false(scalar.as_ref(), &if_true, "y"),
                IfThenElseKernel::if_then_else_broadcast_false(flat.as_ref(), &if_true, "y"),
            );
            assert_eq!(
                PlUtf8ViewArray::if_then_else_broadcast_both(scalar.as_ref(), "y", "n"),
                PlUtf8ViewArray::if_then_else_broadcast_both(flat.as_ref(), "y", "n"),
            );
        }
    }
}
