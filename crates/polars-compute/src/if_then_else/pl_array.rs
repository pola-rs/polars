//! The if-then-else kernels over the arrays of `polars-array`.

#[cfg(feature = "dtype-array")]
use polars_array::PlFixedSizeListArray;
use polars_array::arrow::bridge::{chunk_from_arrow, flat_to_arrow};
use polars_array::arrow::export;
use polars_array::bitmap::{combine_validities_and, invert};
use polars_array::{
    Flat, PlArray, PlBinaryViewArray, PlBitmap, PlBitmapRef, PlListArray, PlPrimitiveArray,
    PlUtf8ViewArray, StaticArray,
};
use polars_arrow::array::{Array, LIST_VALUES_NAME};
use polars_arrow::bitmap::Bitmap;
use polars_arrow::datatypes::{ArrowDataType, Field};
use polars_arrow::types::NativeType;

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

        // A mask that picks the same side at every element answers with that side itself.
        match mask.agreed_value() {
            Some(true) => return if_true.clone(),
            Some(false) => return if_false.clone(),
            None => {},
        }

        // A side that is null throughout holds no element to pick: what it leaves behind is the
        // other side with the elements this one would have picked masked off. The other side's
        // buffers are handed over as they are, in whatever representation they are in.
        if if_true.scalar_value().is_some_and(|value| value.is_none()) {
            let picked = invert(mask);
            let validity = combine_validities_and(if_false.validity(), Some(picked.as_ref()));
            return if_false.clone().with_validity_typed(validity);
        }
        if if_false.scalar_value().is_some_and(|value| value.is_none()) {
            let validity = combine_validities_and(if_true.validity(), Some(mask));
            return if_true.clone().with_validity_typed(validity);
        }

        // Which side each element is taken from is settled, so the mask the answer carries is
        // settled too -- and the values below are then read on their own. This is what lets a
        // side whose values repeat one value reach a broadcast kernel while a mask of one bit
        // per element sits over it: `scalar_value` answers for both axes at once, and would have
        // had those values written out one slot per element first.
        let validity = pick_validity(mask, if_true.validity(), if_false.validity());
        let (if_true, if_false) = (values_only(if_true), values_only(if_false));

        // The mask holds one bit per element: a repeated one was answered above.
        let mask = mask.flat_bitmap().expect("a scalar mask is answered above");

        // A side that repeats one element hands the kernel that element rather than the array
        // written out one slot per element, which is what the broadcast kernels take.
        let values = match (
            if_true.scalar_value().flatten(),
            if_false.scalar_value().flatten(),
        ) {
            (Some(if_true), Some(if_false)) => {
                Self::if_then_else_flat_broadcast_both(mask, if_true, if_false)
            },
            (Some(if_true), None) => {
                Self::if_then_else_flat_broadcast_true(mask, if_true, &if_false.to_flat())
            },
            (None, Some(if_false)) => {
                Self::if_then_else_flat_broadcast_false(mask, &if_true.to_flat(), if_false)
            },
            (None, None) => Self::if_then_else_flat(mask, &if_true.to_flat(), &if_false.to_flat()),
        };
        values.with_validity_typed(validity)
    }

    /// As [`Self::if_then_else`], with a single value standing for every element of `if_true`.
    fn if_then_else_broadcast_true(
        mask: PlBitmapRef<'_>,
        if_true: Self::ValueT<'_>,
        if_false: &Self,
    ) -> Self {
        assert_eq!(mask.len(), if_false.len(), "{LENGTH_MISMATCH}");

        match mask.agreed_value() {
            // A mask that is unset everywhere is `if_false` itself.
            Some(false) => return if_false.clone(),
            // A mask that is set everywhere picks `if_true` at every element, so the one value is
            // written out once and the result repeats it rather than holding a slot per element.
            // No element of `if_false` is picked, so none of its nulls reaches the answer either.
            Some(true) => {
                let single = Bitmap::new_with_value(true, 1);
                // `if_false` is never read through a set bit; it is here because the kernel takes
                // an array, and one element of it is as good as any other.
                let unpicked = if_false.new_from_index_typed(0, 1);
                let element =
                    Self::if_then_else_flat_broadcast_true(&single, if_true, &unpicked.to_flat());
                debug_assert_eq!(element.len(), 1);

                return element.new_from_index_typed(0, mask.len());
            },
            None => {},
        }

        // `if_true` is a value of its own and never null, so the answer is null only where
        // `if_false` is and is not picked over. That settles the mask, and the values of
        // `if_false` are then read apart from the mask that was over them -- as in
        // [`Self::if_then_else`], this is what lets a repeated value stay one value.
        let validity = pick_validity(mask, None, if_false.validity());
        let if_false = values_only(if_false);

        // Neither side is an array to hand back once `if_false` repeats one element too: both are
        // the single value the kernel writes out, which the two-value kernel does once for a mask
        // that repeats a single bit.
        let values = match if_false.scalar_value().flatten() {
            Some(if_false) => Self::if_then_else_broadcast_both(mask, if_true, if_false),
            None => Self::if_then_else_flat_broadcast_true(
                &mask.to_flat(),
                if_true,
                &if_false.to_flat(),
            ),
        };
        values.with_validity_typed(validity)
    }

    /// As [`Self::if_then_else`], with a single value standing for every element of `if_false`.
    fn if_then_else_broadcast_false(
        mask: PlBitmapRef<'_>,
        if_true: &Self,
        if_false: Self::ValueT<'_>,
    ) -> Self {
        assert_eq!(mask.len(), if_true.len(), "{LENGTH_MISMATCH}");

        // As above, with the sides the other way around.
        match mask.agreed_value() {
            Some(true) => return if_true.clone(),
            // A mask that is unset everywhere picks `if_false` at every element.
            Some(false) => {
                let single = Bitmap::new_with_value(false, 1);
                let unpicked = if_true.new_from_index_typed(0, 1);
                let element =
                    Self::if_then_else_flat_broadcast_false(&single, &unpicked.to_flat(), if_false);
                debug_assert_eq!(element.len(), 1);

                return element.new_from_index_typed(0, mask.len());
            },
            None => {},
        }

        // As above, with the sides the other way around.
        let validity = pick_validity(mask, if_true.validity(), None);
        let if_true = values_only(if_true);

        // Neither side is an array to hand back once `if_true` repeats one element too.
        let values = match if_true.scalar_value().flatten() {
            Some(if_true) => Self::if_then_else_broadcast_both(mask, if_true, if_false),
            None => Self::if_then_else_flat_broadcast_false(
                &mask.to_flat(),
                &if_true.to_flat(),
                if_false,
            ),
        };
        values.with_validity_typed(validity)
    }

    /// As [`Self::if_then_else`], with a single value standing for either side.
    fn if_then_else_broadcast_both(
        mask: PlBitmapRef<'_>,
        if_true: Self::ValueT<'_>,
        if_false: Self::ValueT<'_>,
    ) -> Self {
        // Neither side is an array here, so there is no array for a repeated bit to hand back —
        // but that one bit picks the same value at every element, which is the one element the
        // kernel writes out below and the result repeats from there.
        if let Some(bit) = mask.agreed_value() {
            let single = Bitmap::new_with_value(bit, 1);
            let element = Self::if_then_else_flat_broadcast_both(&single, if_true, if_false);
            debug_assert_eq!(element.len(), 1);

            return element.new_from_index_typed(0, mask.len());
        }

        // The mask holds one bit per element: a repeated one was answered above.
        let mask = mask.flat_bitmap().expect("a scalar mask is answered above");

        Self::if_then_else_flat_broadcast_both(mask, if_true, if_false)
    }
}

/// `arr` with the mask over its values dropped, in `O(1)`.
///
/// The values axis and the mask are separate: a chunk that repeats one value under a mask of one
/// bit per element is not scalar by [`PlArray::is_scalar`], which answers for both at once.
/// Dropping the mask leaves the values axis alone to answer, which is what the kernels below read;
/// the caller settles the answer's own mask separately.
fn values_only<A: StaticArray>(arr: &A) -> A {
    arr.clone().with_validity_typed(None)
}

/// The mask an if-then-else leaves: `if_true`'s where `mask` is set, `if_false`'s where it is not.
///
/// `None` on either side means that side is valid throughout, and a mask that repeats a single bit
/// stays a single bit all the way through.
fn pick_validity(
    mask: PlBitmapRef<'_>,
    if_true: Option<PlBitmapRef<'_>>,
    if_false: Option<PlBitmapRef<'_>>,
) -> Option<PlBitmap> {
    match (if_true, if_false) {
        // Neither side has a null, so nothing the answer takes from either is one.
        (None, None) => None,
        // The side with no mask of its own is valid wherever it is picked, and the other side
        // answers for the elements it is picked at.
        (Some(if_true), None) => Some(invert(mask).or(&PlBitmap::from(if_true))),
        (None, Some(if_false)) => Some(PlBitmap::from(mask).or(&PlBitmap::from(if_false))),
        (Some(if_true), Some(if_false)) => {
            let picked =
                combine_validities_and(Some(mask), Some(if_true)).expect("both masks are there");
            let unpicked = combine_validities_and(Some(invert(mask).as_ref()), Some(if_false))
                .expect("both masks are there");
            Some(picked.or(&unpicked))
        },
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
    polars_arrow::array::PrimitiveArray<T>: for<'a> IfThenElseArrowKernel<Scalar<'a> = T>,
{
    arrow_if_then_else_kernel!(std::convert::identity, |_t, _f| T::PRIMITIVE.into());
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
