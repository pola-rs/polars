//! The if-then-else kernel over a [`PlBooleanArray`], whose values are a mask of their own.

use arrow::bitmap::{self, Bitmap};
use polars_array::{Flat, PlBitmap, PlBitmapRef, PlBooleanArray};

use super::{IfThenElseKernel, if_then_else_validity};

/// The values of a flat chunk, which hold one bit per element like its mask.
#[inline]
fn values(array: &Flat<PlBooleanArray>) -> &Bitmap {
    array.as_array().values().into_inner().0
}

/// The mask of a flat chunk, which holds one bit per element.
#[inline]
fn validity(array: &Flat<PlBooleanArray>) -> Option<&Bitmap> {
    array
        .as_array()
        .validity()
        .map(|validity| validity.into_inner().0)
}

/// The values `mask` picks between `if_true` and `if_false`, for two single values.
#[inline]
fn pick(mask: PlBitmap, if_true: bool, if_false: bool) -> PlBitmap {
    match (if_true, if_false) {
        // The two sides agree, so every element takes the one value they agree on and the mask
        // says nothing: this is the same bit for every element.
        (false, false) => PlBitmap::new_scalar(false, mask.len()),
        (true, true) => PlBitmap::new_scalar(true, mask.len()),
        // They differ, so an element takes `true` exactly where the mask picks the true side.
        (true, false) => mask,
        (false, true) => mask.not(),
    }
}

impl IfThenElseKernel for PlBooleanArray {
    fn if_then_else_flat(mask: &Bitmap, if_true: &Flat<Self>, if_false: &Flat<Self>) -> Self {
        let out = bitmap::ternary(mask, values(if_true), values(if_false), |m, t, f| {
            (m & t) | (!m & f)
        });
        PlBooleanArray::from_values(out).with_validity(
            if_then_else_validity(mask, validity(if_true), validity(if_false)).map(PlBitmap::from),
        )
    }

    fn if_then_else_flat_broadcast_true(
        mask: &Bitmap,
        if_true: bool,
        if_false: &Flat<Self>,
    ) -> Self {
        let out = if if_true {
            bitmap::or(values(if_false), mask) // (m & true)  | (!m & f)  ->  f | m
        } else {
            bitmap::and_not(values(if_false), mask) // (m & false) | (!m & f)  ->  f & !m
        };
        PlBooleanArray::from_values(out).with_validity(
            if_then_else_validity(mask, None, validity(if_false)).map(PlBitmap::from),
        )
    }

    fn if_then_else_flat_broadcast_false(
        mask: &Bitmap,
        if_true: &Flat<Self>,
        if_false: bool,
    ) -> Self {
        let out = if if_false {
            bitmap::or_not(values(if_true), mask) // (m & t) | (!m & true)   ->  t | !m
        } else {
            bitmap::and(values(if_true), mask) // (m & t) | (!m & false)  ->  t & m
        };
        PlBooleanArray::from_values(out)
            .with_validity(if_then_else_validity(mask, validity(if_true), None).map(PlBitmap::from))
    }

    fn if_then_else_flat_broadcast_both(mask: &Bitmap, if_true: bool, if_false: bool) -> Self {
        // The mask is flat, but the answer need not be: two sides that agree leave one value for
        // every element, which a scalar chunk holds in a single bit.
        let length = mask.len();
        PlBooleanArray::from_pl_bitmap(pick(PlBitmap::new(mask.clone(), length), if_true, if_false))
    }

    fn if_then_else_broadcast_both(mask: PlBitmapRef<'_>, if_true: bool, if_false: bool) -> Self {
        // Overridden because the mask is the answer here, so a mask that repeats a single bit is
        // handed back as the single bit it is rather than written out.
        PlBooleanArray::from_pl_bitmap(pick(PlBitmap::from(mask), if_true, if_false))
    }
}
