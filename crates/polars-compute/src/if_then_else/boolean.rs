use arrow::array::BooleanArray;
use arrow::bitmap::{self, Bitmap};
use arrow::datatypes::ArrowDataType;

use super::{if_then_else_validity, IfThenElseKernel};

impl IfThenElseKernel for BooleanArray {
    type Scalar<'a> = bool;

    fn if_then_else(mask: &Bitmap, if_true: &Self, if_false: &Self) -> Self {
        let values = bitmap::ternary(mask, if_true.values(), if_false.values(), |m, t, f| {
            (m & t) | (!m & f)
        });
        let validity = if_then_else_validity(mask, if_true.validity(), if_false.validity());
        BooleanArray::from(values).with_validity(validity)
    }

    fn if_then_else_broadcast_true(
        mask: &Bitmap,
        if_true: Self::Scalar<'_>,
        if_false: &Self,
    ) -> Self {
        let values = if if_true {
            bitmap::or(if_false.values(), mask) // (m & true)  | (!m & f)  ->  f | m
        } else {
            bitmap::and_not(if_false.values(), mask) // (m & false) | (!m & f)  ->  f & !m
        };
        let validity = if_then_else_validity(mask, None, if_false.validity());
        BooleanArray::from(values).with_validity(validity)
    }

    fn if_then_else_broadcast_false(
        mask: &Bitmap,
        if_true: &Self,
        if_false: Self::Scalar<'_>,
    ) -> Self {
        let values = if if_false {
            bitmap::or_not(if_true.values(), mask) // (m & t) | (!m & true)   ->  t | !m
        } else {
            bitmap::and(if_true.values(), mask) // (m & t) | (!m & false)  ->  t & m
        };
        let validity = if_then_else_validity(mask, if_true.validity(), None);
        BooleanArray::from(values).with_validity(validity)
    }

    fn if_then_else_broadcast_both(
        _dtype: ArrowDataType,
        mask: &Bitmap,
        if_true: Self::Scalar<'_>,
        if_false: Self::Scalar<'_>,
    ) -> Self {
        let values = match (if_true, if_false) {
            (false, false) => Bitmap::new_with_value(false, mask.len()),
            (false, true) => !mask,
            (true, false) => mask.clone(),
            (true, true) => Bitmap::new_with_value(true, mask.len()),
        };
        BooleanArray::from(values)
    }
}
