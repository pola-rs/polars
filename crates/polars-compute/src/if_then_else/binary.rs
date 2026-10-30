use polars_arrow::array::builder::{ShareStrategy, StaticArrayBuilder};
use polars_arrow::array::{BinaryArray, BinaryArrayBuilder};
use polars_arrow::bitmap::Bitmap;
use polars_arrow::datatypes::ArrowDataType;

use super::{IfThenElseKernel, if_then_else_extend};

impl IfThenElseKernel for BinaryArray<i64> {
    type Scalar<'a> = &'a [u8];

    fn if_then_else(mask: &Bitmap, if_true: &Self, if_false: &Self) -> Self {
        let mut builder = BinaryArrayBuilder::new(if_true.dtype().clone());
        builder.reserve(mask.len());
        if_then_else_extend(
            &mut builder,
            mask,
            |b, off, len| b.subslice_extend(if_true, off, len, ShareStrategy::Always),
            |b, off, len| b.subslice_extend(if_false, off, len, ShareStrategy::Always),
        );
        builder.freeze()
    }

    fn if_then_else_broadcast_true(
        mask: &Bitmap,
        if_true: Self::Scalar<'_>,
        if_false: &Self,
    ) -> Self {
        let if_true_arr = BinaryArray::<i64>::from_slice([if_true]);
        let mut builder = BinaryArrayBuilder::new(if_false.dtype().clone());
        builder.reserve(mask.len());
        if_then_else_extend(
            &mut builder,
            mask,
            |b, _, len| b.subslice_extend_repeated(&if_true_arr, 0, 1, len, ShareStrategy::Always),
            |b, off, len| b.subslice_extend(if_false, off, len, ShareStrategy::Always),
        );
        builder.freeze()
    }

    fn if_then_else_broadcast_false(
        mask: &Bitmap,
        if_true: &Self,
        if_false: Self::Scalar<'_>,
    ) -> Self {
        let if_false_arr = BinaryArray::<i64>::from_slice([if_false]);
        let mut builder = BinaryArrayBuilder::new(if_true.dtype().clone());
        builder.reserve(mask.len());
        if_then_else_extend(
            &mut builder,
            mask,
            |b, off, len| b.subslice_extend(if_true, off, len, ShareStrategy::Always),
            |b, _, len| b.subslice_extend_repeated(&if_false_arr, 0, 1, len, ShareStrategy::Always),
        );
        builder.freeze()
    }

    fn if_then_else_broadcast_both(
        dtype: ArrowDataType,
        mask: &Bitmap,
        if_true: Self::Scalar<'_>,
        if_false: Self::Scalar<'_>,
    ) -> Self {
        let if_true_arr = BinaryArray::<i64>::from_slice([if_true]);
        let if_false_arr = BinaryArray::<i64>::from_slice([if_false]);
        let mut builder = BinaryArrayBuilder::new(dtype);
        builder.reserve(mask.len());
        if_then_else_extend(
            &mut builder,
            mask,
            |b, _, len| b.subslice_extend_repeated(&if_true_arr, 0, 1, len, ShareStrategy::Always),
            |b, _, len| b.subslice_extend_repeated(&if_false_arr, 0, 1, len, ShareStrategy::Always),
        );
        builder.freeze()
    }
}
