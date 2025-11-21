use arrow::array::builder::{ShareStrategy, StaticArrayBuilder, make_builder};
use arrow::array::{Array, ArrayCollectIterExt, ListArray, ListArrayBuilder};
use arrow::bitmap::Bitmap;

use super::{IfThenElseKernel, if_then_else_extend};

impl IfThenElseKernel for ListArray<i64> {
    type Scalar<'a> = Box<dyn Array>;

    fn if_then_else(mask: &Bitmap, if_true: &Self, if_false: &Self) -> Self {
        let inner_dt = if_true.dtype().inner_dtype().unwrap();
        let mut builder = ListArrayBuilder::new(if_true.dtype().clone(), make_builder(inner_dt));
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
        let if_true_list: ListArray<i64> =
            std::iter::once(if_true).collect_arr_trusted_with_dtype(if_false.dtype().clone());
        let inner_dt = if_false.dtype().inner_dtype().unwrap();
        let mut builder = ListArrayBuilder::new(if_false.dtype().clone(), make_builder(inner_dt));
        builder.reserve(mask.len());
        if_then_else_extend(
            &mut builder,
            mask,
            |b, _, len| b.subslice_extend_repeated(&if_true_list, 0, 1, len, ShareStrategy::Always),
            |b, off, len| b.subslice_extend(if_false, off, len, ShareStrategy::Always),
        );
        builder.freeze()
    }

    fn if_then_else_broadcast_false(
        mask: &Bitmap,
        if_true: &Self,
        if_false: Self::Scalar<'_>,
    ) -> Self {
        let if_false_list: ListArray<i64> =
            std::iter::once(if_false).collect_arr_trusted_with_dtype(if_true.dtype().clone());
        let inner_dt = if_true.dtype().inner_dtype().unwrap();
        let mut builder = ListArrayBuilder::new(if_true.dtype().clone(), make_builder(inner_dt));
        builder.reserve(mask.len());
        if_then_else_extend(
            &mut builder,
            mask,
            |b, off, len| b.subslice_extend(if_true, off, len, ShareStrategy::Always),
            |b, _, len| {
                b.subslice_extend_repeated(&if_false_list, 0, 1, len, ShareStrategy::Always)
            },
        );
        builder.freeze()
    }

    fn if_then_else_broadcast_both(
        dtype: arrow::datatypes::ArrowDataType,
        mask: &Bitmap,
        if_true: Self::Scalar<'_>,
        if_false: Self::Scalar<'_>,
    ) -> Self {
        let if_true_list: ListArray<i64> =
            std::iter::once(if_true).collect_arr_trusted_with_dtype(dtype.clone());
        let if_false_list: ListArray<i64> =
            std::iter::once(if_false).collect_arr_trusted_with_dtype(dtype.clone());
        let inner_dt = dtype.inner_dtype().unwrap();
        let mut builder = ListArrayBuilder::new(dtype.clone(), make_builder(inner_dt));
        builder.reserve(mask.len());
        if_then_else_extend(
            &mut builder,
            mask,
            |b, _, len| b.subslice_extend_repeated(&if_true_list, 0, 1, len, ShareStrategy::Always),
            |b, _, len| {
                b.subslice_extend_repeated(&if_false_list, 0, 1, len, ShareStrategy::Always)
            },
        );
        builder.freeze()
    }
}
