use arrow::array::{Array, PrimitiveArray};
use polars_compute::min_max::MinMaxKernel;
use polars_core::prelude::*;
use polars_core::with_match_physical_numeric_polars_type;

fn array_agg<T, S, F1, F2>(
    values: &PrimitiveArray<T>,
    width: usize,
    slice_agg: F1,
    arr_agg: F2,
) -> PrimitiveArray<S>
where
    T: NumericNative,
    S: NumericNative,
    F1: Fn(&[T]) -> Option<S>,
    F2: Fn(&PrimitiveArray<T>) -> Option<S>,
{
    if values.null_count() == 0 {
        let values = values.values().as_slice();
        values
            .chunks_exact(width)
            .map(|sl| slice_agg(sl).unwrap())
            .collect_arr()
    } else {
        (0..values.len())
            .step_by(width)
            .map(|start| {
                // SAFETY: This value array from a FixedSizeListArray,
                // we can ensure that `start + width` will not out out range
                let sliced = unsafe { values.clone().sliced_unchecked(start, width) };
                arr_agg(&sliced)
            })
            .collect_arr()
    }
}

pub(super) enum AggType {
    Min,
    Max,
}

fn agg_min<T>(values: &PrimitiveArray<T>, width: usize) -> PrimitiveArray<T>
where
    T: NumericNative,
    PrimitiveArray<T>: for<'a> MinMaxKernel<Scalar<'a> = T>,
    [T]: for<'a> MinMaxKernel<Scalar<'a> = T>,
{
    array_agg(
        values,
        width,
        MinMaxKernel::min_ignore_nan_kernel,
        MinMaxKernel::min_ignore_nan_kernel,
    )
}

fn agg_max<T>(values: &PrimitiveArray<T>, width: usize) -> PrimitiveArray<T>
where
    T: NumericNative,
    PrimitiveArray<T>: for<'a> MinMaxKernel<Scalar<'a> = T>,
    [T]: for<'a> MinMaxKernel<Scalar<'a> = T>,
{
    array_agg(
        values,
        width,
        MinMaxKernel::max_ignore_nan_kernel,
        MinMaxKernel::max_ignore_nan_kernel,
    )
}

pub(super) fn array_dispatch(
    name: &str,
    values: &Series,
    width: usize,
    agg_type: AggType,
) -> Series {
    let chunks: Vec<ArrayRef> = with_match_physical_numeric_polars_type!(values.dtype(), |$T| {
        let ca: &ChunkedArray<$T> = values.as_ref().as_ref().as_ref();
        ca.downcast_iter().map(|arr| {
            match agg_type {
                AggType::Min => Box::new(agg_min(arr, width)) as ArrayRef,
                AggType::Max => Box::new(agg_max(arr, width)) as ArrayRef,
            }
        }).collect()
    });
    Series::try_from((name, chunks)).unwrap()
}
