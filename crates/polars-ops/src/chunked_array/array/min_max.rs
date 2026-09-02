use arrow::array::{Array, PrimitiveArray};
use polars_compute::min_max::MinMaxKernel;
use polars_core::chunked_array::arrow_bridge::chunk_to_arrow;
use polars_core::prelude::*;
use polars_core::with_match_physical_numeric_polars_type;

fn array_agg<T, S, F1, F2>(
    values: &PrimitiveArray<T>,
    width: usize,
    slice_agg: F1,
    arr_agg: F2,
) -> PlPrimitiveArray<S>
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

fn agg_min<T>(values: &PrimitiveArray<T>, width: usize) -> PlPrimitiveArray<T>
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

fn agg_max<T>(values: &PrimitiveArray<T>, width: usize) -> PlPrimitiveArray<T>
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
    name: PlSmallStr,
    values: &Series,
    width: usize,
    agg_type: AggType,
) -> Series {
    with_match_physical_numeric_polars_type!(values.dtype(), |$T| {
        let ca: &ChunkedArray<$T> = values.as_ref().as_ref().as_ref();
        let chunks = ca.downcast_iter().map(|arr| {
            // TODO(polars-array-scalar): the min/max kernels are Arrow ones that read the values
            // as a slice, so a scalar chunk is written out here rather than reduced once.
            let arr = chunk_to_arrow(arr);
            match agg_type {
                AggType::Min => agg_min(&arr, width),
                AggType::Max => agg_max(&arr, width),
            }
        });

        ChunkedArray::<$T>::from_chunk_iter(name, chunks).into_series()
    })
}
