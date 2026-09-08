use polars_compute::min_max::MinMaxKernel;
use polars_core::prelude::*;
use polars_core::with_match_physical_numeric_polars_type;

/// Reduces every run of `width` consecutive elements of `values` to one element.
fn array_agg<T, S, F1, F2>(
    values: &PlPrimitiveArray<T>,
    width: usize,
    slice_agg: F1,
    arr_agg: F2,
) -> PlPrimitiveArray<S>
where
    T: NumericNative,
    S: NumericNative,
    F1: Fn(&[T]) -> Option<S>,
    F2: Fn(&PlPrimitiveArray<T>) -> Option<S>,
{
    // Without a null anywhere the rows are read straight out of the values buffer, in whichever
    // representation it is in.
    if !values.has_nulls() {
        return match values.scalar_values() {
            // Every row is the same `width` copies of the one value, and so reduces to it — as
            // does the answer, which repeats a single value in turn.
            Some(value) => {
                let reduced =
                    slice_agg(&[value]).expect("a row of one value reduces to that value");
                PlPrimitiveArray::new_scalar(reduced, values.len() / width)
            },
            // The rows are runs of the values buffer, which the kernel that reads a slice reduces
            // without a validity mask to consult.
            None => values
                .flat_values()
                .unwrap()
                .as_slice()
                .chunks_exact(width)
                .map(|sl| slice_agg(sl).unwrap())
                .collect_arr(),
        };
    }

    (0..values.len())
        .step_by(width)
        .map(|start| arr_agg(&values.sliced(start, width)))
        .collect_arr()
}

pub(super) enum AggType {
    Min,
    Max,
}

fn agg_min<T>(values: &PlPrimitiveArray<T>, width: usize) -> PlPrimitiveArray<T>
where
    T: NumericNative,
    PlPrimitiveArray<T>: for<'a> MinMaxKernel<Scalar<'a> = T>,
    [T]: for<'a> MinMaxKernel<Scalar<'a> = T>,
{
    array_agg(
        values,
        width,
        MinMaxKernel::min_ignore_nan_kernel,
        MinMaxKernel::min_ignore_nan_kernel,
    )
}

fn agg_max<T>(values: &PlPrimitiveArray<T>, width: usize) -> PlPrimitiveArray<T>
where
    T: NumericNative,
    PlPrimitiveArray<T>: for<'a> MinMaxKernel<Scalar<'a> = T>,
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
            match agg_type {
                AggType::Min => agg_min(arr, width),
                AggType::Max => agg_max(arr, width),
            }
        });

        ChunkedArray::<$T>::from_chunk_iter(name, chunks).into_series()
    })
}
