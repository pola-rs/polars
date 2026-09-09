use polars_array::bitmap::combine_validities_and;
use polars_compute::min_max::MinMaxKernel;
use polars_core::prelude::*;
use polars_core::with_match_physical_numeric_polars_type;

/// Reduces every one of the `length` runs of `width` consecutive elements of `values` to one
/// element.
fn array_agg<T, S, F1, F2>(
    values: &PlPrimitiveArray<T>,
    width: usize,
    length: usize,
    slice_agg: F1,
    arr_agg: F2,
) -> PlPrimitiveArray<S>
where
    T: NumericNative,
    S: NumericNative,
    F1: Fn(&[T]) -> Option<S>,
    F2: Fn(&PlPrimitiveArray<T>) -> Option<S>,
{
    // A row of no values holds nothing to reduce, so every element reduces to nothing.
    if width == 0 {
        return PlPrimitiveArray::new_full_null(length);
    }

    // Without a null anywhere the rows are read straight out of the values buffer, in whichever
    // representation it is in.
    if !values.has_nulls() {
        return match values.scalar_value_ignore_validity() {
            // Every row is the same `width` copies of the one value, and so reduces to it — as
            // does the answer, which repeats a single value in turn.
            Some(value) => {
                let reduced =
                    slice_agg(&[value]).expect("a row of one value reduces to that value");
                PlPrimitiveArray::new_scalar(reduced, length)
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

fn agg_min<T>(values: &PlPrimitiveArray<T>, width: usize, length: usize) -> PlPrimitiveArray<T>
where
    T: NumericNative,
    PlPrimitiveArray<T>: for<'a> MinMaxKernel<Scalar<'a> = T>,
    [T]: for<'a> MinMaxKernel<Scalar<'a> = T>,
{
    array_agg(
        values,
        width,
        length,
        MinMaxKernel::min_ignore_nan_kernel,
        MinMaxKernel::min_ignore_nan_kernel,
    )
}

fn agg_max<T>(values: &PlPrimitiveArray<T>, width: usize, length: usize) -> PlPrimitiveArray<T>
where
    T: NumericNative,
    PlPrimitiveArray<T>: for<'a> MinMaxKernel<Scalar<'a> = T>,
    [T]: for<'a> MinMaxKernel<Scalar<'a> = T>,
{
    array_agg(
        values,
        width,
        length,
        MinMaxKernel::max_ignore_nan_kernel,
        MinMaxKernel::max_ignore_nan_kernel,
    )
}

/// Reduces every element of `ca` — which `values` holds the values of — to one element.
pub(super) fn array_dispatch(ca: &ArrayChunked, values: &Series, agg_type: AggType) -> Series {
    let width = ca.width();

    with_match_physical_numeric_polars_type!(values.dtype(), |$T| {
        let inner: &ChunkedArray<$T> = values.as_ref().as_ref().as_ref();

        // The values were taken chunk for chunk off `ca`, so every chunk holds the values of the
        // elements of the one beside it.
        let chunks = ca.downcast_iter().zip(inner.downcast_iter()).map(|(arr, values)| {
            let out = match agg_type {
                AggType::Min => agg_min(values, width, arr.len()),
                AggType::Max => agg_max(values, width, arr.len()),
            };

            // An element that is null holds no values to reduce, whatever the values under it
            // read as, so it reduces to nothing in turn.
            let validity = combine_validities_and(out.validity(), arr.validity());
            out.with_validity(validity)
        });

        ChunkedArray::<$T>::from_chunk_iter(ca.name().clone(), chunks).into_series()
    })
}
