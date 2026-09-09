use polars_array::bitmap::combine_validities_and;
use polars_compute::min_max::MinMaxKernel;
use polars_core::prelude::*;
use polars_core::with_match_physical_numeric_polars_type;

/// Reduces every element of `arr` — whose values `values` holds — to one element.
fn array_agg<T, S, F1, F2>(
    arr: &PlFixedSizeListArray,
    values: &PlPrimitiveArray<T>,
    slice_agg: F1,
    arr_agg: F2,
) -> PlPrimitiveArray<S>
where
    T: NumericNative,
    S: NumericNative,
    F1: Fn(&[T]) -> Option<S>,
    F2: Fn(&PlPrimitiveArray<T>) -> Option<S>,
{
    let (width, length) = (arr.width(), arr.len());

    // A row of no values holds nothing to reduce, so every element reduces to nothing.
    if width == 0 {
        return PlPrimitiveArray::new_full_null(length);
    }

    // The values hold the one list every element reads, so they all reduce to the same element:
    // that list is reduced once and the answer repeats it, rather than the list being read —
    // and reduced — once per element.
    if arr.values_are_scalar() {
        return match arr_agg(values) {
            Some(value) => PlPrimitiveArray::new_scalar(value, length),
            // The one list every element reads reduces to nothing, and so does every element.
            None => PlPrimitiveArray::new_full_null(length),
        };
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

fn agg_min<T>(arr: &PlFixedSizeListArray, values: &PlPrimitiveArray<T>) -> PlPrimitiveArray<T>
where
    T: NumericNative,
    PlPrimitiveArray<T>: for<'a> MinMaxKernel<Scalar<'a> = T>,
    [T]: for<'a> MinMaxKernel<Scalar<'a> = T>,
{
    array_agg(
        arr,
        values,
        MinMaxKernel::min_ignore_nan_kernel,
        MinMaxKernel::min_ignore_nan_kernel,
    )
}

fn agg_max<T>(arr: &PlFixedSizeListArray, values: &PlPrimitiveArray<T>) -> PlPrimitiveArray<T>
where
    T: NumericNative,
    PlPrimitiveArray<T>: for<'a> MinMaxKernel<Scalar<'a> = T>,
    [T]: for<'a> MinMaxKernel<Scalar<'a> = T>,
{
    array_agg(
        arr,
        values,
        MinMaxKernel::max_ignore_nan_kernel,
        MinMaxKernel::max_ignore_nan_kernel,
    )
}

/// Reduces every element of `ca` to one element.
pub(super) fn array_dispatch(ca: &ArrayChunked, agg_type: AggType) -> Series {
    with_match_physical_numeric_polars_type!(ca.inner_dtype(), |$T| {
        type N = <$T as PolarsNumericType>::Native;

        let chunks = ca.downcast_iter().map(|arr| {
            // The values a chunk holds, in the representation it holds them in: reading them off
            // the chunk rather than off a flattened copy of it is what leaves a chunk that
            // repeats a single list holding that list once.
            let values = arr
                .values()
                .as_any()
                .downcast_ref::<PlPrimitiveArray<N>>()
                .unwrap();

            let out = match agg_type {
                AggType::Min => agg_min(arr, values),
                AggType::Max => agg_max(arr, values),
            };

            // An element that is null holds no values to reduce, whatever the values under it
            // read as, so it reduces to nothing in turn.
            let validity = combine_validities_and(out.validity(), arr.validity());
            out.with_validity(validity)
        });

        ChunkedArray::<$T>::from_chunk_iter(ca.name().clone(), chunks).into_series()
    })
}
