use polars_arrow::array::{Array, FixedSizeListArray, PrimitiveArray};
use polars_arrow::compute::utils::combine_validities_and;
use polars_compute::min_max::MinMaxKernel;
use polars_core::prelude::*;
use polars_core::with_match_physical_numeric_polars_type;
use polars_utils::arg_min_max::ArgMinMax;

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
                // we can ensure that `start + width` will not go out of range
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

/// Reduces each row with `min_reduce`/`max_reduce`, for inner dtypes without a numeric kernel.
fn array_dispatch_generic(ca: &ArrayChunked, agg_type: AggType) -> PolarsResult<Series> {
    let values = ca
        .amortized_iter()
        .map(|opt_s| {
            let Some(s) = opt_s else {
                return Ok(AnyValue::Null);
            };
            let sc = match agg_type {
                AggType::Min => s.as_ref().min_reduce()?,
                AggType::Max => s.as_ref().max_reduce()?,
            };
            Ok(sc.into_value())
        })
        .collect::<PolarsResult<Vec<_>>>()?;
    Series::from_any_values_and_dtype(ca.name().clone(), &values, ca.inner_dtype(), true)
}

pub(super) fn array_dispatch(ca: &ArrayChunked, agg_type: AggType) -> PolarsResult<Series> {
    let dtype = ca.inner_dtype();
    // Categoricals order lexically, not by their physical representation.
    if dtype.is_categorical() || !dtype.to_physical().is_primitive_numeric() {
        return array_dispatch_generic(ca, agg_type);
    }

    let values = ca.get_inner().to_physical_repr().into_owned();
    let width = ca.width();
    let chunks: Vec<ArrayRef> = with_match_physical_numeric_polars_type!(values.dtype(), |$T| {
        let values: &ChunkedArray<$T> = values.as_ref().as_ref().as_ref();
        // The inner chunks line up with the outer ones.
        values.downcast_iter().zip(ca.downcast_iter()).map(|(arr, outer)| {
            let out = match agg_type {
                AggType::Min => agg_min(arr, width),
                AggType::Max => agg_max(arr, width),
            };
            // Null rows may store any values.
            let validity = combine_validities_and(out.validity(), outer.validity());
            Box::new(out.with_validity(validity)) as ArrayRef
        }).collect()
    });
    let out = Series::try_from((ca.name().clone(), chunks))?;
    // SAFETY: The aggregate of the physical values is a valid physical value of `dtype`.
    unsafe { out.from_physical_unchecked(dtype) }
}

fn arg_per_row<T>(arr: &FixedSizeListArray, width: usize, is_max: bool) -> PrimitiveArray<IdxSize>
where
    T: NumericNative,
    for<'a> &'a [T]: ArgMinMax,
{
    let values = arr
        .values()
        .as_any()
        .downcast_ref::<PrimitiveArray<T>>()
        .unwrap();
    let out: Vec<IdxSize> = values
        .values()
        .as_slice()
        .chunks_exact(width)
        .map(|row| (if is_max { row.argmax() } else { row.argmin() }) as IdxSize)
        .collect();
    PrimitiveArray::from_vec(out).with_validity(arr.validity().cloned())
}

/// The arg min or arg max of every array, or `None` if the inner dtype has no flat kernel.
pub(super) fn array_arg_min_max_flat(ca: &ArrayChunked, is_max: bool) -> Option<IdxCa> {
    let dt = ca.inner_dtype();
    let width = ca.width();
    if width == 0
        || width > IdxSize::MAX as usize
        || !dt.is_primitive_numeric()
        || super::namespace::has_inner_nulls(ca)
    {
        return None;
    }
    let chunks = ca.downcast_iter().map(|arr| {
        with_match_physical_numeric_polars_type!(dt, |$T| {
            arg_per_row::<<$T as PolarsNumericType>::Native>(arr, width, is_max)
        })
    });
    Some(IdxCa::from_chunk_iter(ca.name().clone(), chunks))
}
