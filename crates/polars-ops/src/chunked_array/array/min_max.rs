use arrow::array::{Array, PrimitiveArray};
use arrow::compute::aggregate::SimdOrd;
use arrow::legacy::prelude::FromData;
use arrow::legacy::slice::ExtremaNanAware;
use arrow::types::simd::Simd;
use arrow::types::NativeType;
use polars_core::prelude::*;
use polars_core::with_match_physical_numeric_polars_type;

fn array_agg<T, S, F1, F2>(
    values: &PrimitiveArray<T>,
    width: usize,
    slice_agg: F1,
    arr_agg: F2,
) -> PrimitiveArray<S>
where
    T: NativeType,
    S: NativeType,
    F1: Fn(&[T]) -> S,
    F2: Fn(PrimitiveArray<T>) -> Option<S>,
{
    if values.null_count() == 0 {
        let values = values.values().as_slice();
        let agg = values
            .chunks_exact(width)
            .map(slice_agg)
            .collect::<Vec<_>>();
        PrimitiveArray::from_data_default(agg.into(), None)
    } else {
        (0..values.len())
            .step_by(width)
            .map(|start| {
                let sliced = unsafe { values.clone().sliced_unchecked(start, start + width) };
                arr_agg(sliced)
            })
            .collect()
    }
}

pub(super) enum AggType {
    Min,
    Max,
}

fn agg_min<T>(values: &PrimitiveArray<T>, width: usize) -> PrimitiveArray<T>
where
    T: NativeType + PartialOrd + IsFloat + NativeType + Simd,
    T::Simd: SimdOrd<T>,
{
    array_agg(
        values,
        width,
        |v| *v.min_value_nan_aware().unwrap(),
        |arr| arrow::compute::aggregate::min_primitive(&arr),
    )
}

fn agg_max<T>(values: &PrimitiveArray<T>, width: usize) -> PrimitiveArray<T>
where
    T: NativeType + PartialOrd + IsFloat + NativeType + Simd,
    T::Simd: SimdOrd<T>,
{
    array_agg(
        values,
        width,
        |v| *v.max_value_nan_aware().unwrap(),
        |arr| arrow::compute::aggregate::max_primitive(&arr),
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
