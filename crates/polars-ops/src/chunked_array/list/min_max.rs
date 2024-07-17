use arrow::array::{Array, PrimitiveArray};
use arrow::bitmap::Bitmap;
use arrow::compute::utils::combine_validities_and;
use arrow::types::NativeType;
use polars_compute::min_max::MinMaxKernel;
use polars_core::prelude::*;
use polars_core::with_match_physical_numeric_polars_type;

use crate::chunked_array::list::namespace::has_inner_nulls;

fn min_between_offsets<T>(values: &[T], offset: &[i64]) -> PrimitiveArray<T>
where
    T: NativeType,
    [T]: for<'a> MinMaxKernel<Scalar<'a> = T>,
{
    let mut running_offset = offset[0];

    (offset[1..])
        .iter()
        .map(|end| {
            let current_offset = running_offset;
            running_offset = *end;
            if current_offset == *end {
                return None;
            }

            let slice = unsafe { values.get_unchecked(current_offset as usize..*end as usize) };
            slice.min_ignore_nan_kernel()
        })
        .collect()
}

fn dispatch_min<T>(arr: &dyn Array, offsets: &[i64], validity: Option<&Bitmap>) -> ArrayRef
where
    T: NativeType,
    [T]: for<'a> MinMaxKernel<Scalar<'a> = T>,
{
    let values = arr.as_any().downcast_ref::<PrimitiveArray<T>>().unwrap();
    let values = values.values().as_slice();
    let out = min_between_offsets(values, offsets);
    let new_validity = combine_validities_and(out.validity(), validity);
    out.with_validity(new_validity).to_boxed()
}

fn min_list_numerical(ca: &ListChunked, inner_type: &DataType) -> Series {
    use DataType::*;
    let chunks = ca
        .downcast_iter()
        .map(|arr| {
            let offsets = arr.offsets().as_slice();
            let values = arr.values().as_ref();

            match inner_type {
                Int8 => dispatch_min::<i8>(values, offsets, arr.validity()),
                Int16 => dispatch_min::<i16>(values, offsets, arr.validity()),
                Int32 => dispatch_min::<i32>(values, offsets, arr.validity()),
                Int64 => dispatch_min::<i64>(values, offsets, arr.validity()),
                UInt8 => dispatch_min::<u8>(values, offsets, arr.validity()),
                UInt16 => dispatch_min::<u16>(values, offsets, arr.validity()),
                UInt32 => dispatch_min::<u32>(values, offsets, arr.validity()),
                UInt64 => dispatch_min::<u64>(values, offsets, arr.validity()),
                Float32 => dispatch_min::<f32>(values, offsets, arr.validity()),
                Float64 => dispatch_min::<f64>(values, offsets, arr.validity()),
                _ => unimplemented!(),
            }
        })
        .collect::<Vec<_>>();

    Series::try_from((ca.name(), chunks)).unwrap()
}

pub(super) fn list_min_function(ca: &ListChunked) -> PolarsResult<Series> {
    fn inner(ca: &ListChunked) -> PolarsResult<Series> {
        match ca.inner_dtype() {
            DataType::Boolean => {
                let out: BooleanChunked = ca
                    .apply_amortized_generic(|s| s.and_then(|s| s.as_ref().bool().unwrap().min()));
                Ok(out.into_series())
            },
            dt if dt.is_numeric() => {
                with_match_physical_numeric_polars_type!(dt, |$T| {

                    let out: ChunkedArray<$T> = ca.apply_amortized_generic(|opt_s| {
                            let s = opt_s?;
                            let ca: &ChunkedArray<$T> = s.as_ref().as_ref().as_ref();
                            ca.min()
                    });
                    Ok(out.into_series())
                })
            },
            _ => Ok(ca
                .try_apply_amortized(|s| {
                    let s = s.as_ref();
                    let sc = s.min_reduce()?;
                    Ok(sc.into_series(s.name()))
                })?
                .explode()
                .unwrap()
                .into_series()),
        }
    }

    if has_inner_nulls(ca) {
        return inner(ca);
    };

    match ca.inner_dtype() {
        dt if dt.is_numeric() => Ok(min_list_numerical(ca, dt)),
        _ => inner(ca),
    }
}

fn max_between_offsets<T>(values: &[T], offset: &[i64]) -> PrimitiveArray<T>
where
    T: NativeType,
    [T]: for<'a> MinMaxKernel<Scalar<'a> = T>,
{
    let mut running_offset = offset[0];

    (offset[1..])
        .iter()
        .map(|end| {
            let current_offset = running_offset;
            running_offset = *end;
            if current_offset == *end {
                return None;
            }

            let slice = unsafe { values.get_unchecked(current_offset as usize..*end as usize) };
            slice.max_ignore_nan_kernel()
        })
        .collect()
}

fn dispatch_max<T>(arr: &dyn Array, offsets: &[i64], validity: Option<&Bitmap>) -> ArrayRef
where
    T: NativeType,
    [T]: for<'a> MinMaxKernel<Scalar<'a> = T>,
{
    let values = arr.as_any().downcast_ref::<PrimitiveArray<T>>().unwrap();
    let values = values.values().as_slice();
    let mut out = max_between_offsets(values, offsets);

    if let Some(validity) = validity {
        if out.null_count() > 0 {
            out.apply_validity(|other_validity| validity & &other_validity)
        } else {
            out = out.with_validity(Some(validity.clone()));
        }
    }
    Box::new(out)
}

fn max_list_numerical(ca: &ListChunked, inner_type: &DataType) -> Series {
    use DataType::*;
    let chunks = ca
        .downcast_iter()
        .map(|arr| {
            let offsets = arr.offsets().as_slice();
            let values = arr.values().as_ref();

            match inner_type {
                Int8 => dispatch_max::<i8>(values, offsets, arr.validity()),
                Int16 => dispatch_max::<i16>(values, offsets, arr.validity()),
                Int32 => dispatch_max::<i32>(values, offsets, arr.validity()),
                Int64 => dispatch_max::<i64>(values, offsets, arr.validity()),
                UInt8 => dispatch_max::<u8>(values, offsets, arr.validity()),
                UInt16 => dispatch_max::<u16>(values, offsets, arr.validity()),
                UInt32 => dispatch_max::<u32>(values, offsets, arr.validity()),
                UInt64 => dispatch_max::<u64>(values, offsets, arr.validity()),
                Float32 => dispatch_max::<f32>(values, offsets, arr.validity()),
                Float64 => dispatch_max::<f64>(values, offsets, arr.validity()),
                _ => unimplemented!(),
            }
        })
        .collect::<Vec<_>>();

    Series::try_from((ca.name(), chunks)).unwrap()
}

pub(super) fn list_max_function(ca: &ListChunked) -> PolarsResult<Series> {
    fn inner(ca: &ListChunked) -> PolarsResult<Series> {
        match ca.inner_dtype() {
            DataType::Boolean => {
                let out: BooleanChunked = ca
                    .apply_amortized_generic(|s| s.and_then(|s| s.as_ref().bool().unwrap().max()));
                Ok(out.into_series())
            },
            dt if dt.is_numeric() => {
                with_match_physical_numeric_polars_type!(dt, |$T| {

                    let out: ChunkedArray<$T> = ca.apply_amortized_generic(|opt_s| {
                            let s = opt_s?;
                            let ca: &ChunkedArray<$T> = s.as_ref().as_ref().as_ref();
                            ca.max()
                    });
                    Ok(out.into_series())

                })
            },
            _ => Ok(ca
                .try_apply_amortized(|s| {
                    let s = s.as_ref();
                    let sc = s.max_reduce()?;
                    Ok(sc.into_series(s.name()))
                })?
                .explode()
                .unwrap()
                .into_series()),
        }
    }

    if has_inner_nulls(ca) {
        return inner(ca);
    };

    match ca.inner_dtype() {
        dt if dt.is_numeric() => Ok(max_list_numerical(ca, dt)),
        _ => inner(ca),
    }
}
