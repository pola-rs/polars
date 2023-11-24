use arrow::array::{Array, PrimitiveArray, ArrayRef};
use arrow::bitmap::Bitmap;
use arrow::legacy::array::PolarsArray;
use arrow::legacy::data_types::{IsFloat};
use arrow::legacy::slice::ExtremaNanAware;
use arrow::legacy::utils::CustomIterTools;
use arrow::types::NativeType;
use polars_core::prelude::*;
use polars_core::with_match_physical_numeric_polars_type;

use crate::chunked_array::list::namespace::has_inner_nulls;

fn min_between_offsets<T>(values: &[T], offset: &[i64]) -> PrimitiveArray<T>
where
    T: NativeType + PartialOrd + IsFloat,
{
    let mut running_offset = offset[0];

    (offset[1..])
        .iter()
        .map(|end| {
            let current_offset = running_offset;
            running_offset = *end;

            let slice = unsafe { values.get_unchecked(current_offset as usize..*end as usize) };
            slice.min_value_nan_aware().copied()
        })
        .collect_trusted()
}

fn dispatch_min<T>(arr: &dyn Array, offsets: &[i64], validity: Option<&Bitmap>) -> ArrayRef
where
    T: NativeType + PartialOrd + IsFloat,
{
    let values = arr.as_any().downcast_ref::<PrimitiveArray<T>>().unwrap();
    let values = values.values().as_slice();
    let mut out = min_between_offsets(values, offsets);

    if let Some(validity) = validity {
        if out.has_validity() {
            out.apply_validity(|other_validity| validity & &other_validity)
        } else {
            out = out.with_validity(Some(validity.clone()));
        }
    }
    Box::new(out)
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

pub(super) fn list_min_function(ca: &ListChunked) -> Series {
    fn inner(ca: &ListChunked) -> Series {
        match ca.inner_dtype() {
            DataType::Boolean => {
                let out: BooleanChunked = ca
                    .apply_amortized_generic(|s| s.and_then(|s| s.as_ref().bool().unwrap().min()));
                out.into_series()
            },
            dt if dt.is_numeric() => {
                with_match_physical_numeric_polars_type!(dt, |$T| {

                    let out: ChunkedArray<$T> = ca.apply_amortized_generic(|opt_s| {
                            let s = opt_s?;
                            let ca: &ChunkedArray<$T> = s.as_ref().as_ref().as_ref();
                            ca.min()
                    });
                    out.into_series()
                })
            },
            _ => ca
                .apply_amortized(|s| s.as_ref().min_as_series())
                .explode()
                .unwrap()
                .into_series(),
        }
    }

    if has_inner_nulls(ca) {
        return inner(ca);
    };

    match ca.inner_dtype() {
        dt if dt.is_numeric() => min_list_numerical(ca, &dt),
        _ => inner(ca),
    }
}

fn max_between_offsets<T>(values: &[T], offset: &[i64]) -> PrimitiveArray<T>
where
    T: NativeType + PartialOrd + IsFloat,
{
    let mut running_offset = offset[0];

    (offset[1..])
        .iter()
        .map(|end| {
            let current_offset = running_offset;
            running_offset = *end;

            let slice = unsafe { values.get_unchecked(current_offset as usize..*end as usize) };
            slice.max_value_nan_aware().copied()
        })
        .collect_trusted()
}

fn dispatch_max<T>(arr: &dyn Array, offsets: &[i64], validity: Option<&Bitmap>) -> ArrayRef
where
    T: NativeType + PartialOrd + IsFloat,
{
    let values = arr.as_any().downcast_ref::<PrimitiveArray<T>>().unwrap();
    let values = values.values().as_slice();
    let mut out = max_between_offsets(values, offsets);

    if let Some(validity) = validity {
        if out.has_validity() {
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

pub(super) fn list_max_function(ca: &ListChunked) -> Series {
    fn inner(ca: &ListChunked) -> Series {
        match ca.inner_dtype() {
            DataType::Boolean => {
                let out: BooleanChunked = ca
                    .apply_amortized_generic(|s| s.and_then(|s| s.as_ref().bool().unwrap().max()));
                out.into_series()
            },
            dt if dt.is_numeric() => {
                with_match_physical_numeric_polars_type!(dt, |$T| {

                    let out: ChunkedArray<$T> = ca.apply_amortized_generic(|opt_s| {
                            let s = opt_s?;
                            let ca: &ChunkedArray<$T> = s.as_ref().as_ref().as_ref();
                            ca.max()
                    });
                    out.into_series()

                })
            },
            _ => ca
                .apply_amortized(|s| s.as_ref().max_as_series())
                .explode()
                .unwrap()
                .into_series(),
        }
    }

    if has_inner_nulls(ca) {
        return inner(ca);
    };

    match ca.inner_dtype() {
        dt if dt.is_numeric() => max_list_numerical(ca, &dt),
        _ => inner(ca),
    }
}
