use polars_core::datatypes::ArrayChunked;

use super::*;

pub(super) fn median_with_nulls(ca: &ArrayChunked) -> PolarsResult<Series> {
    let mut out = match ca.inner_dtype() {
        DataType::UInt32
        | DataType::UInt64
        | DataType::Int32
        | DataType::Int64
        | DataType::Float64 => {
            let out: Float64Chunked = ca
                .amortized_iter()
                .map(|s| s.and_then(|s| s.as_ref().median()))
                .collect();
            out.into_series()
        },
        DataType::Float32 => {
            let out: Float32Chunked = ca
                .amortized_iter()
                .map(|s| s.and_then(|s| s.as_ref().median().map(|v| v as f32)))
                .collect();
            out.into_series()
        },
        _ => {
            polars_bail!(ComputeError: "median on array with dtype: {} not yet supported", ca.dtype())
        },
    };
    out.rename(ca.name());
    Ok(out)
}

pub(super) fn std_with_nulls(ca: &ArrayChunked, ddof: u8) -> PolarsResult<Series> {
    let mut out = match ca.inner_dtype() {
        DataType::UInt32
        | DataType::UInt64
        | DataType::Int32
        | DataType::Int64
        | DataType::Float64 => {
            let out: Float64Chunked = ca
                .amortized_iter()
                .map(|s| s.and_then(|s| s.as_ref().std(ddof)))
                .collect();
            out.into_series()
        },
        DataType::Float32 => {
            let out: Float32Chunked = ca
                .amortized_iter()
                .map(|s| s.and_then(|s| s.as_ref().std(ddof).map(|v| v as f32)))
                .collect();
            out.into_series()
        },
        _ => {
            polars_bail!(ComputeError: "median on array with dtype: {} not yet supported", ca.dtype())
        },
    };
    out.rename(ca.name());
    Ok(out)
}

pub(super) fn var_with_nulls(ca: &ArrayChunked, ddof: u8) -> PolarsResult<Series> {
    let mut out = match ca.inner_dtype() {
        DataType::UInt32
        | DataType::UInt64
        | DataType::Int32
        | DataType::Int64
        | DataType::Float64 => {
            let out: Float64Chunked = ca
                .amortized_iter()
                .map(|s| s.and_then(|s| s.as_ref().var(ddof)))
                .collect();
            out.into_series()
        },
        DataType::Float32 => {
            let out: Float32Chunked = ca
                .amortized_iter()
                .map(|s| s.and_then(|s| s.as_ref().var(ddof).map(|v| v as f32)))
                .collect();
            out.into_series()
        },
        _ => {
            polars_bail!(ComputeError: "median on array with dtype: {} not yet supported", ca.dtype())
        },
    };
    out.rename(ca.name());
    Ok(out)
}
