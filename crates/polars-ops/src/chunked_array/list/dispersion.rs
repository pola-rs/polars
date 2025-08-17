use arrow::temporal_conversions::MICROSECONDS_IN_DAY as US_IN_DAY;

use super::*;

pub(super) fn median_with_nulls(ca: &ListChunked) -> Series {
    match ca.inner_dtype() {
        DataType::Float32 => {
            let out: Float32Chunked = ca
                .apply_amortized_generic(|s| s.and_then(|s| s.as_ref().median().map(|v| v as f32)))
                .with_name(ca.name().clone());
            out.into_series()
        },
        #[cfg(feature = "dtype-datetime")]
        DataType::Date => {
            let out: Int64Chunked = ca
                .apply_amortized_generic(|s| {
                    s.and_then(|s| s.as_ref().median().map(|v| (v * (US_IN_DAY as f64)) as i64))
                })
                .with_name(ca.name().clone());
            out.into_datetime(TimeUnit::Microseconds, None)
                .into_series()
        },
        dt if dt.is_temporal() => {
            let out: Int64Chunked = ca
                .apply_amortized_generic(|s| s.and_then(|s| s.as_ref().median().map(|v| v as i64)))
                .with_name(ca.name().clone());
            out.cast(dt).unwrap()
        },
        _ => {
            let out: Float64Chunked = ca
                .apply_amortized_generic(|s| s.and_then(|s| s.as_ref().median()))
                .with_name(ca.name().clone());
            out.into_series()
        },
    }
}

pub(super) fn std_with_nulls(ca: &ListChunked, ddof: u8) -> Series {
    match ca.inner_dtype() {
        DataType::Float32 => {
            let out: Float32Chunked = ca
                .apply_amortized_generic(|s| s.and_then(|s| s.as_ref().std(ddof).map(|v| v as f32)))
                .with_name(ca.name().clone());
            out.into_series()
        },
        #[cfg(feature = "dtype-duration")]
        DataType::Duration(tu) => {
            let out: Int64Chunked = ca
                .apply_amortized_generic(|s| s.and_then(|s| s.as_ref().std(ddof).map(|v| v as i64)))
                .with_name(ca.name().clone());
            out.into_duration(*tu).into_series()
        },
        _ => {
            let out: Float64Chunked = ca
                .apply_amortized_generic(|s| s.and_then(|s| s.as_ref().std(ddof)))
                .with_name(ca.name().clone());
            out.into_series()
        },
    }
}

pub(super) fn var_with_nulls(ca: &ListChunked, ddof: u8) -> PolarsResult<Series> {
    match ca.inner_dtype() {
        DataType::Float32 => {
            let out: Float32Chunked = ca
                .apply_amortized_generic(|s| s.and_then(|s| s.as_ref().var(ddof).map(|v| v as f32)))
                .with_name(ca.name().clone());
            Ok(out.into_series())
        },
        dt if dt.is_temporal() => {
            polars_bail!(InvalidOperation: "variance of type {dt} is not supported")
        },
        _ => {
            let out: Float64Chunked = ca
                .apply_amortized_generic(|s| s.and_then(|s| s.as_ref().var(ddof)))
                .with_name(ca.name().clone());
            Ok(out.into_series())
        },
    }
}
