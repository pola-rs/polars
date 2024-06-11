use super::*;

pub(super) fn median_with_nulls(ca: &ArrayChunked) -> PolarsResult<Series> {
    let mut out = match ca.inner_dtype() {
        DataType::Float32 => {
            let out: Float32Chunked = ca
                .apply_amortized_generic(|s| s.and_then(|s| s.as_ref().median().map(|v| v as f32)))
                .with_name(ca.name());
            out.into_series()
        },
        #[cfg(feature = "dtype-duration")]
        DataType::Duration(tu) => {
            let out: Int64Chunked = ca
                .apply_amortized_generic(|s| s.and_then(|s| s.as_ref().median().map(|v| v as i64)))
                .with_name(ca.name());
            out.into_duration(*tu).into_series()
        },
        _ => {
            let out: Float64Chunked = ca
                .apply_amortized_generic(|s| s.and_then(|s| s.as_ref().median()))
                .with_name(ca.name());
            out.into_series()
        },
    };
    out.rename(ca.name());
    Ok(out)
}

pub(super) fn std_with_nulls(ca: &ArrayChunked, ddof: u8) -> PolarsResult<Series> {
    let mut out = match ca.inner_dtype() {
        DataType::Float32 => {
            let out: Float32Chunked = ca
                .apply_amortized_generic(|s| s.and_then(|s| s.as_ref().std(ddof).map(|v| v as f32)))
                .with_name(ca.name());
            out.into_series()
        },
        #[cfg(feature = "dtype-duration")]
        DataType::Duration(tu) => {
            let out: Int64Chunked = ca
                .apply_amortized_generic(|s| s.and_then(|s| s.as_ref().std(ddof).map(|v| v as i64)))
                .with_name(ca.name());
            out.into_duration(*tu).into_series()
        },
        _ => {
            let out: Float64Chunked = {
                ca.amortized_iter()
                    .map(|s| s.and_then(|s| s.as_ref().std(ddof)))
                    .collect()
            };
            out.into_series()
        },
    };
    out.rename(ca.name());
    Ok(out)
}

pub(super) fn var_with_nulls(ca: &ArrayChunked, ddof: u8) -> PolarsResult<Series> {
    let mut out = match ca.inner_dtype() {
        DataType::Float32 => {
            let out: Float32Chunked = ca
                .apply_amortized_generic(|s| s.and_then(|s| s.as_ref().var(ddof).map(|v| v as f32)))
                .with_name(ca.name());
            out.into_series()
        },
        #[cfg(feature = "dtype-duration")]
        DataType::Duration(TimeUnit::Milliseconds) => {
            let out: Int64Chunked = ca
                .apply_amortized_generic(|s| s.and_then(|s| s.as_ref().var(ddof).map(|v| v as i64)))
                .with_name(ca.name());
            out.into_duration(TimeUnit::Milliseconds).into_series()
        },
        #[cfg(feature = "dtype-duration")]
        DataType::Duration(TimeUnit::Microseconds | TimeUnit::Nanoseconds) => {
            let out: Int64Chunked = ca
                .cast(&DataType::Array(
                    Box::new(DataType::Duration(TimeUnit::Milliseconds)),
                    ca.width(),
                ))
                .unwrap()
                .array()
                .unwrap()
                .apply_amortized_generic(|s| s.and_then(|s| s.as_ref().var(ddof).map(|v| v as i64)))
                .with_name(ca.name());
            out.into_duration(TimeUnit::Milliseconds).into_series()
        },
        _ => {
            let out: Float64Chunked = ca
                .apply_amortized_generic(|s| s.and_then(|s| s.as_ref().var(ddof)))
                .with_name(ca.name());
            out.into_series()
        },
    };
    out.rename(ca.name());
    Ok(out)
}
