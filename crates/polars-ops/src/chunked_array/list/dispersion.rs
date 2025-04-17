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
            const MS_IN_DAY: i64 = 86_400_000;
            let out: Int64Chunked = ca
                .apply_amortized_generic(|s| {
                    s.and_then(|s| s.as_ref().median().map(|v| (v * (MS_IN_DAY as f64)) as i64))
                })
                .with_name(ca.name().clone());
            out.into_datetime(TimeUnit::Milliseconds, None)
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

pub(super) fn quantile_with_nulls(
    ca: &ListChunked,
    quantile: f64,
    method: QuantileMethod,
) -> Series {
    match ca.inner_dtype() {
        DataType::Float32 => {
            let out: Float32Chunked = ca
                .apply_amortized_generic(|s| {
                    s.and_then(|s| {
                        s.as_ref()
                            .quantile(quantile, method)
                            .unwrap_or(Some(f64::NAN))
                            .map(|v| v as f32)
                    })
                })
                .with_name(ca.name().clone());
            out.into_series()
        },
        #[cfg(feature = "dtype-duration")]
        DataType::Duration(tu) => {
            let out: Int64Chunked = ca
                .apply_amortized_generic(|s| {
                    s.and_then(|s| {
                        s.as_ref()
                            .quantile(quantile, method)
                            .unwrap_or(Some(f64::NAN))
                            .map(|v| v as i64)
                    })
                })
                .with_name(ca.name().clone());
            out.into_duration(*tu).into_series()
        },
        _ => {
            let out: Float64Chunked = ca
                .apply_amortized_generic(|s| {
                    s.and_then(|s| {
                        s.as_ref()
                            .quantile(quantile, method)
                            .unwrap_or(Some(f64::NAN))
                    })
                })
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

pub(super) fn var_with_nulls(ca: &ListChunked, ddof: u8) -> Series {
    match ca.inner_dtype() {
        DataType::Float32 => {
            let out: Float32Chunked = ca
                .apply_amortized_generic(|s| s.and_then(|s| s.as_ref().var(ddof).map(|v| v as f32)))
                .with_name(ca.name().clone());
            out.into_series()
        },
        #[cfg(feature = "dtype-duration")]
        DataType::Duration(TimeUnit::Milliseconds) => {
            let out: Int64Chunked = ca
                .apply_amortized_generic(|s| s.and_then(|s| s.as_ref().var(ddof).map(|v| v as i64)))
                .with_name(ca.name().clone());
            out.into_duration(TimeUnit::Milliseconds).into_series()
        },
        #[cfg(feature = "dtype-duration")]
        DataType::Duration(TimeUnit::Microseconds | TimeUnit::Nanoseconds) => {
            let out: Int64Chunked = ca
                .cast(&DataType::List(Box::new(DataType::Duration(
                    TimeUnit::Milliseconds,
                ))))
                .unwrap()
                .list()
                .unwrap()
                .apply_amortized_generic(|s| s.and_then(|s| s.as_ref().var(ddof).map(|v| v as i64)))
                .with_name(ca.name().clone());
            out.into_duration(TimeUnit::Milliseconds).into_series()
        },
        _ => {
            let out: Float64Chunked = ca
                .apply_amortized_generic(|s| s.and_then(|s| s.as_ref().var(ddof)))
                .with_name(ca.name().clone());
            out.into_series()
        },
    }
}
