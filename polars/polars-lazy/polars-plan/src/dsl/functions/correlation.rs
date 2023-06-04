use super::*;

/// Compute the covariance between two columns.
pub fn cov(a: Expr, b: Expr) -> Expr {
    let name = "cov";
    let function = move |a: Series, b: Series| {
        let s = match a.dtype() {
            DataType::Float32 => {
                let ca_a = a.f32().unwrap();
                let ca_b = b.f32().unwrap();
                Series::new(name, &[polars_core::functions::cov_f(ca_a, ca_b)])
            }
            DataType::Float64 => {
                let ca_a = a.f64().unwrap();
                let ca_b = b.f64().unwrap();
                Series::new(name, &[polars_core::functions::cov_f(ca_a, ca_b)])
            }
            DataType::Int32 => {
                let ca_a = a.i32().unwrap();
                let ca_b = b.i32().unwrap();
                Series::new(name, &[polars_core::functions::cov_i(ca_a, ca_b)])
            }
            DataType::Int64 => {
                let ca_a = a.i64().unwrap();
                let ca_b = b.i64().unwrap();
                Series::new(name, &[polars_core::functions::cov_i(ca_a, ca_b)])
            }
            DataType::UInt32 => {
                let ca_a = a.u32().unwrap();
                let ca_b = b.u32().unwrap();
                Series::new(name, &[polars_core::functions::cov_i(ca_a, ca_b)])
            }
            DataType::UInt64 => {
                let ca_a = a.u64().unwrap();
                let ca_b = b.u64().unwrap();
                Series::new(name, &[polars_core::functions::cov_i(ca_a, ca_b)])
            }
            _ => {
                let a = a.cast(&DataType::Float64)?;
                let b = b.cast(&DataType::Float64)?;
                let ca_a = a.f64().unwrap();
                let ca_b = b.f64().unwrap();
                Series::new(name, &[polars_core::functions::cov_f(ca_a, ca_b)])
            }
        };
        Ok(Some(s))
    };
    apply_binary(
        a,
        b,
        function,
        GetOutput::map_dtype(|dt| {
            if matches!(dt, DataType::Float32) {
                DataType::Float32
            } else {
                DataType::Float64
            }
        }),
    )
    .with_function_options(|mut options| {
        options.auto_explode = true;
        options.fmt_str = "cov";
        options
    })
}

/// Compute the pearson correlation between two columns.
pub fn pearson_corr(a: Expr, b: Expr, ddof: u8) -> Expr {
    let name = "pearson_corr";
    let function = move |a: Series, b: Series| {
        let s = match a.dtype() {
            DataType::Float32 => {
                let ca_a = a.f32().unwrap();
                let ca_b = b.f32().unwrap();
                Series::new(
                    name,
                    &[polars_core::functions::pearson_corr_f(ca_a, ca_b, ddof)],
                )
            }
            DataType::Float64 => {
                let ca_a = a.f64().unwrap();
                let ca_b = b.f64().unwrap();
                Series::new(
                    name,
                    &[polars_core::functions::pearson_corr_f(ca_a, ca_b, ddof)],
                )
            }
            DataType::Int32 => {
                let ca_a = a.i32().unwrap();
                let ca_b = b.i32().unwrap();
                Series::new(
                    name,
                    &[polars_core::functions::pearson_corr_i(ca_a, ca_b, ddof)],
                )
            }
            DataType::Int64 => {
                let ca_a = a.i64().unwrap();
                let ca_b = b.i64().unwrap();
                Series::new(
                    name,
                    &[polars_core::functions::pearson_corr_i(ca_a, ca_b, ddof)],
                )
            }
            DataType::UInt32 => {
                let ca_a = a.u32().unwrap();
                let ca_b = b.u32().unwrap();
                Series::new(
                    name,
                    &[polars_core::functions::pearson_corr_i(ca_a, ca_b, ddof)],
                )
            }
            DataType::UInt64 => {
                let ca_a = a.u64().unwrap();
                let ca_b = b.u64().unwrap();
                Series::new(
                    name,
                    &[polars_core::functions::pearson_corr_i(ca_a, ca_b, ddof)],
                )
            }
            _ => {
                let a = a.cast(&DataType::Float64)?;
                let b = b.cast(&DataType::Float64)?;
                let ca_a = a.f64().unwrap();
                let ca_b = b.f64().unwrap();
                Series::new(
                    name,
                    &[polars_core::functions::pearson_corr_f(ca_a, ca_b, ddof)],
                )
            }
        };
        Ok(Some(s))
    };
    apply_binary(
        a,
        b,
        function,
        GetOutput::map_dtype(|dt| {
            if matches!(dt, DataType::Float32) {
                DataType::Float32
            } else {
                DataType::Float64
            }
        }),
    )
    .with_function_options(|mut options| {
        options.auto_explode = true;
        options.fmt_str = "pearson_corr";
        options
    })
}

/// Compute the spearman rank correlation between two columns.
/// Missing data will be excluded from the computation.
/// # Arguments
/// * ddof
///     Delta degrees of freedom
/// * propagate_nans
///     If `true` any `NaN` encountered will lead to `NaN` in the output.
///     If to `false` then `NaN` are regarded as larger than any finite number
///     and thus lead to the highest rank.
#[cfg(all(feature = "rank", feature = "propagate_nans"))]
pub fn spearman_rank_corr(a: Expr, b: Expr, ddof: u8, propagate_nans: bool) -> Expr {
    use polars_core::utils::coalesce_nulls_series;
    use polars_ops::prelude::nan_propagating_aggregate::nan_max_s;

    let function = move |a: Series, b: Series| {
        let (a, b) = coalesce_nulls_series(&a, &b);

        let name = "spearman_rank_correlation";
        if propagate_nans && a.dtype().is_float() {
            for s in [&a, &b] {
                if nan_max_s(s, "")
                    .get(0)
                    .unwrap()
                    .extract::<f64>()
                    .unwrap()
                    .is_nan()
                {
                    return Ok(Some(Series::new(name, &[f64::NAN])));
                }
            }
        }

        // drop nulls so that they are excluded
        let a = a.drop_nulls();
        let b = b.drop_nulls();

        let a_idx = a.rank(
            RankOptions {
                method: RankMethod::Min,
                ..Default::default()
            },
            None,
        );
        let b_idx = b.rank(
            RankOptions {
                method: RankMethod::Min,
                ..Default::default()
            },
            None,
        );
        let a_idx = a_idx.idx().unwrap();
        let b_idx = b_idx.idx().unwrap();

        Ok(Some(Series::new(
            name,
            &[polars_core::functions::pearson_corr_i(a_idx, b_idx, ddof)],
        )))
    };

    apply_binary(a, b, function, GetOutput::from_type(DataType::Float64)).with_function_options(
        |mut options| {
            options.auto_explode = true;
            options.fmt_str = "spearman_rank_correlation";
            options
        },
    )
}

#[cfg(feature = "rolling_window")]
pub fn rolling_corr(x: Expr, y: Expr, options: RollingCovOptions) -> Expr {
    let x = x.cache();
    let y = y.cache();
    // see: https://github.com/pandas-dev/pandas/blob/v1.5.1/pandas/core/window/rolling.py#L1780-L1804
    let rolling_options = RollingOptions {
        window_size: Duration::new(options.window_size as i64),
        min_periods: options.min_periods as usize,
        ..Default::default()
    };

    let mean_x_y = (x.clone() * y.clone()).rolling_mean(rolling_options.clone());
    let mean_x = x.clone().rolling_mean(rolling_options.clone());
    let mean_y = y.clone().rolling_mean(rolling_options.clone());
    let var_x = x.clone().rolling_var(rolling_options.clone());
    let var_y = y.clone().rolling_var(rolling_options);

    let rolling_options_count = RollingOptions {
        window_size: Duration::new(options.window_size as i64),
        min_periods: 0,
        ..Default::default()
    };
    let ddof = options.ddof as f64;
    let count_x_y = (x + y)
        .is_not_null()
        .cast(DataType::Float64)
        .rolling_sum(rolling_options_count)
        .cache();
    let numerator = (mean_x_y - mean_x * mean_y) * (count_x_y.clone() / (count_x_y - lit(ddof)));
    let denominator = (var_x * var_y).pow(lit(0.5));

    numerator / denominator
}

#[cfg(feature = "rolling_window")]
pub fn rolling_cov(x: Expr, y: Expr, options: RollingCovOptions) -> Expr {
    let x = x.cache();
    let y = y.cache();
    // see: https://github.com/pandas-dev/pandas/blob/91111fd99898d9dcaa6bf6bedb662db4108da6e6/pandas/core/window/rolling.py#L1700
    let rolling_options = RollingOptions {
        window_size: Duration::new(options.window_size as i64),
        min_periods: options.min_periods as usize,
        ..Default::default()
    };

    let mean_x_y = (x.clone() * y.clone()).rolling_mean(rolling_options.clone());
    let mean_x = x.clone().rolling_mean(rolling_options.clone());
    let mean_y = y.clone().rolling_mean(rolling_options);
    let rolling_options_count = RollingOptions {
        window_size: Duration::new(options.window_size as i64),
        min_periods: 0,
        ..Default::default()
    };
    let count_x_y = (x + y)
        .is_not_null()
        .cast(DataType::Float64)
        .rolling_sum(rolling_options_count)
        .cache();

    let ddof = options.ddof as f64;

    (mean_x_y - mean_x * mean_y) * (count_x_y.clone() / (count_x_y - lit(ddof)))
}
