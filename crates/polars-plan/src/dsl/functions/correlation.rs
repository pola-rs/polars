use super::*;

/// Compute the covariance between two columns.
pub fn cov(a: Expr, b: Expr, ddof: u8) -> Expr {
    let input = vec![a, b];
    let function = FunctionExpr::Correlation {
        method: CorrelationMethod::Covariance,
        ddof,
    };
    Expr::Function {
        input,
        function,
        options: FunctionOptions {
            collect_groups: ApplyOptions::GroupWise,
            cast_to_supertypes: Some(Default::default()),
            flags: FunctionFlags::default() | FunctionFlags::RETURNS_SCALAR,
            ..Default::default()
        },
    }
}

/// Compute the pearson correlation between two columns.
///
/// # Arguments
/// * ddof
///     Delta degrees of freedom
pub fn pearson_corr(a: Expr, b: Expr, ddof: u8) -> Expr {
    let input = vec![a, b];
    let function = FunctionExpr::Correlation {
        method: CorrelationMethod::Pearson,
        ddof,
    };
    Expr::Function {
        input,
        function,
        options: FunctionOptions {
            collect_groups: ApplyOptions::GroupWise,
            cast_to_supertypes: Some(Default::default()),
            flags: FunctionFlags::default() | FunctionFlags::RETURNS_SCALAR,
            ..Default::default()
        },
    }
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
    let input = vec![a, b];
    let function = FunctionExpr::Correlation {
        method: CorrelationMethod::SpearmanRank(propagate_nans),
        ddof,
    };
    Expr::Function {
        input,
        function,
        options: FunctionOptions {
            collect_groups: ApplyOptions::GroupWise,
            cast_to_supertypes: Some(Default::default()),
            flags: FunctionFlags::default() | FunctionFlags::RETURNS_SCALAR,
            ..Default::default()
        },
    }
}

#[cfg(feature = "rolling_window")]
pub fn rolling_corr(x: Expr, y: Expr, options: RollingCovOptions) -> Expr {
    // see: https://github.com/pandas-dev/pandas/blob/v1.5.1/pandas/core/window/rolling.py#L1780-L1804
    let rolling_options = RollingOptionsFixedWindow {
        window_size: options.window_size as usize,
        min_periods: options.min_periods as usize,
        ..Default::default()
    };

    let non_null_mask = when(x.clone().is_not_null().and(y.clone().is_not_null()))
        .then(lit(1.0))
        .otherwise(lit(Null {}));

    let mean_x_y = (x.clone() * y.clone()).rolling_mean(rolling_options.clone());
    let mean_x = (x.clone() * non_null_mask.clone()).rolling_mean(rolling_options.clone());
    let mean_y = (y.clone() * non_null_mask.clone()).rolling_mean(rolling_options.clone());
    let var_x = (x.clone() * non_null_mask.clone()).rolling_var(rolling_options.clone());
    let var_y = (y.clone() * non_null_mask.clone()).rolling_var(rolling_options);

    let rolling_options_count = RollingOptionsFixedWindow {
        window_size: options.window_size as usize,
        min_periods: 0,
        ..Default::default()
    };
    let ddof = options.ddof as f64;
    let count_x_y = (x + y)
        .is_not_null()
        .cast(DataType::Float64)
        .rolling_sum(rolling_options_count);
    let numerator = (mean_x_y - mean_x * mean_y) * (count_x_y.clone() / (count_x_y - lit(ddof)));
    let denominator = (var_x * var_y).pow(lit(0.5));

    numerator / denominator
}

#[cfg(feature = "rolling_window")]
pub fn rolling_cov(x: Expr, y: Expr, options: RollingCovOptions) -> Expr {
    // see: https://github.com/pandas-dev/pandas/blob/91111fd99898d9dcaa6bf6bedb662db4108da6e6/pandas/core/window/rolling.py#L1700
    let rolling_options = RollingOptionsFixedWindow {
        window_size: options.window_size as usize,
        min_periods: options.min_periods as usize,
        ..Default::default()
    };

    let non_null_mask = when(x.clone().is_not_null().and(y.clone().is_not_null()))
        .then(lit(1.0))
        .otherwise(lit(Null {}));

    let mean_x_y = (x.clone() * y.clone()).rolling_mean(rolling_options.clone());
    let mean_x = (x.clone() * non_null_mask.clone()).rolling_mean(rolling_options.clone());
    let mean_y = (y.clone() * non_null_mask.clone()).rolling_mean(rolling_options);
    let rolling_options_count = RollingOptionsFixedWindow {
        window_size: options.window_size as usize,
        min_periods: 0,
        ..Default::default()
    };
    let count_x_y = (x + y)
        .is_not_null()
        .cast(DataType::Float64)
        .rolling_sum(rolling_options_count);

    let ddof = options.ddof as f64;

    (mean_x_y - mean_x * mean_y) * (count_x_y.clone() / (count_x_y - lit(ddof)))
}
