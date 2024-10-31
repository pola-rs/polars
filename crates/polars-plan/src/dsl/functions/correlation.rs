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

#[cfg(all(feature = "rolling_window", feature = "cov"))]
fn dispatch_corr_cov(x: Expr, y: Expr, options: RollingCovOptions, is_corr: bool) -> Expr {
    // see: https://github.com/pandas-dev/pandas/blob/v1.5.1/pandas/core/window/rolling.py#L1780-L1804
    let rolling_options = RollingOptionsFixedWindow {
        window_size: options.window_size as usize,
        min_periods: options.min_periods as usize,
        ..Default::default()
    };

    Expr::Function {
        input: vec![x, y],
        function: FunctionExpr::RollingExpr(RollingFunction::CorrCov {
            rolling_options,
            corr_cov_options: options,
            is_corr,
        }),
        options: Default::default(),
    }
}

#[cfg(all(feature = "rolling_window", feature = "cov"))]
pub fn rolling_corr(x: Expr, y: Expr, options: RollingCovOptions) -> Expr {
    dispatch_corr_cov(x, y, options, true)
}

#[cfg(all(feature = "rolling_window", feature = "cov"))]
pub fn rolling_cov(x: Expr, y: Expr, options: RollingCovOptions) -> Expr {
    dispatch_corr_cov(x, y, options, false)
}
