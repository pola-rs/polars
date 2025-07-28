use super::*;

/// Compute the covariance between two columns.
pub fn cov(a: Expr, b: Expr, ddof: u8) -> Expr {
    let function = FunctionExpr::Correlation {
        method: CorrelationMethod::Covariance(ddof),
    };
    a.map_binary(function, b)
}

/// Compute the pearson correlation between two columns.
pub fn pearson_corr(a: Expr, b: Expr) -> Expr {
    let function = FunctionExpr::Correlation {
        method: CorrelationMethod::Pearson,
    };
    a.map_binary(function, b)
}

/// Compute the spearman rank correlation between two columns.
/// Missing data will be excluded from the computation.
/// # Arguments
/// * propagate_nans
///   If `true` any `NaN` encountered will lead to `NaN` in the output.
///   If to `false` then `NaN` are regarded as larger than any finite number
///   and thus lead to the highest rank.
#[cfg(all(feature = "rank", feature = "propagate_nans"))]
pub fn spearman_rank_corr(a: Expr, b: Expr, propagate_nans: bool) -> Expr {
    let function = FunctionExpr::Correlation {
        method: CorrelationMethod::SpearmanRank(propagate_nans),
    };
    a.map_binary(function, b)
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
        function: FunctionExpr::RollingExpr {
            function: RollingFunction::CorrCov {
                corr_cov_options: options,
                is_corr,
            },
            options: rolling_options,
        },
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
