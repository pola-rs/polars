use super::*;

impl Expr {
    /// Standard deviation of the values of the Series.
    pub fn std(self, ddof: u8) -> Self {
        AggExpr::Std(Box::new(self), ddof).into()
    }

    /// Variance of the values of the Series.
    pub fn var(self, ddof: u8) -> Self {
        AggExpr::Var(Box::new(self), ddof).into()
    }

    /// Reduce groups to minimal value.
    pub fn min(self) -> Self {
        AggExpr::Min {
            input: Box::new(self),
            propagate_nans: false,
        }
        .into()
    }

    /// Reduce groups to maximum value.
    pub fn max(self) -> Self {
        AggExpr::Max {
            input: Box::new(self),
            propagate_nans: false,
        }
        .into()
    }

    /// Reduce groups to minimal value.
    pub fn nan_min(self) -> Self {
        AggExpr::Min {
            input: Box::new(self),
            propagate_nans: true,
        }
        .into()
    }

    /// Reduce groups to maximum value.
    pub fn nan_max(self) -> Self {
        AggExpr::Max {
            input: Box::new(self),
            propagate_nans: true,
        }
        .into()
    }

    /// Reduce groups to the mean value.
    pub fn mean(self) -> Self {
        AggExpr::Mean(Box::new(self)).into()
    }

    /// Reduce groups to the median value.
    pub fn median(self) -> Self {
        AggExpr::Median(Box::new(self)).into()
    }

    /// Reduce groups to the sum of all the values.
    pub fn sum(self) -> Self {
        AggExpr::Sum(Box::new(self)).into()
    }

    /// Compute the histogram of a dataset.
    #[cfg(feature = "hist")]
    pub fn hist(self,
                bins: Option<Expr>,
                bin_count: Option<usize>,
                include_category: bool,
                include_breakpoint: bool,
    ) -> Self {
        let mut input = vec![self];
        if let Some(bins) = bins {
            input.push(bins)
        }

        Expr::Function {
            input,
            function: FunctionExpr::Hist{
                bin_count,
                include_category,
                include_breakpoint
            },
            options: FunctionOptions {
                collect_groups: ApplyOptions::GroupWise,
                ..Default::default()
            },
        }
    }
}
