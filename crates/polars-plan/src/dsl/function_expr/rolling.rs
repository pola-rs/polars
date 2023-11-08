use super::*;

#[derive(Clone, PartialEq, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum RollingFunction {
    Min,
    MinBy,
    Max,
    MaxBy,
    Mean,
    MeanBy,
    Sum,
    SumBy,
    Median,
    MedianBy,
    Quantile,
    QuantileBy,
    Var,
    VarBy,
    Std,
    StdBy,
    #[cfg(feature = "moment")]
    Skew(usize, bool),
}

impl Display for RollingFunction {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        use RollingFunction::*;

        let name = match self {
            Min => "rolling_min",
            MinBy => "rolling_min_by",
            Max => "rolling_max",
            MaxBy => "rolling_max_by",
            Mean => "rolling_mean",
            MeanBy => "rolling_mean_by",
            Sum => "rolling_sum",
            SumBy => "rolling_sum_by",
            Median => "rolling_median",
            MedianBy => "rolling_median_by",
            Quantile => "rolling_quantile",
            QuantileBy => "rolling_quantile_by",
            Var => "rolling_var",
            VarBy => "rolling_var_by",
            Std => "rolling_std",
            StdBy => "rolling_std_by",
            Skew(..) => "rolling_skew",
        };

        write!(f, "{name}")
    }
}

macro_rules! convert {
    ($func:ident, $s:expr, $options:expr) => {{
        let mut by = $s[1].clone();
        by = by.rechunk();
        let s = &$s[0];

        polars_ensure!(
            $options.weights.is_none(),
            ComputeError: "`weights` is not supported in 'rolling by' expression"
        );
        let (by, tz) = match by.dtype() {
            DataType::Datetime(_, tz) => (
                by.cast(&DataType::Datetime(TimeUnit::Microseconds, None))?,
                tz,
            ),
            _ => (by.clone(), &None),
        };
        let by = by.datetime().unwrap();
        let by_values = by.cont_slice().map_err(|_| {
            polars_err!(
                ComputeError:
                "`by` column should not have null values in 'rolling by' expression"
            )
        })?;
        let tu = by.time_unit();

        let options = RollingOptionsImpl {
            window_size: $options.window_size,
            min_periods: $options.min_periods,
            weights: None,
            center: $options.center,
            by: Some(by_values),
            tu: Some(tu),
            tz: tz.as_ref(),
            closed_window: $options.closed_window,
            fn_params: $options.fn_params.clone(),
        };

        s.$func(options)
    }};
}

pub(super) fn rolling_min(s: &Series, options: RollingOptions) -> PolarsResult<Series> {
    s.rolling_min(options.clone().into())
}

pub(super) fn rolling_min_by(s: &[Series], options: RollingOptions) -> PolarsResult<Series> {
    convert!(rolling_min, s, options)
}

pub(super) fn rolling_max(s: &Series, options: RollingOptions) -> PolarsResult<Series> {
    s.rolling_max(options.clone().into())
}

pub(super) fn rolling_max_by(s: &[Series], options: RollingOptions) -> PolarsResult<Series> {
    convert!(rolling_max, s, options)
}

pub(super) fn rolling_mean(s: &Series, options: RollingOptions) -> PolarsResult<Series> {
    s.rolling_mean(options.clone().into())
}

pub(super) fn rolling_mean_by(s: &[Series], options: RollingOptions) -> PolarsResult<Series> {
    convert!(rolling_mean, s, options)
}

pub(super) fn rolling_sum(s: &Series, options: RollingOptions) -> PolarsResult<Series> {
    s.rolling_sum(options.clone().into())
}

pub(super) fn rolling_sum_by(s: &[Series], options: RollingOptions) -> PolarsResult<Series> {
    convert!(rolling_sum, s, options)
}

pub(super) fn rolling_median(s: &Series, options: RollingOptions) -> PolarsResult<Series> {
    s.rolling_median(options.clone().into())
}

pub(super) fn rolling_median_by(s: &[Series], options: RollingOptions) -> PolarsResult<Series> {
    convert!(rolling_median, s, options)
}

pub(super) fn rolling_quantile(s: &Series, options: RollingOptions) -> PolarsResult<Series> {
    s.rolling_quantile(options.clone().into())
}

pub(super) fn rolling_quantile_by(s: &[Series], options: RollingOptions) -> PolarsResult<Series> {
    convert!(rolling_quantile, s, options)
}

pub(super) fn rolling_var(s: &Series, options: RollingOptions) -> PolarsResult<Series> {
    s.rolling_var(options.clone().into())
}

pub(super) fn rolling_var_by(s: &[Series], options: RollingOptions) -> PolarsResult<Series> {
    convert!(rolling_var, s, options)
}

pub(super) fn rolling_std(s: &Series, options: RollingOptions) -> PolarsResult<Series> {
    s.rolling_std(options.clone().into())
}

pub(super) fn rolling_std_by(s: &[Series], options: RollingOptions) -> PolarsResult<Series> {
    convert!(rolling_std, s, options)
}

pub(super) fn rolling_skew(s: &Series, window_size: usize, bias: bool) -> PolarsResult<Series> {
    s.rolling_skew(window_size, bias)
}
