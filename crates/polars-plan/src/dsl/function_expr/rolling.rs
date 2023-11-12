use super::*;

#[derive(Clone, PartialEq, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum RollingFunction {
    Min(RollingOptions),
    MinBy(RollingOptions),
    Max(RollingOptions),
    MaxBy(RollingOptions),
    Mean(RollingOptions),
    MeanBy(RollingOptions),
    Sum(RollingOptions),
    SumBy(RollingOptions),
    Median(RollingOptions),
    MedianBy(RollingOptions),
    Quantile(RollingOptions),
    QuantileBy(RollingOptions),
    Var(RollingOptions),
    VarBy(RollingOptions),
    Std(RollingOptions),
    StdBy(RollingOptions),
    #[cfg(feature = "moment")]
    Skew(usize, bool),
}

impl Display for RollingFunction {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        use RollingFunction::*;

        let name = match self {
            Min(_) => "rolling_min",
            MinBy(_) => "rolling_min_by",
            Max(_) => "rolling_max",
            MaxBy(_) => "rolling_max_by",
            Mean(_) => "rolling_mean",
            MeanBy(_) => "rolling_mean_by",
            Sum(_) => "rolling_sum",
            SumBy(_) => "rolling_sum_by",
            Median(_) => "rolling_median",
            MedianBy(_) => "rolling_median_by",
            Quantile(_) => "rolling_quantile",
            QuantileBy(_) => "rolling_quantile_by",
            Var(_) => "rolling_var",
            VarBy(_) => "rolling_var_by",
            Std(_) => "rolling_std",
            StdBy(_) => "rolling_std_by",
            #[cfg(feature = "moment")]
            Skew(..) => "rolling_skew",
        };

        write!(f, "{name}")
    }
}

impl Hash for RollingFunction {
    fn hash<H: Hasher>(&self, state: &mut H) {
        use RollingFunction::*;

        std::mem::discriminant(self).hash(state);
        match self {
            #[cfg(feature = "moment")]
            Skew(window_size, bias) => {
                window_size.hash(state);
                bias.hash(state)
            },
            _ => {},
        }
    }
}

fn convert<'a>(
    f: impl Fn(RollingOptionsImpl) -> PolarsResult<Series> + 'a,
    ss: &'a [Series],
    expr_name: &'static str,
) -> impl Fn(RollingOptions) -> PolarsResult<Series> + 'a {
    move |options| {
        let mut by = ss[1].clone();
        by = by.rechunk();

        polars_ensure!(
            options.weights.is_none(),
            ComputeError: "`weights` is not supported in 'rolling by' expression"
        );
        let (by, tz) = match by.dtype() {
            DataType::Datetime(tu, tz) => (by.cast(&DataType::Datetime(*tu, None))?, tz),
            DataType::Date => (
                by.cast(&DataType::Datetime(TimeUnit::Milliseconds, None))?,
                &None,
            ),
            dt => polars_bail!(opq = expr_name, got = dt, expected = "date/datetime"),
        };
        ensure_sorted_arg(&by, expr_name)?;
        let by = by.datetime().unwrap();
        let by_values = by.cont_slice().map_err(|_| {
            polars_err!(
                ComputeError:
                "`by` column should not have null values in 'rolling by' expression"
            )
        })?;
        let tu = by.time_unit();

        let options = RollingOptionsImpl {
            window_size: options.window_size,
            min_periods: options.min_periods,
            weights: None,
            center: options.center,
            by: Some(by_values),
            tu: Some(tu),
            tz: tz.as_ref(),
            closed_window: options.closed_window,
            fn_params: options.fn_params.clone(),
        };

        f(options)
    }
}

pub(super) fn rolling_min(s: &Series, options: RollingOptions) -> PolarsResult<Series> {
    s.rolling_min(options.clone().into())
}

pub(super) fn rolling_min_by(s: &[Series], options: RollingOptions) -> PolarsResult<Series> {
    convert(|options| s[0].rolling_min(options), s, "rolling_min")(options)
}

pub(super) fn rolling_max(s: &Series, options: RollingOptions) -> PolarsResult<Series> {
    s.rolling_max(options.clone().into())
}

pub(super) fn rolling_max_by(s: &[Series], options: RollingOptions) -> PolarsResult<Series> {
    convert(|options| s[0].rolling_max(options), s, "rolling_max")(options)
}

pub(super) fn rolling_mean(s: &Series, options: RollingOptions) -> PolarsResult<Series> {
    s.rolling_mean(options.clone().into())
}

pub(super) fn rolling_mean_by(s: &[Series], options: RollingOptions) -> PolarsResult<Series> {
    convert(|options| s[0].rolling_mean(options), s, "rolling_mean")(options)
}

pub(super) fn rolling_sum(s: &Series, options: RollingOptions) -> PolarsResult<Series> {
    s.rolling_sum(options.clone().into())
}

pub(super) fn rolling_sum_by(s: &[Series], options: RollingOptions) -> PolarsResult<Series> {
    convert(|options| s[0].rolling_sum(options), s, "rolling_sum")(options)
}

pub(super) fn rolling_median(s: &Series, options: RollingOptions) -> PolarsResult<Series> {
    s.rolling_median(options.clone().into())
}

pub(super) fn rolling_median_by(s: &[Series], options: RollingOptions) -> PolarsResult<Series> {
    convert(|options| s[0].rolling_median(options), s, "rolling_median")(options)
}

pub(super) fn rolling_quantile(s: &Series, options: RollingOptions) -> PolarsResult<Series> {
    s.rolling_quantile(options.clone().into())
}

pub(super) fn rolling_quantile_by(s: &[Series], options: RollingOptions) -> PolarsResult<Series> {
    convert(
        |options| s[0].rolling_quantile(options),
        s,
        "rolling_quantile",
    )(options)
}

pub(super) fn rolling_var(s: &Series, options: RollingOptions) -> PolarsResult<Series> {
    s.rolling_var(options.clone().into())
}

pub(super) fn rolling_var_by(s: &[Series], options: RollingOptions) -> PolarsResult<Series> {
    convert(|options| s[0].rolling_var(options), s, "rolling_var")(options)
}

pub(super) fn rolling_std(s: &Series, options: RollingOptions) -> PolarsResult<Series> {
    s.rolling_std(options.clone().into())
}

pub(super) fn rolling_std_by(s: &[Series], options: RollingOptions) -> PolarsResult<Series> {
    convert(|options| s[0].rolling_std(options), s, "rolling_std")(options)
}

#[cfg(feature = "moment")]
pub(super) fn rolling_skew(s: &Series, window_size: usize, bias: bool) -> PolarsResult<Series> {
    s.rolling_skew(window_size, bias)
}
