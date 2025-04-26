#[cfg(feature = "cov")]
use std::ops::BitAnd;

use polars_core::utils::Container;
use polars_time::chunkedarray::*;

use super::*;
#[cfg(feature = "cov")]
use crate::dsl::pow::pow;

#[derive(Clone, PartialEq, Debug, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum RollingFunction {
    Min(RollingOptionsFixedWindow),
    Max(RollingOptionsFixedWindow),
    Mean(RollingOptionsFixedWindow),
    Sum(RollingOptionsFixedWindow),
    Quantile(RollingOptionsFixedWindow),
    Var(RollingOptionsFixedWindow),
    Std(RollingOptionsFixedWindow),
    #[cfg(feature = "moment")]
    Skew(RollingOptionsFixedWindow),
    #[cfg(feature = "moment")]
    Kurtosis(RollingOptionsFixedWindow),
    #[cfg(feature = "cov")]
    CorrCov {
        rolling_options: RollingOptionsFixedWindow,
        corr_cov_options: RollingCovOptions,
        // Whether is Corr or Cov
        is_corr: bool,
    },
}

impl Display for RollingFunction {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        use RollingFunction::*;

        let name = match self {
            Min(_) => "min",
            Max(_) => "max",
            Mean(_) => "mean",
            Sum(_) => "rsum",
            Quantile(_) => "quantile",
            Var(_) => "var",
            Std(_) => "std",
            #[cfg(feature = "moment")]
            Skew(..) => "skew",
            #[cfg(feature = "moment")]
            Kurtosis(..) => "kurtosis",
            #[cfg(feature = "cov")]
            CorrCov { is_corr, .. } => {
                if *is_corr {
                    "corr"
                } else {
                    "cov"
                }
            },
        };

        write!(f, "rolling_{name}")
    }
}

pub(super) fn rolling_min(s: &Column, options: RollingOptionsFixedWindow) -> PolarsResult<Column> {
    // @scalar-opt
    s.as_materialized_series()
        .rolling_min(options)
        .map(Column::from)
}

pub(super) fn rolling_max(s: &Column, options: RollingOptionsFixedWindow) -> PolarsResult<Column> {
    // @scalar-opt
    s.as_materialized_series()
        .rolling_max(options)
        .map(Column::from)
}

pub(super) fn rolling_mean(s: &Column, options: RollingOptionsFixedWindow) -> PolarsResult<Column> {
    // @scalar-opt
    s.as_materialized_series()
        .rolling_mean(options)
        .map(Column::from)
}

pub(super) fn rolling_sum(s: &Column, options: RollingOptionsFixedWindow) -> PolarsResult<Column> {
    // @scalar-opt
    s.as_materialized_series()
        .rolling_sum(options)
        .map(Column::from)
}

pub(super) fn rolling_quantile(
    s: &Column,
    options: RollingOptionsFixedWindow,
) -> PolarsResult<Column> {
    // @scalar-opt
    s.as_materialized_series()
        .rolling_quantile(options)
        .map(Column::from)
}

pub(super) fn rolling_var(s: &Column, options: RollingOptionsFixedWindow) -> PolarsResult<Column> {
    // @scalar-opt
    s.as_materialized_series()
        .rolling_var(options)
        .map(Column::from)
}

pub(super) fn rolling_std(s: &Column, options: RollingOptionsFixedWindow) -> PolarsResult<Column> {
    // @scalar-opt
    s.as_materialized_series()
        .rolling_std(options)
        .map(Column::from)
}

#[cfg(feature = "moment")]
pub(super) fn rolling_skew(s: &Column, options: RollingOptionsFixedWindow) -> PolarsResult<Column> {
    // @scalar-opt
    let s = s.as_materialized_series();
    polars_ops::series::rolling_skew(s, options).map(Column::from)
}

#[cfg(feature = "moment")]
pub(super) fn rolling_kurtosis(
    s: &Column,
    options: RollingOptionsFixedWindow,
) -> PolarsResult<Column> {
    // @scalar-opt
    let s = s.as_materialized_series();
    polars_ops::series::rolling_kurtosis(s, options).map(Column::from)
}

#[cfg(feature = "cov")]
fn det_count_x_y(window_size: usize, len: usize, dtype: &DataType) -> Series {
    match dtype {
        DataType::Float64 => {
            let values = (0..len)
                .map(|v| std::cmp::min(window_size, v + 1) as f64)
                .collect::<Vec<_>>();
            Series::new(PlSmallStr::EMPTY, values)
        },
        DataType::Float32 => {
            let values = (0..len)
                .map(|v| std::cmp::min(window_size, v + 1) as f32)
                .collect::<Vec<_>>();
            Series::new(PlSmallStr::EMPTY, values)
        },
        _ => unreachable!(),
    }
}

#[cfg(feature = "cov")]
pub(super) fn rolling_corr_cov(
    s: &[Column],
    rolling_options: RollingOptionsFixedWindow,
    cov_options: RollingCovOptions,
    is_corr: bool,
) -> PolarsResult<Column> {
    let mut x = s[0].as_materialized_series().rechunk();
    let mut y = s[1].as_materialized_series().rechunk();

    if !x.dtype().is_float() {
        x = x.cast(&DataType::Float64)?;
    }
    if !y.dtype().is_float() {
        y = y.cast(&DataType::Float64)?;
    }
    let dtype = x.dtype().clone();

    let mean_x_y = (&x * &y)?.rolling_mean(rolling_options.clone())?;
    let rolling_options_count = RollingOptionsFixedWindow {
        window_size: rolling_options.window_size,
        min_periods: 0,
        ..Default::default()
    };

    let count_x_y = if (x.null_count() + y.null_count()) > 0 {
        // mask out nulls on both sides before compute mean/var
        let valids = x.is_not_null().bitand(y.is_not_null());
        let valids_arr = valids.downcast_as_array();
        let valids_bitmap = valids_arr.values();

        unsafe {
            let xarr = &mut x.chunks_mut()[0];
            *xarr = xarr.with_validity(Some(valids_bitmap.clone()));
            let yarr = &mut y.chunks_mut()[0];
            *yarr = yarr.with_validity(Some(valids_bitmap.clone()));
            x.compute_len();
            y.compute_len();
        }
        valids
            .cast(&dtype)
            .unwrap()
            .rolling_sum(rolling_options_count)?
    } else {
        det_count_x_y(rolling_options.window_size, x.len(), &dtype)
    };

    let mean_x = x.rolling_mean(rolling_options.clone())?;
    let mean_y = y.rolling_mean(rolling_options.clone())?;
    let ddof = Series::new(
        PlSmallStr::EMPTY,
        &[AnyValue::from(cov_options.ddof).cast(&dtype)],
    );

    let numerator = ((mean_x_y - (mean_x * mean_y).unwrap()).unwrap()
        * (count_x_y.clone() / (count_x_y - ddof).unwrap()).unwrap())
    .unwrap();

    if is_corr {
        let var_x = x.rolling_var(rolling_options.clone())?;
        let var_y = y.rolling_var(rolling_options.clone())?;

        let base = (var_x * var_y).unwrap();
        let sc = Scalar::new(
            base.dtype().clone(),
            AnyValue::Float64(0.5).cast(&dtype).into_static(),
        );
        let denominator = pow(&mut [base.into_column(), sc.into_column("".into())])
            .unwrap()
            .unwrap()
            .take_materialized_series();

        Ok((numerator / denominator)?.into_column())
    } else {
        Ok(numerator.into_column())
    }
}
