use polars_core::{with_match_physical_float_polars_type, with_match_physical_numeric_polars_type};
use polars_ops::series::SeriesMethods;

use super::*;
use crate::prelude::*;
use crate::series::AsSeries;

#[cfg(feature = "rolling_window")]
#[allow(clippy::type_complexity)]
fn rolling_agg<T>(
    ca: &ChunkedArray<T>,
    options: RollingOptionsFixedWindow,
    rolling_agg_fn: &dyn Fn(
        &[T::Native],
        usize,
        usize,
        bool,
        Option<&[f64]>,
        DynArgs,
    ) -> PolarsResult<ArrayRef>,
    rolling_agg_fn_nulls: &dyn Fn(
        &PrimitiveArray<T::Native>,
        usize,
        usize,
        bool,
        Option<&[f64]>,
        DynArgs,
    ) -> ArrayRef,
) -> PolarsResult<Series>
where
    T: PolarsNumericType,
{
    polars_ensure!(options.min_periods <= options.window_size, InvalidOperation: "`min_periods` should be <= `window_size`");
    if ca.is_empty() {
        return Ok(Series::new_empty(ca.name(), ca.dtype()));
    }
    let ca = ca.rechunk();

    let arr = ca.downcast_iter().next().unwrap();
    let arr = match ca.null_count() {
        0 => rolling_agg_fn(
            arr.values().as_slice(),
            options.window_size,
            options.min_periods,
            options.center,
            options.weights.as_deref(),
            options.fn_params,
        )?,
        _ => rolling_agg_fn_nulls(
            arr,
            options.window_size,
            options.min_periods,
            options.center,
            options.weights.as_deref(),
            options.fn_params,
        ),
    };
    Series::try_from((ca.name(), arr))
}

#[cfg(feature = "rolling_window_by")]
#[allow(clippy::type_complexity)]
fn rolling_agg_by<T>(
    ca: &ChunkedArray<T>,
    by: &Series,
    options: RollingOptionsDynamicWindow,
    rolling_agg_fn_dynamic: &dyn Fn(
        &[T::Native],
        Duration,
        &[i64],
        ClosedWindow,
        usize,
        TimeUnit,
        Option<&TimeZone>,
        DynArgs,
        Option<&[IdxSize]>,
    ) -> PolarsResult<ArrayRef>,
) -> PolarsResult<Series>
where
    T: PolarsNumericType,
{
    if ca.is_empty() {
        return Ok(Series::new_empty(ca.name(), ca.dtype()));
    }
    polars_ensure!(by.null_count() == 0 && ca.null_count() == 0, InvalidOperation: "'Expr.rolling_*_by(...)' not yet supported for series with null values, consider using 'DataFrame.rolling' or 'Expr.rolling'");
    polars_ensure!(ca.len() == by.len(), InvalidOperation: "`by` column in `rolling_*_by` must be the same length as values column");
    ensure_duration_matches_data_type(options.window_size, by.dtype(), "window_size")?;
    polars_ensure!(!options.window_size.is_zero() && !options.window_size.negative, InvalidOperation: "`window_size` must be strictly positive");
    let (by, tz) = match by.dtype() {
        DataType::Datetime(tu, tz) => (by.cast(&DataType::Datetime(*tu, None))?, tz),
        DataType::Date => (
            by.cast(&DataType::Datetime(TimeUnit::Milliseconds, None))?,
            &None,
        ),
        dt => polars_bail!(InvalidOperation:
            "in `rolling_*_by` operation, `by` argument of dtype `{}` is not supported (expected `{}`)",
            dt,
            "date/datetime"),
    };
    let ca = ca.rechunk();
    let by = by.rechunk();
    let by_is_sorted = by.is_sorted(SortOptions {
        descending: false,
        ..Default::default()
    })?;
    let by = by.datetime().unwrap();
    let tu = by.time_unit();

    let func = rolling_agg_fn_dynamic;
    let out: ArrayRef = if by_is_sorted {
        let arr = ca.downcast_iter().next().unwrap();
        let by_values = by.cont_slice().unwrap();
        let values = arr.values().as_slice();
        func(
            values,
            options.window_size,
            by_values,
            options.closed_window,
            options.min_periods,
            tu,
            tz.as_ref(),
            options.fn_params,
            None,
        )?
    } else {
        let sorting_indices = by.arg_sort(Default::default());
        let ca = unsafe { ca.take_unchecked(&sorting_indices) };
        let by = unsafe { by.take_unchecked(&sorting_indices) };
        let arr = ca.downcast_iter().next().unwrap();
        let by_values = by.cont_slice().unwrap();
        let values = arr.values().as_slice();
        func(
            values,
            options.window_size,
            by_values,
            options.closed_window,
            options.min_periods,
            tu,
            tz.as_ref(),
            options.fn_params,
            Some(sorting_indices.cont_slice().unwrap()),
        )?
    };
    Series::try_from((ca.name(), out))
}

pub trait SeriesOpsTime: AsSeries {
    /// Apply a rolling mean to a Series based on another Series.
    #[cfg(feature = "rolling_window_by")]
    fn rolling_mean_by(
        &self,
        by: &Series,
        options: RollingOptionsDynamicWindow,
    ) -> PolarsResult<Series> {
        let s = self.as_series().to_float()?;
        with_match_physical_float_polars_type!(s.dtype(), |$T| {
            let ca: &ChunkedArray<$T> = s.as_ref().as_ref().as_ref();
            rolling_agg_by(
                ca,
                by,
                options,
                &super::rolling_kernels::no_nulls::rolling_mean,
            )
        })
    }
    /// Apply a rolling mean to a Series.
    ///
    /// See: [`RollingAgg::rolling_mean`]
    #[cfg(feature = "rolling_window")]
    fn rolling_mean(&self, options: RollingOptionsFixedWindow) -> PolarsResult<Series> {
        let s = self.as_series().to_float()?;
        with_match_physical_float_polars_type!(s.dtype(), |$T| {
            let ca: &ChunkedArray<$T> = s.as_ref().as_ref().as_ref();
            rolling_agg(
                ca,
                options,
                &rolling::no_nulls::rolling_mean,
                &rolling::nulls::rolling_mean,
            )
        })
    }
    /// Apply a rolling sum to a Series based on another Series.
    #[cfg(feature = "rolling_window_by")]
    fn rolling_sum_by(
        &self,
        by: &Series,
        options: RollingOptionsDynamicWindow,
    ) -> PolarsResult<Series> {
        let s = self.as_series().clone();
        with_match_physical_numeric_polars_type!(s.dtype(), |$T| {
            let ca: &ChunkedArray<$T> = s.as_ref().as_ref().as_ref();
            rolling_agg_by(
                ca,
                by,
                options,
                &super::rolling_kernels::no_nulls::rolling_sum,
            )
        })
    }

    /// Apply a rolling sum to a Series.
    #[cfg(feature = "rolling_window")]
    fn rolling_sum(&self, options: RollingOptionsFixedWindow) -> PolarsResult<Series> {
        let mut s = self.as_series().clone();
        if options.weights.is_some() {
            s = s.to_float()?;
        }

        with_match_physical_numeric_polars_type!(s.dtype(), |$T| {
            let ca: &ChunkedArray<$T> = s.as_ref().as_ref().as_ref();
            rolling_agg(
                ca,
                options,
                &rolling::no_nulls::rolling_sum,
                &rolling::nulls::rolling_sum,
            )
        })
    }

    /// Apply a rolling quantile to a Series based on another Series.
    #[cfg(feature = "rolling_window_by")]
    fn rolling_quantile_by(
        &self,
        by: &Series,
        options: RollingOptionsDynamicWindow,
    ) -> PolarsResult<Series> {
        let s = self.as_series().to_float()?;
        with_match_physical_float_polars_type!(s.dtype(), |$T| {
            let ca: &ChunkedArray<$T> = s.as_ref().as_ref().as_ref();
        rolling_agg_by(
            ca,
            by,
            options,
            &super::rolling_kernels::no_nulls::rolling_quantile,
        )
        })
    }

    /// Apply a rolling quantile to a Series.
    #[cfg(feature = "rolling_window")]
    fn rolling_quantile(&self, options: RollingOptionsFixedWindow) -> PolarsResult<Series> {
        let s = self.as_series().to_float()?;
        with_match_physical_float_polars_type!(s.dtype(), |$T| {
            let ca: &ChunkedArray<$T> = s.as_ref().as_ref().as_ref();
        rolling_agg(
            ca,
            options,
            &rolling::no_nulls::rolling_quantile,
            &rolling::nulls::rolling_quantile,
        )
        })
    }

    /// Apply a rolling min to a Series based on another Series.
    #[cfg(feature = "rolling_window_by")]
    fn rolling_min_by(
        &self,
        by: &Series,
        options: RollingOptionsDynamicWindow,
    ) -> PolarsResult<Series> {
        let s = self.as_series().clone();
        with_match_physical_numeric_polars_type!(s.dtype(), |$T| {
            let ca: &ChunkedArray<$T> = s.as_ref().as_ref().as_ref();
            rolling_agg_by(
                ca,
                by,
                options,
                &super::rolling_kernels::no_nulls::rolling_min,
            )
        })
    }

    /// Apply a rolling min to a Series.
    #[cfg(feature = "rolling_window")]
    fn rolling_min(&self, options: RollingOptionsFixedWindow) -> PolarsResult<Series> {
        let mut s = self.as_series().clone();
        if options.weights.is_some() {
            s = s.to_float()?;
        }

        with_match_physical_numeric_polars_type!(s.dtype(), |$T| {
            let ca: &ChunkedArray<$T> = s.as_ref().as_ref().as_ref();
            rolling_agg(
                ca,
                options,
                &rolling::no_nulls::rolling_min,
                &rolling::nulls::rolling_min,
            )
        })
    }

    /// Apply a rolling max to a Series based on another Series.
    #[cfg(feature = "rolling_window_by")]
    fn rolling_max_by(
        &self,
        by: &Series,
        options: RollingOptionsDynamicWindow,
    ) -> PolarsResult<Series> {
        let s = self.as_series().clone();
        with_match_physical_numeric_polars_type!(s.dtype(), |$T| {
            let ca: &ChunkedArray<$T> = s.as_ref().as_ref().as_ref();
            rolling_agg_by(
                ca,
                by,
                options,
                &super::rolling_kernels::no_nulls::rolling_max,
            )
        })
    }

    /// Apply a rolling max to a Series.
    #[cfg(feature = "rolling_window")]
    fn rolling_max(&self, options: RollingOptionsFixedWindow) -> PolarsResult<Series> {
        let mut s = self.as_series().clone();
        if options.weights.is_some() {
            s = s.to_float()?;
        }

        with_match_physical_numeric_polars_type!(s.dtype(), |$T| {
            let ca: &ChunkedArray<$T> = s.as_ref().as_ref().as_ref();
            rolling_agg(
                ca,
                options,
                &rolling::no_nulls::rolling_max,
                &rolling::nulls::rolling_max,
            )
        })
    }

    /// Apply a rolling variance to a Series based on another Series.
    #[cfg(feature = "rolling_window_by")]
    fn rolling_var_by(
        &self,
        by: &Series,
        options: RollingOptionsDynamicWindow,
    ) -> PolarsResult<Series> {
        let s = self.as_series().to_float()?;

        with_match_physical_float_polars_type!(s.dtype(), |$T| {
            let ca: &ChunkedArray<$T> = s.as_ref().as_ref().as_ref();
            let mut ca = ca.clone();

            if let Some(idx) = ca.first_non_null() {
                let k = ca.get(idx).unwrap();
                // TODO! remove this!
                // This is a temporary hack to improve numeric stability.
                // var(X) = var(X - k)
                // This is temporary as we will rework the rolling methods
                // the 100.0 absolute boundary is arbitrarily chosen.
                // the algorithm will square numbers, so it loses precision rapidly
                if k.abs() > 100.0 {
                    ca = ca - k;
                }
            }

            rolling_agg_by(
                &ca,
                by,
                options,
                &super::rolling_kernels::no_nulls::rolling_var,
            )
        })
    }

    /// Apply a rolling variance to a Series.
    #[cfg(feature = "rolling_window")]
    fn rolling_var(&self, options: RollingOptionsFixedWindow) -> PolarsResult<Series> {
        let s = self.as_series().to_float()?;

        with_match_physical_float_polars_type!(s.dtype(), |$T| {
            let ca: &ChunkedArray<$T> = s.as_ref().as_ref().as_ref();
            let mut ca = ca.clone();

            if let Some(idx) = ca.first_non_null() {
                let k = ca.get(idx).unwrap();
                // TODO! remove this!
                // This is a temporary hack to improve numeric stability.
                // var(X) = var(X - k)
                // This is temporary as we will rework the rolling methods
                // the 100.0 absolute boundary is arbitrarily chosen.
                // the algorithm will square numbers, so it loses precision rapidly
                if k.abs() > 100.0 {
                    ca = ca - k;
                }
            }

            rolling_agg(
                &ca,
                options,
                &rolling::no_nulls::rolling_var,
                &rolling::nulls::rolling_var,
            )
        })
    }

    /// Apply a rolling std_dev to a Series based on another Series.
    #[cfg(feature = "rolling_window_by")]
    fn rolling_std_by(
        &self,
        by: &Series,
        options: RollingOptionsDynamicWindow,
    ) -> PolarsResult<Series> {
        self.rolling_var_by(by, options).map(|mut s| {
            match s.dtype().clone() {
                DataType::Float32 => {
                    let ca: &mut ChunkedArray<Float32Type> = s._get_inner_mut().as_mut();
                    ca.apply_mut(|v| v.powf(0.5))
                },
                DataType::Float64 => {
                    let ca: &mut ChunkedArray<Float64Type> = s._get_inner_mut().as_mut();
                    ca.apply_mut(|v| v.powf(0.5))
                },
                _ => unreachable!(),
            }
            s
        })
    }

    /// Apply a rolling std_dev to a Series.
    #[cfg(feature = "rolling_window")]
    fn rolling_std(&self, options: RollingOptionsFixedWindow) -> PolarsResult<Series> {
        self.rolling_var(options).map(|mut s| {
            match s.dtype().clone() {
                DataType::Float32 => {
                    let ca: &mut ChunkedArray<Float32Type> = s._get_inner_mut().as_mut();
                    ca.apply_mut(|v| v.powf(0.5))
                },
                DataType::Float64 => {
                    let ca: &mut ChunkedArray<Float64Type> = s._get_inner_mut().as_mut();
                    ca.apply_mut(|v| v.powf(0.5))
                },
                _ => unreachable!(),
            }
            s
        })
    }
}

impl SeriesOpsTime for Series {}
