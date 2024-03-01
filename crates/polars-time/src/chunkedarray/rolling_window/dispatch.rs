use polars_core::{with_match_physical_float_polars_type, with_match_physical_numeric_polars_type};

use super::*;
use crate::prelude::*;
use crate::series::AsSeries;

#[allow(clippy::type_complexity)]
fn rolling_agg<T>(
    ca: &ChunkedArray<T>,
    options: RollingOptionsImpl,
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
    rolling_agg_fn_dynamic: Option<
        &dyn Fn(
            &[T::Native],
            Duration,
            &[i64],
            ClosedWindow,
            usize,
            TimeUnit,
            Option<&TimeZone>,
            DynArgs,
        ) -> PolarsResult<ArrayRef>,
    >,
) -> PolarsResult<Series>
where
    T: PolarsNumericType,
{
    if ca.is_empty() {
        return Ok(Series::new_empty(ca.name(), ca.dtype()));
    }
    let ca = ca.rechunk();

    let arr = ca.downcast_iter().next().unwrap();
    // "5i" is a window size of 5, e.g. fixed
    let arr = if options.window_size.parsed_int {
        let options: RollingOptionsFixedWindow = options.into();
        check_input(options.window_size, options.min_periods)?;

        Ok(match ca.null_count() {
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
        })
    } else {
        if arr.null_count() > 0 {
            panic!("'rolling by' not yet supported for series with null values, consider using 'group_by_rolling'")
        }
        let values = arr.values().as_slice();
        let duration = options.window_size;
        polars_ensure!(duration.duration_ns() > 0 && !duration.negative, ComputeError:"window size should be strictly positive");
        let tu = options.tu.unwrap();
        let by = options.by.unwrap();
        let closed_window = options.closed_window.expect("closed window  must be set");
        let func = rolling_agg_fn_dynamic.expect(
            "'rolling by' not yet supported for this expression, consider using 'group_by_rolling'",
        );

        func(
            values,
            duration,
            by,
            closed_window,
            options.min_periods,
            tu,
            options.tz,
            options.fn_params,
        )
    }?;
    Series::try_from((ca.name(), arr))
}

pub trait SeriesOpsTime: AsSeries {
    /// Apply a rolling mean to a Series.
    ///
    /// See: [`RollingAgg::rolling_mean`]
    #[cfg(feature = "rolling_window")]
    fn rolling_mean(&self, options: RollingOptionsImpl) -> PolarsResult<Series> {
        let s = self.as_series().to_float()?;
        with_match_physical_float_polars_type!(s.dtype(), |$T| {
            let ca: &ChunkedArray<$T> = s.as_ref().as_ref().as_ref();
            rolling_agg(
                ca,
                options,
                &rolling::no_nulls::rolling_mean,
                &rolling::nulls::rolling_mean,
                Some(&super::rolling_kernels::no_nulls::rolling_mean),
            )
        })
    }
    /// Apply a rolling sum to a Series.
    #[cfg(feature = "rolling_window")]
    fn rolling_sum(&self, options: RollingOptionsImpl) -> PolarsResult<Series> {
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
                Some(&super::rolling_kernels::no_nulls::rolling_sum),
            )
        })
    }

    /// Apply a rolling quantile to a Series.
    #[cfg(feature = "rolling_window")]
    fn rolling_quantile(&self, options: RollingOptionsImpl) -> PolarsResult<Series> {
        let s = self.as_series().to_float()?;
        with_match_physical_float_polars_type!(s.dtype(), |$T| {
            let ca: &ChunkedArray<$T> = s.as_ref().as_ref().as_ref();
        rolling_agg(
            ca,
            options,
            &rolling::no_nulls::rolling_quantile,
            &rolling::nulls::rolling_quantile,
            Some(&super::rolling_kernels::no_nulls::rolling_quantile),
        )
        })
    }

    /// Apply a rolling min to a Series.
    #[cfg(feature = "rolling_window")]
    fn rolling_min(&self, options: RollingOptionsImpl) -> PolarsResult<Series> {
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
                Some(&super::rolling_kernels::no_nulls::rolling_min),
            )
        })
    }
    /// Apply a rolling max to a Series.
    #[cfg(feature = "rolling_window")]
    fn rolling_max(&self, options: RollingOptionsImpl) -> PolarsResult<Series> {
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
                Some(&super::rolling_kernels::no_nulls::rolling_max),
            )
        })
    }

    /// Apply a rolling variance to a Series.
    #[cfg(feature = "rolling_window")]
    fn rolling_var(&self, options: RollingOptionsImpl) -> PolarsResult<Series> {
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
                Some(&super::rolling_kernels::no_nulls::rolling_var),
            )
        })
    }

    /// Apply a rolling std_dev to a Series.
    #[cfg(feature = "rolling_window")]
    fn rolling_std(&self, options: RollingOptionsImpl) -> PolarsResult<Series> {
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
