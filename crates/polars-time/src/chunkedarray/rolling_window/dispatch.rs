use std::borrow::Cow;

use arrow::types::NativeType;
#[cfg(feature = "dtype-f16")]
use num_traits::real::Real;
use polars_compute::rolling::no_nulls::RollingAggWindowNoNulls;
use polars_compute::rolling::nulls::RollingAggWindowNulls;
use polars_compute::rolling::{MeanWindow, SumWindow, no_nulls, nulls};
use polars_core::{with_match_physical_float_polars_type, with_match_physical_numeric_polars_type};
use polars_ops::series::SeriesMethods;
use polars_utils::float::IsFloat;

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
        Option<RollingFnParams>,
    ) -> PolarsResult<ArrayRef>,
    rolling_agg_fn_nulls: &dyn Fn(
        &PrimitiveArray<T::Native>,
        usize,
        usize,
        bool,
        Option<&[f64]>,
        Option<RollingFnParams>,
    ) -> ArrayRef,
) -> PolarsResult<Series>
where
    T: PolarsNumericType,
{
    polars_ensure!(options.min_periods <= options.window_size, InvalidOperation: "`min_periods` should be <= `window_size`");
    if ca.is_empty() {
        return Ok(Series::new_empty(ca.name().clone(), ca.dtype()));
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
    Series::try_from((ca.name().clone(), arr))
}

#[cfg(feature = "rolling_window_by")]
fn rolling_agg_by<T, Out, NoNullsAgg, NullsAgg>(
    ca: &ChunkedArray<T>,
    by: &Series,
    options: RollingOptionsDynamicWindow,
) -> PolarsResult<Series>
where
    T: PolarsNumericType,
    T::Native: NativeType + IsFloat,
    Out: NativeType,
    NoNullsAgg: RollingAggWindowNoNulls<T::Native, Out>,
    NullsAgg: RollingAggWindowNulls<T::Native, Out>,
{
    use crate::chunkedarray::rolling_window::rolling_kernels::shared::{
        RollingAggWindowNoNullsWrapper, RollingAggWindowNullsWrapper, rolling_apply_agg,
    };

    if ca.is_empty() {
        return Ok(Series::new_empty(ca.name().clone(), ca.dtype()));
    }

    polars_ensure!(
        ca.len() == by.len(),
        InvalidOperation: "`by` column in `rolling_*_by` must be the same length as values column"
    );
    ensure_duration_matches_dtype(options.window_size, by.dtype(), "window_size")?;
    polars_ensure!(
        !options.window_size.is_zero() && !options.window_size.negative,
        InvalidOperation: "`window_size` must be strictly positive"
    );

    let (by, tz) = match by.dtype() {
        DataType::Datetime(tu, tz) => (by.cast(&DataType::Datetime(*tu, None))?, tz),
        DataType::Date => (
            by.cast(&DataType::Datetime(TimeUnit::Microseconds, None))?,
            &None,
        ),
        DataType::Int64 => (
            by.cast(&DataType::Datetime(TimeUnit::Nanoseconds, None))?,
            &None,
        ),
        DataType::Int32 | DataType::UInt64 | DataType::UInt32 => (
            by.cast(&DataType::Int64)?
                .cast(&DataType::Datetime(TimeUnit::Nanoseconds, None))?,
            &None,
        ),
        dt => polars_bail!(InvalidOperation:
            "in `rolling_*_by` operation, `by` argument of dtype `{}` is not supported (expected `{}`)",
            dt,
            "Date/Datetime/Int64/Int32/UInt64/UInt32"),
    };
    let mut ca_rechunked = ca.rechunk();
    let by = by.rechunk();
    let by_is_sorted = by.is_sorted(SortOptions {
        descending: false,
        ..Default::default()
    })?;
    let by_logical = by.datetime().unwrap();
    let tu = by_logical.time_unit();
    let mut by_physical = Cow::Borrowed(by_logical.physical());
    let sorting_indices_opt = (!by_is_sorted).then(|| by_physical.arg_sort(Default::default()));

    if let Some(sorting_indices) = &sorting_indices_opt {
        // SAFETY: `sorting_indices` is in-bounds because we checked that `ca.len() == by.len()` and
        // they are derived from `by`.
        ca_rechunked = Cow::Owned(unsafe { ca_rechunked.take_unchecked(sorting_indices) });
        // SAFETY: `sorting_indices` is in-bounds because they are derived from `by`.
        by_physical = Cow::Owned(unsafe { by_physical.take_unchecked(sorting_indices) });
    }

    let by_values = by_physical.cont_slice().unwrap();
    let arr = ca_rechunked.downcast_iter().next().unwrap();
    let values = arr.values().as_slice();

    // We explicitly branch here because we want to compile different versions based on the no_nulls
    // or nulls kernel.
    let out: ArrayRef = if ca.null_count() == 0 {
        let mut agg_window =
            RollingAggWindowNoNullsWrapper(NoNullsAgg::new(values, 0, 0, options.fn_params, None));

        rolling_apply_agg(
            &mut agg_window,
            options.window_size,
            by_values,
            options.closed_window,
            options.min_periods,
            tu,
            tz.as_ref(),
            sorting_indices_opt
                .as_ref()
                .map(|s| s.cont_slice().unwrap()),
        )?
    } else {
        let validity = arr.validity().unwrap();
        let mut agg_window = RollingAggWindowNullsWrapper(NullsAgg::new(
            values,
            validity,
            0,
            0,
            options.fn_params,
            None,
        ));

        rolling_apply_agg(
            &mut agg_window,
            options.window_size,
            by_values,
            options.closed_window,
            options.min_periods,
            tu,
            tz.as_ref(),
            sorting_indices_opt
                .as_ref()
                .map(|s| s.cont_slice().unwrap()),
        )?
    };

    Series::try_from((ca.name().clone(), out))
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
            rolling_agg_by::<$T, _, MeanWindow<_>, MeanWindow<_>>(ca, by, options)
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
        let mut s = self.as_series().clone();
        if s.dtype() == &DataType::Boolean {
            s = s.cast(&DataType::IDX_DTYPE).unwrap();
        }
        if matches!(
            s.dtype(),
            DataType::Int8 | DataType::UInt8 | DataType::Int16 | DataType::UInt16
        ) {
            s = s.cast(&DataType::Int64).unwrap();
        }

        polars_ensure!(
            s.dtype().is_primitive_numeric() && !s.dtype().is_unknown(),
            op = "rolling_sum_by",
            s.dtype()
        );

        with_match_physical_numeric_polars_type!(s.dtype(), |$T| {
            let ca: &ChunkedArray<$T> = s.as_ref().as_ref().as_ref();
            type Native = <$T as PolarsNumericType>::Native;
            type SM<'a> = SumWindow<'a, Native, Native>;
            rolling_agg_by::<$T, _, SM, SM>(ca, by, options)
        })
    }

    /// Apply a rolling sum to a Series.
    #[cfg(feature = "rolling_window")]
    fn rolling_sum(&self, options: RollingOptionsFixedWindow) -> PolarsResult<Series> {
        let mut s = self.as_series().clone();
        if options.weights.is_some() {
            s = s.to_float()?;
        } else if s.dtype() == &DataType::Boolean {
            s = s.cast(&DataType::IDX_DTYPE).unwrap();
        } else if matches!(
            s.dtype(),
            DataType::Int8 | DataType::UInt8 | DataType::Int16 | DataType::UInt16
        ) {
            s = s.cast(&DataType::Int64).unwrap();
        }

        polars_ensure!(
            s.dtype().is_primitive_numeric() && !s.dtype().is_unknown(),
            op = "rolling_sum",
            s.dtype()
        );

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
            rolling_agg_by::<
                $T,
                _,
                no_nulls::QuantileWindow<_>,
                nulls::QuantileWindow<_>
            >(ca, by, options)
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

        let dt = s.dtype();
        match dt {
            // Our rolling kernels don't yet support boolean, use UInt8 as a workaround for now.
            &DataType::Boolean => {
                return s
                    .cast(&DataType::UInt8)?
                    .rolling_min_by(by, options)?
                    .cast(&DataType::Boolean);
            },
            dt if dt.is_temporal() => {
                return s.to_physical_repr().rolling_min_by(by, options)?.cast(dt);
            },
            dt => {
                polars_ensure!(
                    dt.is_primitive_numeric() && !dt.is_unknown(),
                    op = "rolling_min_by",
                    dt
                );
            },
        }

        with_match_physical_numeric_polars_type!(s.dtype(), |$T| {
            let ca: &ChunkedArray<$T> = s.as_ref().as_ref().as_ref();
            rolling_agg_by::<
                $T,
                _,
                no_nulls::MinWindow<_>,
                nulls::MinWindow<_>
            >(ca, by, options)
        })
    }

    /// Apply a rolling min to a Series.
    #[cfg(feature = "rolling_window")]
    fn rolling_min(&self, options: RollingOptionsFixedWindow) -> PolarsResult<Series> {
        let mut s = self.as_series().clone();
        if options.weights.is_some() {
            s = s.to_float()?;
        }

        let dt = s.dtype();
        match dt {
            // Our rolling kernels don't yet support boolean, use UInt8 as a workaround for now.
            &DataType::Boolean => {
                return s
                    .cast(&DataType::UInt8)?
                    .rolling_min(options)?
                    .cast(&DataType::Boolean);
            },
            dt if dt.is_temporal() => {
                return s.to_physical_repr().rolling_min(options)?.cast(dt);
            },
            dt => {
                polars_ensure!(
                    dt.is_primitive_numeric() && !dt.is_unknown(),
                    op = "rolling_min",
                    dt
                );
            },
        }

        with_match_physical_numeric_polars_type!(dt, |$T| {
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

        let dt = s.dtype();
        match dt {
            // Our rolling kernels don't yet support boolean, use UInt8 as a workaround for now.
            &DataType::Boolean => {
                return s
                    .cast(&DataType::UInt8)?
                    .rolling_max_by(by, options)?
                    .cast(&DataType::Boolean);
            },
            dt if dt.is_temporal() => {
                return s.to_physical_repr().rolling_max_by(by, options)?.cast(dt);
            },
            dt => {
                polars_ensure!(
                    dt.is_primitive_numeric() && !dt.is_unknown(),
                    op = "rolling_max_by",
                    dt
                );
            },
        }

        with_match_physical_numeric_polars_type!(s.dtype(), |$T| {
            let ca: &ChunkedArray<$T> = s.as_ref().as_ref().as_ref();
            rolling_agg_by::<
                $T,
                _,
                no_nulls::MaxWindow<_>,
                nulls::MaxWindow<_>
            >(ca, by, options)
        })
    }

    /// Apply a rolling max to a Series.
    #[cfg(feature = "rolling_window")]
    fn rolling_max(&self, options: RollingOptionsFixedWindow) -> PolarsResult<Series> {
        let mut s = self.as_series().clone();
        if options.weights.is_some() {
            s = s.to_float()?;
        }

        let dt = s.dtype();
        match dt {
            // Our rolling kernels don't yet support boolean, use UInt8 as a workaround for now.
            &DataType::Boolean => {
                return s
                    .cast(&DataType::UInt8)?
                    .rolling_max(options)?
                    .cast(&DataType::Boolean);
            },
            dt if dt.is_temporal() => {
                return s.to_physical_repr().rolling_max(options)?.cast(dt);
            },
            dt => {
                polars_ensure!(
                    dt.is_primitive_numeric() && !dt.is_unknown(),
                    op = "rolling_max",
                    dt
                );
            },
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

            rolling_agg_by::<
                $T,
                _,
                no_nulls::MomentWindow<_, no_nulls::VarianceMoment>,
                nulls::MomentWindow<_, nulls::VarianceMoment>
            >(ca, by, options)
        })
    }

    /// Apply a rolling variance to a Series.
    #[cfg(feature = "rolling_window")]
    fn rolling_var(&self, options: RollingOptionsFixedWindow) -> PolarsResult<Series> {
        let s = self.as_series().to_float()?;

        with_match_physical_float_polars_type!(s.dtype(), |$T| {
            let ca: &ChunkedArray<$T> = s.as_ref().as_ref().as_ref();

            rolling_agg(
                ca,
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
            with_match_physical_float_polars_type!(s.dtype(), |$T| {
                let ca: &mut ChunkedArray<$T> = s._get_inner_mut().as_mut();
                ca.apply_mut(|v| v.sqrt());
            });

            s
        })
    }

    /// Apply a rolling std_dev to a Series.
    #[cfg(feature = "rolling_window")]
    fn rolling_std(&self, options: RollingOptionsFixedWindow) -> PolarsResult<Series> {
        self.rolling_var(options).map(|mut s| {
            with_match_physical_float_polars_type!(s.dtype(), |$T| {
                let ca: &mut ChunkedArray<$T> = s._get_inner_mut().as_mut();
                ca.apply_mut(|v| v.sqrt());
            });

            s
        })
    }

    /// Apply a rolling rank to a Series based on another Series.
    #[cfg(feature = "rolling_window_by")]
    fn rolling_rank_by(
        &self,
        by: &Series,
        options: RollingOptionsDynamicWindow,
    ) -> PolarsResult<Series> {
        if !matches!(
            options.closed_window,
            ClosedWindow::Right | ClosedWindow::Both
        ) {
            polars_bail!(InvalidOperation: "`rolling_rank_by` window needs to be closed on the right side (i.e., `closed` must be `right` or `both`)");
        }

        let s = self.as_series().clone();

        match s.dtype() {
            DataType::Boolean => return s.cast(&DataType::UInt8)?.rolling_rank_by(by, options),
            dt if dt.is_temporal() => return s.to_physical_repr().rolling_rank_by(by, options),
            dt => {
                polars_ensure!(
                    dt.is_primitive_numeric() && !dt.is_unknown(),
                    op = "rolling_rank_by",
                    dt
                );
            },
        }

        let method = if let Some(RollingFnParams::Rank { method, .. }) = options.fn_params {
            method
        } else {
            unreachable!("expected RollingFnParams::Rank");
        };

        with_match_physical_numeric_polars_type!(s.dtype(), |$T| {
            let ca: &ChunkedArray<$T> = s.as_ref().as_ref().as_ref();

            match method {
                RollingRankMethod::Average => rolling_agg_by::<
                    $T,
                    _,
                    no_nulls::RankWindowAvg<_>,
                    nulls::RankWindowAvg<_>
                >(ca, by, options),
                RollingRankMethod::Min => rolling_agg_by::<
                    $T,
                    _,
                    no_nulls::RankWindowMin<_>,
                    nulls::RankWindowMin<_>
                >(ca, by, options),
                RollingRankMethod::Max => rolling_agg_by::<
                    $T,
                    _,
                    no_nulls::RankWindowMax<_>,
                    nulls::RankWindowMax<_>
                >(ca, by, options),
                RollingRankMethod::Dense => rolling_agg_by::<
                    $T,
                    _,
                    no_nulls::RankWindowDense<_>,
                    nulls::RankWindowDense<_>
                >(ca, by, options),
                RollingRankMethod::Random => rolling_agg_by::<
                    $T,
                    _,
                    no_nulls::RankWindowRandom<_>,
                    nulls::RankWindowRandom<_>
                >(ca, by, options),
                _ => todo!()
            }
        })
    }

    /// Apply a rolling rank to a Series.
    #[cfg(feature = "rolling_window")]
    fn rolling_rank(&self, options: RollingOptionsFixedWindow) -> PolarsResult<Series> {
        let s = self.as_series();

        match s.dtype() {
            DataType::Boolean => return s.cast(&DataType::UInt8)?.rolling_rank(options),
            dt if dt.is_temporal() => return s.to_physical_repr().rolling_rank(options),
            dt => {
                polars_ensure!(
                    dt.is_primitive_numeric() && !dt.is_unknown(),
                    op = "rolling_rank",
                    dt
                );
            },
        }

        with_match_physical_numeric_polars_type!(s.dtype(), |$T| {
            let ca: &ChunkedArray<$T> = s.as_ref().as_ref().as_ref();
            let mut ca = ca.clone();

            rolling_agg(
                &ca,
                options,
                &rolling::no_nulls::rolling_rank,
                &rolling::nulls::rolling_rank,
            )
        })
    }
}

impl SeriesOpsTime for Series {}
