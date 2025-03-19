use arrow::legacy::time_zone::Tz;
use polars_core::POOL;
use polars_core::prelude::*;
use polars_core::series::IsSorted;
use polars_core::utils::flatten::flatten_par;
use polars_ops::series::SeriesMethods;
use polars_utils::itertools::Itertools;
use polars_utils::pl_str::PlSmallStr;
use polars_utils::slice::SortedSlice;
use rayon::prelude::*;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::prelude::*;

#[repr(transparent)]
struct Wrap<T>(pub T);

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct DynamicGroupOptions {
    /// Time or index column.
    pub index_column: PlSmallStr,
    /// Start a window at this interval.
    pub every: Duration,
    /// Window duration.
    pub period: Duration,
    /// Offset window boundaries.
    pub offset: Duration,
    /// Truncate the time column values to the window.
    pub label: Label,
    /// Add the boundaries to the DataFrame.
    pub include_boundaries: bool,
    pub closed_window: ClosedWindow,
    pub start_by: StartBy,
}

impl Default for DynamicGroupOptions {
    fn default() -> Self {
        Self {
            index_column: "".into(),
            every: Duration::new(1),
            period: Duration::new(1),
            offset: Duration::new(1),
            label: Label::Left,
            include_boundaries: false,
            closed_window: ClosedWindow::Left,
            start_by: Default::default(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct RollingGroupOptions {
    /// Time or index column.
    pub index_column: PlSmallStr,
    /// Window duration.
    pub period: Duration,
    pub offset: Duration,
    pub closed_window: ClosedWindow,
}

impl Default for RollingGroupOptions {
    fn default() -> Self {
        Self {
            index_column: "".into(),
            period: Duration::new(1),
            offset: Duration::new(1),
            closed_window: ClosedWindow::Left,
        }
    }
}

fn check_sortedness_slice(v: &[i64]) -> PolarsResult<()> {
    polars_ensure!(v.is_sorted_ascending(), ComputeError: "input data is not sorted");
    Ok(())
}

const LB_NAME: &str = "_lower_boundary";
const UP_NAME: &str = "_upper_boundary";

pub trait PolarsTemporalGroupby {
    fn rolling(
        &self,
        group_by: Option<GroupsSlice>,
        options: &RollingGroupOptions,
    ) -> PolarsResult<(Column, GroupPositions)>;

    fn group_by_dynamic(
        &self,
        group_by: Option<GroupsSlice>,
        options: &DynamicGroupOptions,
    ) -> PolarsResult<(Column, Vec<Column>, GroupPositions)>;
}

impl PolarsTemporalGroupby for DataFrame {
    fn rolling(
        &self,
        group_by: Option<GroupsSlice>,
        options: &RollingGroupOptions,
    ) -> PolarsResult<(Column, GroupPositions)> {
        Wrap(self).rolling(group_by, options)
    }

    fn group_by_dynamic(
        &self,
        group_by: Option<GroupsSlice>,
        options: &DynamicGroupOptions,
    ) -> PolarsResult<(Column, Vec<Column>, GroupPositions)> {
        Wrap(self).group_by_dynamic(group_by, options)
    }
}

impl Wrap<&DataFrame> {
    fn rolling(
        &self,
        group_by: Option<GroupsSlice>,
        options: &RollingGroupOptions,
    ) -> PolarsResult<(Column, GroupPositions)> {
        polars_ensure!(
                        !options.period.is_zero() && !options.period.negative,
                        ComputeError:
                        "rolling window period should be strictly positive",
        );
        let time = self.0.column(&options.index_column)?.clone();
        if group_by.is_none() {
            // If by is given, the column must be sorted in the 'by' arg, which we can not check now
            // this will be checked when the groups are materialized.
            time.as_materialized_series().ensure_sorted_arg("rolling")?;
        }
        let time_type = time.dtype();

        polars_ensure!(time.null_count() == 0, ComputeError: "null values in `rolling` not supported, fill nulls.");
        ensure_duration_matches_dtype(options.period, time_type, "period")?;
        ensure_duration_matches_dtype(options.offset, time_type, "offset")?;

        use DataType::*;
        let (dt, tu, tz): (Column, TimeUnit, Option<TimeZone>) = match time_type {
            Datetime(tu, tz) => (time.clone(), *tu, tz.clone()),
            Date => (
                time.cast(&Datetime(TimeUnit::Milliseconds, None))?,
                TimeUnit::Milliseconds,
                None,
            ),
            UInt32 | UInt64 | Int32 => {
                let time_type_dt = Datetime(TimeUnit::Nanoseconds, None);
                let dt = time.cast(&Int64).unwrap().cast(&time_type_dt).unwrap();
                let (out, gt) = self.impl_rolling(
                    dt,
                    group_by,
                    options,
                    TimeUnit::Nanoseconds,
                    None,
                    &time_type_dt,
                )?;
                let out = out.cast(&Int64).unwrap().cast(time_type).unwrap();
                return Ok((out, gt));
            },
            Int64 => {
                let time_type = Datetime(TimeUnit::Nanoseconds, None);
                let dt = time.cast(&time_type).unwrap();
                let (out, gt) = self.impl_rolling(
                    dt,
                    group_by,
                    options,
                    TimeUnit::Nanoseconds,
                    None,
                    &time_type,
                )?;
                let out = out.cast(&Int64).unwrap();
                return Ok((out, gt));
            },
            dt => polars_bail!(
                ComputeError:
                "expected any of the following dtypes: {{ Date, Datetime, Int32, Int64, UInt32, UInt64 }}, got {}",
                dt
            ),
        };
        match tz {
            #[cfg(feature = "timezones")]
            Some(tz) => {
                self.impl_rolling(dt, group_by, options, tu, tz.parse::<Tz>().ok(), time_type)
            },
            _ => self.impl_rolling(dt, group_by, options, tu, None, time_type),
        }
    }

    /// Returns: time_keys, keys, groupsproxy.
    fn group_by_dynamic(
        &self,
        group_by: Option<GroupsSlice>,
        options: &DynamicGroupOptions,
    ) -> PolarsResult<(Column, Vec<Column>, GroupPositions)> {
        let time = self.0.column(&options.index_column)?.rechunk();
        if group_by.is_none() {
            // If by is given, the column must be sorted in the 'by' arg, which we can not check now
            // this will be checked when the groups are materialized.
            time.as_materialized_series()
                .ensure_sorted_arg("group_by_dynamic")?;
        }
        let time_type = time.dtype();

        polars_ensure!(time.null_count() == 0, ComputeError: "null values in dynamic group_by not supported, fill nulls.");
        ensure_duration_matches_dtype(options.every, time_type, "every")?;
        ensure_duration_matches_dtype(options.offset, time_type, "offset")?;
        ensure_duration_matches_dtype(options.period, time_type, "period")?;

        use DataType::*;
        let (dt, tu) = match time_type {
            Datetime(tu, _) => (time.clone(), *tu),
            Date => (
                time.cast(&Datetime(TimeUnit::Milliseconds, None))?,
                TimeUnit::Milliseconds,
            ),
            Int32 => {
                let time_type = Datetime(TimeUnit::Nanoseconds, None);
                let dt = time.cast(&Int64).unwrap().cast(&time_type).unwrap();
                let (out, mut keys, gt) = self.impl_group_by_dynamic(
                    dt,
                    group_by,
                    options,
                    TimeUnit::Nanoseconds,
                    &time_type,
                )?;
                let out = out.cast(&Int64).unwrap().cast(&Int32).unwrap();
                for k in &mut keys {
                    if k.name().as_str() == UP_NAME || k.name().as_str() == LB_NAME {
                        *k = k.cast(&Int64).unwrap().cast(&Int32).unwrap()
                    }
                }
                return Ok((out, keys, gt));
            },
            Int64 => {
                let time_type = Datetime(TimeUnit::Nanoseconds, None);
                let dt = time.cast(&time_type).unwrap();
                let (out, mut keys, gt) = self.impl_group_by_dynamic(
                    dt,
                    group_by,
                    options,
                    TimeUnit::Nanoseconds,
                    &time_type,
                )?;
                let out = out.cast(&Int64).unwrap();
                for k in &mut keys {
                    if k.name().as_str() == UP_NAME || k.name().as_str() == LB_NAME {
                        *k = k.cast(&Int64).unwrap()
                    }
                }
                return Ok((out, keys, gt));
            },
            dt => polars_bail!(
                ComputeError:
                "expected any of the following dtypes: {{ Date, Datetime, Int32, Int64 }}, got {}",
                dt
            ),
        };
        self.impl_group_by_dynamic(dt, group_by, options, tu, time_type)
    }

    fn impl_group_by_dynamic(
        &self,
        mut dt: Column,
        group_by: Option<GroupsSlice>,
        options: &DynamicGroupOptions,
        tu: TimeUnit,
        time_type: &DataType,
    ) -> PolarsResult<(Column, Vec<Column>, GroupPositions)> {
        polars_ensure!(!options.every.negative, ComputeError: "'every' argument must be positive");
        if dt.is_empty() {
            return dt.cast(time_type).map(|s| (s, vec![], Default::default()));
        }

        // A requirement for the index so we can set this such that downstream code has this info.
        dt.set_sorted_flag(IsSorted::Ascending);

        let w = Window::new(options.every, options.period, options.offset);
        let dt = dt.datetime().unwrap();
        let tz = dt.time_zone();

        let mut lower_bound = None;
        let mut upper_bound = None;

        let mut include_lower_bound = false;
        let mut include_upper_bound = false;

        if options.include_boundaries {
            include_lower_bound = true;
            include_upper_bound = true;
        }
        if options.label == Label::Left {
            include_lower_bound = true;
        } else if options.label == Label::Right {
            include_upper_bound = true;
        }

        let mut update_bounds =
            |lower: Vec<i64>, upper: Vec<i64>| match (&mut lower_bound, &mut upper_bound) {
                (None, None) => {
                    lower_bound = Some(lower);
                    upper_bound = Some(upper);
                },
                (Some(lower_bound), Some(upper_bound)) => {
                    lower_bound.extend_from_slice(&lower);
                    upper_bound.extend_from_slice(&upper);
                },
                _ => unreachable!(),
            };

        let groups = if group_by.is_none() {
            let vals = dt.downcast_iter().next().unwrap();
            let ts = vals.values().as_slice();
            let (groups, lower, upper) = group_by_windows(
                w,
                ts,
                options.closed_window,
                tu,
                tz,
                include_lower_bound,
                include_upper_bound,
                options.start_by,
            );
            update_bounds(lower, upper);
            PolarsResult::Ok(GroupsType::Slice {
                groups,
                rolling: false,
            })
        } else {
            let vals = dt.downcast_iter().next().unwrap();
            let ts = vals.values().as_slice();

            let groups = group_by.as_ref().unwrap();

            let iter = groups.par_iter().map(|[start, len]| {
                let group_offset = *start;
                let start = *start as usize;
                let end = start + *len as usize;
                let values = &ts[start..end];
                check_sortedness_slice(values)?;

                let (groups, lower, upper) = group_by_windows(
                    w,
                    values,
                    options.closed_window,
                    tu,
                    tz,
                    include_lower_bound,
                    include_upper_bound,
                    options.start_by,
                );

                PolarsResult::Ok((
                    groups
                        .iter()
                        .map(|[start, len]| [*start + group_offset, *len])
                        .collect_vec(),
                    lower,
                    upper,
                ))
            });

            let res = POOL.install(|| iter.collect::<PolarsResult<Vec<_>>>())?;
            let groups = res.iter().map(|g| &g.0).collect_vec();
            let lower = res.iter().map(|g| &g.1).collect_vec();
            let upper = res.iter().map(|g| &g.2).collect_vec();

            let ((groups, upper), lower) = POOL.install(|| {
                rayon::join(
                    || rayon::join(|| flatten_par(&groups), || flatten_par(&upper)),
                    || flatten_par(&lower),
                )
            });

            update_bounds(lower, upper);
            PolarsResult::Ok(GroupsType::Slice {
                groups,
                rolling: false,
            })
        }?;
        // note that if 'group_by' is none we can be sure that the index column, the lower column and the
        // upper column remain/are sorted

        let dt = unsafe { dt.clone().into_series().agg_first(&groups) };
        let mut dt = dt.datetime().unwrap().as_ref().clone();

        let lower =
            lower_bound.map(|lower| Int64Chunked::new_vec(PlSmallStr::from_static(LB_NAME), lower));
        let upper =
            upper_bound.map(|upper| Int64Chunked::new_vec(PlSmallStr::from_static(UP_NAME), upper));

        if options.label == Label::Left {
            let mut lower = lower.clone().unwrap();
            if group_by.is_none() {
                lower.set_sorted_flag(IsSorted::Ascending)
            }
            dt = lower.with_name(dt.name().clone());
        } else if options.label == Label::Right {
            let mut upper = upper.clone().unwrap();
            if group_by.is_none() {
                upper.set_sorted_flag(IsSorted::Ascending)
            }
            dt = upper.with_name(dt.name().clone());
        }

        let mut bounds = vec![];
        if let (true, Some(mut lower), Some(mut upper)) = (options.include_boundaries, lower, upper)
        {
            if group_by.is_none() {
                lower.set_sorted_flag(IsSorted::Ascending);
                upper.set_sorted_flag(IsSorted::Ascending);
            }
            bounds.push(lower.into_datetime(tu, tz.clone()).into_column());
            bounds.push(upper.into_datetime(tu, tz.clone()).into_column());
        }

        dt.into_datetime(tu, None)
            .into_column()
            .cast(time_type)
            .map(|s| (s, bounds, groups.into_sliceable()))
    }

    /// Returns: time_keys, keys, groupsproxy
    fn impl_rolling(
        &self,
        dt: Column,
        group_by: Option<GroupsSlice>,
        options: &RollingGroupOptions,
        tu: TimeUnit,
        tz: Option<Tz>,
        time_type: &DataType,
    ) -> PolarsResult<(Column, GroupPositions)> {
        let mut dt = dt.rechunk();

        let groups = if group_by.is_none() {
            // a requirement for the index
            // so we can set this such that downstream code has this info
            dt.set_sorted_flag(IsSorted::Ascending);
            let dt = dt.datetime().unwrap();
            let vals = dt.downcast_iter().next().unwrap();
            let ts = vals.values().as_slice();
            PolarsResult::Ok(GroupsType::Slice {
                groups: group_by_values(
                    options.period,
                    options.offset,
                    ts,
                    options.closed_window,
                    tu,
                    tz,
                )?,
                rolling: true,
            })
        } else {
            let dt = dt.datetime().unwrap();
            let vals = dt.downcast_iter().next().unwrap();
            let ts = vals.values().as_slice();

            let groups = group_by.unwrap();

            let iter = groups.into_par_iter().map(|[start, len]| {
                let group_offset = start;
                let start = start as usize;
                let end = start + len as usize;
                let values = &ts[start..end];
                check_sortedness_slice(values)?;

                let group = group_by_values(
                    options.period,
                    options.offset,
                    values,
                    options.closed_window,
                    tu,
                    tz,
                )?;

                PolarsResult::Ok(
                    group
                        .iter()
                        .map(|[start, len]| [*start + group_offset, *len])
                        .collect_vec(),
                )
            });

            let groups = POOL.install(|| iter.collect::<PolarsResult<Vec<_>>>())?;
            let groups = POOL.install(|| flatten_par(&groups));
            PolarsResult::Ok(GroupsType::Slice {
                groups,
                rolling: true,
            })
        }?;

        let dt = dt.cast(time_type).unwrap();

        Ok((dt, groups.into_sliceable()))
    }
}

#[cfg(test)]
mod test {
    use polars_compute::rolling::QuantileMethod;
    use polars_ops::prelude::*;

    use super::*;

    #[test]
    fn test_rolling_group_by_tu() -> PolarsResult<()> {
        // test multiple time units
        for tu in [
            TimeUnit::Nanoseconds,
            TimeUnit::Microseconds,
            TimeUnit::Milliseconds,
        ] {
            let mut date = StringChunked::new(
                "dt".into(),
                [
                    "2020-01-01 13:45:48",
                    "2020-01-01 16:42:13",
                    "2020-01-01 16:45:09",
                    "2020-01-02 18:12:48",
                    "2020-01-03 19:45:32",
                    "2020-01-08 23:16:43",
                ],
            )
            .as_datetime(
                None,
                tu,
                false,
                false,
                None,
                &StringChunked::from_iter(std::iter::once("raise")),
            )?
            .into_column();
            date.set_sorted_flag(IsSorted::Ascending);
            let a = Column::new("a".into(), [3, 7, 5, 9, 2, 1]);
            let df = DataFrame::new(vec![date, a.clone()])?;

            let (_, groups) = df
                .rolling(
                    None,
                    &RollingGroupOptions {
                        index_column: "dt".into(),
                        period: Duration::parse("2d"),
                        offset: Duration::parse("-2d"),
                        closed_window: ClosedWindow::Right,
                    },
                )
                .unwrap();

            let sum = unsafe { a.agg_sum(&groups) };
            let expected = Column::new("".into(), [3, 10, 15, 24, 11, 1]);
            assert_eq!(sum, expected);
        }

        Ok(())
    }

    #[test]
    fn test_rolling_group_by_aggs() -> PolarsResult<()> {
        let mut date = StringChunked::new(
            "dt".into(),
            [
                "2020-01-01 13:45:48",
                "2020-01-01 16:42:13",
                "2020-01-01 16:45:09",
                "2020-01-02 18:12:48",
                "2020-01-03 19:45:32",
                "2020-01-08 23:16:43",
            ],
        )
        .as_datetime(
            None,
            TimeUnit::Milliseconds,
            false,
            false,
            None,
            &StringChunked::from_iter(std::iter::once("raise")),
        )?
        .into_column();
        date.set_sorted_flag(IsSorted::Ascending);

        let a = Column::new("a".into(), [3, 7, 5, 9, 2, 1]);
        let df = DataFrame::new(vec![date, a.clone()])?;

        let (_, groups) = df
            .rolling(
                None,
                &RollingGroupOptions {
                    index_column: "dt".into(),
                    period: Duration::parse("2d"),
                    offset: Duration::parse("-2d"),
                    closed_window: ClosedWindow::Right,
                },
            )
            .unwrap();

        let nulls = Series::new(
            "".into(),
            [Some(3), Some(7), None, Some(9), Some(2), Some(1)],
        );

        let min = unsafe { a.as_materialized_series().agg_min(&groups) };
        let expected = Series::new("".into(), [3, 3, 3, 3, 2, 1]);
        assert_eq!(min, expected);

        // Expected for nulls is equality.
        let min = unsafe { nulls.agg_min(&groups) };
        assert_eq!(min, expected);

        let max = unsafe { a.as_materialized_series().agg_max(&groups) };
        let expected = Series::new("".into(), [3, 7, 7, 9, 9, 1]);
        assert_eq!(max, expected);

        let max = unsafe { nulls.agg_max(&groups) };
        assert_eq!(max, expected);

        let var = unsafe { a.as_materialized_series().agg_var(&groups, 1) };
        let expected = Series::new(
            "".into(),
            [0.0, 8.0, 4.000000000000002, 6.666666666666667, 24.5, 0.0],
        );
        assert!(abs(&(var - expected)?).unwrap().lt(1e-12).unwrap().all());

        let var = unsafe { nulls.agg_var(&groups, 1) };
        let expected = Series::new("".into(), [0.0, 8.0, 8.0, 9.333333333333343, 24.5, 0.0]);
        assert!(abs(&(var - expected)?).unwrap().lt(1e-12).unwrap().all());

        let quantile = unsafe {
            a.as_materialized_series()
                .agg_quantile(&groups, 0.5, QuantileMethod::Linear)
        };
        let expected = Series::new("".into(), [3.0, 5.0, 5.0, 6.0, 5.5, 1.0]);
        assert_eq!(quantile, expected);

        let quantile = unsafe { nulls.agg_quantile(&groups, 0.5, QuantileMethod::Linear) };
        let expected = Series::new("".into(), [3.0, 5.0, 5.0, 7.0, 5.5, 1.0]);
        assert_eq!(quantile, expected);

        Ok(())
    }
}
