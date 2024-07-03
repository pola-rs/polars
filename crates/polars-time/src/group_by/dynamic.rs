use arrow::legacy::time_zone::Tz;
use arrow::legacy::utils::CustomIterTools;
use polars_core::export::rayon::prelude::*;
use polars_core::prelude::*;
use polars_core::series::IsSorted;
use polars_core::utils::flatten::flatten_par;
use polars_core::POOL;
use polars_ops::series::SeriesMethods;
use polars_utils::idx_vec::IdxVec;
use polars_utils::slice::{GetSaferUnchecked, SortedSlice};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use smartstring::alias::String as SmartString;

use crate::prelude::*;

#[repr(transparent)]
struct Wrap<T>(pub T);

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct DynamicGroupOptions {
    /// Time or index column.
    pub index_column: SmartString,
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
    pub index_column: SmartString,
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
        group_by: Vec<Series>,
        options: &RollingGroupOptions,
    ) -> PolarsResult<(Series, Vec<Series>, GroupsProxy)>;

    fn group_by_dynamic(
        &self,
        group_by: Vec<Series>,
        options: &DynamicGroupOptions,
    ) -> PolarsResult<(Series, Vec<Series>, GroupsProxy)>;
}

impl PolarsTemporalGroupby for DataFrame {
    fn rolling(
        &self,
        group_by: Vec<Series>,
        options: &RollingGroupOptions,
    ) -> PolarsResult<(Series, Vec<Series>, GroupsProxy)> {
        Wrap(self).rolling(group_by, options)
    }

    fn group_by_dynamic(
        &self,
        group_by: Vec<Series>,
        options: &DynamicGroupOptions,
    ) -> PolarsResult<(Series, Vec<Series>, GroupsProxy)> {
        Wrap(self).group_by_dynamic(group_by, options)
    }
}

impl Wrap<&DataFrame> {
    fn rolling(
        &self,
        group_by: Vec<Series>,
        options: &RollingGroupOptions,
    ) -> PolarsResult<(Series, Vec<Series>, GroupsProxy)> {
        polars_ensure!(
                        !options.period.is_zero() && !options.period.negative,
                        ComputeError:
                        "rolling window period should be strictly positive",
        );
        let time = self.0.column(&options.index_column)?.clone();
        if group_by.is_empty() {
            // If by is given, the column must be sorted in the 'by' arg, which we can not check now
            // this will be checked when the groups are materialized.
            time.ensure_sorted_arg("rolling")?;
        }
        let time_type = time.dtype();

        polars_ensure!(time.null_count() == 0, ComputeError: "null values in `rolling` not supported, fill nulls.");
        ensure_duration_matches_data_type(options.period, time_type, "period")?;
        ensure_duration_matches_data_type(options.offset, time_type, "offset")?;

        use DataType::*;
        let (dt, tu, tz): (Series, TimeUnit, Option<TimeZone>) = match time_type {
            Datetime(tu, tz) => (time.clone(), *tu, tz.clone()),
            Date => (
                time.cast(&Datetime(TimeUnit::Milliseconds, None))?,
                TimeUnit::Milliseconds,
                None,
            ),
            UInt32 | UInt64 | Int32 => {
                let time_type_dt = Datetime(TimeUnit::Nanoseconds, None);
                let dt = time.cast(&Int64).unwrap().cast(&time_type_dt).unwrap();
                let (out, by, gt) = self.impl_rolling(
                    dt,
                    group_by,
                    options,
                    TimeUnit::Nanoseconds,
                    None,
                    &time_type_dt,
                )?;
                let out = out.cast(&Int64).unwrap().cast(time_type).unwrap();
                return Ok((out, by, gt));
            },
            Int64 => {
                let time_type = Datetime(TimeUnit::Nanoseconds, None);
                let dt = time.cast(&time_type).unwrap();
                let (out, by, gt) = self.impl_rolling(
                    dt,
                    group_by,
                    options,
                    TimeUnit::Nanoseconds,
                    None,
                    &time_type,
                )?;
                let out = out.cast(&Int64).unwrap();
                return Ok((out, by, gt));
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
        group_by: Vec<Series>,
        options: &DynamicGroupOptions,
    ) -> PolarsResult<(Series, Vec<Series>, GroupsProxy)> {
        let time = self.0.column(&options.index_column)?.rechunk();
        if group_by.is_empty() {
            // If by is given, the column must be sorted in the 'by' arg, which we can not check now
            // this will be checked when the groups are materialized.
            time.ensure_sorted_arg("group_by_dynamic")?;
        }
        let time_type = time.dtype();

        polars_ensure!(time.null_count() == 0, ComputeError: "null values in dynamic group_by not supported, fill nulls.");
        ensure_duration_matches_data_type(options.every, time_type, "every")?;
        ensure_duration_matches_data_type(options.offset, time_type, "offset")?;
        ensure_duration_matches_data_type(options.period, time_type, "period")?;

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
                    if k.name() == UP_NAME || k.name() == LB_NAME {
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
                    if k.name() == UP_NAME || k.name() == LB_NAME {
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
        mut dt: Series,
        mut by: Vec<Series>,
        options: &DynamicGroupOptions,
        tu: TimeUnit,
        time_type: &DataType,
    ) -> PolarsResult<(Series, Vec<Series>, GroupsProxy)> {
        polars_ensure!(!options.every.negative, ComputeError: "'every' argument must be positive");
        if dt.is_empty() {
            return dt.cast(time_type).map(|s| (s, by, GroupsProxy::default()));
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

        let groups = if by.is_empty() {
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
            PolarsResult::Ok(GroupsProxy::Slice {
                groups,
                rolling: false,
            })
        } else {
            let groups = self
                .0
                .group_by_with_series(by.clone(), true, true)?
                .take_groups();

            // Include boundaries cannot be parallel (easily).
            if include_lower_bound | include_upper_bound {
                POOL.install(|| match groups {
                    GroupsProxy::Idx(groups) => {
                        let ir = groups
                            .par_iter()
                            .map(|base_g| {
                                let dt = unsafe { dt.take_unchecked(base_g.1) };
                                let vals = dt.downcast_iter().next().unwrap();
                                let ts = vals.values().as_slice();
                                if !matches!(dt.is_sorted_flag(), IsSorted::Ascending) {
                                    check_sortedness_slice(ts)?
                                }
                                let (sub_groups, lower, upper) = group_by_windows(
                                    w,
                                    ts,
                                    options.closed_window,
                                    tu,
                                    tz,
                                    include_lower_bound,
                                    include_upper_bound,
                                    options.start_by,
                                );

                                Ok((lower, upper, update_subgroups_idx(&sub_groups, base_g)))
                            })
                            .collect::<PolarsResult<Vec<_>>>()?;

                        // flatten in 2 stages
                        // first flatten the groups
                        let mut groups = Vec::with_capacity(ir.len());

                        ir.into_iter().for_each(|(lower, upper, g)| {
                            update_bounds(lower, upper);
                            groups.push(g);
                        });

                        // then parallelize the flatten in the `from` impl
                        Ok(GroupsProxy::Idx(GroupsIdx::from(groups)))
                    },
                    GroupsProxy::Slice { groups, .. } => {
                        let mut ir = groups
                            .par_iter()
                            .map(|base_g| {
                                let dt = dt.slice(base_g[0] as i64, base_g[1] as usize);
                                let vals = dt.downcast_iter().next().unwrap();
                                let ts = vals.values().as_slice();
                                let (sub_groups, lower, upper) = group_by_windows(
                                    w,
                                    ts,
                                    options.closed_window,
                                    tu,
                                    tz,
                                    include_lower_bound,
                                    include_upper_bound,
                                    options.start_by,
                                );
                                (lower, upper, update_subgroups_slice(&sub_groups, *base_g))
                            })
                            .collect::<Vec<_>>();

                        let mut capacity = 0;
                        ir.iter_mut().for_each(|(lower, upper, g)| {
                            let lower = std::mem::take(lower);
                            let upper = std::mem::take(upper);
                            update_bounds(lower, upper);
                            capacity += g.len();
                        });

                        let mut groups = Vec::with_capacity(capacity);
                        ir.iter().for_each(|(_, _, g)| groups.extend_from_slice(g));

                        Ok(GroupsProxy::Slice {
                            groups,
                            rolling: false,
                        })
                    },
                })
            } else {
                POOL.install(|| match groups {
                    GroupsProxy::Idx(groups) => {
                        let groupsidx = groups
                            .par_iter()
                            .map(|base_g| {
                                let dt = unsafe { dt.take_unchecked(base_g.1) };
                                let vals = dt.downcast_iter().next().unwrap();
                                let ts = vals.values().as_slice();
                                if !matches!(dt.is_sorted_flag(), IsSorted::Ascending) {
                                    check_sortedness_slice(ts)?
                                }
                                let (sub_groups, _, _) = group_by_windows(
                                    w,
                                    ts,
                                    options.closed_window,
                                    tu,
                                    tz,
                                    include_lower_bound,
                                    include_upper_bound,
                                    options.start_by,
                                );
                                Ok(update_subgroups_idx(&sub_groups, base_g))
                            })
                            .collect::<PolarsResult<Vec<_>>>()?;
                        Ok(GroupsProxy::Idx(GroupsIdx::from(groupsidx)))
                    },
                    GroupsProxy::Slice { groups, .. } => {
                        let groups = groups
                            .par_iter()
                            .map(|base_g| {
                                let dt = dt.slice(base_g[0] as i64, base_g[1] as usize);
                                let vals = dt.downcast_iter().next().unwrap();
                                let ts = vals.values().as_slice();
                                let (sub_groups, _, _) = group_by_windows(
                                    w,
                                    ts,
                                    options.closed_window,
                                    tu,
                                    tz,
                                    include_lower_bound,
                                    include_upper_bound,
                                    options.start_by,
                                );
                                update_subgroups_slice(&sub_groups, *base_g)
                            })
                            .collect::<Vec<_>>();

                        let groups = flatten_par(&groups);

                        Ok(GroupsProxy::Slice {
                            groups,
                            rolling: false,
                        })
                    },
                })
            }
        }?;
        // note that if 'by' is empty we can be sure that the index column, the lower column and the
        // upper column remain/are sorted

        let dt = unsafe { dt.clone().into_series().agg_first(&groups) };
        let mut dt = dt.datetime().unwrap().as_ref().clone();
        for key in by.iter_mut() {
            *key = unsafe { key.agg_first(&groups) };
        }

        let lower = lower_bound.map(|lower| Int64Chunked::new_vec(LB_NAME, lower));
        let upper = upper_bound.map(|upper| Int64Chunked::new_vec(UP_NAME, upper));

        if options.label == Label::Left {
            let mut lower = lower.clone().unwrap();
            if by.is_empty() {
                lower.set_sorted_flag(IsSorted::Ascending)
            }
            dt = lower.with_name(dt.name());
        } else if options.label == Label::Right {
            let mut upper = upper.clone().unwrap();
            if by.is_empty() {
                upper.set_sorted_flag(IsSorted::Ascending)
            }
            dt = upper.with_name(dt.name());
        }

        if let (true, Some(mut lower), Some(mut upper)) = (options.include_boundaries, lower, upper)
        {
            if by.is_empty() {
                lower.set_sorted_flag(IsSorted::Ascending);
                upper.set_sorted_flag(IsSorted::Ascending);
            }
            by.push(lower.into_datetime(tu, tz.clone()).into_series());
            by.push(upper.into_datetime(tu, tz.clone()).into_series());
        }

        dt.into_datetime(tu, None)
            .into_series()
            .cast(time_type)
            .map(|s| (s, by, groups))
    }

    /// Returns: time_keys, keys, groupsproxy
    fn impl_rolling(
        &self,
        dt: Series,
        group_by: Vec<Series>,
        options: &RollingGroupOptions,
        tu: TimeUnit,
        tz: Option<Tz>,
        time_type: &DataType,
    ) -> PolarsResult<(Series, Vec<Series>, GroupsProxy)> {
        let mut dt = dt.rechunk();

        let groups = if group_by.is_empty() {
            // a requirement for the index
            // so we can set this such that downstream code has this info
            dt.set_sorted_flag(IsSorted::Ascending);
            let dt = dt.datetime().unwrap();
            let vals = dt.downcast_iter().next().unwrap();
            let ts = vals.values().as_slice();
            PolarsResult::Ok(GroupsProxy::Slice {
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
            let groups = self
                .0
                .group_by_with_series(group_by.clone(), true, true)?
                .take_groups();

            // we keep a local copy, as we are reordering on next operation.
            let dt_local = dt.datetime().unwrap().clone();

            // make sure that the output order is correct
            dt = unsafe { dt.agg_list(&groups).explode().unwrap() };

            // continue determining the rolling indexes.

            POOL.install(|| match groups {
                GroupsProxy::Idx(groups) => {
                    let idx = groups
                        .par_iter()
                        .map(|base_g| {
                            let dt = unsafe { dt_local.take_unchecked(base_g.1) };
                            let vals = dt.downcast_iter().next().unwrap();
                            let ts = vals.values().as_slice();
                            if !matches!(dt.is_sorted_flag(), IsSorted::Ascending) {
                                check_sortedness_slice(ts)?
                            }

                            let sub_groups = group_by_values(
                                options.period,
                                options.offset,
                                ts,
                                options.closed_window,
                                tu,
                                tz,
                            )?;
                            Ok(update_subgroups_idx(&sub_groups, base_g))
                        })
                        .collect::<PolarsResult<Vec<_>>>()?;

                    Ok(GroupsProxy::Idx(GroupsIdx::from(idx)))
                },
                GroupsProxy::Slice { groups, .. } => {
                    let slice_groups = groups
                        .par_iter()
                        .map(|base_g| {
                            let dt = dt_local.slice(base_g[0] as i64, base_g[1] as usize);
                            let vals = dt.downcast_iter().next().unwrap();
                            let ts = vals.values().as_slice();
                            let sub_groups = group_by_values(
                                options.period,
                                options.offset,
                                ts,
                                options.closed_window,
                                tu,
                                tz,
                            )?;
                            Ok(update_subgroups_slice(&sub_groups, *base_g))
                        })
                        .collect::<PolarsResult<Vec<_>>>()?;

                    let slice_groups = flatten_par(&slice_groups);
                    Ok(GroupsProxy::Slice {
                        groups: slice_groups,
                        rolling: false,
                    })
                },
            })
        }?;

        let dt = dt.cast(time_type).unwrap();

        Ok((dt, group_by, groups))
    }
}

fn update_subgroups_slice(sub_groups: &[[IdxSize; 2]], base_g: [IdxSize; 2]) -> Vec<[IdxSize; 2]> {
    sub_groups
        .iter()
        .map(|&[first, len]| {
            let new_first = if len == 0 {
                // In case the group is empty, keep the original first so that the
                // group_by keys still point to the original group.
                base_g[0]
            } else {
                base_g[0] + first
            };
            [new_first, len]
        })
        .collect_trusted::<Vec<_>>()
}

fn update_subgroups_idx(
    sub_groups: &[[IdxSize; 2]],
    base_g: (IdxSize, &IdxVec),
) -> Vec<(IdxSize, IdxVec)> {
    sub_groups
        .iter()
        .map(|&[first, len]| {
            let new_first = if len == 0 {
                // In case the group is empty, keep the original first so that the
                // group_by keys still point to the original group.
                base_g.0
            } else {
                unsafe { *base_g.1.get_unchecked_release(first as usize) }
            };

            let first = first as usize;
            let len = len as usize;
            let idx = (first..first + len)
                .map(|i| unsafe { *base_g.1.get_unchecked_release(i) })
                .collect::<IdxVec>();
            (new_first, idx)
        })
        .collect_trusted::<Vec<_>>()
}

#[cfg(test)]
mod test {
    use chrono::prelude::*;
    use polars_ops::prelude::*;
    use polars_utils::unitvec;

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
                "dt",
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
            .into_series();
            date.set_sorted_flag(IsSorted::Ascending);
            let a = Series::new("a", [3, 7, 5, 9, 2, 1]);
            let df = DataFrame::new(vec![date, a.clone()])?;

            let (_, _, groups) = df
                .rolling(
                    vec![],
                    &RollingGroupOptions {
                        index_column: "dt".into(),
                        period: Duration::parse("2d"),
                        offset: Duration::parse("-2d"),
                        closed_window: ClosedWindow::Right,
                    },
                )
                .unwrap();

            let sum = unsafe { a.agg_sum(&groups) };
            let expected = Series::new("", [3, 10, 15, 24, 11, 1]);
            assert_eq!(sum, expected);
        }

        Ok(())
    }

    #[test]
    fn test_rolling_group_by_aggs() -> PolarsResult<()> {
        let mut date = StringChunked::new(
            "dt",
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
        .into_series();
        date.set_sorted_flag(IsSorted::Ascending);

        let a = Series::new("a", [3, 7, 5, 9, 2, 1]);
        let df = DataFrame::new(vec![date, a.clone()])?;

        let (_, _, groups) = df
            .rolling(
                vec![],
                &RollingGroupOptions {
                    index_column: "dt".into(),
                    period: Duration::parse("2d"),
                    offset: Duration::parse("-2d"),
                    closed_window: ClosedWindow::Right,
                },
            )
            .unwrap();

        let nulls = Series::new("", [Some(3), Some(7), None, Some(9), Some(2), Some(1)]);

        let min = unsafe { a.agg_min(&groups) };
        let expected = Series::new("", [3, 3, 3, 3, 2, 1]);
        assert_eq!(min, expected);

        // Expected for nulls is equality.
        let min = unsafe { nulls.agg_min(&groups) };
        assert_eq!(min, expected);

        let max = unsafe { a.agg_max(&groups) };
        let expected = Series::new("", [3, 7, 7, 9, 9, 1]);
        assert_eq!(max, expected);

        let max = unsafe { nulls.agg_max(&groups) };
        assert_eq!(max, expected);

        let var = unsafe { a.agg_var(&groups, 1) };
        let expected = Series::new(
            "",
            [0.0, 8.0, 4.000000000000002, 6.666666666666667, 24.5, 0.0],
        );
        assert!(abs(&(var - expected)?).unwrap().lt(1e-12).unwrap().all());

        let var = unsafe { nulls.agg_var(&groups, 1) };
        let expected = Series::new("", [0.0, 8.0, 8.0, 9.333333333333343, 24.5, 0.0]);
        assert!(abs(&(var - expected)?).unwrap().lt(1e-12).unwrap().all());

        let quantile = unsafe { a.agg_quantile(&groups, 0.5, QuantileInterpolOptions::Linear) };
        let expected = Series::new("", [3.0, 5.0, 5.0, 6.0, 5.5, 1.0]);
        assert_eq!(quantile, expected);

        let quantile = unsafe { nulls.agg_quantile(&groups, 0.5, QuantileInterpolOptions::Linear) };
        let expected = Series::new("", [3.0, 5.0, 5.0, 7.0, 5.5, 1.0]);
        assert_eq!(quantile, expected);

        Ok(())
    }

    #[test]
    fn test_dynamic_group_by_window() -> PolarsResult<()> {
        let start = NaiveDate::from_ymd_opt(2021, 12, 16)
            .unwrap()
            .and_hms_opt(0, 0, 0)
            .unwrap()
            .and_utc()
            .timestamp_millis();
        let stop = NaiveDate::from_ymd_opt(2021, 12, 16)
            .unwrap()
            .and_hms_opt(3, 0, 0)
            .unwrap()
            .and_utc()
            .timestamp_millis();
        let range = datetime_range_impl(
            "date",
            start,
            stop,
            Duration::parse("30m"),
            ClosedWindow::Both,
            TimeUnit::Milliseconds,
            None,
        )?
        .into_series();

        let groups = Series::new("groups", ["a", "a", "a", "b", "b", "a", "a"]);
        let df = DataFrame::new(vec![range, groups.clone()]).unwrap();

        let (time_key, mut keys, groups) = df
            .group_by_dynamic(
                vec![groups],
                &DynamicGroupOptions {
                    index_column: "date".into(),
                    every: Duration::parse("1h"),
                    period: Duration::parse("1h"),
                    offset: Duration::parse("0h"),
                    label: Label::Left,
                    include_boundaries: true,
                    closed_window: ClosedWindow::Both,
                    start_by: Default::default(),
                },
            )
            .unwrap();

        let ca = time_key.datetime().unwrap();
        let years = ca.year();
        assert_eq!(years.get(0), Some(2021i32));

        keys.push(time_key);
        let out = DataFrame::new(keys).unwrap();
        let g = out.column("groups").unwrap();
        let g = g.str().unwrap();
        let g = g.into_no_null_iter().collect::<Vec<_>>();
        assert_eq!(g, &["a", "a", "a", "a", "b", "b"]);

        let upper = out.column("_upper_boundary").unwrap().slice(0, 3);
        let start = NaiveDate::from_ymd_opt(2021, 12, 16)
            .unwrap()
            .and_hms_opt(1, 0, 0)
            .unwrap()
            .and_utc()
            .timestamp_millis();
        let stop = NaiveDate::from_ymd_opt(2021, 12, 16)
            .unwrap()
            .and_hms_opt(3, 0, 0)
            .unwrap()
            .and_utc()
            .timestamp_millis();
        let range = datetime_range_impl(
            "_upper_boundary",
            start,
            stop,
            Duration::parse("1h"),
            ClosedWindow::Both,
            TimeUnit::Milliseconds,
            None,
        )?
        .into_series();
        assert_eq!(&upper, &range);

        let upper = out.column("_lower_boundary").unwrap().slice(0, 3);
        let start = NaiveDate::from_ymd_opt(2021, 12, 16)
            .unwrap()
            .and_hms_opt(0, 0, 0)
            .unwrap()
            .and_utc()
            .timestamp_millis();
        let stop = NaiveDate::from_ymd_opt(2021, 12, 16)
            .unwrap()
            .and_hms_opt(2, 0, 0)
            .unwrap()
            .and_utc()
            .timestamp_millis();
        let range = datetime_range_impl(
            "_lower_boundary",
            start,
            stop,
            Duration::parse("1h"),
            ClosedWindow::Both,
            TimeUnit::Milliseconds,
            None,
        )?
        .into_series();
        assert_eq!(&upper, &range);

        let expected = GroupsProxy::Idx(
            vec![
                (0 as IdxSize, unitvec![0 as IdxSize, 1, 2]),
                (2, unitvec![2]),
                (5, unitvec![5, 6]),
                (6, unitvec![6]),
                (3, unitvec![3, 4]),
                (4, unitvec![4]),
            ]
            .into(),
        );
        assert_eq!(expected, groups);
        Ok(())
    }

    #[test]
    fn test_truncate_offset() -> PolarsResult<()> {
        let start = NaiveDate::from_ymd_opt(2021, 3, 1)
            .unwrap()
            .and_hms_opt(12, 0, 0)
            .unwrap()
            .and_utc()
            .timestamp_millis();
        let stop = NaiveDate::from_ymd_opt(2021, 3, 7)
            .unwrap()
            .and_hms_opt(12, 0, 0)
            .unwrap()
            .and_utc()
            .timestamp_millis();
        let range = datetime_range_impl(
            "date",
            start,
            stop,
            Duration::parse("1d"),
            ClosedWindow::Both,
            TimeUnit::Milliseconds,
            None,
        )?
        .into_series();

        let groups = Series::new("groups", ["a", "a", "a", "b", "b", "a", "a"]);
        let df = DataFrame::new(vec![range, groups.clone()]).unwrap();

        let (mut time_key, keys, _groups) = df
            .group_by_dynamic(
                vec![groups],
                &DynamicGroupOptions {
                    index_column: "date".into(),
                    every: Duration::parse("6d"),
                    period: Duration::parse("6d"),
                    offset: Duration::parse("0h"),
                    label: Label::Left,
                    include_boundaries: true,
                    closed_window: ClosedWindow::Both,
                    start_by: Default::default(),
                },
            )
            .unwrap();
        time_key.rename("");
        let lower_bound = keys[1].clone().with_name("");
        assert!(time_key.equals(&lower_bound));
        Ok(())
    }
}
