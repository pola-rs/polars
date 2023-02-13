use polars_arrow::utils::CustomIterTools;
use polars_core::export::rayon::prelude::*;
use polars_core::frame::groupby::GroupsProxy;
use polars_core::prelude::*;
use polars_core::POOL;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::prelude::*;

#[repr(transparent)]
struct Wrap<T>(pub T);

#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct DynamicGroupOptions {
    /// Time or index column
    pub index_column: String,
    /// start a window at this interval
    pub every: Duration,
    /// window duration
    pub period: Duration,
    /// offset window boundaries
    pub offset: Duration,
    /// truncate the time column values to the window
    pub truncate: bool,
    // add the boundaries to the dataframe
    pub include_boundaries: bool,
    pub closed_window: ClosedWindow,
    pub start_by: StartBy,
}

impl Default for DynamicGroupOptions {
    fn default() -> Self {
        Self {
            index_column: "".to_string(),
            every: Duration::new(1),
            period: Duration::new(1),
            offset: Duration::new(1),
            truncate: true,
            include_boundaries: false,
            closed_window: ClosedWindow::Left,
            start_by: Default::default(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct RollingGroupOptions {
    /// Time or index column
    pub index_column: String,
    /// window duration
    pub period: Duration,
    pub offset: Duration,
    pub closed_window: ClosedWindow,
}

const LB_NAME: &str = "_lower_boundary";
const UP_NAME: &str = "_upper_boundary";

pub trait PolarsTemporalGroupby {
    fn groupby_rolling(
        &self,
        by: Vec<Series>,
        options: &RollingGroupOptions,
    ) -> PolarsResult<(Series, Vec<Series>, GroupsProxy)>;

    fn groupby_dynamic(
        &self,
        by: Vec<Series>,
        options: &DynamicGroupOptions,
    ) -> PolarsResult<(Series, Vec<Series>, GroupsProxy)>;
}

impl PolarsTemporalGroupby for DataFrame {
    fn groupby_rolling(
        &self,
        by: Vec<Series>,
        options: &RollingGroupOptions,
    ) -> PolarsResult<(Series, Vec<Series>, GroupsProxy)> {
        Wrap(self).groupby_rolling(by, options)
    }

    fn groupby_dynamic(
        &self,
        by: Vec<Series>,
        options: &DynamicGroupOptions,
    ) -> PolarsResult<(Series, Vec<Series>, GroupsProxy)> {
        Wrap(self).groupby_dynamic(by, options)
    }
}

impl Wrap<&DataFrame> {
    fn groupby_rolling(
        &self,
        by: Vec<Series>,
        options: &RollingGroupOptions,
    ) -> PolarsResult<(Series, Vec<Series>, GroupsProxy)> {
        let time = self.0.column(&options.index_column)?;
        let time_type = time.dtype();

        if time.null_count() > 0 {
            panic!("null values in dynamic groupby not yet supported, fill nulls.")
        }

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
                let (out, by, gt) =
                    self.impl_groupby_rolling(dt, by, options, TimeUnit::Nanoseconds, &time_type)?;
                let out = out.cast(&Int64).unwrap().cast(&Int32).unwrap();
                return Ok((out, by, gt));
            }
            Int64 => {
                let time_type = Datetime(TimeUnit::Nanoseconds, None);
                let dt = time.cast(&time_type).unwrap();
                let (out, by, gt) =
                    self.impl_groupby_rolling(dt, by, options, TimeUnit::Nanoseconds, &time_type)?;
                let out = out.cast(&Int64).unwrap();
                return Ok((out, by, gt));
            }
            dt => {
                return Err(PolarsError::ComputeError(
                    format!(
                    "expected any of the following dtypes {{Date, Datetime, Int32, Int64}}, got {dt}",
                )
                    .into(),
                ))
            }
        };
        self.impl_groupby_rolling(dt, by, options, tu, time_type)
    }

    /// Returns: time_keys, keys, groupsproxy
    fn groupby_dynamic(
        &self,
        by: Vec<Series>,
        options: &DynamicGroupOptions,
    ) -> PolarsResult<(Series, Vec<Series>, GroupsProxy)> {
        if options.offset.parsed_int || options.every.parsed_int || options.period.parsed_int {
            assert!(
                (options.offset.parsed_int || options.offset.is_zero())
                    && (options.every.parsed_int || options.every.is_zero())
                    && (options.period.parsed_int || options.period.is_zero()),
                "you cannot combine time durations like '2h' with integer durations like '3i'"
            )
        }

        let time = self.0.column(&options.index_column)?.rechunk();
        let time_type = time.dtype();

        if time.null_count() > 0 {
            panic!("null values in dynamic groupby not yet supported, fill nulls.")
        }

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
                let (out, mut keys, gt) =
                    self.impl_groupby_dynamic(dt, by, options, TimeUnit::Nanoseconds, &time_type)?;
                let out = out.cast(&Int64).unwrap().cast(&Int32).unwrap();
                for k in &mut keys {
                    if k.name() == UP_NAME || k.name() == LB_NAME {
                        *k = k.cast(&Int64).unwrap().cast(&Int32).unwrap()
                    }
                }
                return Ok((out, keys, gt));
            }
            Int64 => {
                let time_type = Datetime(TimeUnit::Nanoseconds, None);
                let dt = time.cast(&time_type).unwrap();
                let (out, mut keys, gt) =
                    self.impl_groupby_dynamic(dt, by, options, TimeUnit::Nanoseconds, &time_type)?;
                let out = out.cast(&Int64).unwrap();
                for k in &mut keys {
                    if k.name() == UP_NAME || k.name() == LB_NAME {
                        *k = k.cast(&Int64).unwrap()
                    }
                }
                return Ok((out, keys, gt));
            }
            dt => {
                return Err(PolarsError::ComputeError(
                    format!(
                    "expected any of the following dtypes {{Date, Datetime, Int32, Int64}}, got {dt}",
                )
                    .into(),
                ))
            }
        };
        self.impl_groupby_dynamic(dt, by, options, tu, time_type)
    }

    fn impl_groupby_dynamic(
        &self,
        dt: Series,
        mut by: Vec<Series>,
        options: &DynamicGroupOptions,
        tu: TimeUnit,
        time_type: &DataType,
    ) -> PolarsResult<(Series, Vec<Series>, GroupsProxy)> {
        if dt.is_empty() {
            return dt.cast(time_type).map(|s| (s, by, GroupsProxy::default()));
        }

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
        if options.truncate {
            include_lower_bound = true;
        }

        let mut update_bounds =
            |lower: Vec<i64>, upper: Vec<i64>| match (&mut lower_bound, &mut upper_bound) {
                (None, None) => {
                    lower_bound = Some(lower);
                    upper_bound = Some(upper);
                }
                (Some(lower_bound), Some(upper_bound)) => {
                    lower_bound.extend_from_slice(&lower);
                    upper_bound.extend_from_slice(&upper);
                }
                _ => unreachable!(),
            };

        let groups = if by.is_empty() {
            let vals = dt.downcast_iter().next().unwrap();
            let ts = vals.values().as_slice();
            partially_check_sorted(ts);
            let (groups, lower, upper) = groupby_windows(
                w,
                ts,
                options.closed_window,
                tu,
                include_lower_bound,
                include_upper_bound,
                options.start_by,
            );
            update_bounds(lower, upper);
            GroupsProxy::Slice {
                groups,
                rolling: false,
            }
        } else {
            let groups = self
                .0
                .groupby_with_series(by.clone(), true, true)?
                .take_groups();

            // include boundaries cannot be parallel (easily)
            if include_lower_bound {
                POOL.install(|| match groups {
                    GroupsProxy::Idx(groups) => {
                        let mut ir = groups
                            .par_iter()
                            .map(|base_g| {
                                let dt = unsafe { dt.take_unchecked(base_g.1.into()) };

                                let vals = dt.downcast_iter().next().unwrap();
                                let ts = vals.values().as_slice();
                                let (sub_groups, lower, upper) = groupby_windows(
                                    w,
                                    ts,
                                    options.closed_window,
                                    tu,
                                    include_lower_bound,
                                    include_upper_bound,
                                    options.start_by,
                                );

                                (lower, upper, update_subgroups_idx(&sub_groups, base_g))
                            })
                            .collect::<Vec<_>>();

                        ir.iter_mut().for_each(|(lower, upper, _)| {
                            let lower = std::mem::take(lower);
                            let upper = std::mem::take(upper);
                            update_bounds(lower, upper)
                        });

                        GroupsProxy::Idx(ir.into_iter().flat_map(|(_, _, groups)| groups).collect())
                    }
                    GroupsProxy::Slice { groups, .. } => {
                        let mut ir = groups
                            .par_iter()
                            .map(|base_g| {
                                let dt = dt.slice(base_g[0] as i64, base_g[1] as usize);
                                let vals = dt.downcast_iter().next().unwrap();
                                let ts = vals.values().as_slice();
                                let (sub_groups, lower, upper) = groupby_windows(
                                    w,
                                    ts,
                                    options.closed_window,
                                    tu,
                                    include_lower_bound,
                                    include_upper_bound,
                                    options.start_by,
                                );
                                (lower, upper, update_subgroups_slice(&sub_groups, *base_g))
                            })
                            .collect::<Vec<_>>();

                        ir.iter_mut().for_each(|(lower, upper, _)| {
                            let lower = std::mem::take(lower);
                            let upper = std::mem::take(upper);
                            update_bounds(lower, upper)
                        });

                        GroupsProxy::Slice {
                            groups: ir.into_iter().flat_map(|(_, _, groups)| groups).collect(),
                            rolling: false,
                        }
                    }
                })
            } else {
                POOL.install(|| match groups {
                    GroupsProxy::Idx(groups) => {
                        let groupsidx = groups
                            .par_iter()
                            .flat_map(|base_g| {
                                let dt = unsafe { dt.take_unchecked(base_g.1.into()) };
                                let vals = dt.downcast_iter().next().unwrap();
                                let ts = vals.values().as_slice();
                                let (sub_groups, _, _) = groupby_windows(
                                    w,
                                    ts,
                                    options.closed_window,
                                    tu,
                                    include_lower_bound,
                                    include_upper_bound,
                                    options.start_by,
                                );
                                update_subgroups_idx(&sub_groups, base_g)
                            })
                            .collect();
                        GroupsProxy::Idx(groupsidx)
                    }
                    GroupsProxy::Slice { groups, .. } => {
                        let groups = groups
                            .par_iter()
                            .flat_map(|base_g| {
                                let dt = dt.slice(base_g[0] as i64, base_g[1] as usize);
                                let vals = dt.downcast_iter().next().unwrap();
                                let ts = vals.values().as_slice();
                                let (sub_groups, _, _) = groupby_windows(
                                    w,
                                    ts,
                                    options.closed_window,
                                    tu,
                                    include_lower_bound,
                                    include_upper_bound,
                                    options.start_by,
                                );
                                update_subgroups_slice(&sub_groups, *base_g)
                            })
                            .collect();
                        GroupsProxy::Slice {
                            groups,
                            rolling: false,
                        }
                    }
                })
            }
        };

        let dt = unsafe { dt.clone().into_series().agg_first(&groups) };
        let mut dt = dt.datetime().unwrap().as_ref().clone();
        for key in by.iter_mut() {
            *key = unsafe { key.agg_first(&groups) };
        }

        let lower = lower_bound.map(|lower| Int64Chunked::new_vec(LB_NAME, lower));

        if options.truncate {
            let mut lower = lower.clone().unwrap();
            lower.rename(dt.name());
            dt = lower;
        }

        if let (true, Some(lower), Some(higher)) = (options.include_boundaries, lower, upper_bound)
        {
            by.push(lower.into_datetime(tu, tz.clone()).into_series());
            let s = Int64Chunked::new_vec(UP_NAME, higher)
                .into_datetime(tu, tz.clone())
                .into_series();
            by.push(s);
        }

        dt.into_datetime(tu, None)
            .into_series()
            .cast(time_type)
            .map(|s| (s, by, groups))
    }

    /// Returns: time_keys, keys, groupsproxy
    fn impl_groupby_rolling(
        &self,
        dt: Series,
        by: Vec<Series>,
        options: &RollingGroupOptions,
        tu: TimeUnit,
        time_type: &DataType,
    ) -> PolarsResult<(Series, Vec<Series>, GroupsProxy)> {
        let mut dt = dt.rechunk();

        let groups = if by.is_empty() {
            let dt = dt.datetime().unwrap();
            let vals = dt.downcast_iter().next().unwrap();
            let ts = vals.values().as_slice();
            GroupsProxy::Slice {
                groups: groupby_values(
                    options.period,
                    options.offset,
                    ts,
                    options.closed_window,
                    tu,
                ),
                rolling: true,
            }
        } else {
            let groups = self
                .0
                .groupby_with_series(by.clone(), true, true)?
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
                        .flat_map(|base_g| {
                            let dt = unsafe { dt_local.take_unchecked(base_g.1.into()) };
                            let vals = dt.downcast_iter().next().unwrap();
                            let ts = vals.values().as_slice();
                            let sub_groups = groupby_values(
                                options.period,
                                options.offset,
                                ts,
                                options.closed_window,
                                tu,
                            );
                            update_subgroups_idx(&sub_groups, base_g)
                        })
                        .collect();

                    GroupsProxy::Idx(idx)
                }
                GroupsProxy::Slice { groups, .. } => {
                    let slice_groups = groups
                        .par_iter()
                        .flat_map(|base_g| {
                            let dt = dt_local.slice(base_g[0] as i64, base_g[1] as usize);
                            let vals = dt.downcast_iter().next().unwrap();
                            let ts = vals.values().as_slice();
                            let sub_groups = groupby_values(
                                options.period,
                                options.offset,
                                ts,
                                options.closed_window,
                                tu,
                            );
                            update_subgroups_slice(&sub_groups, *base_g)
                        })
                        .collect();

                    GroupsProxy::Slice {
                        groups: slice_groups,
                        rolling: false,
                    }
                }
            })
        };

        let dt = dt.cast(time_type).unwrap();

        Ok((dt, by, groups))
    }
}

fn update_subgroups_slice(sub_groups: &[[IdxSize; 2]], base_g: [IdxSize; 2]) -> Vec<[IdxSize; 2]> {
    sub_groups
        .iter()
        .map(|&[first, len]| {
            let new_first = base_g[0] + first;
            [new_first, len]
        })
        .collect_trusted::<Vec<_>>()
}

fn update_subgroups_idx(
    sub_groups: &[[IdxSize; 2]],
    base_g: (IdxSize, &Vec<IdxSize>),
) -> Vec<(IdxSize, Vec<IdxSize>)> {
    sub_groups
        .iter()
        .map(|&[first, len]| {
            let new_first = unsafe { *base_g.1.get_unchecked(first as usize) };

            let first = first as usize;
            let len = len as usize;
            let idx = (first..first + len)
                .map(|i| {
                    debug_assert!(i < base_g.1.len());
                    unsafe { *base_g.1.get_unchecked(i) }
                })
                .collect_trusted::<Vec<_>>();
            (new_first, idx)
        })
        .collect_trusted::<Vec<_>>()
}

#[cfg(test)]
mod test {
    use chrono::prelude::*;

    use super::*;

    #[test]
    fn test_rolling_groupby_tu() -> PolarsResult<()> {
        // test multiple time units
        for tu in [
            TimeUnit::Nanoseconds,
            TimeUnit::Microseconds,
            TimeUnit::Milliseconds,
        ] {
            let date = Utf8Chunked::new(
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
            .as_datetime(None, tu, false, false, false, None)?
            .into_series();
            let a = Series::new("a", [3, 7, 5, 9, 2, 1]);
            let df = DataFrame::new(vec![date, a.clone()])?;

            let (_, _, groups) = df
                .groupby_rolling(
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
    fn test_rolling_groupby_aggs() -> PolarsResult<()> {
        let date = Utf8Chunked::new(
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
        .as_datetime(None, TimeUnit::Milliseconds, false, false, false, None)?
        .into_series();
        let a = Series::new("a", [3, 7, 5, 9, 2, 1]);
        let df = DataFrame::new(vec![date, a.clone()])?;

        let (_, _, groups) = df
            .groupby_rolling(
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

        // expected for nulls is equal
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
        assert_eq!(var, expected);

        let var = unsafe { nulls.agg_var(&groups, 1) };
        let expected = Series::new("", [0.0, 8.0, 8.0, 9.333333333333343, 24.5, 0.0]);
        assert_eq!(var, expected);

        let quantile = unsafe { a.agg_quantile(&groups, 0.5, QuantileInterpolOptions::Linear) };
        let expected = Series::new("", [3.0, 5.0, 5.0, 6.0, 5.5, 1.0]);
        assert_eq!(quantile, expected);

        let quantile = unsafe { nulls.agg_quantile(&groups, 0.5, QuantileInterpolOptions::Linear) };
        let expected = Series::new("", [3.0, 5.0, 5.0, 7.0, 5.5, 1.0]);
        assert_eq!(quantile, expected);

        Ok(())
    }

    #[test]
    fn test_dynamic_groupby_window() -> PolarsResult<()> {
        let start = NaiveDate::from_ymd_opt(2021, 12, 16)
            .unwrap()
            .and_hms_opt(0, 0, 0)
            .unwrap()
            .timestamp_millis();
        let stop = NaiveDate::from_ymd_opt(2021, 12, 16)
            .unwrap()
            .and_hms_opt(3, 0, 0)
            .unwrap()
            .timestamp_millis();
        let range = date_range_impl(
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
            .groupby_dynamic(
                vec![groups],
                &DynamicGroupOptions {
                    index_column: "date".into(),
                    every: Duration::parse("1h"),
                    period: Duration::parse("1h"),
                    offset: Duration::parse("0h"),
                    truncate: true,
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
        let g = g.utf8().unwrap();
        let g = g.into_no_null_iter().collect::<Vec<_>>();
        assert_eq!(g, &["a", "a", "a", "a", "b", "b"]);

        let upper = out.column("_upper_boundary").unwrap().slice(0, 3);
        let start = NaiveDate::from_ymd_opt(2021, 12, 16)
            .unwrap()
            .and_hms_opt(1, 0, 0)
            .unwrap()
            .timestamp_millis();
        let stop = NaiveDate::from_ymd_opt(2021, 12, 16)
            .unwrap()
            .and_hms_opt(3, 0, 0)
            .unwrap()
            .timestamp_millis();
        let range = date_range_impl(
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
            .timestamp_millis();
        let stop = NaiveDate::from_ymd_opt(2021, 12, 16)
            .unwrap()
            .and_hms_opt(2, 0, 0)
            .unwrap()
            .timestamp_millis();
        let range = date_range_impl(
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
                (0 as IdxSize, vec![0 as IdxSize, 1, 2]),
                (2, vec![2]),
                (5, vec![5, 6]),
                (6, vec![6]),
                (3, vec![3, 4]),
                (4, vec![4]),
            ]
            .into(),
        );
        assert_eq!(expected, groups);
        Ok(())
    }

    #[test]
    #[should_panic]
    fn test_panic_integer_temporal_combine() {
        let df = DataFrame::new_no_checks(vec![]);
        let _ = df.groupby_dynamic(
            vec![],
            &DynamicGroupOptions {
                index_column: "date".into(),
                every: Duration::parse("1h"),
                period: Duration::parse("1i"),
                offset: Duration::parse("0h"),
                truncate: true,
                include_boundaries: true,
                closed_window: ClosedWindow::Both,
                start_by: Default::default(),
            },
        );
    }

    #[test]
    fn test_truncate_offset() -> PolarsResult<()> {
        let start = NaiveDate::from_ymd_opt(2021, 3, 1)
            .unwrap()
            .and_hms_opt(12, 0, 0)
            .unwrap()
            .timestamp_millis();
        let stop = NaiveDate::from_ymd_opt(2021, 3, 7)
            .unwrap()
            .and_hms_opt(12, 0, 0)
            .unwrap()
            .timestamp_millis();
        let range = date_range_impl(
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
            .groupby_dynamic(
                vec![groups],
                &DynamicGroupOptions {
                    index_column: "date".into(),
                    every: Duration::parse("6d"),
                    period: Duration::parse("6d"),
                    offset: Duration::parse("0h"),
                    truncate: true,
                    include_boundaries: true,
                    closed_window: ClosedWindow::Both,
                    start_by: Default::default(),
                },
            )
            .unwrap();
        let mut lower_bound = keys[1].clone();
        time_key.rename("");
        lower_bound.rename("");
        assert!(time_key.series_equal(&lower_bound));
        Ok(())
    }
}
