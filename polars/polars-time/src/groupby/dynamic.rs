use crate::prelude::*;
use polars_arrow::utils::CustomIterTools;
use polars_core::export::rayon::prelude::*;
use polars_core::frame::groupby::GroupsProxy;
use polars_core::prelude::*;
use polars_core::POOL;

#[repr(transparent)]
struct Wrap<T>(pub T);

#[derive(Clone, Debug)]
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
}

#[derive(Clone, Debug)]
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
    fn groupby_rolling(&self, options: &RollingGroupOptions) -> Result<(Series, GroupsProxy)>;

    fn groupby_dynamic(
        &self,
        by: Vec<Series>,
        options: &DynamicGroupOptions,
    ) -> Result<(Series, Vec<Series>, GroupsProxy)>;
}

impl PolarsTemporalGroupby for DataFrame {
    fn groupby_rolling(&self, options: &RollingGroupOptions) -> Result<(Series, GroupsProxy)> {
        Wrap(self).groupby_rolling(options)
    }

    fn groupby_dynamic(
        &self,
        by: Vec<Series>,
        options: &DynamicGroupOptions,
    ) -> Result<(Series, Vec<Series>, GroupsProxy)> {
        Wrap(self).groupby_dynamic(by, options)
    }
}

impl Wrap<&DataFrame> {
    fn groupby_rolling(&self, options: &RollingGroupOptions) -> Result<(Series, GroupsProxy)> {
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
                let (out, gt) =
                    self.impl_groupby_rolling(dt, options, TimeUnit::Nanoseconds, &time_type)?;
                let out = out.cast(&Int64).unwrap().cast(&Int32).unwrap();
                return Ok((out, gt));
            }
            Int64 => {
                let time_type = Datetime(TimeUnit::Nanoseconds, None);
                let dt = time.cast(&time_type).unwrap();
                let (out, gt) =
                    self.impl_groupby_rolling(dt, options, TimeUnit::Nanoseconds, &time_type)?;
                let out = out.cast(&Int64).unwrap();
                return Ok((out, gt));
            }
            dt => {
                return Err(PolarsError::ValueError(
                    format!(
                    "expected any of the following dtypes {{Date, Datetime, Int32, Int64}}, got {}",
                    dt
                )
                    .into(),
                ))
            }
        };
        self.impl_groupby_rolling(dt, options, tu, time_type)
    }

    /// Returns: time_keys, keys, groupsproxy
    fn groupby_dynamic(
        &self,
        by: Vec<Series>,
        options: &DynamicGroupOptions,
    ) -> Result<(Series, Vec<Series>, GroupsProxy)> {
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
                return Err(PolarsError::ValueError(
                    format!(
                    "expected any of the following dtypes {{Date, Datetime, Int32, Int64}}, got {}",
                    dt
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
    ) -> Result<(Series, Vec<Series>, GroupsProxy)> {
        let w = Window::new(options.every, options.period, options.offset);
        let dt = dt.datetime().unwrap();

        let mut lower_bound = None;
        let mut upper_bound = None;

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
            let (groups, lower, upper) =
                groupby_windows(w, ts, options.include_boundaries, options.closed_window, tu);
            update_bounds(lower, upper);
            GroupsProxy::Slice(groups)
        } else {
            let groups = self
                .0
                .groupby_with_series(by.clone(), true, true)?
                .take_groups();
            let groups = groups.into_idx();

            // include boundaries cannot be parallel (easily)
            if options.include_boundaries {
                let groupsidx = groups
                    .iter()
                    // we just flat map, because iterate over groups so we almost always need to reallocate
                    .flat_map(|base_g| {
                        let dt = unsafe { dt.take_unchecked(base_g.1.into()) };

                        let vals = dt.downcast_iter().next().unwrap();
                        let ts = vals.values().as_slice();
                        let (sub_groups, lower, upper) = groupby_windows(
                            w,
                            ts,
                            options.include_boundaries,
                            options.closed_window,
                            tu,
                        );
                        let _lower = Int64Chunked::new_vec("lower", lower.clone())
                            .into_datetime(tu, None)
                            .into_series();
                        let _higher = Int64Chunked::new_vec("upper", upper.clone())
                            .into_datetime(tu, None)
                            .into_series();

                        update_bounds(lower, upper);
                        update_subgroups(&sub_groups, base_g)
                    })
                    .collect();
                GroupsProxy::Idx(groupsidx)
            } else {
                let groupsidx = POOL.install(|| {
                    groups
                        .par_iter()
                        .flat_map(|base_g| {
                            let dt = unsafe { dt.take_unchecked(base_g.1.into()) };
                            let vals = dt.downcast_iter().next().unwrap();
                            let ts = vals.values().as_slice();
                            let (sub_groups, _, _) = groupby_windows(
                                w,
                                ts,
                                options.include_boundaries,
                                options.closed_window,
                                tu,
                            );
                            update_subgroups(&sub_groups, base_g)
                        })
                        .collect()
                });
                GroupsProxy::Idx(groupsidx)
            }
        };

        let dt = dt.clone().into_series().agg_first(&groups);
        let mut dt = dt.datetime().unwrap().as_ref().clone();
        for key in by.iter_mut() {
            *key = key.agg_first(&groups)
        }

        if options.truncate {
            let w = Window::new(options.every, options.period, options.offset);
            let truncate_fn = match tu {
                TimeUnit::Nanoseconds => Window::truncate_no_offset_ns,
                TimeUnit::Microseconds => Window::truncate_no_offset_us,
                TimeUnit::Milliseconds => Window::truncate_no_offset_ms,
            };
            dt = dt.apply(|v| truncate_fn(&w, v));
        }

        if let (true, Some(lower), Some(higher)) =
            (options.include_boundaries, lower_bound, upper_bound)
        {
            let s = Int64Chunked::new_vec(LB_NAME, lower)
                .into_datetime(tu, None)
                .into_series();
            by.push(s);
            let s = Int64Chunked::new_vec(UP_NAME, higher)
                .into_datetime(tu, None)
                .into_series();
            by.push(s);
        }

        dt.into_datetime(tu, None)
            .into_series()
            .cast(time_type)
            .map(|s| (s, by, groups))
    }

    fn impl_groupby_rolling(
        &self,
        dt: Series,
        options: &RollingGroupOptions,
        tu: TimeUnit,
        time_type: &DataType,
    ) -> Result<(Series, GroupsProxy)> {
        let dt = dt.datetime().unwrap().clone();

        let mut groups = dt
            .downcast_iter()
            .map(|vals| {
                let ts = vals.values().as_slice();
                groupby_values(
                    options.period,
                    options.offset,
                    ts,
                    options.closed_window,
                    tu,
                )
            })
            .collect::<Vec<_>>();

        // we don't flatmap because in case of a single chunk we don't need to reallocate the inner vec,
        // just pop it.
        let groups = if groups.len() == 1 {
            GroupsProxy::Slice(groups.pop().unwrap())
        } else {
            GroupsProxy::Slice(groups.into_iter().flatten().collect())
        };
        dt.cast(time_type).map(|s| (s, groups))
    }
}

fn update_subgroups(
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
    use super::*;
    use crate::prelude::*;
    use chrono::prelude::*;

    #[test]
    fn test_rolling_groupby() -> Result<()> {
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
            .as_datetime(None, tu)?
            .into_series();
            let a = Series::new("a", [3, 7, 5, 9, 2, 1]);
            let df = DataFrame::new(vec![date, a.clone()])?;

            let (_, groups) = df
                .groupby_rolling(&RollingGroupOptions {
                    index_column: "dt".into(),
                    period: Duration::parse("2d"),
                    offset: Duration::parse("-2d"),
                    closed_window: ClosedWindow::Right,
                })
                .unwrap();
            let sum = a.agg_sum(&groups).unwrap();
            let expected = Series::new("", [3, 10, 15, 24, 11, 1]);
            assert_eq!(sum, expected);
        }

        Ok(())
    }

    #[test]
    fn test_dynamic_groupby_window() {
        let start = NaiveDate::from_ymd(2021, 12, 16)
            .and_hms(0, 0, 0)
            .timestamp_millis();
        let stop = NaiveDate::from_ymd(2021, 12, 16)
            .and_hms(3, 0, 0)
            .timestamp_millis();
        let range = date_range_impl(
            "date",
            start,
            stop,
            Duration::parse("30m"),
            ClosedWindow::Both,
            TimeUnit::Milliseconds,
        )
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
        let start = NaiveDate::from_ymd(2021, 12, 16)
            .and_hms(1, 0, 0)
            .timestamp_millis();
        let stop = NaiveDate::from_ymd(2021, 12, 16)
            .and_hms(3, 0, 0)
            .timestamp_millis();
        let range = date_range_impl(
            "_upper_boundary",
            start,
            stop,
            Duration::parse("1h"),
            ClosedWindow::Both,
            TimeUnit::Milliseconds,
        )
        .into_series();
        assert_eq!(&upper, &range);

        let upper = out.column("_lower_boundary").unwrap().slice(0, 3);
        let start = NaiveDate::from_ymd(2021, 12, 16)
            .and_hms(0, 0, 0)
            .timestamp_millis();
        let stop = NaiveDate::from_ymd(2021, 12, 16)
            .and_hms(2, 0, 0)
            .timestamp_millis();
        let range = date_range_impl(
            "_lower_boundary",
            start,
            stop,
            Duration::parse("1h"),
            ClosedWindow::Both,
            TimeUnit::Milliseconds,
        )
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
            },
        );
    }
}
