use polars_arrow::legacy::time_zone::Tz;
use polars_core::prelude::*;
use polars_core::runtime::RAYON;
use polars_core::series::IsSorted;
use polars_core::utils::flatten::flatten_par;
use polars_defs::time::duration::ensure_duration_matches_dtype;
use polars_defs::time::group_by::{
    ClosedWindow, DynamicGroupOptionsIR, Label, RollingGroupOptionsIR,
};
use polars_ops::series::SeriesMethods;
use polars_utils::itertools::Itertools;
use polars_utils::pl_str::PlSmallStr;
use polars_utils::slice::SortedSlice;
use rayon::prelude::*;

use crate::prelude::*;
use crate::windows::index_space::IndexSpace;

#[repr(transparent)]
struct Wrap<T>(pub T);

fn check_sortedness_slice(v: &[i64]) -> PolarsResult<()> {
    polars_ensure!(v.is_sorted_ascending(), ComputeError: "input data is not sorted");
    Ok(())
}

pub const LB_NAME: &str = "_lower_boundary";
pub const UB_NAME: &str = "_upper_boundary";

pub trait PolarsTemporalGroupby {
    fn rolling(
        &self,
        group_by: Option<GroupsSlice>,
        options: &RollingGroupOptionsIR,
    ) -> PolarsResult<(Column, GroupPositions)>;

    fn group_by_dynamic(
        &self,
        group_by: Option<GroupsSlice>,
        options: &DynamicGroupOptionsIR,
    ) -> PolarsResult<(Column, Vec<Column>, GroupPositions)>;
}

impl PolarsTemporalGroupby for DataFrame {
    fn rolling(
        &self,
        group_by: Option<GroupsSlice>,
        options: &RollingGroupOptionsIR,
    ) -> PolarsResult<(Column, GroupPositions)> {
        Wrap(self).rolling(group_by, options)
    }

    fn group_by_dynamic(
        &self,
        group_by: Option<GroupsSlice>,
        options: &DynamicGroupOptionsIR,
    ) -> PolarsResult<(Column, Vec<Column>, GroupPositions)> {
        Wrap(self).group_by_dynamic(group_by, options)
    }
}

impl Wrap<&DataFrame> {
    fn rolling(
        &self,
        group_by: Option<GroupsSlice>,
        options: &RollingGroupOptionsIR,
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

        let space = IndexSpace::rolling(time_type)?;
        let dt = space.cast_to_space(&time)?;
        let (out, gt) =
            self.impl_rolling(dt, group_by, options, space.time_unit, space.tz().cloned())?;
        Ok((space.cast_from_space(&out)?, gt))
    }

    /// Returns: time_keys, keys, groupsproxy.
    fn group_by_dynamic(
        &self,
        group_by: Option<GroupsSlice>,
        options: &DynamicGroupOptionsIR,
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

        let space = IndexSpace::dynamic(time_type)?;
        let dt = space.cast_to_space(&time)?;
        let (out, mut keys, gt) =
            self.impl_group_by_dynamic(dt, group_by, options, space.time_unit)?;
        let out = space.cast_from_space(&out)?;
        for k in &mut keys {
            if k.name().as_str() == UB_NAME || k.name().as_str() == LB_NAME {
                *k = space.cast_to_boundary(k)?;
            }
        }
        Ok((out, keys, gt))
    }

    fn impl_group_by_dynamic(
        &self,
        mut dt: Column,
        group_by: Option<GroupsSlice>,
        options: &DynamicGroupOptionsIR,
        tu: TimeUnit,
    ) -> PolarsResult<(Column, Vec<Column>, GroupPositions)> {
        polars_ensure!(!options.every.negative, ComputeError: "'every' argument must be positive");
        if dt.is_empty() {
            let mut bounds = vec![];
            if options.include_boundaries {
                bounds.push(Column::new_empty(
                    PlSmallStr::from_static(LB_NAME),
                    dt.dtype(),
                ));
                bounds.push(Column::new_empty(
                    PlSmallStr::from_static(UB_NAME),
                    dt.dtype(),
                ));
            }
            return Ok((dt, bounds, Default::default()));
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

        let overlapping = match options.closed_window {
            ClosedWindow::Both => options.period >= options.every,
            _ => options.period > options.every,
        };

        let groups = if let Some(groups) = group_by.as_ref() {
            let vals = dt.physical().downcast_iter().next().unwrap();
            let ts = vals.values().as_slice();

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
                )?;

                PolarsResult::Ok((
                    groups
                        .iter()
                        .map(|[start, len]| [*start + group_offset, *len])
                        .collect_vec(),
                    lower,
                    upper,
                ))
            });

            let res = RAYON.install(|| iter.collect::<PolarsResult<Vec<_>>>())?;
            let groups = res.iter().map(|g| &g.0).collect_vec();
            let lower = res.iter().map(|g| &g.1).collect_vec();
            let upper = res.iter().map(|g| &g.2).collect_vec();

            let ((groups, upper), lower) = RAYON.install(|| {
                rayon::join(
                    || rayon::join(|| flatten_par(&groups), || flatten_par(&upper)),
                    || flatten_par(&lower),
                )
            });

            update_bounds(lower, upper);
            // The upper bound of a window is not a monotonic function of its lower
            // bound (month clamping, DST), so the group ends may move backwards.
            let monotonic = slice_groups_are_monotonic(&groups);
            PolarsResult::Ok(GroupsType::new_slice(groups, overlapping, monotonic))
        } else {
            let vals = dt.physical().downcast_iter().next().unwrap();
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
            )?;
            update_bounds(lower, upper);
            let monotonic = slice_groups_are_monotonic(&groups);
            PolarsResult::Ok(GroupsType::new_slice(groups, overlapping, monotonic))
        }?;
        // note that if 'group_by' is none we can be sure that the index column, the lower column and the
        // upper column remain/are sorted

        let dt = unsafe { dt.clone().into_series().agg_first(&groups) };
        let mut dt = dt.datetime().unwrap().physical().clone();

        let lower =
            lower_bound.map(|lower| Int64Chunked::new_vec(PlSmallStr::from_static(LB_NAME), lower));
        let upper =
            upper_bound.map(|upper| Int64Chunked::new_vec(PlSmallStr::from_static(UB_NAME), upper));

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

        let dt = dt.into_datetime(tu, tz.clone()).into_column();
        Ok((dt, bounds, groups.into_sliceable()))
    }

    /// Returns: time_keys, keys, groupsproxy
    fn impl_rolling(
        &self,
        dt: Column,
        group_by: Option<GroupsSlice>,
        options: &RollingGroupOptionsIR,
        tu: TimeUnit,
        tz: Option<Tz>,
    ) -> PolarsResult<(Column, GroupPositions)> {
        let mut dt = dt.rechunk();

        let groups = if let Some(groups) = group_by {
            let dt = dt.datetime().unwrap();
            let vals = dt.physical().downcast_iter().next().unwrap();
            let ts = vals.values().as_slice();

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

            let groups = RAYON.install(|| iter.collect::<PolarsResult<Vec<_>>>())?;
            PolarsResult::Ok(RAYON.install(|| flatten_par(&groups)))
        } else {
            // a requirement for the index
            // so we can set this such that downstream code has this info
            dt.set_sorted_flag(IsSorted::Ascending);
            let dt = dt.datetime().unwrap();
            let vals = dt.physical().downcast_iter().next().unwrap();
            let ts = vals.values().as_slice();
            group_by_values(
                options.period,
                options.offset,
                ts,
                options.closed_window,
                tu,
                tz,
            )
        }?;

        let groups = GroupsType::new_slice(groups, true, true);

        Ok((dt, groups.into_sliceable()))
    }
}

#[cfg(test)]
mod test {
    use polars_compute::rolling::QuantileMethod;
    use polars_core::chunked_array::temporal::string::StringMethods;
    use polars_defs::time::duration::Duration;
    use polars_defs::time::group_by::RollingGroupOptionsIR;
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
            let df = DataFrame::new_infer_height(vec![date, a.clone()])?;

            let (_, groups) = df
                .rolling(
                    None,
                    &RollingGroupOptionsIR {
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
        let df = DataFrame::new_infer_height(vec![date, a.clone()])?;

        let (_, groups) = df
            .rolling(
                None,
                &RollingGroupOptionsIR {
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
