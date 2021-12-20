use crate::frame::groupby::GroupTuples;
use crate::prelude::*;
use crate::POOL;
use polars_time::groupby::ClosedWindow;
use polars_time::{Duration, Window};
use rayon::prelude::*;

#[derive(Clone, Debug)]
pub struct DynamicGroupOptions {
    pub time_column: String,
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

impl DataFrame {
    /// Returns: time_keys, keys, grouptuples
    pub fn groupby_dynamic(
        &self,
        mut by: Vec<Series>,
        options: &DynamicGroupOptions,
    ) -> Result<(Series, Vec<Series>, GroupTuples)> {
        let w = Window::new(options.every, options.period, options.offset);
        let time = self.column(&options.time_column)?;
        let time_type = time.dtype();
        if time.null_count() > 0 {
            panic!("null values in dynamic groupby not yet supported, fill nulls.")
        }
        let dt = time.cast(&DataType::Datetime)?;
        let dt = dt.datetime().unwrap();

        let mut lower_bound = None;
        let mut upper_bound = None;

        let groups = if by.is_empty() {
            dt.downcast_iter()
                .map(|vals| {
                    let ts = vals.values().as_slice();
                    let (groups, lower, upper) = polars_time::groupby::groupby(
                        w,
                        ts,
                        options.include_boundaries,
                        options.closed_window,
                    );
                    match (&mut lower_bound, &mut upper_bound) {
                        (None, None) => {
                            lower_bound = Some(lower);
                            upper_bound = Some(upper);
                        }
                        (Some(lower_bound), Some(upper_bound)) => {
                            lower_bound.extend_from_slice(&lower);
                            upper_bound.extend_from_slice(&upper);
                        }
                        _ => unreachable!(),
                    }
                    groups
                })
                .flatten()
                .collect::<Vec<_>>()
        } else {
            let mut groups = self.groupby_with_series(by.clone(), true)?.groups;
            groups.sort_unstable_by_key(|g| g.0);

            // include boundaries cannot be parallel (easily)
            if options.include_boundaries {
                groups
                    .iter()
                    .map(|base_g| {
                        let dt = unsafe {
                            dt.take_unchecked((base_g.1.iter().map(|i| *i as usize)).into())
                        };

                        let vals = dt.downcast_iter().next().unwrap();
                        let ts = vals.values().as_slice();
                        let (mut sub_groups, lower, upper) = polars_time::groupby::groupby(
                            w,
                            ts,
                            options.include_boundaries,
                            options.closed_window,
                        );
                        let _lower = Int64Chunked::new_vec("lower", lower.clone())
                            .into_date()
                            .into_series();
                        let _higher = Int64Chunked::new_vec("upper", upper.clone())
                            .into_date()
                            .into_series();

                        match (&mut lower_bound, &mut upper_bound) {
                            (None, None) => {
                                lower_bound = Some(lower);
                                upper_bound = Some(upper);
                            }
                            (Some(lower_bound), Some(upper_bound)) => {
                                lower_bound.extend_from_slice(&lower);
                                upper_bound.extend_from_slice(&upper);
                            }
                            _ => unreachable!(),
                        }

                        sub_groups.iter_mut().for_each(|g| {
                            g.0 = unsafe { *base_g.1.get_unchecked(g.0 as usize) };
                            for x in g.1.iter_mut() {
                                debug_assert!((*x as usize) < base_g.1.len());
                                unsafe { *x = *base_g.1.get_unchecked(*x as usize) }
                            }
                        });
                        sub_groups
                    })
                    .flatten()
                    .collect::<Vec<_>>()
            } else {
                POOL.install(|| {
                    groups
                        .par_iter()
                        .map(|base_g| {
                            let dt = unsafe {
                                dt.take_unchecked((base_g.1.iter().map(|i| *i as usize)).into())
                            };
                            let vals = dt.downcast_iter().next().unwrap();
                            let ts = vals.values().as_slice();
                            let (mut sub_groups, _, _) = polars_time::groupby::groupby(
                                w,
                                ts,
                                options.include_boundaries,
                                options.closed_window,
                            );

                            sub_groups.iter_mut().for_each(|g| {
                                g.0 = unsafe { *base_g.1.get_unchecked(g.0 as usize) };
                                for x in g.1.iter_mut() {
                                    debug_assert!((*x as usize) < base_g.1.len());
                                    unsafe { *x = *base_g.1.get_unchecked(*x as usize) }
                                }
                            });
                            sub_groups
                        })
                        .flatten()
                        .collect::<Vec<_>>()
                })
            }
        };

        // Safety:
        // within bounds
        let mut dt = unsafe { dt.take_unchecked(groups.iter().map(|g| g.0 as usize).into()) };
        for key in by.iter_mut() {
            *key = unsafe { key.take_iter_unchecked(&mut groups.iter().map(|g| g.0 as usize)) };
        }

        if options.truncate {
            dt = dt.apply(|v| w.truncate(v));
        }

        if let (true, Some(lower), Some(higher)) =
            (options.include_boundaries, lower_bound, upper_bound)
        {
            let s = Int64Chunked::new_vec("_lower_boundary", lower)
                .into_date()
                .into_series();
            by.push(s);
            let s = Int64Chunked::new_vec("_upper_boundary", higher)
                .into_date()
                .into_series();
            by.push(s);
        }

        dt.into_date()
            .into_series()
            .cast(time_type)
            .map(|s| (s, by, groups))
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::time::date_range;
    use polars_time::export::chrono::prelude::*;

    #[test]
    fn test_dynamic_groupby_window() {
        let start = NaiveDate::from_ymd(2021, 12, 16)
            .and_hms(0, 0, 0)
            .timestamp_nanos();
        let stop = NaiveDate::from_ymd(2021, 12, 16)
            .and_hms(3, 0, 0)
            .timestamp_nanos();
        let range = date_range(
            start,
            stop,
            Duration::parse("30m"),
            ClosedWindow::Both,
            "date",
        )
        .into_series();

        let groups = Series::new("groups", ["a", "a", "a", "b", "b", "a", "a"]);
        let df = DataFrame::new(vec![range, groups.clone()]).unwrap();

        let (time_key, mut keys, groups) = df
            .groupby_dynamic(
                vec![groups],
                &DynamicGroupOptions {
                    time_column: "date".into(),
                    every: Duration::parse("1h"),
                    period: Duration::parse("1h"),
                    offset: Duration::parse("0h"),
                    truncate: true,
                    include_boundaries: true,
                    closed_window: ClosedWindow::Both,
                },
            )
            .unwrap();

        keys.push(time_key);
        let out = DataFrame::new(keys).unwrap();
        let g = out.column("groups").unwrap();
        let g = g.utf8().unwrap();
        let g = g.into_no_null_iter().collect::<Vec<_>>();
        assert_eq!(g, &["a", "a", "a", "b"]);

        let upper = out.column("_upper_boundary").unwrap().slice(0, 3);
        let start = NaiveDate::from_ymd(2021, 12, 16)
            .and_hms(1, 0, 0)
            .timestamp_nanos();
        let stop = NaiveDate::from_ymd(2021, 12, 16)
            .and_hms(3, 0, 0)
            .timestamp_nanos();
        let range = date_range(
            start,
            stop,
            Duration::parse("1h"),
            ClosedWindow::Both,
            "_upper_boundary",
        )
        .into_series();
        assert_eq!(&upper, &range);

        let upper = out.column("_lower_boundary").unwrap().slice(0, 3);
        let start = NaiveDate::from_ymd(2021, 12, 16)
            .and_hms(0, 0, 0)
            .timestamp_nanos();
        let stop = NaiveDate::from_ymd(2021, 12, 16)
            .and_hms(2, 0, 0)
            .timestamp_nanos();
        let range = date_range(
            start,
            stop,
            Duration::parse("1h"),
            ClosedWindow::Both,
            "_lower_boundary",
        )
        .into_series();
        assert_eq!(&upper, &range);

        let expected = vec![
            (0u32, vec![0u32, 1, 2]),
            (2u32, vec![2]),
            (5u32, vec![5, 6]),
            (3u32, vec![3, 4]),
        ];
        assert_eq!(expected, groups);
    }
}
