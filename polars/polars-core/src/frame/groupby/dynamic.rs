use crate::frame::groupby::GroupTuples;
use crate::prelude::*;
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

        let groups = if by.is_empty() {
            dt.downcast_iter()
                .map(|vals| {
                    let ts = vals.values().as_slice();
                    polars_time::groupby::groupby(w, ts)
                })
                .flatten()
                .collect::<Vec<_>>()
        } else {
            let mut groups = self.groupby_with_series(by.clone(), true)?.groups;
            groups.sort_unstable_by_key(|g| g.0);
            groups
                .par_iter()
                .map(|g| {
                    let offset = g.0;
                    let dt = unsafe { dt.take_unchecked((g.1.iter().map(|i| *i as usize)).into()) };
                    let vals = dt.downcast_iter().next().unwrap();
                    let ts = vals.values().as_slice();
                    let mut sub_groups = polars_time::groupby::groupby(w, ts);

                    sub_groups.iter_mut().for_each(|g| {
                        g.0 += offset;
                        for x in g.1.iter_mut() {
                            *x += offset
                        }
                    });
                    sub_groups
                })
                .flatten()
                .collect::<Vec<_>>()
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
        dt.into_date()
            .into_series()
            .cast(time_type)
            .map(|s| (s, by, groups))
    }
}
