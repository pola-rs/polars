use crate::prelude::*;
use polars_time::{
    Duration, Window
};
use crate::frame::groupby::{
    GroupTuples,
    GroupBy
};


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
    pub truncate: bool
}

impl DataFrame {
    pub fn groupby_dynamic(&self, options: &DynamicGroupOptions) -> Result<(Series, GroupTuples)> {
        let w = Window::new(options.every, options.period, options.offset);


        let time = self.column(&options.time_column)?;
        if time.null_count() > 0 {
            panic!("null values in dynamic groupby not yet supported, fill nulls.")
        }

        let dt = time.cast(&DataType::Datetime)?;
        let dt = dt.datetime().unwrap();

        let gt = dt.downcast_iter().map(|vals| {
            let ts = vals.values().as_slice();
            polars_time::groupby::groupby(w, ts)
        }).flatten().collect::<Vec<_>>();

        // Safety:
        // within bounds
        let mut dt = unsafe {
            dt.take_unchecked(gt.iter().map(|g| g.0 as usize).into())
        };

        if options.truncate {
            dt = dt.apply(|v| w.truncate(v));
        }


        Ok((dt.into_date().into_series(), gt))
    }
}
