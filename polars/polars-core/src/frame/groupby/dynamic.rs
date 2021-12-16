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
    every: Duration,
    /// window duration
    period: Duration,
    /// offset window boundaries
    offset: Duration,
    /// truncate the time column values to the window
    truncate: bool
}

impl DataFrame {
    pub fn groupby_dynamic(&self, options: &DynamicGroupOptions) -> Result<(Self, GroupTuples)> {
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

        let mut df = self.clone();
        if options.truncate {
            let out = dt.apply(|v| w.truncate(v));
            let out = out.cast(&DataType::Datetime).unwrap();
            df.with_column(out)?;
        }


        Ok((df, gt))
    }
}
