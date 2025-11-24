use std::ops::RangeInclusive;
use std::rc::Rc;

use proptest::prelude::*;

use crate::prelude::{Column, DataFrame};
use crate::series::proptest::{SeriesArbitraryOptions, series_strategy};

pub struct DataFrameArbitraryOptions {
    pub series_options: SeriesArbitraryOptions,
    pub num_columns: RangeInclusive<usize>,
}

impl Default for DataFrameArbitraryOptions {
    fn default() -> Self {
        Self {
            series_options: SeriesArbitraryOptions::default(),
            num_columns: 1..=5,
        }
    }
}

pub fn dataframe_strategy(
    options: Rc<DataFrameArbitraryOptions>,
    nesting_level: usize,
) -> impl Strategy<Value = DataFrame> {
    options
        .series_options
        .series_length_range
        .clone()
        .prop_flat_map(move |series_length| {
            let mut opts = options.series_options.clone();
            opts.series_length_range = series_length..=series_length;

            prop::collection::vec(
                series_strategy(Rc::new(opts), nesting_level),
                options.num_columns.clone(),
            )
        })
        .prop_map(|series| DataFrame::new(series.into_iter().map(Column::from).collect()).unwrap())
}
