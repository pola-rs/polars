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
        .num_columns
        .clone()
        .prop_flat_map(move |num_cols| {
            prop::collection::vec(
                series_strategy(options.series_options.clone().into(), nesting_level),
                num_cols,
            )
        })
        .prop_map(|series| {
            let columns: Vec<Column> = series.into_iter().map(|series| series.into()).collect();
            DataFrame::new(columns).unwrap()
        })
}
