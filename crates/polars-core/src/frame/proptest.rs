use std::ops::RangeInclusive;

use crate::series::proptest::SeriesArbitraryOptions;

pub struct DataFrameArbitraryOptions {
    pub series_options: SeriesArbitraryOptions,
    pub num_columns: RangeInclusive<usize>,
}

impl Default for DataFrameArbitraryOptions {
    fn default() -> Self {
        Self {
            series_options: SeriesArbitraryOptions::default(),
            num_columns: 0..=5,
        }
    }
}
