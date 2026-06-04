use polars_core::chunked_array::ops::SortOptions;
use polars_core::chunked_array::ops::row_encode::_get_rows_encoded_ca;
use polars_core::frame::DataFrame;
use polars_core::prelude::*;
use polars_error::{PolarsResult, polars_ensure};
use polars_utils::itertools::Itertools;
use polars_utils::pl_str::PlSmallStr;

use crate::prelude::*;

pub trait DataFrameIsSorted {
    fn is_sorted(
        &self,
        by: &[PlSmallStr],
        descending: &[bool],
        nulls_last: &[bool],
    ) -> PolarsResult<bool>;
}

impl DataFrameIsSorted for DataFrame {
    fn is_sorted(
        &self,
        by: &[PlSmallStr],
        descending: &[bool],
        nulls_last: &[bool],
    ) -> PolarsResult<bool> {
        polars_ensure!(!by.is_empty(), InvalidOperation: "by must specify at least one column");
        polars_ensure!(descending.len() == by.len(), InvalidOperation: "descending must be of same length as by");
        polars_ensure!(nulls_last.len() == by.len(), InvalidOperation: "nulls_last must be of same length as by");

        if let [by] = by {
            let col = self.column(by)?;
            return SeriesMethods::is_sorted(
                col.as_materialized_series(),
                SortOptions::new()
                    .with_order_descending(descending[0])
                    .with_nulls_last(nulls_last[0]),
            );
        }

        let by_vec = by
            .iter()
            .map(|name| self.column(name).cloned())
            .try_collect_vec()?;
        let row_encoded =
            _get_rows_encoded_ca(PlSmallStr::EMPTY, &by_vec, descending, nulls_last, false)?;
        let s = Series::new(PlSmallStr::EMPTY, row_encoded);
        SeriesMethods::is_sorted(&s, SortOptions::new())
    }
}
