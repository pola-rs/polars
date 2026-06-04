use std::cmp::Ordering;

use polars_core::chunked_array::ops::SortOptions;
#[cfg(feature = "dtype-struct")]
use polars_core::chunked_array::ops::row_encode::_get_rows_encoded_ca;
use polars_core::frame::DataFrame;
use polars_core::prelude::*;
use polars_core::with_match_physical_numeric_polars_type;
use polars_error::{PolarsResult, polars_bail};
use polars_utils::itertools::Itertools;
use polars_utils::pl_str::PlSmallStr;
use polars_utils::scratch_vec::ScratchVec;
use polars_utils::sort::reorder_tot_cmp;
use polars_utils::total_ord::{ToTotalOrd, TotalEq, TotalHash, TotalOrd};
use recursive::recursive;

use crate::prelude::*;
use crate::series::ops::rle::rle_lengths_helper_ca;

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
        if by.is_empty() {
            polars_bail!(InvalidOperation: "by must specify at least one column");
        }
        if descending.len() != by.len() {
            polars_bail!(InvalidOperation: "descending must be of same length as by");
        }
        if nulls_last.len() != by.len() {
            polars_bail!(InvalidOperation: "nulls_last must be of same length as by");
        }

        if let [single_by] = by {
            let col = self.column(single_by)?;
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
