use polars_ops::prelude::search_sorted;

use super::*;

pub(super) fn search_sorted_impl(s: &mut [Series], side: SearchSortedSide) -> PolarsResult<Series> {
    let sorted_array = &s[0];
    let search_value = &s[1];

    search_sorted(sorted_array, search_value, side, false).map(|ca| ca.into_series())
}
