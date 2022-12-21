use polars_ops::prelude::search_sorted;

use super::*;

pub(super) fn search_sorted_impl(s: &mut [Series]) -> PolarsResult<Series> {
    let sorted_array = &s[0];
    let search_value = s[1].get(0).unwrap();

    search_sorted(sorted_array, &search_value).map(|idx| Series::new(sorted_array.name(), &[idx]))
}
