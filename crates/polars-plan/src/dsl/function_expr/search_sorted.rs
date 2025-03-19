use super::*;

pub(super) fn search_sorted_impl(s: &mut [Column], side: SearchSortedSide) -> PolarsResult<Column> {
    let sorted_array = &s[0];
    let search_value = &s[1];

    let sorted_series = sorted_array.as_materialized_series();
    let descending = matches!(sorted_series.is_sorted_flag(), IsSorted::Descending);
    search_sorted(
        sorted_series,
        search_value.as_materialized_series(),
        side,
        descending,
    )
    .map(|ca| ca.into_column())
}
