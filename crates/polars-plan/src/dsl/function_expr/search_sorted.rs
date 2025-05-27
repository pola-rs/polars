use super::*;

pub(super) fn search_sorted_impl(
    s: &mut [Column],
    side: SearchSortedSide,
    descending: bool,
) -> PolarsResult<Column> {
    let sorted_array = &s[0];
    let search_value = &s[1];

    search_sorted(
        sorted_array.as_materialized_series(),
        search_value.as_materialized_series(),
        side,
        descending,
    )
    .map(|ca| ca.into_column())
}
