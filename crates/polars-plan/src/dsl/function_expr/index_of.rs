use polars_ops::series::index_of as index_of_op;

use super::*;

/// Given two columns, find the index of a value (the second column) within the
/// first column. Will use binary search if possible, as an optimization.
pub(super) fn index_of(s: &mut [Column]) -> PolarsResult<Column> {
    let series = if let Column::Scalar(ref sc) = s[0] {
        // We only care about the first value:
        &sc.as_single_value_series()
    } else {
        s[0].as_materialized_series()
    };

    let needle_s = &s[1];
    polars_ensure!(
        needle_s.len() == 1,
        InvalidOperation: "needle of `index_of` can only contain a single value, found {} values",
        needle_s.len()
    );
    let needle = Scalar::new(
        needle_s.dtype().clone(),
        needle_s.get(0).unwrap().into_static(),
    );

    let is_sorted_flag = series.is_sorted_flag();
    let result = match is_sorted_flag {
        // If the Series is sorted, we can use an optimized binary search to
        // find the value.
        IsSorted::Ascending | IsSorted::Descending
            if !needle.is_null() &&
            // search_sorted() doesn't support decimals at the moment.
            !series.dtype().is_decimal() =>
        {
            search_sorted(
                series,
                needle_s.as_materialized_series(),
                SearchSortedSide::Left,
                IsSorted::Descending == is_sorted_flag,
            )?
            .get(0)
            .and_then(|idx| {
                // search_sorted() gives an index even if it's not an exact
                // match! So we want to make sure it actually found the value.
                if series.get(idx as usize).ok()? == needle.as_any_value() {
                    Some(idx as usize)
                } else {
                    None
                }
            })
        },
        _ => index_of_op(series, needle)?,
    };

    let av = match result {
        None => AnyValue::Null,
        Some(idx) => AnyValue::from(idx as IdxSize),
    };
    let scalar = Scalar::new(IDX_DTYPE, av);
    Ok(Column::new_scalar(series.name().clone(), scalar, 1))
}
