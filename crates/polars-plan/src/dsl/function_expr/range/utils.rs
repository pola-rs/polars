use polars_core::series::Series;

pub(super) fn temporal_series_to_i64_scalar(s: &Series) -> i64 {
    s.to_physical_repr()
        .get(0)
        .unwrap()
        .extract::<i64>()
        .unwrap()
}
