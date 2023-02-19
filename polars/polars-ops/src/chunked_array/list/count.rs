use super::namespace::ListNameSpaceImpl;
use super::*;

pub fn list_count_match(ca: &ListChunked, value: AnyValue) -> PolarsResult<Series> {
    let value = Series::new("", [value]);

    let ca = ca.apply_to_inner(&|s| {
        ChunkCompare::<&Series>::equal(&s, &value).map(|ca| ca.into_series())
    })?;
    ca.lst_sum().fill_null(FillNullStrategy::Zero)
}
