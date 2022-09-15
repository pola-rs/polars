use super::*;

pub(super) fn clip(
    s: Series,
    min: Option<AnyValue<'_>>,
    max: Option<AnyValue<'_>>,
) -> PolarsResult<Series> {
    match (min, max) {
        (Some(min), Some(max)) => s.clip(min, max),
        (Some(min), None) => s.clip_min(min),
        (None, Some(max)) => s.clip_max(max),
        _ => unreachable!(),
    }
}
