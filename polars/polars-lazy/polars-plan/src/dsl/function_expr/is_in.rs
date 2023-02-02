use super::*;

pub(super) fn is_in(s: &mut [Series]) -> PolarsResult<Option<Series>> {
    let left = &s[0];
    let other = &s[1];

    left.is_in(other).map(|ca| Some(ca.into_series()))
}
