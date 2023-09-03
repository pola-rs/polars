use super::*;

pub(super) fn abs(s: &Series) -> PolarsResult<Series> {
    s.abs()
}
