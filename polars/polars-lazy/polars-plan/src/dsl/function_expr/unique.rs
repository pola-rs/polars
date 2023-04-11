use super::*;

pub(super) fn unique(s: &Series, stable: bool) -> PolarsResult<Series> {
    if stable {
        s.unique_stable()
    } else {
        s.unique()
    }
}
