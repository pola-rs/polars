use super::*;

pub(super) fn unique(s: &Column, stable: bool) -> PolarsResult<Column> {
    if stable {
        s.unique_stable()
    } else {
        s.unique()
    }
}
