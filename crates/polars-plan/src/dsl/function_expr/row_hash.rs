use super::*;

pub(super) fn row_hash(c: &Column, k0: u64, k1: u64, k2: u64, k3: u64) -> PolarsResult<Column> {
    // @scalar-opt
    Ok(c.as_materialized_series()
        .hash(PlRandomState::with_seeds(k0, k1, k2, k3))
        .into_column())
}
