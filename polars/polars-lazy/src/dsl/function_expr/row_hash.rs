use super::*;

pub(super) fn row_hash(s: &Series, k0: u64, k1: u64, k2: u64, k3: u64) -> PolarsResult<Series> {
    Ok(s.hash(ahash::RandomState::with_seeds(k0, k1, k2, k3))
        .into_series())
}
