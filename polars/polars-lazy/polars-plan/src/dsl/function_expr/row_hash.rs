use super::*;

pub(super) fn row_hash(s: &Series, k0: u64, _k1: u64, _k2: u64, _k3: u64) -> PolarsResult<Series> {
    Ok(s.hash(PlHasherBuilder::with_seed(k0)).into_series())
}
