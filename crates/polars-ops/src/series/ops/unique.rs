use std::hash::Hash;

use polars_core::hashing::_HASHMAP_INIT_SIZE;
use polars_core::prelude::*;
use polars_core::utils::NoNull;

fn unique_counts_helper<I, J>(items: I) -> IdxCa
where
    I: Iterator<Item = J>,
    J: Hash + Eq,
{
    let mut map = PlIndexMap::with_capacity_and_hasher(_HASHMAP_INIT_SIZE, Default::default());
    for item in items {
        map.entry(item)
            .and_modify(|cnt| {
                *cnt += 1;
            })
            .or_insert(1 as IdxSize);
    }
    let out: NoNull<IdxCa> = map.into_values().collect();
    out.into_inner()
}

/// Returns a count of the unique values in the order of appearance.
pub fn unique_counts(s: &Series) -> PolarsResult<Series> {
    if s.dtype().to_physical().is_numeric() {
        if s.bit_repr_is_large() {
            let ca = s.bit_repr_large();
            Ok(unique_counts_helper(ca.into_iter()).into_series())
        } else {
            let ca = s.bit_repr_small();
            Ok(unique_counts_helper(ca.into_iter()).into_series())
        }
    } else {
        match s.dtype() {
            DataType::Utf8 => Ok(unique_counts_helper(s.utf8().unwrap().into_iter()).into_series()),
            dt => {
                polars_bail!(opq = unique_counts, dt)
            },
        }
    }
}
