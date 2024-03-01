use std::hash::Hash;

use polars_core::hashing::_HASHMAP_INIT_SIZE;
use polars_core::prelude::*;
use polars_core::utils::NoNull;
use polars_core::with_match_physical_numeric_polars_type;
use polars_utils::total_ord::{ToTotalOrd, TotalEq, TotalHash};

fn unique_counts_helper<I, J>(items: I) -> IdxCa
where
    I: Iterator<Item = J>,
    J: TotalHash + TotalEq + ToTotalOrd,
    <J as ToTotalOrd>::TotalOrdItem: Hash + Eq,
{
    let mut map = PlIndexMap::with_capacity_and_hasher(_HASHMAP_INIT_SIZE, Default::default());
    for item in items {
        let item = item.to_total_ord();
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
        let s_physical = s.to_physical_repr();

        with_match_physical_numeric_polars_type!(s_physical.dtype(), |$T| {
            let ca: &ChunkedArray<$T> = s_physical.as_ref().as_ref().as_ref();
            Ok(unique_counts_helper(ca.iter()).into_series())
        })
    } else {
        match s.dtype() {
            DataType::String => {
                Ok(unique_counts_helper(s.str().unwrap().into_iter()).into_series())
            },
            DataType::Null => {
                let ca = if s.is_empty() {
                    IdxCa::new(s.name(), [] as [IdxSize; 0])
                } else {
                    IdxCa::new(s.name(), [s.len() as IdxSize])
                };
                Ok(ca.into_series())
            },
            dt => {
                polars_bail!(opq = unique_counts, dt)
            },
        }
    }
}
