use std::hash::Hash;

use arrow::legacy::bit_util::find_first_true_false_null;
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

fn unique_counts_boolean_helper(ca: &BooleanChunked) -> IdxCa {
    if ca.is_empty() {
        return IdxCa::new(ca.name(), [] as [IdxSize; 0]);
    }

    let ca = ca.rechunk();
    let arr = ca.downcast_iter().next().unwrap();

    if let Some(validity) = arr.validity() {
        let num_true = arr.values().set_bits();
        let num_null = validity.unset_bits();
        let num_false = arr.len() - num_true - num_null;

        let (true_index, false_index, null_index) = find_first_true_false_null(
            arr.values().chunks::<u64>(),
            arr.validity().unwrap().chunks::<u64>(),
        );

        let mut idx_vec = vec![
            (true_index, num_true),
            (false_index, num_false),
            (null_index, num_null),
        ];

        idx_vec.retain(|(idx, _)| idx.is_some());
        idx_vec.sort_by_key(|(idx, _)| idx.unwrap());

        let out = idx_vec
            .into_iter()
            .map(|(_, cnt)| cnt as IdxSize)
            .collect::<Vec<IdxSize>>();

        IdxCa::from_vec(ca.name(), out)
    } else {
        let num_true = arr.values().set_bits();
        let num_false = arr.len() - num_true;

        if num_true == 0 {
            return IdxCa::new(ca.name(), [num_false as IdxSize]);
        }

        if num_false == 0 {
            return IdxCa::new(ca.name(), [num_true as IdxSize]);
        }

        let first_is_true = arr.value(0);
        if first_is_true {
            IdxCa::new(ca.name(), [num_true as IdxSize, num_false as IdxSize])
        } else {
            IdxCa::new(ca.name(), [num_false as IdxSize, num_true as IdxSize])
        }
    }
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
            DataType::Boolean => {
                let ca = s.bool().unwrap();
                Ok(unique_counts_boolean_helper(ca).into_series())
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
