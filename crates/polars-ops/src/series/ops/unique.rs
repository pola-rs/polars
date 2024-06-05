use std::hash::Hash;

use arrow::array::Array;
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

    let (n_true, n_null);
    if let Some(validity) = arr.validity() {
        n_null = validity.unset_bits();
        if n_null < arr.len() {
            n_true = (arr.values() & validity).set_bits();
        } else {
            n_true = 0;
        }
    } else {
        n_null = 0;
        n_true = arr.values().set_bits();
    }
    let n_false = arr.len() - n_true - n_null;
    let (n_true, n_false, n_null) = (n_true as IdxSize, n_false as IdxSize, n_null as IdxSize);

    if n_true == 0 && n_false == 0 {
        return IdxCa::new(ca.name(), [n_null]);
    }
    if n_true == 0 && n_null == 0 {
        return IdxCa::new(ca.name(), [n_false]);
    }
    if n_false == 0 && n_null == 0 {
        return IdxCa::new(ca.name(), [n_true]);
    }

    if n_true == 0 {
        match arr.is_null(0) {
            true => return IdxCa::new(ca.name(), [n_null, n_false]),
            false => return IdxCa::new(ca.name(), [n_false, n_null]),
        }
    } else if n_false == 0 {
        match arr.is_null(0) {
            true => return IdxCa::new(ca.name(), [n_null, n_true]),
            false => return IdxCa::new(ca.name(), [n_true, n_null]),
        }
    } else if n_null == 0 {
        match arr.value(0) {
            true => return IdxCa::new(ca.name(), [n_true, n_false]),
            false => return IdxCa::new(ca.name(), [n_false, n_true]),
        }
    }

    if arr.is_null(0) {
        let first_non_null = arr.validity().unwrap().iter().position(|v| v).unwrap();
        match arr.value(first_non_null) {
            true => return IdxCa::new(ca.name(), [n_null, n_true, n_false]),
            false => return IdxCa::new(ca.name(), [n_null, n_false, n_true]),
        }
    } else {
        let first_unique = arr.value(0);
        let second_unique = arr
            .validity()
            .unwrap()
            .iter()
            .zip(arr.values())
            .position(|(v, val)| !v || val != first_unique)
            .unwrap();

        match (first_unique, arr.is_null(second_unique)) {
            (true, true) => return IdxCa::new(ca.name(), [n_true, n_null, n_false]),
            (true, false) => return IdxCa::new(ca.name(), [n_true, n_false, n_null]),
            (false, true) => return IdxCa::new(ca.name(), [n_false, n_null, n_true]),
            (false, false) => return IdxCa::new(ca.name(), [n_false, n_true, n_null]),
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
