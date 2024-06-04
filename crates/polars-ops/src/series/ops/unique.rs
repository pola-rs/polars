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

    let (true_count, null_count);
    if let Some(validity) = arr.validity() {
        null_count = validity.unset_bits();
        true_count = (arr.values() & validity).set_bits();
    } else {
        null_count = 0;
        true_count = arr.values().set_bits();
    }
    let false_count = arr.len() - true_count - null_count;

    if true_count == 0 && false_count == 0 {
        return IdxCa::new(ca.name(), [null_count as IdxSize]);
    }
    if true_count == 0 && null_count == 0 {
        return IdxCa::new(ca.name(), [false_count as IdxSize]);
    }
    if false_count == 0 && null_count == 0 {
        return IdxCa::new(ca.name(), [true_count as IdxSize]);
    }

    if true_count == 0 {
        match arr.is_null(0) {
            true => return IdxCa::new(ca.name(), [null_count as IdxSize, false_count as IdxSize]),
            false => return IdxCa::new(ca.name(), [false_count as IdxSize, null_count as IdxSize]),
        }
    } else if false_count == 0 {
        match arr.is_null(0) {
            true => return IdxCa::new(ca.name(), [null_count as IdxSize, true_count as IdxSize]),
            false => return IdxCa::new(ca.name(), [true_count as IdxSize, null_count as IdxSize]),
        }
    } else if null_count == 0 {
        match arr.value(0) {
            true => return IdxCa::new(ca.name(), [true_count as IdxSize, false_count as IdxSize]),
            false => return IdxCa::new(ca.name(), [false_count as IdxSize, true_count as IdxSize]),
        }
    }

    let (mut true_index, mut null_index): (Option<usize>, Option<usize>) = (None, None);

    if arr.is_null(0) {
        null_index = Some(0);
    } else if arr.value(0) {
        true_index = Some(0);
    }

    if let Some(0) = null_index {
        let first_non_null = arr.validity().unwrap().iter().position(|v| v).unwrap();
        match arr.value(first_non_null) {
            true => {
                return IdxCa::new(
                    ca.name(),
                    [
                        null_count as IdxSize,
                        true_count as IdxSize,
                        false_count as IdxSize,
                    ],
                )
            },
            false => {
                return IdxCa::new(
                    ca.name(),
                    [
                        null_count as IdxSize,
                        false_count as IdxSize,
                        true_count as IdxSize,
                    ],
                )
            },
        }
    } else if let Some(0) = true_index {
        let first_non_true = arr
            .validity()
            .unwrap()
            .iter()
            .zip(arr.values())
            .position(|(v, val)| (v && !val) || !v)
            .unwrap();
        match arr.is_null(first_non_true) {
            true => {
                return IdxCa::new(
                    ca.name(),
                    [
                        true_count as IdxSize,
                        null_count as IdxSize,
                        false_count as IdxSize,
                    ],
                )
            },
            false => {
                return IdxCa::new(
                    ca.name(),
                    [
                        true_count as IdxSize,
                        false_count as IdxSize,
                        null_count as IdxSize,
                    ],
                )
            },
        }
    } else {
        let first_non_false = arr
            .validity()
            .unwrap()
            .iter()
            .zip(arr.values())
            .position(|(v, val)| (v && val) || !v)
            .unwrap();
        match arr.is_null(first_non_false) {
            true => {
                return IdxCa::new(
                    ca.name(),
                    [
                        false_count as IdxSize,
                        null_count as IdxSize,
                        true_count as IdxSize,
                    ],
                )
            },
            false => {
                return IdxCa::new(
                    ca.name(),
                    [
                        false_count as IdxSize,
                        true_count as IdxSize,
                        null_count as IdxSize,
                    ],
                )
            },
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
