use std::borrow::Cow;
use std::hash::Hash;

use polars_core::hashing::_HASHMAP_INIT_SIZE;
use polars_core::prelude::row_encode::encode_rows_unordered;
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
    if s.is_empty() {
        return Ok(IdxCa::new(s.name().clone(), [] as [IdxSize; 0]).into_series());
    } else if s.null_count() == s.len() {
        return Ok(IdxCa::new(s.name().clone(), [s.len() as IdxSize]).into_series());
    }

    let mut s = Cow::Borrowed(s);

    if s.dtype().is_nested() {
        s = Cow::Owned(encode_rows_unordered(&[s.into_owned().into_column()])?.into_series());
    }

    match s.dtype().to_physical() {
        dt if dt.is_primitive_numeric() => {
            let s_physical = s.to_physical_repr();
            with_match_physical_numeric_polars_type!(s_physical.dtype(), |$T| {
                let ca: &ChunkedArray<$T> = s_physical.as_ref().as_ref().as_ref();
                Ok(unique_counts_helper(ca.iter()).into_series())
            })
        },
        DataType::Null => unreachable!("handled before"),
        DataType::BinaryOffset => {
            let ca = s.binary_offset()?;
            Ok(unique_counts_helper(ca.into_iter()).into_series())
        },
        DataType::Binary => {
            let ca = s.binary()?;
            Ok(unique_counts_helper(ca.into_iter()).into_series())
        },
        DataType::String => {
            let ca = s.str()?.as_binary();
            Ok(unique_counts_helper(ca.into_iter()).into_series())
        },
        DataType::Boolean => {
            let ca = s.bool()?;

            let num_trues = ca.num_trues() as IdxSize;
            let num_nulls = ca.null_count() as IdxSize;
            let num_falses = ca.len() as IdxSize - num_trues - num_nulls;

            let values: Vec<IdxSize> = match ca.get(0) {
                Some(false) if num_nulls == 0 && num_trues == 0 => vec![num_falses],
                Some(false) if num_nulls == 0 => vec![num_falses, num_trues],
                Some(false) if num_trues == 0 => vec![num_falses, num_nulls],

                Some(true) if num_nulls == 0 && num_falses == 0 => vec![num_trues],
                Some(true) if num_nulls == 0 => vec![num_trues, num_falses],
                Some(true) if num_falses == 0 => vec![num_trues, num_nulls],

                None if num_trues == 0 && num_falses == 0 => unreachable!(),
                None if num_trues == 0 => vec![num_nulls, num_falses],
                None if num_falses == 0 => vec![num_nulls, num_trues],

                Some(false) => {
                    let first_true = ca.first_true_idx().unwrap();
                    let first_null = ca.first_null().unwrap();

                    if first_true < first_null {
                        vec![num_falses, num_trues, num_nulls]
                    } else {
                        vec![num_falses, num_nulls, num_trues]
                    }
                },
                Some(true) => {
                    let first_false = ca.first_false_idx().unwrap();
                    let first_null = ca.first_null().unwrap();

                    if first_false < first_null {
                        vec![num_trues, num_falses, num_nulls]
                    } else {
                        vec![num_trues, num_nulls, num_falses]
                    }
                },
                None => {
                    if ca.get(ca.first_non_null().unwrap()).unwrap() {
                        vec![num_nulls, num_trues, num_falses]
                    } else {
                        vec![num_nulls, num_falses, num_trues]
                    }
                },
            };
            Ok(IdxCa::new(s.name().clone(), values).into_series())
        },

        #[cfg(feature = "dtype-extension")]
        DataType::Extension(_, _) => unique_counts(s.ext().unwrap().storage()),

        DataType::UInt8
        | DataType::UInt16
        | DataType::UInt32
        | DataType::UInt64
        | DataType::UInt128
        | DataType::Int8
        | DataType::Int16
        | DataType::Int32
        | DataType::Int64
        | DataType::Int128
        | DataType::Float16
        | DataType::Float32
        | DataType::Float64
        | DataType::Date
        | DataType::Datetime(..)
        | DataType::Duration(..)
        | DataType::Time => unreachable!("primitive numeric"),
        #[cfg(feature = "dtype-decimal")]
        DataType::Decimal(..) => unreachable!("primitive numeric"),
        #[cfg(feature = "dtype-categorical")]
        DataType::Categorical(..) | DataType::Enum(..) => unreachable!("primitive numeric"),
        #[cfg(feature = "dtype-array")]
        DataType::Array(..) => unreachable!("row encoded"),
        #[cfg(feature = "dtype-struct")]
        DataType::Struct(..) => unreachable!("row encoded"),
        DataType::List(..) => {
            unreachable!("row encoded")
        },
        #[cfg(feature = "object")]
        dt @ DataType::Object(..) => polars_bail!(opq = unique_counts, dt),
        dt @ DataType::Unknown(..) => polars_bail!(opq = unique_counts, dt),
    }
}
