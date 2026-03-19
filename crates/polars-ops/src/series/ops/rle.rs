use std::hash::Hash;

use polars_core::prelude::*;
use polars_core::series::IsSorted;
use polars_core::with_match_physical_numeric_polars_type;
use polars_utils::select::select_unpredictable;
use polars_utils::total_ord::{ToTotalOrd, TotalEq, TotalHash};

pub static RLE_VALUE_COLUMN_NAME: &str = "value";
pub static RLE_LENGTH_COLUMN_NAME: &str = "len";

/// Get the run-lengths of values.
pub fn rle_lengths(s: &Column, lengths: &mut Vec<IdxSize>) -> PolarsResult<()> {
    lengths.clear();
    if s.is_empty() {
        return Ok(());
    }
    if s.len() == 1 {
        lengths.push(1);
        return Ok(());
    }

    if let Some(sc) = s.as_scalar_column() {
        lengths.push(sc.len() as IdxSize);
        return Ok(());
    }

    let s = s.as_materialized_series().to_physical_repr();
    match s.dtype() {
        DataType::Boolean => {
            let ca: &BooleanChunked = s.as_ref().as_ref().as_ref();
            rle_lengths_helper_ca(ca, lengths);
            return Ok(());
        },
        dt if dt.is_numeric() => {
            with_match_physical_numeric_polars_type!(dt, |$T| {
                let ca: &ChunkedArray<$T> = s.as_ref().as_ref().as_ref();
                rle_lengths_helper_ca(ca, lengths);
                return Ok(());
            })
        },
        DataType::String => {
            let ca: &StringChunked = s.as_ref().as_ref().as_ref();
            rle_lengths_helper_ca(&ca.as_binary(), lengths);
            return Ok(());
        },
        DataType::Binary => {
            let ca: &BinaryChunked = s.as_ref().as_ref().as_ref();
            rle_lengths_helper_ca(ca, lengths);
            return Ok(());
        },
        _ => {},
    }

    let (s1, s2) = (s.slice(0, s.len() - 1), s.slice(1, s.len()));
    let s_neq = s1.not_equal_missing(&s2)?;
    let n_runs = s_neq.sum().unwrap() + 1;

    lengths.reserve(n_runs as usize);
    lengths.push(1);

    assert!(!s_neq.has_nulls());
    for arr in s_neq.downcast_iter() {
        let mut values = arr.values().clone();
        while !values.is_empty() {
            // @NOTE: This `as IdxSize` is safe because it is less than or equal to the a ChunkedArray
            // length.
            *lengths.last_mut().unwrap() += values.take_leading_zeros() as IdxSize;

            if !values.is_empty() {
                lengths.push(1);
                values.slice(1, values.len() - 1);
            }
        }
    }
    Ok(())
}

fn rle_lengths_helper_ca<'a, T>(ca: &'a ChunkedArray<T>, lengths: &mut Vec<IdxSize>)
where
    T: PolarsDataType,
    T::Physical<'a>: TotalHash + TotalEq + ToTotalOrd + Copy,
    <T::Physical<'a> as ToTotalOrd>::TotalOrdItem: Hash + Eq + Copy,
{
    lengths.clear();
    if ca.is_empty() {
        return;
    }

    unsafe {
        lengths.reserve(ca.len());

        if ca.has_nulls() {
            let mut prev = ca.get_unchecked(0).map(|v| v.to_total_ord());
            let mut out_idx = 0;
            let mut run_len = 0;
            for arr in ca.downcast_iter() {
                for val in arr.iter() {
                    let val = val.map(|v| v.to_total_ord());
                    let diff = val != prev;
                    run_len = 1 + select_unpredictable(diff, 0, run_len);
                    out_idx += diff as usize;
                    lengths.as_mut_ptr().add(out_idx).write(run_len);
                    prev = val;
                }
            }
            lengths.set_len(out_idx + 1);
        } else {
            let mut prev = ca.value_unchecked(0).to_total_ord();
            let mut out_idx = 0;
            let mut run_len = 0;
            for arr in ca.downcast_iter() {
                for val in arr.values_iter() {
                    let val = val.to_total_ord();
                    let diff = val != prev;
                    run_len = 1 + select_unpredictable(diff, 0, run_len);
                    out_idx += diff as usize;
                    lengths.as_mut_ptr().add(out_idx).write(run_len);
                    prev = val;
                }
            }
            lengths.set_len(out_idx + 1);
        }
    }
}

/// Get the lengths of runs of identical values.
pub fn rle(s: &Column) -> PolarsResult<Column> {
    let mut lengths = Vec::new();
    rle_lengths(s, &mut lengths)?;

    let mut idxs = Vec::with_capacity(lengths.len());
    if !lengths.is_empty() {
        idxs.push(0);
        for length in &lengths[..lengths.len() - 1] {
            idxs.push(*idxs.last().unwrap() + length);
        }
    }

    let vals = s
        .take_slice(&idxs)
        .unwrap()
        .with_name(PlSmallStr::from_static(RLE_VALUE_COLUMN_NAME));
    let outvals = vec![
        Series::from_vec(PlSmallStr::from_static(RLE_LENGTH_COLUMN_NAME), lengths).into(),
        vals,
    ];
    Ok(StructChunked::from_columns(s.name().clone(), idxs.len(), &outvals)?.into_column())
}

/// Similar to `rle`, but maps values to run IDs.
pub fn rle_id(s: &Column) -> PolarsResult<Column> {
    if s.is_empty() {
        return Ok(Column::new_empty(s.name().clone(), &IDX_DTYPE));
    }

    let (s1, s2) = (s.slice(0, s.len() - 1), s.slice(1, s.len()));
    let s_neq = s1
        .as_materialized_series()
        .not_equal_missing(s2.as_materialized_series())?;

    let mut out = Vec::<IdxSize>::with_capacity(s.len());
    let mut last = 0;
    out.push(last); // Run numbers start at zero
    assert_eq!(s_neq.null_count(), 0);
    for a in s_neq.downcast_iter() {
        for aa in a.values_iter() {
            last += aa as IdxSize;
            out.push(last);
        }
    }
    Ok(IdxCa::from_vec(s.name().clone(), out)
        .with_sorted_flag(IsSorted::Ascending)
        .into_column())
}
