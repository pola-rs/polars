use polars_core::prelude::*;
use polars_core::series::IsSorted;

/// Get the run-Lengths of values.
pub fn rle_lengths(s: &Column, lengths: &mut Vec<IdxSize>) -> PolarsResult<()> {
    lengths.clear();
    if s.is_empty() {
        return Ok(());
    }

    if let Some(sc) = s.as_scalar_column() {
        lengths.push(sc.len() as IdxSize);
        return Ok(());
    }

    let (s1, s2) = (s.slice(0, s.len() - 1), s.slice(1, s.len()));
    let s_neq = s1
        .as_materialized_series()
        .not_equal_missing(s2.as_materialized_series())?;
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
        .with_name(PlSmallStr::from_static("value"));
    let outvals = vec![
        Series::from_vec(PlSmallStr::from_static("len"), lengths).into(),
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
