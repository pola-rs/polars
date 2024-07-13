use polars_core::prelude::*;
use polars_core::series::IsSorted;

/// Get the lengths of runs of identical values.
pub fn rle(s: &Series) -> PolarsResult<Series> {
    let (s1, s2) = (s.slice(0, s.len() - 1), s.slice(1, s.len()));
    let s_neq = s1.not_equal_missing(&s2)?;
    let n_runs = s_neq.sum().ok_or_else(|| polars_err!(InvalidOperation: "could not evaluate 'rle_id' on series of dtype: {}", s.dtype()))? + 1;

    let mut lengths = Vec::<IdxSize>::with_capacity(n_runs as usize);
    lengths.push(1);
    let mut vals = Series::new_empty("value", s.dtype());
    let vals = vals.extend(&s.head(Some(1)))?.extend(&s2.filter(&s_neq)?)?;
    let mut idx = 0;

    assert_eq!(s_neq.null_count(), 0);
    for arr in s_neq.downcast_iter() {
        for v in arr.values_iter() {
            if v {
                idx += 1;
                lengths.push(1)
            } else {
                lengths[idx] += 1;
            }
        }
    }

    let outvals = vec![Series::from_vec("len", lengths), vals.to_owned()];
    Ok(StructChunked::from_series(s.name(), &outvals)?.into_series())
}

/// Similar to `rle`, but maps values to run IDs.
pub fn rle_id(s: &Series) -> PolarsResult<Series> {
    if s.len() == 0 {
        return Ok(Series::new_empty(s.name(), &IDX_DTYPE));
    }
    let (s1, s2) = (s.slice(0, s.len() - 1), s.slice(1, s.len()));
    let s_neq = s1.not_equal_missing(&s2)?;

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
    Ok(IdxCa::from_vec(s.name(), out)
        .with_sorted_flag(IsSorted::Ascending)
        .into_series())
}
