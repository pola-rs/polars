use polars_core::prelude::*;

/// Get the lengths of runs of identical values.
pub fn rle(s: &Series) -> PolarsResult<Series> {
    let (s1, s2) = (s.slice(0, s.len() - 1), s.slice(1, s.len()));
    let s_neq = s1.not_equal_missing(&s2)?;
    let n_runs = s_neq.sum().unwrap() + 1;
    let mut lengths = Vec::with_capacity(n_runs as usize);
    lengths.push(1);
    let mut vals = Series::new_empty("values", s.dtype());
    let vals = vals.extend(&s.head(Some(1)))?.extend(&s2.filter(&s_neq)?)?;
    let mut idx = 0;
    for v in s_neq.into_iter() {
        if v.unwrap() {
            idx += 1;
            lengths.push(1);
        } else {
            lengths[idx] += 1;
        }
    }

    let outvals = vec![Series::from_vec("lengths", lengths), vals.to_owned()];
    Ok(StructChunked::new("rle", &outvals)?.into_series())
}

/// Similar to `rle`, but maps values to run IDs.
pub fn rle_id(s: &Series) -> PolarsResult<Series> {
    if s.len() == 0 {
        return Ok(Series::new_empty("id", &DataType::UInt32));
    }
    let (s1, s2) = (s.slice(0, s.len() - 1), s.slice(1, s.len()));
    let s_neq = s1.not_equal_missing(&s2)?;

    let mut out = Vec::with_capacity(s.len());
    let mut last = 0;
    out.push(last); // Run numbers start at zero
    for a in s_neq.downcast_iter() {
        for aa in a.values_iter() {
            last += aa as u32;
            out.push(last);
        }
    }
    Ok(Series::from_vec("id", out))
}
