use polars_core::prelude::*;

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

pub fn rleid(s: &Series) -> PolarsResult<Series> {
    let (s1, s2) = (s.slice(0, s.len() - 1), s.slice(1, s.len()));
    // Run numbers start at zero
    Ok(std::iter::once(false)
        .chain(s1.not_equal_missing(&s2)?.into_iter().flatten())
        .scan(0_u32, |acc, x| {
            *acc = *acc + x as u32;
            Some(*acc)
        })
        .collect())
}
