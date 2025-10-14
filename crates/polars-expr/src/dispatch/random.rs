use polars_core::error::{PolarsResult, polars_ensure};
use polars_core::prelude::DataType::Float64;
use polars_core::prelude::{Column, IDX_DTYPE};

pub(super) fn shuffle(s: &Column, seed: Option<u64>) -> PolarsResult<Column> {
    Ok(s.shuffle(seed))
}

pub(super) fn sample_frac(
    s: &[Column],
    with_replacement: bool,
    shuffle: bool,
    seed: Option<u64>,
) -> PolarsResult<Column> {
    let src = &s[0];
    let frac_s = &s[1];

    polars_ensure!(
        frac_s.len() == 1,
        ComputeError: "Sample fraction must be a single value."
    );

    let frac_s = frac_s.cast(&Float64)?;
    let frac = frac_s.f64()?;

    match frac.get(0) {
        Some(frac) => src.sample_frac(frac, with_replacement, shuffle, seed),
        None => Ok(Column::new_empty(src.name().clone(), src.dtype())),
    }
}

pub(super) fn sample_n(
    s: &[Column],
    with_replacement: bool,
    shuffle: bool,
    seed: Option<u64>,
) -> PolarsResult<Column> {
    let src = &s[0];
    let n_s = &s[1];

    polars_ensure!(
        n_s.len() == 1,
        ComputeError: "Sample size must be a single value."
    );

    let n_s = n_s.cast(&IDX_DTYPE)?;
    let n = n_s.idx()?;

    match n.get(0) {
        Some(n) => src.sample_n(n as usize, with_replacement, shuffle, seed),
        None => Ok(Column::new_empty(src.name().clone(), src.dtype())),
    }
}
