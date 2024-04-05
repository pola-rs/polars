use polars_core::prelude::*;

pub fn top_k(s: &[Series], descending: bool) -> PolarsResult<Series> {
    let src = &s[0];
    let k_s = &s[1];

    if src.is_empty() {
        return Ok(src.clone());
    }

    polars_ensure!(
        k_s.len() == 1,
        ComputeError: "`k` must be a single value for `top_k`."
    );

    let k_s = k_s.cast(&IDX_DTYPE)?;
    let k = k_s.idx()?;

    if let Some(k) = k.get(0) {
        let s = src.to_physical_repr();
        Ok(s.sort(!descending, false)?.head(Some(k as usize)))
    } else {
        polars_bail!(ComputeError: "`k` must be set for `top_k`")
    }
}
