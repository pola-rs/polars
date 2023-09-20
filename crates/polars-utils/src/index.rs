use polars_error::{polars_bail, polars_ensure, PolarsResult};

use crate::IdxSize;

pub fn check_bounds(idx: &[IdxSize], len: IdxSize) -> PolarsResult<()> {
    let mut inbounds = true;

    for &i in idx {
        if i >= len {
            // we will not break here as that prevents SIMD
            inbounds = false;
        }
    }
    polars_ensure!(inbounds, ComputeError: "indices are out of bounds");
    Ok(())
}
