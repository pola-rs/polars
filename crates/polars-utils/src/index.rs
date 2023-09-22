use polars_error::{polars_bail, polars_ensure, PolarsResult};

use crate::IdxSize;

pub fn check_bounds(idx: &[IdxSize], len: IdxSize) -> PolarsResult<()> {
    // We iterate in large uninterrupted chunks to help auto-vectorization.
    let mut in_bounds = true;
    for chunk in idx.chunks(1024) {
        for i in chunk {
            if *i >= len {
                in_bounds = false;
            }
        }
        if !in_bounds {
            break;
        }
    }
    polars_ensure!(in_bounds, ComputeError: "indices are out of bounds");
    Ok(())
}
