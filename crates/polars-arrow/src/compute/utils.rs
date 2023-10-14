use polars_error::{polars_bail, polars_ensure, PolarsResult};

use crate::array::Array;
use crate::bitmap::Bitmap;

pub fn combine_validities(lhs: Option<&Bitmap>, rhs: Option<&Bitmap>) -> Option<Bitmap> {
    match (lhs, rhs) {
        (Some(lhs), None) => Some(lhs.clone()),
        (None, Some(rhs)) => Some(rhs.clone()),
        (None, None) => None,
        (Some(lhs), Some(rhs)) => Some(lhs & rhs),
    }
}

// Errors iff the two arrays have a different length.
#[inline]
pub fn check_same_len(lhs: &dyn Array, rhs: &dyn Array) -> PolarsResult<()> {
    polars_ensure!(lhs.len() == rhs.len(), ComputeError:
            "arrays must have the same length"
    );
    Ok(())
}
