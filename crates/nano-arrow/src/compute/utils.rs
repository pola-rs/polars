use crate::{
    array::Array,
    bitmap::Bitmap,
    error::{Error, Result},
};

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
pub fn check_same_len(lhs: &dyn Array, rhs: &dyn Array) -> Result<()> {
    if lhs.len() != rhs.len() {
        return Err(Error::InvalidArgumentError(
            "Arrays must have the same length".to_string(),
        ));
    }
    Ok(())
}
