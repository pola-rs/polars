//! Contains the operator [`if_then_else`].
use crate::array::{growable, Array, BooleanArray};
use crate::bitmap::utils::SlicesIterator;
use crate::error::{Error, Result};

/// Returns the values from `lhs` if the predicate is `true` or from the `rhs` if the predicate is false
/// Returns `None` if the predicate is `None`.
/// # Example
/// ```rust
/// # use arrow2::error::Result;
/// use arrow2::compute::if_then_else::if_then_else;
/// use arrow2::array::{Int32Array, BooleanArray};
///
/// # fn main() -> Result<()> {
/// let lhs = Int32Array::from_slice(&[1, 2, 3]);
/// let rhs = Int32Array::from_slice(&[4, 5, 6]);
/// let predicate = BooleanArray::from(&[Some(true), None, Some(false)]);
/// let result = if_then_else(&predicate, &lhs, &rhs)?;
///
/// let expected = Int32Array::from(&[Some(1), None, Some(6)]);
///
/// assert_eq!(expected, result.as_ref());
/// # Ok(())
/// # }
/// ```
pub fn if_then_else(
    predicate: &BooleanArray,
    lhs: &dyn Array,
    rhs: &dyn Array,
) -> Result<Box<dyn Array>> {
    if lhs.data_type() != rhs.data_type() {
        return Err(Error::InvalidArgumentError(format!(
            "If then else requires the arguments to have the same datatypes ({:?} != {:?})",
            lhs.data_type(),
            rhs.data_type()
        )));
    }
    if (lhs.len() != rhs.len()) | (lhs.len() != predicate.len()) {
        return Err(Error::InvalidArgumentError(format!(
            "If then else requires all arguments to have the same length (predicate = {}, lhs = {}, rhs = {})",
            predicate.len(),
            lhs.len(),
            rhs.len()
        )));
    }

    let result = if predicate.null_count() > 0 {
        let mut growable = growable::make_growable(&[lhs, rhs], true, lhs.len());
        for (i, v) in predicate.iter().enumerate() {
            match v {
                Some(v) => growable.extend(!v as usize, i, 1),
                None => growable.extend_validity(1),
            }
        }
        growable.as_box()
    } else {
        let mut growable = growable::make_growable(&[lhs, rhs], false, lhs.len());
        let mut start_falsy = 0;
        let mut total_len = 0;
        for (start, len) in SlicesIterator::new(predicate.values()) {
            if start != start_falsy {
                growable.extend(1, start_falsy, start - start_falsy);
                total_len += start - start_falsy;
            };
            growable.extend(0, start, len);
            total_len += len;
            start_falsy = start + len;
        }
        if total_len != lhs.len() {
            growable.extend(1, total_len, lhs.len() - total_len);
        }
        growable.as_box()
    };
    Ok(result)
}
