//! Contains the operator [`if_then_else`].
use polars_error::{polars_bail, PolarsResult};

use crate::array::{growable, Array, BooleanArray};
use crate::bitmap::utils::SlicesIterator;

/// Returns the values from `lhs` if the predicate is `true` or from the `rhs` if the predicate is false
/// Returns `None` if the predicate is `None`.
pub fn if_then_else(
    predicate: &BooleanArray,
    lhs: &dyn Array,
    rhs: &dyn Array,
) -> PolarsResult<Box<dyn Array>> {
    if lhs.data_type() != rhs.data_type() {
        polars_bail!(InvalidOperation:
            "If then else requires the arguments to have the same datatypes ({:?} != {:?})",
            lhs.data_type(),
            rhs.data_type()
        )
    }
    if (lhs.len() != rhs.len()) | (lhs.len() != predicate.len()) {
        polars_bail!(ComputeError:
            "If then else requires all arguments to have the same length (predicate = {}, lhs = {}, rhs = {})",
            predicate.len(),
            lhs.len(),
            rhs.len()
        );
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
