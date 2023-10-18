use commutative::{
    commutative, commutative_scalar, non_commutative, non_commutative_scalar,
    non_commutative_scalar_swapped,
};
use polars_error::{PolarsError, PolarsResult};

use crate::array::PrimitiveArray;
use crate::datatypes::DataType;

mod add;
mod commutative;
mod div;
mod mul;
mod sub;

pub use add::*;
pub use div::*;
pub use mul::*;
pub use sub::*;

/// Maximum value that can exist with a selected precision
#[inline]
fn max_value(precision: usize) -> i128 {
    10i128.pow(precision as u32) - 1
}

fn get_parameters(lhs: &DataType, rhs: &DataType) -> PolarsResult<(usize, usize)> {
    if let (DataType::Decimal(lhs_p, lhs_s), DataType::Decimal(rhs_p, rhs_s)) =
        (lhs.to_logical_type(), rhs.to_logical_type())
    {
        if lhs_p == rhs_p && lhs_s == rhs_s {
            Ok((*lhs_p, *lhs_s))
        } else {
            Err(PolarsError::InvalidOperation(
                "arrays must have the same precision and scale".into(),
            ))
        }
    } else {
        unreachable!()
    }
}
