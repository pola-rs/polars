use arrow::datatypes::DataType;
use polars_error::{PolarsError, PolarsResult};
use arrow::array::PrimitiveArray;
use commutative::{
    commutative_scalar, commutative, non_commutative, non_commutative_scalar_swapped, non_commutative_scalar
};

mod add;
mod sub;
mod mul;
mod commutative;

pub use add::*;
pub use sub::*;
pub use mul::*;

/// Maximum value that can exist with a selected precision
#[inline]
fn max_value(precision: usize) -> i128 {
    10i128.pow(precision as u32) - 1
}

// Calculates the number of digits in a i128 number
fn number_digits(num: i128) -> usize {
    let mut num = num.abs();
    let mut digit: i128 = 0;
    let base = 10i128;

    while num != 0 {
        num /= base;
        digit += 1;
    }

    digit as usize
}

fn get_parameters(lhs: &DataType, rhs: &DataType) -> PolarsResult<(usize, usize)> {
    if let (DataType::Decimal(lhs_p, lhs_s), DataType::Decimal(rhs_p, rhs_s)) =
        (lhs.to_logical_type(), rhs.to_logical_type())
    {
        if lhs_p == rhs_p && lhs_s == rhs_s {
            Ok((*lhs_p, *lhs_s))
        } else {
            Err(PolarsError::InvalidOperation(
                "Arrays must have the same precision and scale".into(),
            ))
        }
    } else {
        unreachable!()
    }
}

/// Returns the adjusted precision and scale for the lhs and rhs precision and
/// scale
fn adjusted_precision_scale(
    lhs_p: usize,
    lhs_s: usize,
    rhs_p: usize,
    rhs_s: usize,
) -> (usize, usize, usize) {
    // The initial new precision and scale is based on the number of digits
    // that lhs and rhs number has before and after the point. The max
    // number of digits before and after the point will make the last
    // precision and scale of the result

    //                        Digits before/after point
    //                        before    after
    //    11.1111 -> 5, 4  ->   2        4
    // 11111.01   -> 7, 2  ->   5        2
    // -----------------
    // 11122.1211 -> 9, 4  ->   5        4
    let lhs_digits_before = lhs_p - lhs_s;
    let rhs_digits_before = rhs_p - rhs_s;

    let res_digits_before = std::cmp::max(lhs_digits_before, rhs_digits_before);

    let (res_s, diff) = if lhs_s > rhs_s {
        (lhs_s, lhs_s - rhs_s)
    } else {
        (rhs_s, rhs_s - lhs_s)
    };

    let res_p = res_digits_before + res_s;

    (res_p, res_s, diff)
}
