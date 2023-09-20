//! Defines the arithmetic kernels for Decimal `PrimitiveArrays`. The
//! [`Decimal`](crate::datatypes::DataType::Decimal) type specifies the
//! precision and scale parameters. These affect the arithmetic operations and
//! need to be considered while doing operations with Decimal numbers.

mod add;
pub use add::*;
mod div;
pub use div::*;
mod mul;
pub use mul::*;
mod sub;
pub use sub::*;

use crate::datatypes::DataType;
use crate::error::{Error, Result};

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

fn get_parameters(lhs: &DataType, rhs: &DataType) -> Result<(usize, usize)> {
    if let (DataType::Decimal(lhs_p, lhs_s), DataType::Decimal(rhs_p, rhs_s)) =
        (lhs.to_logical_type(), rhs.to_logical_type())
    {
        if lhs_p == rhs_p && lhs_s == rhs_s {
            Ok((*lhs_p, *lhs_s))
        } else {
            Err(Error::InvalidArgumentError(
                "Arrays must have the same precision and scale".to_string(),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_max_value() {
        assert_eq!(999, max_value(3));
        assert_eq!(99999, max_value(5));
        assert_eq!(999999, max_value(6));
    }

    #[test]
    fn test_number_digits() {
        assert_eq!(2, number_digits(12i128));
        assert_eq!(3, number_digits(123i128));
        assert_eq!(4, number_digits(1234i128));
        assert_eq!(6, number_digits(123456i128));
        assert_eq!(7, number_digits(1234567i128));
        assert_eq!(7, number_digits(-1234567i128));
        assert_eq!(3, number_digits(-123i128));
    }

    #[test]
    fn test_adjusted_precision_scale() {
        //    11.1111 -> 5, 4  ->   2        4
        // 11111.01   -> 7, 2  ->   5        2
        // -----------------
        // 11122.1211 -> 9, 4  ->   5        4
        assert_eq!((9, 4, 2), adjusted_precision_scale(5, 4, 7, 2))
    }
}
