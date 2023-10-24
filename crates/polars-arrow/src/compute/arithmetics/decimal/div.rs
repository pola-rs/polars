//! Defines the division arithmetic kernels for Decimal
//! `PrimitiveArrays`.

use polars_error::{polars_bail, PolarsResult};

use super::{adjusted_precision_scale, get_parameters, max_value, number_digits};
use crate::array::PrimitiveArray;
use crate::compute::arithmetics::{ArrayCheckedDiv, ArrayDiv};
use crate::compute::arity::{binary, binary_checked, unary};
use crate::compute::utils::{check_same_len, combine_validities};
use crate::datatypes::DataType;
use crate::scalar::{PrimitiveScalar, Scalar};

/// Divide two decimal primitive arrays with the same precision and scale. If
/// the precision and scale is different, then an InvalidArgumentError is
/// returned. This function panics if the dividend is divided by 0 or None.
/// This function also panics if the division produces a number larger
/// than the possible number for the array precision.
///
/// # Examples
/// ```
/// use polars_arrow::compute::arithmetics::decimal::div;
/// use polars_arrow::array::PrimitiveArray;
/// use polars_arrow::datatypes::DataType;
///
/// let a = PrimitiveArray::from([Some(1_00i128), Some(4_00i128), Some(6_00i128)]).to(DataType::Decimal(5, 2));
/// let b = PrimitiveArray::from([Some(1_00i128), Some(2_00i128), Some(2_00i128)]).to(DataType::Decimal(5, 2));
///
/// let result = div(&a, &b);
/// let expected = PrimitiveArray::from([Some(1_00i128), Some(2_00i128), Some(3_00i128)]).to(DataType::Decimal(5, 2));
///
/// assert_eq!(result, expected);
/// ```
pub fn div(lhs: &PrimitiveArray<i128>, rhs: &PrimitiveArray<i128>) -> PrimitiveArray<i128> {
    let (precision, scale) = get_parameters(lhs.data_type(), rhs.data_type()).unwrap();

    let scale = 10i128.pow(scale as u32);
    let max = max_value(precision);
    let op = move |a: i128, b: i128| {
        // The division is done using the numbers without scale.
        // The dividend is scaled up to maintain precision after the
        // division

        //   222.222 -->  222222000
        //   123.456 -->     123456
        // --------       ---------
        //     1.800 <--       1800
        let numeral: i128 = a * scale;

        // The division can overflow if the dividend is divided
        // by zero.
        let res: i128 = numeral.checked_div(b).expect("Found division by zero");

        assert!(
            res.abs() <= max,
            "Overflow in multiplication presented for precision {precision}"
        );

        res
    };

    binary(lhs, rhs, lhs.data_type().clone(), op)
}

/// Multiply a decimal [`PrimitiveArray`] with a [`PrimitiveScalar`] with the same precision and scale. If
/// the precision and scale is different, then an InvalidArgumentError is
/// returned. This function panics if the multiplied numbers result in a number
/// larger than the possible number for the selected precision.
pub fn div_scalar(lhs: &PrimitiveArray<i128>, rhs: &PrimitiveScalar<i128>) -> PrimitiveArray<i128> {
    let (precision, scale) = get_parameters(lhs.data_type(), rhs.data_type()).unwrap();

    let rhs = if let Some(rhs) = *rhs.value() {
        rhs
    } else {
        return PrimitiveArray::<i128>::new_null(lhs.data_type().clone(), lhs.len());
    };

    let scale = 10i128.pow(scale as u32);
    let max = max_value(precision);

    let op = move |a: i128| {
        // The division is done using the numbers without scale.
        // The dividend is scaled up to maintain precision after the
        // division

        //   222.222 -->  222222000
        //   123.456 -->     123456
        // --------       ---------
        //     1.800 <--       1800
        let numeral: i128 = a * scale;

        // The division can overflow if the dividend is divided
        // by zero.
        let res: i128 = numeral.checked_div(rhs).expect("Found division by zero");

        assert!(
            res.abs() <= max,
            "Overflow in multiplication presented for precision {precision}"
        );

        res
    };

    unary(lhs, op, lhs.data_type().clone())
}

/// Saturated division of two decimal primitive arrays with the same
/// precision and scale. If the precision and scale is different, then an
/// InvalidArgumentError is returned. If the result from the division is
/// larger than the possible number with the selected precision then the
/// resulted number in the arrow array is the maximum number for the selected
/// precision. The function panics if divided by zero.
///
/// # Examples
/// ```
/// use polars_arrow::compute::arithmetics::decimal::saturating_div;
/// use polars_arrow::array::PrimitiveArray;
/// use polars_arrow::datatypes::DataType;
///
/// let a = PrimitiveArray::from([Some(999_99i128), Some(4_00i128), Some(6_00i128)]).to(DataType::Decimal(5, 2));
/// let b = PrimitiveArray::from([Some(000_01i128), Some(2_00i128), Some(2_00i128)]).to(DataType::Decimal(5, 2));
///
/// let result = saturating_div(&a, &b);
/// let expected = PrimitiveArray::from([Some(999_99i128), Some(2_00i128), Some(3_00i128)]).to(DataType::Decimal(5, 2));
///
/// assert_eq!(result, expected);
/// ```
pub fn saturating_div(
    lhs: &PrimitiveArray<i128>,
    rhs: &PrimitiveArray<i128>,
) -> PrimitiveArray<i128> {
    let (precision, scale) = get_parameters(lhs.data_type(), rhs.data_type()).unwrap();

    let scale = 10i128.pow(scale as u32);
    let max = max_value(precision);

    let op = move |a: i128, b: i128| {
        let numeral: i128 = a * scale;

        match numeral.checked_div(b) {
            Some(res) => match res {
                res if res.abs() > max => {
                    if res > 0 {
                        max
                    } else {
                        -max
                    }
                },
                _ => res,
            },
            None => 0,
        }
    };

    binary(lhs, rhs, lhs.data_type().clone(), op)
}

/// Checked division of two decimal primitive arrays with the same precision
/// and scale. If the precision and scale is different, then an
/// InvalidArgumentError is returned. If the divisor is zero, then the
/// validity for that index is changed to None
///
/// # Examples
/// ```
/// use polars_arrow::compute::arithmetics::decimal::checked_div;
/// use polars_arrow::array::PrimitiveArray;
/// use polars_arrow::datatypes::DataType;
///
/// let a = PrimitiveArray::from([Some(1_00i128), Some(4_00i128), Some(6_00i128)]).to(DataType::Decimal(5, 2));
/// let b = PrimitiveArray::from([Some(000_00i128), None, Some(2_00i128)]).to(DataType::Decimal(5, 2));
///
/// let result = checked_div(&a, &b);
/// let expected = PrimitiveArray::from([None, None, Some(3_00i128)]).to(DataType::Decimal(5, 2));
///
/// assert_eq!(result, expected);
/// ```
pub fn checked_div(lhs: &PrimitiveArray<i128>, rhs: &PrimitiveArray<i128>) -> PrimitiveArray<i128> {
    let (precision, scale) = get_parameters(lhs.data_type(), rhs.data_type()).unwrap();

    let scale = 10i128.pow(scale as u32);
    let max = max_value(precision);

    let op = move |a: i128, b: i128| {
        let numeral: i128 = a * scale;

        match numeral.checked_div(b) {
            Some(res) => match res {
                res if res.abs() > max => None,
                _ => Some(res),
            },
            None => None,
        }
    };

    binary_checked(lhs, rhs, lhs.data_type().clone(), op)
}

// Implementation of ArrayDiv trait for PrimitiveArrays
impl ArrayDiv<PrimitiveArray<i128>> for PrimitiveArray<i128> {
    fn div(&self, rhs: &PrimitiveArray<i128>) -> Self {
        div(self, rhs)
    }
}

// Implementation of ArrayCheckedDiv trait for PrimitiveArrays
impl ArrayCheckedDiv<PrimitiveArray<i128>> for PrimitiveArray<i128> {
    fn checked_div(&self, rhs: &PrimitiveArray<i128>) -> Self {
        checked_div(self, rhs)
    }
}

/// Adaptive division of two decimal primitive arrays with different precision
/// and scale. If the precision and scale is different, then the smallest scale
/// and precision is adjusted to the largest precision and scale. If during the
/// division one of the results is larger than the max possible value, the
/// result precision is changed to the precision of the max value. The function
/// panics when divided by zero.
///
/// ```nocode
///  1000.00   -> 7, 2
///    10.0000 -> 6, 4
/// -----------------
///   100.0000 -> 9, 4
/// ```
/// # Examples
/// ```
/// use polars_arrow::compute::arithmetics::decimal::adaptive_div;
/// use polars_arrow::array::PrimitiveArray;
/// use polars_arrow::datatypes::DataType;
///
/// let a = PrimitiveArray::from([Some(1000_00i128)]).to(DataType::Decimal(7, 2));
/// let b = PrimitiveArray::from([Some(10_0000i128)]).to(DataType::Decimal(6, 4));
/// let result = adaptive_div(&a, &b).unwrap();
/// let expected = PrimitiveArray::from([Some(100_0000i128)]).to(DataType::Decimal(9, 4));
///
/// assert_eq!(result, expected);
/// ```
pub fn adaptive_div(
    lhs: &PrimitiveArray<i128>,
    rhs: &PrimitiveArray<i128>,
) -> PolarsResult<PrimitiveArray<i128>> {
    check_same_len(lhs, rhs)?;

    let (lhs_p, lhs_s, rhs_p, rhs_s) =
        if let (DataType::Decimal(lhs_p, lhs_s), DataType::Decimal(rhs_p, rhs_s)) =
            (lhs.data_type(), rhs.data_type())
        {
            (*lhs_p, *lhs_s, *rhs_p, *rhs_s)
        } else {
            polars_bail!(ComputeError: "Incorrect data type for the array")
        };

    // The resulting precision is mutable because it could change while
    // looping through the iterator
    let (mut res_p, res_s, diff) = adjusted_precision_scale(lhs_p, lhs_s, rhs_p, rhs_s);

    let shift = 10i128.pow(diff as u32);
    let shift_1 = 10i128.pow(res_s as u32);
    let mut max = max_value(res_p);

    let values = lhs
        .values()
        .iter()
        .zip(rhs.values().iter())
        .map(|(l, r)| {
            let numeral: i128 = l * shift_1;

            // Based on the array's scales one of the arguments in the sum has to be shifted
            // to the left to match the final scale
            let res = if lhs_s > rhs_s {
                numeral.checked_div(r * shift)
            } else {
                (numeral * shift).checked_div(*r)
            }
            .expect("Found division by zero");

            // The precision of the resulting array will change if one of the
            // multiplications during the iteration produces a value bigger
            // than the possible value for the initial precision

            //  10.0000 -> 6, 4
            //  00.1000 -> 6, 4
            // -----------------
            // 100.0000 -> 7, 4
            if res.abs() > max {
                res_p = number_digits(res);
                max = max_value(res_p);
            }

            res
        })
        .collect::<Vec<_>>();

    let validity = combine_validities(lhs.validity(), rhs.validity());

    Ok(PrimitiveArray::<i128>::new(
        DataType::Decimal(res_p, res_s),
        values.into(),
        validity,
    ))
}
