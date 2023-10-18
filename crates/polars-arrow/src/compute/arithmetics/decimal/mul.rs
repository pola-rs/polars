//! Defines the multiplication arithmetic kernels for Decimal
//! `PrimitiveArrays`.

use polars_error::{polars_bail, PolarsResult};

use super::{adjusted_precision_scale, get_parameters, max_value, number_digits};
use crate::array::PrimitiveArray;
use crate::compute::arithmetics::{ArrayCheckedMul, ArrayMul, ArraySaturatingMul};
use crate::compute::arity::{binary, binary_checked, unary};
use crate::compute::utils::{check_same_len, combine_validities};
use crate::datatypes::DataType;
use crate::scalar::{PrimitiveScalar, Scalar};

/// Multiply two decimal primitive arrays with the same precision and scale. If
/// the precision and scale is different, then an InvalidArgumentError is
/// returned. This function panics if the multiplied numbers result in a number
/// larger than the possible number for the selected precision.
///
/// # Examples
/// ```
/// use polars_arrow::compute::arithmetics::decimal::mul;
/// use polars_arrow::array::PrimitiveArray;
/// use polars_arrow::datatypes::DataType;
///
/// let a = PrimitiveArray::from([Some(1_00i128), Some(1_00i128), None, Some(2_00i128)]).to(DataType::Decimal(5, 2));
/// let b = PrimitiveArray::from([Some(1_00i128), Some(2_00i128), None, Some(2_00i128)]).to(DataType::Decimal(5, 2));
///
/// let result = mul(&a, &b);
/// let expected = PrimitiveArray::from([Some(1_00i128), Some(2_00i128), None, Some(4_00i128)]).to(DataType::Decimal(5, 2));
///
/// assert_eq!(result, expected);
/// ```
pub fn mul(lhs: &PrimitiveArray<i128>, rhs: &PrimitiveArray<i128>) -> PrimitiveArray<i128> {
    let (precision, scale) = get_parameters(lhs.data_type(), rhs.data_type()).unwrap();

    let scale = 10i128.pow(scale as u32);
    let max = max_value(precision);

    let op = move |a: i128, b: i128| {
        // The multiplication between i128 can overflow if they are
        // very large numbers. For that reason a checked
        // multiplication is used.
        let res: i128 = a.checked_mul(b).expect("major overflow for multiplication");

        // The multiplication is done using the numbers without scale.
        // The resulting scale of the value has to be corrected by
        // dividing by (10^scale)

        //   111.111 -->      111111
        //   222.222 -->      222222
        // --------          -------
        // 24691.308 <-- 24691308642
        let res = res / scale;

        assert!(
            res.abs() <= max,
            "overflow in multiplication presented for precision {precision}"
        );

        res
    };

    binary(lhs, rhs, lhs.data_type().clone(), op)
}

/// Multiply a decimal [`PrimitiveArray`] with a [`PrimitiveScalar`] with the same precision and scale. If
/// the precision and scale is different, then an InvalidArgumentError is
/// returned. This function panics if the multiplied numbers result in a number
/// larger than the possible number for the selected precision.
pub fn mul_scalar(lhs: &PrimitiveArray<i128>, rhs: &PrimitiveScalar<i128>) -> PrimitiveArray<i128> {
    let (precision, scale) = get_parameters(lhs.data_type(), rhs.data_type()).unwrap();

    let rhs = if let Some(rhs) = *rhs.value() {
        rhs
    } else {
        return PrimitiveArray::<i128>::new_null(lhs.data_type().clone(), lhs.len());
    };

    let scale = 10i128.pow(scale as u32);
    let max = max_value(precision);

    let op = move |a: i128| {
        // The multiplication between i128 can overflow if they are
        // very large numbers. For that reason a checked
        // multiplication is used.
        let res: i128 = a
            .checked_mul(rhs)
            .expect("major overflow for multiplication");

        // The multiplication is done using the numbers without scale.
        // The resulting scale of the value has to be corrected by
        // dividing by (10^scale)

        //   111.111 -->      111111
        //   222.222 -->      222222
        // --------          -------
        // 24691.308 <-- 24691308642
        let res = res / scale;

        assert!(
            res.abs() <= max,
            "overflow in multiplication presented for precision {precision}"
        );

        res
    };

    unary(lhs, op, lhs.data_type().clone())
}

/// Saturated multiplication of two decimal primitive arrays with the same
/// precision and scale. If the precision and scale is different, then an
/// InvalidArgumentError is returned. If the result from the multiplication is
/// larger than the possible number with the selected precision then the
/// resulted number in the arrow array is the maximum number for the selected
/// precision.
///
/// # Examples
/// ```
/// use polars_arrow::compute::arithmetics::decimal::saturating_mul;
/// use polars_arrow::array::PrimitiveArray;
/// use polars_arrow::datatypes::DataType;
///
/// let a = PrimitiveArray::from([Some(999_99i128), Some(1_00i128), None, Some(2_00i128)]).to(DataType::Decimal(5, 2));
/// let b = PrimitiveArray::from([Some(10_00i128), Some(2_00i128), None, Some(2_00i128)]).to(DataType::Decimal(5, 2));
///
/// let result = saturating_mul(&a, &b);
/// let expected = PrimitiveArray::from([Some(999_99i128), Some(2_00i128), None, Some(4_00i128)]).to(DataType::Decimal(5, 2));
///
/// assert_eq!(result, expected);
/// ```
pub fn saturating_mul(
    lhs: &PrimitiveArray<i128>,
    rhs: &PrimitiveArray<i128>,
) -> PrimitiveArray<i128> {
    let (precision, scale) = get_parameters(lhs.data_type(), rhs.data_type()).unwrap();

    let scale = 10i128.pow(scale as u32);
    let max = max_value(precision);

    let op = move |a: i128, b: i128| match a.checked_mul(b) {
        Some(res) => {
            let res = res / scale;

            match res {
                res if res.abs() > max => {
                    if res > 0 {
                        max
                    } else {
                        -max
                    }
                },
                _ => res,
            }
        },
        None => max,
    };

    binary(lhs, rhs, lhs.data_type().clone(), op)
}

/// Checked multiplication of two decimal primitive arrays with the same
/// precision and scale. If the precision and scale is different, then an
/// InvalidArgumentError is returned. If the result from the mul is larger than
/// the possible number with the selected precision (overflowing), then the
/// validity for that index is changed to None
///
/// # Examples
/// ```
/// use polars_arrow::compute::arithmetics::decimal::checked_mul;
/// use polars_arrow::array::PrimitiveArray;
/// use polars_arrow::datatypes::DataType;
///
/// let a = PrimitiveArray::from([Some(999_99i128), Some(1_00i128), None, Some(2_00i128)]).to(DataType::Decimal(5, 2));
/// let b = PrimitiveArray::from([Some(10_00i128), Some(2_00i128), None, Some(2_00i128)]).to(DataType::Decimal(5, 2));
///
/// let result = checked_mul(&a, &b);
/// let expected = PrimitiveArray::from([None, Some(2_00i128), None, Some(4_00i128)]).to(DataType::Decimal(5, 2));
///
/// assert_eq!(result, expected);
/// ```
pub fn checked_mul(lhs: &PrimitiveArray<i128>, rhs: &PrimitiveArray<i128>) -> PrimitiveArray<i128> {
    let (precision, scale) = get_parameters(lhs.data_type(), rhs.data_type()).unwrap();

    let scale = 10i128.pow(scale as u32);
    let max = max_value(precision);

    let op = move |a: i128, b: i128| match a.checked_mul(b) {
        Some(res) => {
            let res = res / scale;

            match res {
                res if res.abs() > max => None,
                _ => Some(res),
            }
        },
        None => None,
    };

    binary_checked(lhs, rhs, lhs.data_type().clone(), op)
}

// Implementation of ArrayMul trait for PrimitiveArrays
impl ArrayMul<PrimitiveArray<i128>> for PrimitiveArray<i128> {
    fn mul(&self, rhs: &PrimitiveArray<i128>) -> Self {
        mul(self, rhs)
    }
}

// Implementation of ArrayCheckedMul trait for PrimitiveArrays
impl ArrayCheckedMul<PrimitiveArray<i128>> for PrimitiveArray<i128> {
    fn checked_mul(&self, rhs: &PrimitiveArray<i128>) -> Self {
        checked_mul(self, rhs)
    }
}

// Implementation of ArraySaturatingMul trait for PrimitiveArrays
impl ArraySaturatingMul<PrimitiveArray<i128>> for PrimitiveArray<i128> {
    fn saturating_mul(&self, rhs: &PrimitiveArray<i128>) -> Self {
        saturating_mul(self, rhs)
    }
}

/// Adaptive multiplication of two decimal primitive arrays with different
/// precision and scale. If the precision and scale is different, then the
/// smallest scale and precision is adjusted to the largest precision and
/// scale. If during the multiplication one of the results is larger than the
/// max possible value, the result precision is changed to the precision of the
/// max value
///
/// ```nocode
///   11111.0    -> 6, 1
///      10.002  -> 5, 3
/// -----------------
///  111132.222  -> 9, 3
/// ```
/// # Examples
/// ```
/// use polars_arrow::compute::arithmetics::decimal::adaptive_mul;
/// use polars_arrow::array::PrimitiveArray;
/// use polars_arrow::datatypes::DataType;
///
/// let a = PrimitiveArray::from([Some(11111_0i128), Some(1_0i128)]).to(DataType::Decimal(6, 1));
/// let b = PrimitiveArray::from([Some(10_002i128), Some(2_000i128)]).to(DataType::Decimal(5, 3));
/// let result = adaptive_mul(&a, &b).unwrap();
/// let expected = PrimitiveArray::from([Some(111132_222i128), Some(2_000i128)]).to(DataType::Decimal(9, 3));
///
/// assert_eq!(result, expected);
/// ```
pub fn adaptive_mul(
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
            polars_bail!(ComputeError: "incorrect data type for the array")
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
            // Based on the array's scales one of the arguments in the sum has to be shifted
            // to the left to match the final scale
            let res = if lhs_s > rhs_s {
                l.checked_mul(r * shift)
            } else {
                (l * shift).checked_mul(*r)
            }
            .expect("major overflow for multiplication");

            let res = res / shift_1;

            // The precision of the resulting array will change if one of the
            // multiplications during the iteration produces a value bigger
            // than the possible value for the initial precision

            //  10.0000 -> 6, 4
            //  10.0000 -> 6, 4
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
