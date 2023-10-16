//! Defines the addition arithmetic kernels for [`PrimitiveArray`] representing decimals.
use polars_error::{polars_bail, PolarsResult};

use super::{adjusted_precision_scale, get_parameters, max_value, number_digits};
use crate::array::PrimitiveArray;
use crate::compute::arithmetics::{ArrayAdd, ArrayCheckedAdd, ArraySaturatingAdd};
use crate::compute::arity::{binary, binary_checked};
use crate::compute::utils::{check_same_len, combine_validities};
use crate::datatypes::DataType;

/// Adds two decimal [`PrimitiveArray`] with the same precision and scale.
/// # Error
/// Errors if the precision and scale are different.
/// # Panic
/// This function panics iff the added numbers result in a number larger than
/// the possible number for the precision.
pub fn add(lhs: &PrimitiveArray<i128>, rhs: &PrimitiveArray<i128>) -> PrimitiveArray<i128> {
    let (precision, _) = get_parameters(lhs.data_type(), rhs.data_type()).unwrap();

    let max = max_value(precision);
    let op = move |a, b| {
        let res: i128 = a + b;

        assert!(
            res.abs() <= max,
            "Overflow in addition presented for precision {precision}"
        );

        res
    };

    binary(lhs, rhs, lhs.data_type().clone(), op)
}

/// Saturated addition of two decimal primitive arrays with the same precision
/// and scale. If the precision and scale is different, then an
/// InvalidArgumentError is returned. If the result from the sum is larger than
/// the possible number with the selected precision then the resulted number in
/// the arrow array is the maximum number for the selected precision.
///
/// # Examples
/// ```
/// use polars_arrow::compute::arithmetics::decimal::saturating_add;
/// use polars_arrow::array::PrimitiveArray;
/// use polars_arrow::datatypes::DataType;
///
/// let a = PrimitiveArray::from([Some(99000i128), Some(11100i128), None, Some(22200i128)]).to(DataType::Decimal(5, 2));
/// let b = PrimitiveArray::from([Some(01000i128), Some(22200i128), None, Some(11100i128)]).to(DataType::Decimal(5, 2));
///
/// let result = saturating_add(&a, &b);
/// let expected = PrimitiveArray::from([Some(99999i128), Some(33300i128), None, Some(33300i128)]).to(DataType::Decimal(5, 2));
///
/// assert_eq!(result, expected);
/// ```
pub fn saturating_add(
    lhs: &PrimitiveArray<i128>,
    rhs: &PrimitiveArray<i128>,
) -> PrimitiveArray<i128> {
    let (precision, _) = get_parameters(lhs.data_type(), rhs.data_type()).unwrap();

    let max = max_value(precision);
    let op = move |a, b| {
        let res: i128 = a + b;

        if res.abs() > max {
            if res > 0 {
                max
            } else {
                -max
            }
        } else {
            res
        }
    };

    binary(lhs, rhs, lhs.data_type().clone(), op)
}

/// Checked addition of two decimal primitive arrays with the same precision
/// and scale. If the precision and scale is different, then an
/// InvalidArgumentError is returned. If the result from the sum is larger than
/// the possible number with the selected precision (overflowing), then the
/// validity for that index is changed to None
///
/// # Examples
/// ```
/// use polars_arrow::compute::arithmetics::decimal::checked_add;
/// use polars_arrow::array::PrimitiveArray;
/// use polars_arrow::datatypes::DataType;
///
/// let a = PrimitiveArray::from([Some(99000i128), Some(11100i128), None, Some(22200i128)]).to(DataType::Decimal(5, 2));
/// let b = PrimitiveArray::from([Some(01000i128), Some(22200i128), None, Some(11100i128)]).to(DataType::Decimal(5, 2));
///
/// let result = checked_add(&a, &b);
/// let expected = PrimitiveArray::from([None, Some(33300i128), None, Some(33300i128)]).to(DataType::Decimal(5, 2));
///
/// assert_eq!(result, expected);
/// ```
pub fn checked_add(lhs: &PrimitiveArray<i128>, rhs: &PrimitiveArray<i128>) -> PrimitiveArray<i128> {
    let (precision, _) = get_parameters(lhs.data_type(), rhs.data_type()).unwrap();

    let max = max_value(precision);
    let op = move |a, b| {
        let result: i128 = a + b;

        if result.abs() > max {
            None
        } else {
            Some(result)
        }
    };

    binary_checked(lhs, rhs, lhs.data_type().clone(), op)
}

// Implementation of ArrayAdd trait for PrimitiveArrays
impl ArrayAdd<PrimitiveArray<i128>> for PrimitiveArray<i128> {
    fn add(&self, rhs: &PrimitiveArray<i128>) -> Self {
        add(self, rhs)
    }
}

// Implementation of ArrayCheckedAdd trait for PrimitiveArrays
impl ArrayCheckedAdd<PrimitiveArray<i128>> for PrimitiveArray<i128> {
    fn checked_add(&self, rhs: &PrimitiveArray<i128>) -> Self {
        checked_add(self, rhs)
    }
}

// Implementation of ArraySaturatingAdd trait for PrimitiveArrays
impl ArraySaturatingAdd<PrimitiveArray<i128>> for PrimitiveArray<i128> {
    fn saturating_add(&self, rhs: &PrimitiveArray<i128>) -> Self {
        saturating_add(self, rhs)
    }
}

/// Adaptive addition of two decimal primitive arrays with different precision
/// and scale. If the precision and scale is different, then the smallest scale
/// and precision is adjusted to the largest precision and scale. If during the
/// addition one of the results is larger than the max possible value, the
/// result precision is changed to the precision of the max value
///
/// ```nocode
/// 11111.11   -> 7, 2
/// 11111.111  -> 8, 3
/// ------------------
/// 22222.221  -> 8, 3
/// ```
/// # Examples
/// ```
/// use polars_arrow::compute::arithmetics::decimal::adaptive_add;
/// use polars_arrow::array::PrimitiveArray;
/// use polars_arrow::datatypes::DataType;
///
/// let a = PrimitiveArray::from([Some(11111_11i128)]).to(DataType::Decimal(7, 2));
/// let b = PrimitiveArray::from([Some(11111_111i128)]).to(DataType::Decimal(8, 3));
/// let result = adaptive_add(&a, &b).unwrap();
/// let expected = PrimitiveArray::from([Some(22222_221i128)]).to(DataType::Decimal(8, 3));
///
/// assert_eq!(result, expected);
/// ```
pub fn adaptive_add(
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
    let mut max = max_value(res_p);

    let values = lhs
        .values()
        .iter()
        .zip(rhs.values().iter())
        .map(|(l, r)| {
            // Based on the array's scales one of the arguments in the sum has to be shifted
            // to the left to match the final scale
            let res = if lhs_s > rhs_s {
                l + r * shift
            } else {
                l * shift + r
            };

            // The precision of the resulting array will change if one of the
            // sums during the iteration produces a value bigger than the
            // possible value for the initial precision

            //  99.9999 -> 6, 4
            //  00.0001 -> 6, 4
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
