use arrow::array::PrimitiveArray;
use arrow::compute::arithmetics::{ArrayAdd, ArrayCheckedAdd, ArraySaturatingAdd};
use arrow::compute::arity::{binary, binary_checked, unary};
use arrow::datatypes::DataType;
use arrow::error::{Error, Result};
use polars_error::*;

use super::{adjusted_precision_scale, get_parameters, max_value, number_digits};
use crate::compute::{binary_mut, unary_mut};
use crate::utils::combine_validities_and;

pub fn commutative<F>(
    lhs: &PrimitiveArray<i128>,
    rhs: &PrimitiveArray<i128>,
    op: F
) -> PolarsResult<PrimitiveArray<i128>>
    where F: Fn(i128, i128) -> i128
{
    let (precision, _) = get_parameters(lhs.data_type(), rhs.data_type()).unwrap();

    let max = max_value(precision);
    let mut overflow = false;
    let op = move |a, b| {
        let res = op(a, b);
        overflow |= res.abs() > max;
        res
    };
    polars_ensure!(!overflow, ComputeError: "Decimal overflowed the allowed precision: {precision}");

    Ok(binary_mut(lhs, rhs, lhs.data_type().clone(), op))
}

pub fn commutative_scalar<F>(
    lhs: &PrimitiveArray<i128>,
    rhs: i128,
    rhs_dtype: &DataType,
    op: F
) -> PolarsResult<PrimitiveArray<i128>>
    where F: Fn(i128, i128) -> i128
{
    let (precision, _) = get_parameters(lhs.data_type(), rhs_dtype).unwrap();

    let max = max_value(precision);
    let mut overflow = false;
    let op = move |a| {
        let res = op(a, rhs);
        overflow |= res.abs() > max;
        res
    };
    polars_ensure!(!overflow, ComputeError: "Decimal overflowed the allowed precision: {precision}");

    Ok(unary_mut(lhs, op, lhs.data_type().clone()))
}

pub fn non_commutative<F>(
    lhs: &PrimitiveArray<i128>,
    rhs: &PrimitiveArray<i128>,
    op: F
) -> PolarsResult<PrimitiveArray<i128>>
    where F: Fn(i128, i128) -> i128
{
    let op = move |a, b| {
        op(a, b)
    };

    Ok(binary_mut(lhs, rhs, lhs.data_type().clone(), op))
}

pub fn non_commutative_scalar<F>(lhs: &PrimitiveArray<i128>, rhs: i128,
                              op: F
) -> PolarsResult<PrimitiveArray<i128>>
    where F: Fn(i128, i128) -> i128
{
    let op = move |a| {
        op(a, rhs)
    };

    Ok(unary_mut(lhs, op, lhs.data_type().clone()))
}

pub fn non_commutative_scalar_swapped<F>(
    lhs: i128,
    rhs: &PrimitiveArray<i128>,
    op: F
) -> PolarsResult<PrimitiveArray<i128>>
    where F: Fn(i128, i128) -> i128
{
    let op = move |a| {
        op(lhs, a)
    };

    Ok(unary_mut(rhs, op, rhs.data_type().clone()))
}
