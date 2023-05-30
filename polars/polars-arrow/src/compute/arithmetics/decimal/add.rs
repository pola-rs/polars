//! Defines the addition arithmetic kernels for [`PrimitiveArray`] representing decimals.
use arrow::{
    array::PrimitiveArray,
    compute::{
        arithmetics::{ArrayAdd, ArrayCheckedAdd, ArraySaturatingAdd},
        arity::{binary, binary_checked},
    },
};
use arrow::{
    datatypes::DataType,
    error::{Error, Result},
};
use arrow::compute::arity::unary;
use polars_error::*;
use crate::compute::{binary_mut, unary_mut};
use crate::utils::combine_validities_and;

use super::{adjusted_precision_scale, get_parameters, max_value, number_digits};

pub fn add(lhs: &PrimitiveArray<i128>, rhs: &PrimitiveArray<i128>) -> PolarsResult<PrimitiveArray<i128>> {
    let (precision, _) = get_parameters(lhs.data_type(), rhs.data_type()).unwrap();

    let max = max_value(precision);
    let mut overflow = false;
    let op = move |a, b| {
        let res: i128 = a + b;
        if res.abs() > max {
            overflow = true
        }
        res
    };
    polars_ensure!(!overflow, ComputeError: "Decimal overflowed the allowed precision: {precision}");

    Ok(binary_mut(lhs, rhs, lhs.data_type().clone(), op))
}


pub fn add_scalar(lhs: &PrimitiveArray<i128>, rhs: i128, rhs_dtype: &DataType) -> PolarsResult<PrimitiveArray<i128>> {
    let (precision, _) = get_parameters(lhs.data_type(), rhs_dtype).unwrap();

    let max = max_value(precision);
    let mut overflow = false;
    let op = move |a| {
        let res: i128 = a + rhs;
        if res.abs() > max {
            overflow = true
        }
        res
    };
    polars_ensure!(!overflow, ComputeError: "Decimal overflowed the allowed precision: {precision}");

    Ok(unary_mut(lhs, op, lhs.data_type().clone()))
}
