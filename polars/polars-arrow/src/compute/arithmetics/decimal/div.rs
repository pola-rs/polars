use ethnum::I256;

use super::*;

#[inline]
fn decimal_div(a: i128, b: i128, scale: i128) -> i128 {
    // The division is done using the numbers without scale.
    // The dividend is scaled up to maintain precision after the
    // division

    //   222.222 -->  222222000
    //   123.456 -->     123456
    // --------       ---------
    //     1.800 <--       1800

    // operate in I256 space to reduce overflow
    let a = I256::new(a);
    let b = I256::new(b);
    let scale = I256::new(scale);
    (a * scale / b).as_i128()
}

pub fn div(
    lhs: &PrimitiveArray<i128>,
    rhs: &PrimitiveArray<i128>,
) -> PolarsResult<PrimitiveArray<i128>> {
    let (_, scale) = get_parameters(lhs.data_type(), rhs.data_type())?;
    let scale = 10i128.pow(scale as u32);
    non_commutative(lhs, rhs, |a, b| decimal_div(a, b, scale))
}

pub fn div_scalar(
    lhs: &PrimitiveArray<i128>,
    rhs: i128,
    rhs_dtype: &DataType,
) -> PolarsResult<PrimitiveArray<i128>> {
    let (_, scale) = get_parameters(lhs.data_type(), rhs_dtype)?;
    let scale = 10i128.pow(scale as u32);
    non_commutative_scalar(lhs, rhs, |a, b| decimal_div(a, b, scale))
}

pub fn div_scalar_swapped(
    lhs: i128,
    lhs_dtype: &DataType,
    rhs: &PrimitiveArray<i128>,
) -> PolarsResult<PrimitiveArray<i128>> {
    let (_, scale) = get_parameters(lhs_dtype, rhs.data_type())?;
    let scale = 10i128.pow(scale as u32);
    non_commutative_scalar_swapped(lhs, rhs, |a, b| decimal_div(a, b, scale))
}
