use ethnum::I256;

use super::*;

#[inline]
fn decimal_mul(a: i128, b: i128, scale: i128) -> i128 {
    // The multiplication is done using the numbers without scale.
    // The resulting scale of the value has to be corrected by
    // dividing by (10^scale)

    //   111.111 -->      111111
    //   222.222 -->      222222
    // --------          -------
    // 24691.308 <-- 24691308642

    // operate in I256 space to reduce overflow
    let a = I256::new(a);
    let b = I256::new(b);
    let scale = I256::new(scale);

    (a * b / scale).as_i128()
}

pub fn mul(
    lhs: &PrimitiveArray<i128>,
    rhs: &PrimitiveArray<i128>,
) -> PolarsResult<PrimitiveArray<i128>> {
    let (_, scale) = get_parameters(lhs.data_type(), rhs.data_type())?;
    let scale = 10i128.pow(scale as u32);
    commutative(lhs, rhs, |a, b| decimal_mul(a, b, scale))
}

pub fn mul_scalar(
    lhs: &PrimitiveArray<i128>,
    rhs: i128,
    rhs_dtype: &ArrowDataType,
) -> PolarsResult<PrimitiveArray<i128>> {
    let (_, scale) = get_parameters(lhs.data_type(), rhs_dtype)?;
    let scale = 10i128.pow(scale as u32);
    commutative_scalar(lhs, rhs, rhs_dtype, |a, b| decimal_mul(a, b, scale))
}
