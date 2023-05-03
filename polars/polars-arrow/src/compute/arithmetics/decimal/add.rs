use super::*;

pub fn add(
    lhs: &PrimitiveArray<i128>,
    rhs: &PrimitiveArray<i128>,
) -> PolarsResult<PrimitiveArray<i128>> {
    commutative(lhs, rhs, |a, b| a + b)
}

pub fn add_scalar(
    lhs: &PrimitiveArray<i128>,
    rhs: i128,
    rhs_dtype: &DataType,
) -> PolarsResult<PrimitiveArray<i128>> {
    commutative_scalar(lhs, rhs, rhs_dtype, |a, b| a + b)
}
