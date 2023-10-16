use super::*;

pub fn sub(
    lhs: &PrimitiveArray<i128>,
    rhs: &PrimitiveArray<i128>,
) -> PolarsResult<PrimitiveArray<i128>> {
    non_commutative(lhs, rhs, |a, b| a - b)
}

pub fn sub_scalar(lhs: &PrimitiveArray<i128>, rhs: i128) -> PolarsResult<PrimitiveArray<i128>> {
    non_commutative_scalar(lhs, rhs, |a, b| a - b)
}

pub fn sub_scalar_swapped(
    lhs: i128,
    rhs: &PrimitiveArray<i128>,
) -> PolarsResult<PrimitiveArray<i128>> {
    non_commutative_scalar_swapped(lhs, rhs, |a, b| a - b)
}
