use crate::{prelude::*, series::chunked_array::ChunkedArray};
use arrow::{array::ArrayRef, compute, datatypes::ArrowNumericType};
use std::ops::{Add, Div, Mul, Sub};
use std::sync::Arc;

macro_rules! operand_on_primitive_arr {
    ($_self:expr, $rhs:tt, $operator:expr, $expect:expr) => {{
        let mut new_chunks = Vec::with_capacity($_self.chunks.len());
        $_self
            .downcast_chunks()
            .iter()
            .zip($rhs.downcast_chunks())
            .for_each(|(left, right)| {
                let res = Arc::new($operator(left, right).expect($expect)) as ArrayRef;
                new_chunks.push(res);
            });
        $_self.copy_with_chunks(new_chunks)
    }};
}

impl<T> Add for &ChunkedArray<T>
where
    T: ArrowNumericType,
    T::Native: Add<Output = T::Native>
        + Sub<Output = T::Native>
        + Mul<Output = T::Native>
        + Div<Output = T::Native>
        + num::Zero,
{
    type Output = ChunkedArray<T>;

    fn add(self, rhs: Self) -> Self::Output {
        let expect_str = "Could not add, check data types and length";
        operand_on_primitive_arr![self, rhs, compute::add, expect_str]
    }
}

impl<T> Div for &ChunkedArray<T>
where
    T: ArrowNumericType,
    T::Native: Add<Output = T::Native>
        + Sub<Output = T::Native>
        + Mul<Output = T::Native>
        + Div<Output = T::Native>
        + num::Zero
        + num::One,
{
    type Output = ChunkedArray<T>;

    fn div(self, rhs: Self) -> Self::Output {
        let expect_str = "Could not divide, check data types and length";
        operand_on_primitive_arr!(self, rhs, compute::divide, expect_str)
    }
}

impl<T> Mul for &ChunkedArray<T>
where
    T: ArrowNumericType,
    T::Native: Add<Output = T::Native>
        + Sub<Output = T::Native>
        + Mul<Output = T::Native>
        + Div<Output = T::Native>
        + num::Zero,
{
    type Output = ChunkedArray<T>;

    fn mul(self, rhs: Self) -> Self::Output {
        let expect_str = "Could not multiply, check data types and length";
        operand_on_primitive_arr!(self, rhs, compute::multiply, expect_str)
    }
}

impl<T> Sub for &ChunkedArray<T>
where
    T: ArrowNumericType,
    T::Native: Add<Output = T::Native>
        + Sub<Output = T::Native>
        + Mul<Output = T::Native>
        + Div<Output = T::Native>
        + num::Zero,
{
    type Output = ChunkedArray<T>;

    fn sub(self, rhs: Self) -> Self::Output {
        let expect_str = "Could not subtract, check data types and length";
        operand_on_primitive_arr![self, rhs, compute::subtract, expect_str]
    }
}

impl<T> Add for ChunkedArray<T>
where
    T: ArrowNumericType,
    T::Native: Add<Output = T::Native>
        + Sub<Output = T::Native>
        + Mul<Output = T::Native>
        + Div<Output = T::Native>
        + num::Zero,
{
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        (&self).add(&rhs)
    }
}

impl<T> Div for ChunkedArray<T>
where
    T: ArrowNumericType,
    T::Native: Add<Output = T::Native>
        + Sub<Output = T::Native>
        + Mul<Output = T::Native>
        + Div<Output = T::Native>
        + num::Zero
        + num::One,
{
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        (&self).div(&rhs)
    }
}

impl<T> Mul for ChunkedArray<T>
where
    T: ArrowNumericType,
    T::Native: Add<Output = T::Native>
        + Sub<Output = T::Native>
        + Mul<Output = T::Native>
        + Div<Output = T::Native>
        + num::Zero,
{
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        (&self).mul(&rhs)
    }
}

impl<T> Sub for ChunkedArray<T>
where
    T: ArrowNumericType,
    T::Native: Add<Output = T::Native>
        + Sub<Output = T::Native>
        + Mul<Output = T::Native>
        + Div<Output = T::Native>
        + num::Zero,
{
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        (&self).sub(&rhs)
    }
}
