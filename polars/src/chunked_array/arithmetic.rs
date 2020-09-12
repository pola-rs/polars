//! Implementations of arithmetic operations on ChunkedArray's.
use crate::prelude::*;
use crate::utils::Xob;
use arrow::{array::ArrayRef, compute};
use num::{Num, NumCast, ToPrimitive};
use std::ops::{Add, Div, Mul, Sub};
use std::sync::Arc;

// TODO: Add Boolean arithmetic

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

#[macro_export]
macro_rules! apply_operand_on_chunkedarray_by_iter {

    ($self:ident, $rhs:ident, $operand:tt) => {
            {
                match ($self.null_count(), $rhs.null_count()) {
                    (0, 0) => {
                        let a: Xob<ChunkedArray<_>> = $self
                        .into_no_null_iter()
                        .zip($rhs.into_no_null_iter())
                        .map(|(left, right)| left $operand right)
                        .collect();
                        a.into_inner()
                    },
                    (0, _) => {
                        $self
                        .into_no_null_iter()
                        .zip($rhs.into_iter())
                        .map(|(left, opt_right)| opt_right.map(|right| left $operand right))
                        .collect()
                    },
                    (_, 0) => {
                        $self
                        .into_iter()
                        .zip($rhs.into_no_null_iter())
                        .map(|(opt_left, right)| opt_left.map(|left| left $operand right))
                        .collect()
                    },
                    (_, _) => {
                    $self.into_iter()
                        .zip($rhs.into_iter())
                        .map(|(opt_left, opt_right)| match (opt_left, opt_right) {
                            (None, None) => None,
                            (None, Some(_)) => None,
                            (Some(_), None) => None,
                            (Some(left), Some(right)) => Some(left $operand right),
                        })
                        .collect()

                    }
                }
            }
    }
}

// Operands on ChunkedArray & ChunkedArray

impl<T> Add for &ChunkedArray<T>
where
    T: PolarsNumericType,
    T::Native: Add<Output = T::Native>
        + Sub<Output = T::Native>
        + Mul<Output = T::Native>
        + Div<Output = T::Native>
        + num::Zero,
{
    type Output = ChunkedArray<T>;

    fn add(self, rhs: Self) -> Self::Output {
        if self.chunk_id == rhs.chunk_id {
            let expect_str = "Could not add, check data types and length";
            operand_on_primitive_arr![self, rhs, compute::add, expect_str]
        } else {
            apply_operand_on_chunkedarray_by_iter!(self, rhs, +)
        }
    }
}

impl<T> Div for &ChunkedArray<T>
where
    T: PolarsNumericType,
    T::Native: Add<Output = T::Native>
        + Sub<Output = T::Native>
        + Mul<Output = T::Native>
        + Div<Output = T::Native>
        + num::Zero
        + num::One,
{
    type Output = ChunkedArray<T>;

    fn div(self, rhs: Self) -> Self::Output {
        if self.chunk_id == rhs.chunk_id {
            let expect_str = "Could not divide, check data types and length";
            operand_on_primitive_arr!(self, rhs, compute::divide, expect_str)
        } else {
            apply_operand_on_chunkedarray_by_iter!(self, rhs, /)
        }
    }
}

impl<T> Mul for &ChunkedArray<T>
where
    T: PolarsNumericType,
    T::Native: Add<Output = T::Native>
        + Sub<Output = T::Native>
        + Mul<Output = T::Native>
        + Div<Output = T::Native>
        + num::Zero,
{
    type Output = ChunkedArray<T>;

    fn mul(self, rhs: Self) -> Self::Output {
        if self.chunk_id == rhs.chunk_id {
            let expect_str = "Could not multiply, check data types and length";
            operand_on_primitive_arr!(self, rhs, compute::multiply, expect_str)
        } else {
            apply_operand_on_chunkedarray_by_iter!(self, rhs, *)
        }
    }
}

impl<T> Sub for &ChunkedArray<T>
where
    T: PolarsNumericType,
    T::Native: Add<Output = T::Native>
        + Sub<Output = T::Native>
        + Mul<Output = T::Native>
        + Div<Output = T::Native>
        + num::Zero,
{
    type Output = ChunkedArray<T>;

    fn sub(self, rhs: Self) -> Self::Output {
        if self.chunk_id == rhs.chunk_id {
            let expect_str = "Could not subtract, check data types and length";
            operand_on_primitive_arr![self, rhs, compute::subtract, expect_str]
        } else {
            apply_operand_on_chunkedarray_by_iter!(self, rhs, -)
        }
    }
}

impl<T> Add for ChunkedArray<T>
where
    T: PolarsNumericType,
    T::Native: Add<Output = T::Native>
        + Sub<Output = T::Native>
        + Mul<Output = T::Native>
        + Div<Output = T::Native>
        + num::Zero,
{
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        if self.chunk_id == rhs.chunk_id {
            (&self).add(&rhs)
        } else {
            apply_operand_on_chunkedarray_by_iter!(self, rhs, +)
        }
    }
}

impl<T> Div for ChunkedArray<T>
where
    T: PolarsNumericType,
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
    T: PolarsNumericType,
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
    T: PolarsNumericType,
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

// Operands on ChunkedArray & Num

impl<T, N> Add<N> for &ChunkedArray<T>
where
    T: PolarsNumericType,
    T::Native: NumCast,
    N: Num + ToPrimitive,
    T::Native: Add<Output = T::Native>
        + Sub<Output = T::Native>
        + Mul<Output = T::Native>
        + Div<Output = T::Native>
        + num::Zero,
{
    type Output = ChunkedArray<T>;

    fn add(self, rhs: N) -> Self::Output {
        let adder: T::Native = NumCast::from(rhs).unwrap();
        if self.is_optimal_aligned() {
            let intermed: Xob<_> = self.into_no_null_iter().map(|val| val + adder).collect();
            intermed.into_inner()
        } else {
            self.into_iter()
                .map(|opt_val| opt_val.map(|val| val + adder))
                .collect()
        }
    }
}

impl<T, N> Sub<N> for &ChunkedArray<T>
where
    T: PolarsNumericType,
    T::Native: NumCast,
    N: Num + ToPrimitive,
    T::Native: Add<Output = T::Native>
        + Sub<Output = T::Native>
        + Mul<Output = T::Native>
        + Div<Output = T::Native>
        + num::Zero,
{
    type Output = ChunkedArray<T>;

    fn sub(self, rhs: N) -> Self::Output {
        let subber: T::Native = NumCast::from(rhs).unwrap();
        if self.is_optimal_aligned() {
            let intermed: Xob<_> = self.into_no_null_iter().map(|val| val - subber).collect();
            intermed.into_inner()
        } else {
            self.into_iter()
                .map(|opt_val| opt_val.map(|val| val - subber))
                .collect()
        }
    }
}

impl<T, N> Div<N> for &ChunkedArray<T>
where
    T: PolarsNumericType,
    T::Native: NumCast,
    N: Num + ToPrimitive,
    T::Native: Add<Output = T::Native>
        + Sub<Output = T::Native>
        + Mul<Output = T::Native>
        + Div<Output = T::Native>
        + num::Zero,
{
    type Output = ChunkedArray<T>;

    fn div(self, rhs: N) -> Self::Output {
        let divider: T::Native = NumCast::from(rhs).unwrap();
        if self.is_optimal_aligned() {
            let intermed: Xob<_> = self.into_no_null_iter().map(|val| val / divider).collect();
            intermed.into_inner()
        } else {
            self.into_iter()
                .map(|opt_val| opt_val.map(|val| val / divider))
                .collect()
        }
    }
}

impl<T, N> Mul<N> for &ChunkedArray<T>
where
    T: PolarsNumericType,
    T::Native: NumCast,
    N: Num + ToPrimitive,
    T::Native: Add<Output = T::Native>
        + Sub<Output = T::Native>
        + Mul<Output = T::Native>
        + Div<Output = T::Native>
        + num::Zero,
{
    type Output = ChunkedArray<T>;

    fn mul(self, rhs: N) -> Self::Output {
        let multiplier: T::Native = NumCast::from(rhs).unwrap();
        if self.is_optimal_aligned() {
            let intermed: Xob<_> = self
                .into_no_null_iter()
                .map(|val| val * multiplier)
                .collect();
            intermed.into_inner()
        } else {
            self.into_iter()
                .map(|opt_val| opt_val.map(|val| val * multiplier))
                .collect()
        }
    }
}

pub trait Pow {
    fn pow_f32(&self, exp: f32) -> Float32Chunked;
    fn pow_f64(&self, exp: f64) -> Float64Chunked;
}

macro_rules! power {
    ($self:ident, $exp:ident, $to_primitive:ident, $return:ident) => {{
        if let Ok(slice) = $self.cont_slice() {
            slice
                .iter()
                .map(|&val| val.$to_primitive().unwrap().powf($exp))
                .collect::<Xob<$return>>()
                .into_inner()
        } else {
            $self
                .into_iter()
                .map(|val| val.map(|val| val.$to_primitive().unwrap().powf($exp)))
                .collect()
        }
    }};
}

impl<T> Pow for ChunkedArray<T>
where
    T: PolarsNumericType,
    T::Native: ToPrimitive,
{
    fn pow_f32(&self, exp: f32) -> Float32Chunked {
        power!(self, exp, to_f32, Float32Chunked)
    }

    fn pow_f64(&self, exp: f64) -> Float64Chunked {
        power!(self, exp, to_f64, Float64Chunked)
    }
}

#[cfg(test)]
pub(crate) mod test {
    use crate::prelude::*;

    pub(crate) fn create_two_chunked() -> (Int32Chunked, Int32Chunked) {
        let mut a1 = Int32Chunked::new_from_slice("a", &[1, 2, 3]);
        let a2 = Int32Chunked::new_from_slice("a", &[4, 5, 6]);
        let a3 = Int32Chunked::new_from_slice("a", &[1, 2, 3, 4, 5, 6]);
        a1.append(&a2);
        (a1, a3)
    }

    #[test]
    fn test_chunk_mismatch() {
        let (a1, a2) = create_two_chunked();
        // with different chunks
        let _ = &a1 + &a2;
        let _ = &a1 - &a2;
        let _ = &a1 / &a2;
        let _ = &a1 * &a2;

        // with same chunks
        let _ = &a1 + &a1;
        let _ = &a1 - &a1;
        let _ = &a1 / &a1;
        let _ = &a1 * &a1;
    }

    #[test]
    fn test_power() {
        let a = UInt32Chunked::new_from_slice("", &[1, 2, 3]);
        let b = a.pow_f64(2.);
        println!("{:?}", b);
    }
}
