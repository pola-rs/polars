//! Implementations of arithmetic operations on ChunkedArray's.
use crate::prelude::*;
use crate::utils::align_chunks_binary;
use arrow::array::PrimitiveArray;
use arrow::{array::ArrayRef, compute, compute::arithmetics::basic};
use num::{Num, NumCast, One, ToPrimitive, Zero};
use std::borrow::Cow;
use std::ops::{Add, Div, Mul, Rem, Sub};
use std::sync::Arc;

macro_rules! apply_operand_on_chunkedarray_by_iter {

    ($self:ident, $rhs:ident, $operand:tt) => {
            {
                match ($self.null_count(), $rhs.null_count()) {
                    (0, 0) => {
                        let a: NoNull<ChunkedArray<_>> = $self
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

fn arithmetic_helper<T, Kernel, F>(
    lhs: &ChunkedArray<T>,
    rhs: &ChunkedArray<T>,
    kernel: Kernel,
    operation: F,
) -> ChunkedArray<T>
where
    T: PolarsNumericType,
    T::Native: Add<Output = T::Native>
        + Sub<Output = T::Native>
        + Mul<Output = T::Native>
        + Div<Output = T::Native>
        + num::Zero,
    Kernel: Fn(
        &PrimitiveArray<T::Native>,
        &PrimitiveArray<T::Native>,
    ) -> arrow::error::Result<PrimitiveArray<T::Native>>,
    F: Fn(T::Native, T::Native) -> T::Native,
{
    let mut ca = match (lhs.len(), rhs.len()) {
        (a, b) if a == b => {
            let (lhs, rhs) = align_chunks_binary(lhs, rhs);
            let chunks = lhs
                .downcast_iter()
                .zip(rhs.downcast_iter())
                .map(|(lhs, rhs)| Arc::new(kernel(lhs, rhs).expect("output")) as ArrayRef)
                .collect();
            lhs.copy_with_chunks(chunks)
        }
        // broadcast right path
        (_, 1) => {
            let opt_rhs = rhs.get(0);
            match opt_rhs {
                None => ChunkedArray::full_null(lhs.name(), lhs.len()),
                Some(rhs) => lhs.apply(|lhs| operation(lhs, rhs)),
            }
        }
        (1, _) => {
            let opt_lhs = lhs.get(0);
            match opt_lhs {
                None => ChunkedArray::full_null(lhs.name(), rhs.len()),
                Some(lhs) => rhs.apply(|rhs| operation(lhs, rhs)),
            }
        }
        _ => panic!("Cannot apply operation on arrays of different lengths"),
    };
    ca.rename(lhs.name());
    ca
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
        arithmetic_helper(self, rhs, basic::add::add, |lhs, rhs| lhs + rhs)
    }
}

impl<T> Div for &ChunkedArray<T>
where
    T: PolarsNumericType,
    T::Native: Add<Output = T::Native>
        + Sub<Output = T::Native>
        + Mul<Output = T::Native>
        + Div<Output = T::Native>
        + Rem<Output = T::Native>
        + num::Zero
        + num::One,
{
    type Output = ChunkedArray<T>;

    fn div(self, rhs: Self) -> Self::Output {
        arithmetic_helper(self, rhs, basic::div::div, |lhs, rhs| lhs / rhs)
    }
}

impl<T> Mul for &ChunkedArray<T>
where
    T: PolarsNumericType,
    T::Native: Add<Output = T::Native>
        + Sub<Output = T::Native>
        + Mul<Output = T::Native>
        + Div<Output = T::Native>
        + Rem<Output = T::Native>
        + num::Zero,
{
    type Output = ChunkedArray<T>;

    fn mul(self, rhs: Self) -> Self::Output {
        arithmetic_helper(self, rhs, basic::mul::mul, |lhs, rhs| lhs * rhs)
    }
}

impl<T> Rem for &ChunkedArray<T>
where
    T: PolarsNumericType,
    T::Native: Add<Output = T::Native>
        + Sub<Output = T::Native>
        + Mul<Output = T::Native>
        + Div<Output = T::Native>
        + Rem<Output = T::Native>
        + num::Zero
        + num::One,
{
    type Output = ChunkedArray<T>;

    fn rem(self, rhs: Self) -> Self::Output {
        arithmetic_helper(self, rhs, basic::rem::rem, |lhs, rhs| lhs % rhs)
    }
}

impl<T> Sub for &ChunkedArray<T>
where
    T: PolarsNumericType,
    T::Native: Add<Output = T::Native>
        + Sub<Output = T::Native>
        + Mul<Output = T::Native>
        + Div<Output = T::Native>
        + Rem<Output = T::Native>
        + num::Zero,
{
    type Output = ChunkedArray<T>;

    fn sub(self, rhs: Self) -> Self::Output {
        arithmetic_helper(self, rhs, basic::sub::sub, |lhs, rhs| lhs - rhs)
    }
}

impl<T> Add for ChunkedArray<T>
where
    T: PolarsNumericType,
    T::Native: Add<Output = T::Native>
        + Sub<Output = T::Native>
        + Mul<Output = T::Native>
        + Div<Output = T::Native>
        + Rem<Output = T::Native>
        + num::Zero,
{
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        (&self).add(&rhs)
    }
}

impl<T> Div for ChunkedArray<T>
where
    T: PolarsNumericType,
    T::Native: Add<Output = T::Native>
        + Sub<Output = T::Native>
        + Mul<Output = T::Native>
        + Div<Output = T::Native>
        + Rem<Output = T::Native>
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
        + Rem<Output = T::Native>
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
        + Rem<Output = T::Native>
        + num::Zero,
{
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        (&self).sub(&rhs)
    }
}

impl<T> Rem for ChunkedArray<T>
where
    T: PolarsNumericType,
    T::Native: Add<Output = T::Native>
        + Sub<Output = T::Native>
        + Mul<Output = T::Native>
        + Div<Output = T::Native>
        + Rem<Output = T::Native>
        + num::Zero
        + num::One,
{
    type Output = ChunkedArray<T>;

    fn rem(self, rhs: Self) -> Self::Output {
        (&self).rem(&rhs)
    }
}

// Operands on ChunkedArray & Num

impl<T, N> Add<N> for &ChunkedArray<T>
where
    T: PolarsNumericType,
    T::Native: NumCast,
    N: Num + ToPrimitive,
    T::Native: Add<Output = T::Native>,
{
    type Output = ChunkedArray<T>;

    fn add(self, rhs: N) -> Self::Output {
        let adder: T::Native = NumCast::from(rhs).unwrap();
        self.apply(|val| val + adder)
    }
}

impl<T, N> Sub<N> for &ChunkedArray<T>
where
    T: PolarsNumericType,
    T::Native: NumCast,
    N: Num + ToPrimitive,
    T::Native: Sub<Output = T::Native>,
{
    type Output = ChunkedArray<T>;

    fn sub(self, rhs: N) -> Self::Output {
        let subber: T::Native = NumCast::from(rhs).unwrap();
        self.apply(|val| val - subber)
    }
}

impl<T, N> Div<N> for &ChunkedArray<T>
where
    T: PolarsNumericType,
    T::Native: NumCast
        + Div<Output = T::Native>
        + One
        + Zero
        + Rem<Output = T::Native>
        + Sub<Output = T::Native>,
    N: Num + ToPrimitive,
{
    type Output = ChunkedArray<T>;

    fn div(self, rhs: N) -> Self::Output {
        let rhs: T::Native = NumCast::from(rhs).expect("could not cast");
        self.apply_kernel(|arr| Arc::new(basic::div::div_scalar(arr, &rhs)))
    }
}

impl<T, N> Mul<N> for &ChunkedArray<T>
where
    T: PolarsNumericType,
    T::Native: NumCast,
    N: Num + ToPrimitive,
    T::Native: Mul<Output = T::Native>,
{
    type Output = ChunkedArray<T>;

    fn mul(self, rhs: N) -> Self::Output {
        let multiplier: T::Native = NumCast::from(rhs).unwrap();
        self.apply(|val| val * multiplier)
    }
}

impl<T, N> Rem<N> for &ChunkedArray<T>
where
    T: PolarsNumericType,
    T::Native: NumCast,
    N: Num + ToPrimitive,
    T::Native:
        Div<Output = T::Native> + One + Zero + Rem<Output = T::Native> + Sub<Output = T::Native>,
{
    type Output = ChunkedArray<T>;

    fn rem(self, rhs: N) -> Self::Output {
        let rhs: T::Native = NumCast::from(rhs).expect("could not cast");
        self.apply_kernel(|arr| Arc::new(basic::rem::rem_scalar(arr, &rhs)))
    }
}

impl<T, N> Add<N> for ChunkedArray<T>
where
    T: PolarsNumericType,
    T::Native: NumCast,
    N: Num + ToPrimitive,
    T::Native: Add<Output = T::Native>,
{
    type Output = ChunkedArray<T>;

    fn add(self, rhs: N) -> Self::Output {
        (&self).add(rhs)
    }
}

impl<T, N> Sub<N> for ChunkedArray<T>
where
    T: PolarsNumericType,
    T::Native: NumCast,
    N: Num + ToPrimitive,
    T::Native: Sub<Output = T::Native>,
{
    type Output = ChunkedArray<T>;

    fn sub(self, rhs: N) -> Self::Output {
        (&self).sub(rhs)
    }
}

impl<T, N> Div<N> for ChunkedArray<T>
where
    T: PolarsNumericType,
    T::Native: NumCast
        + Div<Output = T::Native>
        + One
        + Zero
        + Sub<Output = T::Native>
        + Rem<Output = T::Native>,
    N: Num + ToPrimitive,
{
    type Output = ChunkedArray<T>;

    fn div(self, rhs: N) -> Self::Output {
        (&self).div(rhs)
    }
}

impl<T, N> Mul<N> for ChunkedArray<T>
where
    T: PolarsNumericType,
    T::Native: NumCast,
    N: Num + ToPrimitive,
    T::Native: Mul<Output = T::Native>,
{
    type Output = ChunkedArray<T>;

    fn mul(self, rhs: N) -> Self::Output {
        (&self).mul(rhs)
    }
}

impl<T, N> Rem<N> for ChunkedArray<T>
where
    T: PolarsNumericType,
    T::Native: NumCast,
    N: Num + ToPrimitive,
    T::Native:
        Div<Output = T::Native> + One + Zero + Rem<Output = T::Native> + Sub<Output = T::Native>,
{
    type Output = ChunkedArray<T>;

    fn rem(self, rhs: N) -> Self::Output {
        (&self).rem(rhs)
    }
}

fn concat_strings(l: &str, r: &str) -> String {
    // fastest way to concat strings according to https://github.com/hoodie/concatenation_benchmarks-rs
    let mut s = String::with_capacity(l.len() + r.len());
    s.push_str(l);
    s.push_str(r);
    s
}

impl Add for &Utf8Chunked {
    type Output = Utf8Chunked;

    fn add(self, rhs: Self) -> Self::Output {
        // broadcasting path rhs
        if rhs.len() == 1 {
            let rhs = rhs.get(0);
            return match rhs {
                Some(rhs) => self.add(rhs),
                None => Utf8Chunked::full_null(self.name(), self.len()),
            };
        }
        // broadcasting path lhs
        if self.len() == 1 {
            let lhs = self.get(0);
            return match lhs {
                Some(lhs) => rhs.apply(|s| Cow::Owned(concat_strings(lhs, s))),
                None => Utf8Chunked::full_null(self.name(), rhs.len()),
            };
        }

        // todo! add no_null variants. Need 4 paths.
        self.into_iter()
            .zip(rhs.into_iter())
            .map(|(opt_l, opt_r)| match (opt_l, opt_r) {
                (Some(l), Some(r)) => Some(concat_strings(l, r)),
                _ => None,
            })
            .collect()
    }
}

impl Add for Utf8Chunked {
    type Output = Utf8Chunked;

    fn add(self, rhs: Self) -> Self::Output {
        (&self).add(&rhs)
    }
}

impl Add<&str> for &Utf8Chunked {
    type Output = Utf8Chunked;

    fn add(self, rhs: &str) -> Self::Output {
        match self.null_count() {
            0 => self
                .into_no_null_iter()
                .map(|l| concat_strings(l, rhs))
                .collect(),
            _ => self
                .into_iter()
                .map(|opt_l| opt_l.map(|l| concat_strings(l, rhs)))
                .collect(),
        }
    }
}

pub trait Pow {
    fn pow_f32(&self, _exp: f32) -> Float32Chunked {
        unimplemented!()
    }
    fn pow_f64(&self, _exp: f64) -> Float64Chunked {
        unimplemented!()
    }
}

impl<T> Pow for ChunkedArray<T>
where
    T: PolarsNumericType,
    ChunkedArray<T>: ChunkCast,
{
    fn pow_f32(&self, exp: f32) -> Float32Chunked {
        self.cast::<Float32Type>()
            .expect("f32 array")
            .apply_kernel(|arr| {
                Arc::new(compute::arity::unary(
                    arr,
                    |x| x.powf(exp),
                    DataType::Float32.to_arrow(),
                ))
            })
    }

    fn pow_f64(&self, exp: f64) -> Float64Chunked {
        self.cast::<Float64Type>()
            .expect("f64 array")
            .apply_kernel(|arr| {
                Arc::new(compute::arity::unary(
                    arr,
                    |x| x.powf(exp),
                    DataType::Float64.to_arrow(),
                ))
            })
    }
}

impl Pow for BooleanChunked {}
impl Pow for Utf8Chunked {}
impl Pow for ListChunked {}
impl Pow for CategoricalChunked {}

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
    #[allow(clippy::eq_op)]
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
