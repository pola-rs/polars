//! Implementations of arithmetic operations on ChunkedArray's.
use std::borrow::Cow;
use std::ops::{Add, Div, Mul, Rem, Sub};

use arrow::array::PrimitiveArray;
use arrow::compute::arithmetics::basic;
use arrow::compute::arity_assign;
use num::{Num, NumCast, ToPrimitive};

use crate::prelude::*;
use crate::series::IsSorted;
use crate::utils::{align_chunks_binary, align_chunks_binary_owned};

macro_rules! apply_operand_on_chunkedarray_by_iter {

    ($self:ident, $rhs:ident, $operand:tt) => {
            {
                match ($self.has_validity(), $rhs.has_validity()) {
                    (false, false) => {
                        let a: NoNull<ChunkedArray<_>> = $self
                        .into_no_null_iter()
                        .zip($rhs.into_no_null_iter())
                        .map(|(left, right)| left $operand right)
                        .collect_trusted();
                        a.into_inner()
                    },
                    (false, _) => {
                        $self
                        .into_no_null_iter()
                        .zip($rhs.into_iter())
                        .map(|(left, opt_right)| opt_right.map(|right| left $operand right))
                        .collect_trusted()
                    },
                    (_, false) => {
                        $self
                        .into_iter()
                        .zip($rhs.into_no_null_iter())
                        .map(|(opt_left, right)| opt_left.map(|left| left $operand right))
                        .collect_trusted()
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
                        .collect_trusted()

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
    Kernel: Fn(&PrimitiveArray<T::Native>, &PrimitiveArray<T::Native>) -> PrimitiveArray<T::Native>,
    F: Fn(T::Native, T::Native) -> T::Native,
{
    let mut ca = match (lhs.len(), rhs.len()) {
        (a, b) if a == b => {
            let (lhs, rhs) = align_chunks_binary(lhs, rhs);
            let chunks = lhs
                .downcast_iter()
                .zip(rhs.downcast_iter())
                .map(|(lhs, rhs)| Box::new(kernel(lhs, rhs)) as ArrayRef)
                .collect();
            lhs.copy_with_chunks(chunks, false)
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

/// This assigns to the owned buffer if the ref count is 1
fn arithmetic_helper_owned<T, Kernel, F>(
    mut lhs: ChunkedArray<T>,
    mut rhs: ChunkedArray<T>,
    kernel: Kernel,
    operation: F,
) -> ChunkedArray<T>
where
    T: PolarsNumericType,
    Kernel: Fn(&mut PrimitiveArray<T::Native>, &mut PrimitiveArray<T::Native>),
    F: Fn(T::Native, T::Native) -> T::Native,
{
    let ca = match (lhs.len(), rhs.len()) {
        (a, b) if a == b => {
            let (mut lhs, mut rhs) = align_chunks_binary_owned(lhs, rhs);
            // safety, we do no t change the lengths
            unsafe {
                lhs.downcast_iter_mut()
                    .zip(rhs.downcast_iter_mut())
                    .for_each(|(lhs, rhs)| kernel(lhs, rhs));
            }
            lhs.set_sorted2(IsSorted::Not);
            lhs
        }
        // broadcast right path
        (_, 1) => {
            let opt_rhs = rhs.get(0);
            match opt_rhs {
                None => ChunkedArray::full_null(lhs.name(), lhs.len()),
                Some(rhs) => {
                    lhs.apply_mut(|lhs| operation(lhs, rhs));
                    lhs
                }
            }
        }
        (1, _) => {
            let opt_lhs = lhs.get(0);
            match opt_lhs {
                None => ChunkedArray::full_null(lhs.name(), rhs.len()),
                Some(lhs) => {
                    rhs.apply_mut(|rhs| operation(lhs, rhs));
                    rhs
                }
            }
        }
        _ => panic!("Cannot apply operation on arrays of different lengths"),
    };
    ca
}

// Operands on ChunkedArray & ChunkedArray

impl<T> Add for &ChunkedArray<T>
where
    T: PolarsNumericType,
{
    type Output = ChunkedArray<T>;

    fn add(self, rhs: Self) -> Self::Output {
        arithmetic_helper(self, rhs, basic::add, |lhs, rhs| lhs + rhs)
    }
}

impl<T> Div for &ChunkedArray<T>
where
    T: PolarsNumericType,
{
    type Output = ChunkedArray<T>;

    fn div(self, rhs: Self) -> Self::Output {
        arithmetic_helper(self, rhs, basic::div, |lhs, rhs| lhs / rhs)
    }
}

impl<T> Mul for &ChunkedArray<T>
where
    T: PolarsNumericType,
{
    type Output = ChunkedArray<T>;

    fn mul(self, rhs: Self) -> Self::Output {
        arithmetic_helper(self, rhs, basic::mul, |lhs, rhs| lhs * rhs)
    }
}

impl<T> Rem for &ChunkedArray<T>
where
    T: PolarsNumericType,
{
    type Output = ChunkedArray<T>;

    fn rem(self, rhs: Self) -> Self::Output {
        arithmetic_helper(self, rhs, basic::rem, |lhs, rhs| lhs % rhs)
    }
}

impl<T> Sub for &ChunkedArray<T>
where
    T: PolarsNumericType,
{
    type Output = ChunkedArray<T>;

    fn sub(self, rhs: Self) -> Self::Output {
        arithmetic_helper(self, rhs, basic::sub, |lhs, rhs| lhs - rhs)
    }
}

impl<T> Add for ChunkedArray<T>
where
    T: PolarsNumericType,
{
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        arithmetic_helper_owned(
            self,
            rhs,
            |a, b| arity_assign::binary(a, b, |a, b| a + b),
            |lhs, rhs| lhs + rhs,
        )
    }
}

impl<T> Div for ChunkedArray<T>
where
    T: PolarsNumericType,
{
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        arithmetic_helper_owned(
            self,
            rhs,
            |a, b| arity_assign::binary(a, b, |a, b| a / b),
            |lhs, rhs| lhs / rhs,
        )
    }
}

impl<T> Mul for ChunkedArray<T>
where
    T: PolarsNumericType,
{
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        arithmetic_helper_owned(
            self,
            rhs,
            |a, b| arity_assign::binary(a, b, |a, b| a * b),
            |lhs, rhs| lhs * rhs,
        )
    }
}

impl<T> Sub for ChunkedArray<T>
where
    T: PolarsNumericType,
{
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        arithmetic_helper_owned(
            self,
            rhs,
            |a, b| arity_assign::binary(a, b, |a, b| a - b),
            |lhs, rhs| lhs - rhs,
        )
    }
}

impl<T> Rem for ChunkedArray<T>
where
    T: PolarsNumericType,
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
    N: Num + ToPrimitive,
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
    N: Num + ToPrimitive,
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
    N: Num + ToPrimitive,
{
    type Output = ChunkedArray<T>;

    fn div(self, rhs: N) -> Self::Output {
        let rhs: T::Native = NumCast::from(rhs).expect("could not cast");
        self.apply_kernel(&|arr| Box::new(basic::div_scalar(arr, &rhs)))
    }
}

impl<T, N> Mul<N> for &ChunkedArray<T>
where
    T: PolarsNumericType,
    N: Num + ToPrimitive,
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
    N: Num + ToPrimitive,
{
    type Output = ChunkedArray<T>;

    fn rem(self, rhs: N) -> Self::Output {
        let rhs: T::Native = NumCast::from(rhs).expect("could not cast");
        self.apply_kernel(&|arr| Box::new(basic::rem_scalar(arr, &rhs)))
    }
}

impl<T, N> Add<N> for ChunkedArray<T>
where
    T: PolarsNumericType,
    N: Num + ToPrimitive,
{
    type Output = ChunkedArray<T>;

    fn add(mut self, rhs: N) -> Self::Output {
        if std::env::var("ASSIGN").is_ok() {
            let adder: T::Native = NumCast::from(rhs).unwrap();
            self.apply_mut(|val| val + adder);
            self
        } else {
            (&self).add(rhs)
        }
    }
}

impl<T, N> Sub<N> for ChunkedArray<T>
where
    T: PolarsNumericType,
    N: Num + ToPrimitive,
{
    type Output = ChunkedArray<T>;

    fn sub(mut self, rhs: N) -> Self::Output {
        if std::env::var("ASSIGN").is_ok() {
            let subber: T::Native = NumCast::from(rhs).unwrap();
            self.apply_mut(|val| val - subber);
            self
        } else {
            (&self).sub(rhs)
        }
    }
}

impl<T, N> Div<N> for ChunkedArray<T>
where
    T: PolarsNumericType,
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
    N: Num + ToPrimitive,
{
    type Output = ChunkedArray<T>;

    fn mul(mut self, rhs: N) -> Self::Output {
        if std::env::var("ASSIGN").is_ok() {
            let multiplier: T::Native = NumCast::from(rhs).unwrap();
            self.apply_mut(|val| val * multiplier);
            self
        } else {
            (&self).mul(rhs)
        }
    }
}

impl<T, N> Rem<N> for ChunkedArray<T>
where
    T: PolarsNumericType,
    N: Num + ToPrimitive,
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

#[cfg(feature = "dtype-binary")]
fn concat_binary_arrs(l: &[u8], r: &[u8]) -> Vec<u8> {
    let mut v = Vec::with_capacity(l.len() + r.len());
    v.extend_from_slice(l);
    v.extend_from_slice(r);
    v
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
        let mut ca: Self::Output = self
            .into_iter()
            .zip(rhs.into_iter())
            .map(|(opt_l, opt_r)| match (opt_l, opt_r) {
                (Some(l), Some(r)) => Some(concat_strings(l, r)),
                _ => None,
            })
            .collect_trusted();
        ca.rename(self.name());
        ca
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
        let mut ca: Self::Output = match self.has_validity() {
            false => self
                .into_no_null_iter()
                .map(|l| concat_strings(l, rhs))
                .collect_trusted(),
            _ => self
                .into_iter()
                .map(|opt_l| opt_l.map(|l| concat_strings(l, rhs)))
                .collect_trusted(),
        };
        ca.rename(self.name());
        ca
    }
}

#[cfg(feature = "dtype-binary")]
impl Add for &BinaryChunked {
    type Output = BinaryChunked;

    fn add(self, rhs: Self) -> Self::Output {
        // broadcasting path rhs
        if rhs.len() == 1 {
            let rhs = rhs.get(0);
            return match rhs {
                Some(rhs) => self.add(rhs),
                None => BinaryChunked::full_null(self.name(), self.len()),
            };
        }
        // broadcasting path lhs
        if self.len() == 1 {
            let lhs = self.get(0);
            return match lhs {
                Some(lhs) => rhs.apply(|s| Cow::Owned(concat_binary_arrs(lhs, s))),
                None => BinaryChunked::full_null(self.name(), rhs.len()),
            };
        }

        // todo! add no_null variants. Need 4 paths.
        let mut ca: Self::Output = self
            .into_iter()
            .zip(rhs.into_iter())
            .map(|(opt_l, opt_r)| match (opt_l, opt_r) {
                (Some(l), Some(r)) => Some(concat_binary_arrs(l, r)),
                _ => None,
            })
            .collect_trusted();
        ca.rename(self.name());
        ca
    }
}

#[cfg(feature = "dtype-binary")]
impl Add for BinaryChunked {
    type Output = BinaryChunked;

    fn add(self, rhs: Self) -> Self::Output {
        (&self).add(&rhs)
    }
}

#[cfg(feature = "dtype-binary")]
impl Add<&[u8]> for &BinaryChunked {
    type Output = BinaryChunked;

    fn add(self, rhs: &[u8]) -> Self::Output {
        let mut ca: Self::Output = match self.has_validity() {
            false => self
                .into_no_null_iter()
                .map(|l| concat_binary_arrs(l, rhs))
                .collect_trusted(),
            _ => self
                .into_iter()
                .map(|opt_l| opt_l.map(|l| concat_binary_arrs(l, rhs)))
                .collect_trusted(),
        };
        ca.rename(self.name());
        ca
    }
}

#[cfg(test)]
pub(crate) mod test {
    use crate::prelude::*;

    pub(crate) fn create_two_chunked() -> (Int32Chunked, Int32Chunked) {
        let mut a1 = Int32Chunked::new("a", &[1, 2, 3]);
        let a2 = Int32Chunked::new("a", &[4, 5, 6]);
        let a3 = Int32Chunked::new("a", &[1, 2, 3, 4, 5, 6]);
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
}
