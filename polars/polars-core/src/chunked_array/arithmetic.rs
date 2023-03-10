//! Implementations of arithmetic operations on ChunkedArray's.
use std::borrow::Cow;
use std::ops::{Add, Div, Mul, Rem, Sub};

use arrow::array::PrimitiveArray;
use arrow::compute::arithmetics::basic::{self, NativeArithmetics};
use arrow::compute::arity_assign;
use num_traits::{NumCast, ToPrimitive};

use crate::prelude::*;
use crate::series::IsSorted;
use crate::utils::{align_chunks_binary, align_chunks_binary_owned};

fn try_kernel_op_ref<T, K, F>(
    lhs: &ChunkedArray<T>,
    rhs: &ChunkedArray<T>,
    kernel: K,
    op: F,
) -> PolarsResult<ChunkedArray<T>>
where
    T: PolarsNumericType,
    K: Fn(&PrimitiveArray<T::Native>, &PrimitiveArray<T::Native>) -> PrimitiveArray<T::Native>,
    F: Fn(T::Native, T::Native) -> T::Native,
{
    let ca = match (lhs.len(), rhs.len()) {
        (a, b) if a == b => {
            let (lhs, rhs) = align_chunks_binary(lhs, rhs);
            let chunks = lhs
                .downcast_iter()
                .zip(rhs.downcast_iter())
                .map(|(lhs, rhs)| Box::new(kernel(lhs, rhs)) as ArrayRef)
                .collect();
            lhs.copy_with_chunks(chunks, false, false)
        }
        (_, 1) => match rhs.get(0) {
            None => ChunkedArray::full_null(lhs.name(), lhs.len()),
            // TODO: could use <op>_scalar() logic here to speed up div/rem
            Some(rhs) => lhs.apply(|lhs| op(lhs, rhs)),
        },
        (1, _) => match lhs.get(0) {
            None => ChunkedArray::full_null(lhs.name(), rhs.len()),
            Some(lhs_val) => rhs.apply(|rhs| op(lhs_val, rhs)).with_name(lhs.name()),
        },
        _ => {
            polars_bail!(ComputeError: "cannot apply operation on arrays of different lengths")
        }
    };
    Ok(ca)
}

fn try_kernel_op_owned<T, K, F>(
    mut lhs: ChunkedArray<T>,
    mut rhs: ChunkedArray<T>,
    kernel: K,
    op: F,
) -> PolarsResult<ChunkedArray<T>>
where
    T: PolarsNumericType,
    K: Fn(&mut PrimitiveArray<T::Native>, &PrimitiveArray<T::Native>),
    F: Fn(T::Native, T::Native) -> T::Native,
{
    // this assigns to the owned buffer if the ref count is 1
    let ca = match (lhs.len(), rhs.len()) {
        (a, b) if a == b => {
            let (mut lhs, mut rhs) = align_chunks_binary_owned(lhs, rhs);
            // safety: we don't change lengths
            unsafe {
                lhs.downcast_iter_mut()
                    .zip(rhs.downcast_iter_mut())
                    .for_each(|(lhs, rhs)| kernel(lhs, rhs));
            }
            lhs.set_sorted_flag(IsSorted::Not);
            lhs
        }
        (_, 1) => match rhs.get(0) {
            None => ChunkedArray::full_null(lhs.name(), lhs.len()),
            Some(rhs) => {
                lhs.apply_mut(|lhs| op(lhs, rhs));
                lhs
            }
        },
        (1, _) => match lhs.get(0) {
            None => ChunkedArray::full_null(lhs.name(), rhs.len()),
            Some(lhs_val) => {
                rhs.apply_mut(|rhs| op(lhs_val, rhs));
                rhs.with_name(lhs.name())
            }
        },
        _ => {
            polars_bail!(ComputeError: "cannot apply operation on arrays of different lengths")
        }
    };
    Ok(ca)
}

fn try_op_scalar_ref<T, N, F>(lhs: &ChunkedArray<T>, rhs: N, op: F) -> PolarsResult<ChunkedArray<T>>
where
    T: PolarsNumericType,
    T::Native: NativeArithmetics + NumCast + ToPrimitive,
    N: NumCast,
    F: Fn(&PrimitiveArray<T::Native>, &T::Native) -> PrimitiveArray<T::Native>,
{
    let rhs: T::Native = NumCast::from(rhs).ok_or_else(
        || polars_err!(InvalidOperation: "`{}`: unable to cast to native type", stringify!($op)),
    )?;
    let chunks = lhs
        .downcast_iter()
        .map(|arr| Box::new(op(arr, &rhs)) as ArrayRef)
        .collect();
    Ok(unsafe { ChunkedArray::from_chunks(lhs.name(), chunks) })
}

fn try_op_scalar_owned<T, N, F>(
    lhs: ChunkedArray<T>,
    rhs: N,
    op: F,
) -> PolarsResult<ChunkedArray<T>>
where
    T: PolarsNumericType,
    T::Native: NativeArithmetics + NumCast + ToPrimitive,
    N: NumCast,
    F: Fn(&PrimitiveArray<T::Native>, &T::Native) -> PrimitiveArray<T::Native>,
{
    // TODO: for ops other than div_scalar/rem_scalar, it's possible to use apply_mut here instead
    try_op_scalar_ref(&lhs, rhs, op)
}

macro_rules! implement_arithmetic_ops {
    ($Op:ident, $op:ident, $op_scalar:ident, $TryOp:ident, $try_op:ident) => {
        implement_arithmetic_ops!(
            $Op, $op, $op_scalar, $TryOp, $try_op,
            try_kernel_op_ref,
            basic::$op::<T::Native>,
            try_op_scalar_ref,
            &'lhs ChunkedArray<T>, &'rhs ChunkedArray<T>, ['lhs, 'rhs],
        );
        implement_arithmetic_ops!(
            $Op, $op, $op_scalar, $TryOp, $try_op,
            try_kernel_op_owned,
            |a, b| arity_assign::binary(a, b, <T::Native as $Op>::$op),
            try_op_scalar_owned,
            ChunkedArray<T>, ChunkedArray<T>, [],
        );
    };
    (
        $Op:ident, $op:ident, $op_scalar:ident, $TryOp:ident, $try_op:ident,
        $kernel_op:ident, $func_op:expr, $func_op_scalar:expr,
        $lhs:ty, $rhs:ty, [$($llt:lifetime, $rlt:lifetime)?] $(,)?
    ) => {
        impl<$($llt, $rlt,)? T> $TryOp<$rhs> for $lhs
        where
            T: PolarsNumericType,
            T::Native: NativeArithmetics,
        {
            type Output = ChunkedArray<T>;
            type Error = PolarsError;
            fn $try_op(self, rhs: $rhs) -> PolarsResult<Self::Output> {
                $kernel_op(self, rhs, $func_op, <T::Native as $Op>::$op)
            }
        }
        impl<$($llt, $rlt,)? T> $Op<$rhs> for $lhs
        where
            T: PolarsNumericType,
            T::Native: NativeArithmetics,
        {
            type Output = ChunkedArray<T>;
            fn $op(self, rhs: $rhs) -> Self::Output {
                $TryOp::$try_op(self, rhs).unwrap()
            }
        }
        impl<$($llt,)? T, N> $TryOp<N> for $lhs
        where
            T: PolarsNumericType,
            T::Native: NativeArithmetics + ToPrimitive + NumCast,
            N: NumCast,
        {
            type Output = ChunkedArray<T>;
            type Error = PolarsError;
            fn $try_op(self, rhs: N) -> PolarsResult<Self::Output> {
                $func_op_scalar(self, rhs, basic::$op_scalar::<T::Native>)
            }
        }
        impl<$($llt,)? T, N> $Op<N> for $lhs
        where
            T: PolarsNumericType,
            T::Native: NativeArithmetics + ToPrimitive + NumCast,
            N: NumCast,
        {
            type Output = ChunkedArray<T>;
            fn $op(self, rhs: N) -> Self::Output {
                $TryOp::$try_op(self, rhs).unwrap()
            }
        }
    };
}

implement_arithmetic_ops!(Add, add, add_scalar, TryAdd, try_add);
implement_arithmetic_ops!(Sub, sub, sub_scalar, TrySub, try_sub);
implement_arithmetic_ops!(Mul, mul, mul_scalar, TryMul, try_mul);
implement_arithmetic_ops!(Div, div, div_scalar, TryDiv, try_div);
implement_arithmetic_ops!(Rem, rem, rem_scalar, TryRem, try_rem);

fn concat_strings(l: &str, r: &str) -> String {
    // fastest way to concat strings according to https://github.com/hoodie/concatenation_benchmarks-rs
    let mut s = String::with_capacity(l.len() + r.len());
    s.push_str(l);
    s.push_str(r);
    s
}

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

impl Add for BinaryChunked {
    type Output = BinaryChunked;

    fn add(self, rhs: Self) -> Self::Output {
        (&self).add(&rhs)
    }
}

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
