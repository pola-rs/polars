use polars_compute::arithmetic::ArithmeticKernel;

use crate::chunked_array::arity::{apply_binary_kernel_broadcast, apply_binary_kernel_broadcast_owned, unary_kernel, unary_kernel_owned};

use super::*;

pub(crate) fn arithmetic_helper<T, Kernel, F>(
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
        (a, b) if a == b => arity::binary(lhs, rhs, |lhs, rhs| kernel(lhs, rhs)),
        // broadcast right path
        (_, 1) => {
            let opt_rhs = rhs.get(0);
            match opt_rhs {
                None => ChunkedArray::full_null(lhs.name(), lhs.len()),
                Some(rhs) => lhs.apply_values(|lhs| operation(lhs, rhs)),
            }
        },
        (1, _) => {
            let opt_lhs = lhs.get(0);
            match opt_lhs {
                None => ChunkedArray::full_null(lhs.name(), rhs.len()),
                Some(lhs) => rhs.apply_values(|rhs| operation(lhs, rhs)),
            }
        },
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
            lhs.compute_len();
            lhs.set_sorted_flag(IsSorted::Not);
            lhs
        },
        // broadcast right path
        (_, 1) => {
            let opt_rhs = rhs.get(0);
            match opt_rhs {
                None => ChunkedArray::full_null(lhs.name(), lhs.len()),
                Some(rhs) => {
                    lhs.apply_mut(|lhs| operation(lhs, rhs));
                    lhs
                },
            }
        },
        (1, _) => {
            let opt_lhs = lhs.get(0);
            match opt_lhs {
                None => ChunkedArray::full_null(lhs.name(), rhs.len()),
                Some(lhs_val) => {
                    rhs.apply_mut(|rhs| operation(lhs_val, rhs));
                    rhs.rename(lhs.name());
                    rhs
                },
            }
        },
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
        apply_binary_kernel_broadcast(
            self,
            rhs,
            |l, r| ArithmeticKernel::wrapping_add(l.clone(), r.clone()),
            |l, r| ArithmeticKernel::wrapping_add_scalar(r.clone(), l),
            |l, r| ArithmeticKernel::wrapping_add_scalar(l.clone(), r),
        )
    }
}

impl<T> Mul for &ChunkedArray<T>
where
    T: PolarsNumericType,
{
    type Output = ChunkedArray<T>;

    fn mul(self, rhs: Self) -> Self::Output {
        apply_binary_kernel_broadcast(
            self,
            rhs,
            |l, r| ArithmeticKernel::wrapping_mul(l.clone(), r.clone()),
            |l, r| ArithmeticKernel::wrapping_mul_scalar(r.clone(), l),
            |l, r| ArithmeticKernel::wrapping_mul_scalar(l.clone(), r),
        )
    }
}

impl<T> Rem for &ChunkedArray<T>
where
    T: PolarsNumericType,
{
    type Output = ChunkedArray<T>;

    fn rem(self, rhs: Self) -> Self::Output {
        apply_binary_kernel_broadcast(
            self,
            rhs,
            |l, r| ArithmeticKernel::wrapping_mod(l.clone(), r.clone()),
            |l, r| ArithmeticKernel::wrapping_mod_scalar_lhs(l, r.clone()),
            |l, r| ArithmeticKernel::wrapping_mod_scalar(l.clone(), r),
        )
    }
}

impl<T> Sub for &ChunkedArray<T>
where
    T: PolarsNumericType,
{
    type Output = ChunkedArray<T>;

    fn sub(self, rhs: Self) -> Self::Output {
        apply_binary_kernel_broadcast(
            self,
            rhs,
            |l, r| ArithmeticKernel::wrapping_sub(l.clone(), r.clone()),
            |l, r| ArithmeticKernel::wrapping_sub_scalar_lhs(l, r.clone()),
            |l, r| ArithmeticKernel::wrapping_sub_scalar(l.clone(), r),
        )
    }
}

impl<T> Add for ChunkedArray<T>
where
    T: PolarsNumericType,
{
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        apply_binary_kernel_broadcast_owned(
            self,
            rhs,
            |l, r| ArithmeticKernel::wrapping_add(l, r),
            |l, r| ArithmeticKernel::wrapping_add_scalar(r, l),
            |l, r| ArithmeticKernel::wrapping_add_scalar(l, r),
        )
    }
}

impl<T> Mul for ChunkedArray<T>
where
    T: PolarsNumericType,
{
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        apply_binary_kernel_broadcast_owned(
            self,
            rhs,
            |l, r| ArithmeticKernel::wrapping_mul(l, r),
            |l, r| ArithmeticKernel::wrapping_mul_scalar(r, l),
            |l, r| ArithmeticKernel::wrapping_mul_scalar(l, r),
        )
    }
}

impl<T> Sub for ChunkedArray<T>
where
    T: PolarsNumericType,
{
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        apply_binary_kernel_broadcast_owned(
            self,
            rhs,
            |l, r| ArithmeticKernel::wrapping_sub(l, r),
            |l, r| ArithmeticKernel::wrapping_sub_scalar_lhs(l, r),
            |l, r| ArithmeticKernel::wrapping_sub_scalar(l, r),
        )
    }
}

impl<T> Rem for ChunkedArray<T>
where
    T: PolarsNumericType,
{
    type Output = ChunkedArray<T>;

    fn rem(self, rhs: Self) -> Self::Output {
        apply_binary_kernel_broadcast_owned(
            self,
            rhs,
            |l, r| ArithmeticKernel::wrapping_mod(l, r),
            |l, r| ArithmeticKernel::wrapping_mod_scalar_lhs(l, r),
            |l, r| ArithmeticKernel::wrapping_mod_scalar(l, r),
        )
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
        let rhs: T::Native = NumCast::from(rhs).unwrap();
        unary_kernel(self, |a| ArithmeticKernel::wrapping_add_scalar(a.clone(), rhs))
    }
}

impl<T, N> Sub<N> for &ChunkedArray<T>
where
    T: PolarsNumericType,
    N: Num + ToPrimitive,
{
    type Output = ChunkedArray<T>;

    fn sub(self, rhs: N) -> Self::Output {
        let rhs: T::Native = NumCast::from(rhs).unwrap();
        unary_kernel(self, |a| ArithmeticKernel::wrapping_sub_scalar(a.clone(), rhs))
    }
}

impl<T, N> Mul<N> for &ChunkedArray<T>
where
    T: PolarsNumericType,
    N: Num + ToPrimitive,
{
    type Output = ChunkedArray<T>;

    fn mul(self, rhs: N) -> Self::Output {
        let rhs: T::Native = NumCast::from(rhs).unwrap();
        unary_kernel(self, |a| ArithmeticKernel::wrapping_mul_scalar(a.clone(), rhs))
    }
}

impl<T, N> Rem<N> for &ChunkedArray<T>
where
    T: PolarsNumericType,
    N: Num + ToPrimitive,
{
    type Output = ChunkedArray<T>;

    fn rem(self, rhs: N) -> Self::Output {
        let rhs: T::Native = NumCast::from(rhs).unwrap();
        unary_kernel(self, |a| ArithmeticKernel::wrapping_mod_scalar(a.clone(), rhs))
    }
}

impl<T, N> Add<N> for ChunkedArray<T>
where
    T: PolarsNumericType,
    N: Num + ToPrimitive,
{
    type Output = ChunkedArray<T>;

    fn add(self, rhs: N) -> Self::Output {
        let rhs: T::Native = NumCast::from(rhs).unwrap();
        unary_kernel_owned(self, |a| ArithmeticKernel::wrapping_add_scalar(a, rhs))
    }
}

impl<T, N> Sub<N> for ChunkedArray<T>
where
    T: PolarsNumericType,
    N: Num + ToPrimitive,
{
    type Output = ChunkedArray<T>;

    fn sub(self, rhs: N) -> Self::Output {
        let rhs: T::Native = NumCast::from(rhs).unwrap();
        unary_kernel_owned(self, |a| ArithmeticKernel::wrapping_sub_scalar(a, rhs))
    }
}

impl<T, N> Mul<N> for ChunkedArray<T>
where
    T: PolarsNumericType,
    N: Num + ToPrimitive,
{
    type Output = ChunkedArray<T>;

    fn mul(self, rhs: N) -> Self::Output {
        let rhs: T::Native = NumCast::from(rhs).unwrap();
        unary_kernel_owned(self, |a| ArithmeticKernel::wrapping_mul_scalar(a, rhs))
    }
}

impl<T, N> Rem<N> for ChunkedArray<T>
where
    T: PolarsNumericType,
    N: Num + ToPrimitive,
{
    type Output = ChunkedArray<T>;

    fn rem(self, rhs: N) -> Self::Output {
        let rhs: T::Native = NumCast::from(rhs).unwrap();
        unary_kernel_owned(self, |a| ArithmeticKernel::wrapping_mod_scalar(a, rhs))
    }
}



impl<T> Div for &ChunkedArray<T>
where
    T: PolarsNumericType,
{
    type Output = ChunkedArray<T>;

    fn div(self, rhs: Self) -> Self::Output {
        apply_binary_kernel_broadcast(
            self,
            rhs,
            |l, r| ArithmeticKernel::legacy_div(l.clone(), r.clone()),
            |l, r| ArithmeticKernel::legacy_div_scalar_lhs(l, r.clone()),
            |l, r| ArithmeticKernel::legacy_div_scalar(l.clone(), r),
        )
    }
}

impl<T> Div for ChunkedArray<T>
where
    T: PolarsNumericType,
{
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        apply_binary_kernel_broadcast_owned(
            self,
            rhs,
            |l, r| ArithmeticKernel::legacy_div(l, r),
            |l, r| ArithmeticKernel::legacy_div_scalar_lhs(l, r),
            |l, r| ArithmeticKernel::legacy_div_scalar(l, r),
        )
    }
}

impl<T, N> Div<N> for &ChunkedArray<T>
where
    T: PolarsNumericType,
    N: Num + ToPrimitive,
{
    type Output = ChunkedArray<T>;

    fn div(self, rhs: N) -> Self::Output {
        let rhs: T::Native = NumCast::from(rhs).unwrap();
        unary_kernel(self, |a| ArithmeticKernel::legacy_div_scalar(a.clone(), rhs))
    }
}

impl<T, N> Div<N> for ChunkedArray<T>
where
    T: PolarsNumericType,
    N: Num + ToPrimitive,
{
    type Output = ChunkedArray<T>;

    fn div(self, rhs: N) -> Self::Output {
        let rhs: T::Native = NumCast::from(rhs).unwrap();
        unary_kernel_owned(self, |a| ArithmeticKernel::legacy_div_scalar(a, rhs))
    }
}
