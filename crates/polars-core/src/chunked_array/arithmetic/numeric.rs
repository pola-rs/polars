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
        arithmetic_helper(
            self,
            rhs,
            <T::Native as ArrayArithmetics>::add,
            |lhs, rhs| lhs + rhs,
        )
    }
}

impl<T> Div for &ChunkedArray<T>
where
    T: PolarsNumericType,
{
    type Output = ChunkedArray<T>;

    fn div(self, rhs: Self) -> Self::Output {
        arithmetic_helper(
            self,
            rhs,
            <T::Native as ArrayArithmetics>::div,
            |lhs, rhs| lhs / rhs,
        )
    }
}

impl<T> Mul for &ChunkedArray<T>
where
    T: PolarsNumericType,
{
    type Output = ChunkedArray<T>;

    fn mul(self, rhs: Self) -> Self::Output {
        arithmetic_helper(
            self,
            rhs,
            <T::Native as ArrayArithmetics>::mul,
            |lhs, rhs| lhs * rhs,
        )
    }
}

impl<T> Rem for &ChunkedArray<T>
where
    T: PolarsNumericType,
{
    type Output = ChunkedArray<T>;

    fn rem(self, rhs: Self) -> Self::Output {
        arithmetic_helper(
            self,
            rhs,
            <T::Native as ArrayArithmetics>::rem,
            |lhs, rhs| lhs % rhs,
        )
    }
}

impl<T> Sub for &ChunkedArray<T>
where
    T: PolarsNumericType,
{
    type Output = ChunkedArray<T>;

    fn sub(self, rhs: Self) -> Self::Output {
        arithmetic_helper(
            self,
            rhs,
            <T::Native as ArrayArithmetics>::sub,
            |lhs, rhs| lhs - rhs,
        )
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
        let mut out = self.apply_values(|val| val + adder);
        out.set_sorted_flag(self.is_sorted_flag());
        out
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
        let mut out = self.apply_values(|val| val - subber);
        out.set_sorted_flag(self.is_sorted_flag());
        out
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
        let mut out = self
            .apply_kernel(&|arr| Box::new(<T::Native as ArrayArithmetics>::div_scalar(arr, &rhs)));

        if rhs.tot_lt(&T::Native::zero()) {
            out.set_sorted_flag(self.is_sorted_flag().reverse());
        } else {
            out.set_sorted_flag(self.is_sorted_flag());
        }
        out
    }
}

impl<T, N> Mul<N> for &ChunkedArray<T>
where
    T: PolarsNumericType,
    N: Num + ToPrimitive,
{
    type Output = ChunkedArray<T>;

    fn mul(self, rhs: N) -> Self::Output {
        // don't set sorted flag as probability of overflow is higher
        let multiplier: T::Native = NumCast::from(rhs).unwrap();
        let rhs = ChunkedArray::from_vec("", vec![multiplier]);
        self.mul(&rhs)
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
        let rhs = ChunkedArray::from_vec("", vec![rhs]);
        self.rem(&rhs)
    }
}

impl<T, N> Add<N> for ChunkedArray<T>
where
    T: PolarsNumericType,
    N: Num + ToPrimitive,
{
    type Output = ChunkedArray<T>;

    fn add(self, rhs: N) -> Self::Output {
        (&self).add(rhs)
    }
}

impl<T, N> Sub<N> for ChunkedArray<T>
where
    T: PolarsNumericType,
    N: Num + ToPrimitive,
{
    type Output = ChunkedArray<T>;

    fn sub(self, rhs: N) -> Self::Output {
        (&self).sub(rhs)
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
        let multiplier: T::Native = NumCast::from(rhs).unwrap();
        self.apply_mut(|val| val * multiplier);
        self
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
