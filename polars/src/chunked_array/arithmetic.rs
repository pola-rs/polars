use crate::prelude::*;
use arrow::{array::ArrayRef, compute};
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

                if let Ok(slice_lhs) = $self.cont_slice() {
                    slice_lhs
                        .iter()
                        .zip($rhs.into_iter())
                        .map(|(&left, opt_right)| opt_right.map(|right| left $operand right))
                        .collect()
                } else if let Ok(slice_rhs) = $rhs.cont_slice() {
                    slice_rhs
                        .iter()
                        .zip($self.into_iter())
                        .map(|(&right, opt_left)| opt_left.map(|left| left $operand right))
                        .collect()
                } else {
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

impl<T> Add for &ChunkedArray<T>
where
    T: PolarNumericType,
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
    T: PolarNumericType,
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
    T: PolarNumericType,
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
    T: PolarNumericType,
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
    T: PolarNumericType,
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
    T: PolarNumericType,
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
    T: PolarNumericType,
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
    T: PolarNumericType,
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
}
