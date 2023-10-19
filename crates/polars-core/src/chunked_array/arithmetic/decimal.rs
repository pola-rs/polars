use arrow::legacy::compute::arithmetics::decimal;

use super::*;
use crate::prelude::DecimalChunked;
use crate::utils::align_chunks_binary;

// TODO: remove
impl ArrayArithmetics for i128 {
    fn add(_lhs: &PrimitiveArray<Self>, _rhs: &PrimitiveArray<Self>) -> PrimitiveArray<Self> {
        unimplemented!()
    }

    fn sub(_lhs: &PrimitiveArray<Self>, _rhs: &PrimitiveArray<Self>) -> PrimitiveArray<Self> {
        unimplemented!()
    }

    fn mul(_lhs: &PrimitiveArray<Self>, _rhs: &PrimitiveArray<Self>) -> PrimitiveArray<Self> {
        unimplemented!()
    }

    fn div(_lhs: &PrimitiveArray<Self>, _rhs: &PrimitiveArray<Self>) -> PrimitiveArray<Self> {
        unimplemented!()
    }

    fn div_scalar(_lhs: &PrimitiveArray<Self>, _rhs: &Self) -> PrimitiveArray<Self> {
        unimplemented!()
    }

    fn rem(_lhs: &PrimitiveArray<Self>, _rhs: &PrimitiveArray<Self>) -> PrimitiveArray<Self> {
        unimplemented!("requires support in arrow2 crate")
    }

    fn rem_scalar(_lhs: &PrimitiveArray<Self>, _rhs: &Self) -> PrimitiveArray<Self> {
        unimplemented!("requires support in arrow2 crate")
    }
}

impl DecimalChunked {
    fn arithmetic_helper<Kernel, ScalarKernelLhs, ScalarKernelRhs>(
        &self,
        rhs: &DecimalChunked,
        kernel: Kernel,
        operation_lhs: ScalarKernelLhs,
        operation_rhs: ScalarKernelRhs,
    ) -> PolarsResult<Self>
    where
        Kernel:
            Fn(&PrimitiveArray<i128>, &PrimitiveArray<i128>) -> PolarsResult<PrimitiveArray<i128>>,
        ScalarKernelLhs: Fn(&PrimitiveArray<i128>, i128) -> PolarsResult<PrimitiveArray<i128>>,
        ScalarKernelRhs: Fn(i128, &PrimitiveArray<i128>) -> PolarsResult<PrimitiveArray<i128>>,
    {
        let lhs = self;

        let mut ca = match (lhs.len(), rhs.len()) {
            (a, b) if a == b => {
                let (lhs, rhs) = align_chunks_binary(lhs, rhs);
                let chunks = lhs
                    .downcast_iter()
                    .zip(rhs.downcast_iter())
                    .map(|(lhs, rhs)| kernel(lhs, rhs).map(|a| Box::new(a) as ArrayRef))
                    .collect::<PolarsResult<_>>()?;
                unsafe { lhs.copy_with_chunks(chunks, false, false) }
            },
            // broadcast right path
            (_, 1) => {
                let opt_rhs = rhs.get(0);
                match opt_rhs {
                    None => ChunkedArray::full_null(lhs.name(), lhs.len()),
                    Some(rhs_val) => {
                        let chunks = lhs
                            .downcast_iter()
                            .map(|lhs| operation_lhs(lhs, rhs_val).map(|a| Box::new(a) as ArrayRef))
                            .collect::<PolarsResult<_>>()?;
                        unsafe { lhs.copy_with_chunks(chunks, false, false) }
                    },
                }
            },
            (1, _) => {
                let opt_lhs = lhs.get(0);
                match opt_lhs {
                    None => ChunkedArray::full_null(lhs.name(), rhs.len()),
                    Some(lhs_val) => {
                        let chunks = rhs
                            .downcast_iter()
                            .map(|rhs| operation_rhs(lhs_val, rhs).map(|a| Box::new(a) as ArrayRef))
                            .collect::<PolarsResult<_>>()?;
                        unsafe { lhs.copy_with_chunks(chunks, false, false) }
                    },
                }
            },
            _ => {
                polars_bail!(ComputeError: "cannot apply operation on arrays of different lengths")
            },
        };
        ca.rename(lhs.name());
        Ok(ca.into_decimal_unchecked(self.precision(), self.scale()))
    }
}

impl Add for &DecimalChunked {
    type Output = PolarsResult<DecimalChunked>;

    fn add(self, rhs: Self) -> Self::Output {
        self.arithmetic_helper(
            rhs,
            decimal::add,
            |lhs, rhs_val| decimal::add_scalar(lhs, rhs_val, &rhs.dtype().to_arrow()),
            |lhs_val, rhs| decimal::add_scalar(rhs, lhs_val, &self.dtype().to_arrow()),
        )
    }
}

impl Sub for &DecimalChunked {
    type Output = PolarsResult<DecimalChunked>;

    fn sub(self, rhs: Self) -> Self::Output {
        self.arithmetic_helper(
            rhs,
            decimal::sub,
            decimal::sub_scalar,
            decimal::sub_scalar_swapped,
        )
    }
}

impl Mul for &DecimalChunked {
    type Output = PolarsResult<DecimalChunked>;

    fn mul(self, rhs: Self) -> Self::Output {
        self.arithmetic_helper(
            rhs,
            decimal::mul,
            |lhs, rhs_val| decimal::mul_scalar(lhs, rhs_val, &rhs.dtype().to_arrow()),
            |lhs_val, rhs| decimal::mul_scalar(rhs, lhs_val, &self.dtype().to_arrow()),
        )
    }
}

impl Div for &DecimalChunked {
    type Output = PolarsResult<DecimalChunked>;

    fn div(self, rhs: Self) -> Self::Output {
        self.arithmetic_helper(
            rhs,
            decimal::div,
            |lhs, rhs_val| decimal::div_scalar(lhs, rhs_val, &rhs.dtype().to_arrow()),
            |lhs_val, rhs| decimal::div_scalar_swapped(lhs_val, &self.dtype().to_arrow(), rhs),
        )
    }
}
