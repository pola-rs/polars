use crate::prelude::DecimalChunked;
use polars_arrow::compute::arithmetics::decimal;
use super::*;

impl ArrayArithmetics for i128 {
    fn add(lhs: &PrimitiveArray<Self>, rhs: &PrimitiveArray<Self>) -> PrimitiveArray<Self> {
        decimal::add(lhs, rhs).unwrap()
    }

    fn sub(lhs: &PrimitiveArray<Self>, rhs: &PrimitiveArray<Self>) -> PrimitiveArray<Self> {
        todo!()
        // decimal::sub(lhs, rhs)
    }

    fn mul(lhs: &PrimitiveArray<Self>, rhs: &PrimitiveArray<Self>) -> PrimitiveArray<Self> {
        todo!()
        // decimal::mul(lhs, rhs)
    }

    fn div(lhs: &PrimitiveArray<Self>, rhs: &PrimitiveArray<Self>) -> PrimitiveArray<Self> {
        todo!()
        // decimal::div(lhs, rhs)
    }

    fn div_scalar(_lhs: &PrimitiveArray<Self>, _rhs: &Self) -> PrimitiveArray<Self> {
        // decimal::div_scalar(lhs, rhs)
        todo!("decimal::div_scalar exists, but takes &PrimitiveScalar<i128>, not &i128");
    }

    fn rem(_lhs: &PrimitiveArray<Self>, _rhs: &PrimitiveArray<Self>) -> PrimitiveArray<Self> {
        unimplemented!("requires support in arrow2 crate")
    }

    fn rem_scalar(_lhs: &PrimitiveArray<Self>, _rhs: &Self) -> PrimitiveArray<Self> {
        unimplemented!("requires support in arrow2 crate")
    }
}

impl DecimalChunked {
    fn arithmetic_helper<Kernel, ScalarKernelLhs, ScalarKernelRhs>(&self, rhs: &DecimalChunked,
                                                  kernel: Kernel,
                                                  operation_lhs: ScalarKernelLhs,
                                                  operation_rhs: ScalarKernelRhs
    ) -> PolarsResult<Self>
        where
            Kernel: Fn(&PrimitiveArray<i128>, &PrimitiveArray<i128>) -> PolarsResult<PrimitiveArray<i128>>,
            ScalarKernelLhs: Fn(&PrimitiveArray<i128>, i128, &ArrowDataType) -> PolarsResult<PrimitiveArray<i128>>,
            ScalarKernelRhs: Fn(i128, &ArrowDataType, &PrimitiveArray<i128>) -> PolarsResult<PrimitiveArray<i128>>

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
                lhs.copy_with_chunks(chunks, false, false)
            }
            // broadcast right path
            (_, 1) => {
                let opt_rhs = rhs.get(0);
                match opt_rhs {
                    None => ChunkedArray::full_null(lhs.name(), lhs.len()),
                    Some(rhs_val) => {
                        let chunks = lhs.downcast_iter().map(|lhs| {
                            operation_lhs(lhs, rhs_val, &rhs.dtype().to_arrow()).map(|a| Box::new(a) as ArrayRef)
                        }).collect::<PolarsResult<_>>()?;
                        lhs.copy_with_chunks(chunks, false, false)
                    },
                }
            }
            (1, _) => {
                let opt_lhs = lhs.get(0);
                match opt_lhs {
                    None => ChunkedArray::full_null(lhs.name(), rhs.len()),
                    Some(lhs_val) => {
                        let chunks = rhs.downcast_iter().map(|rhs| {
                            operation_rhs(lhs_val, &lhs.dtype().to_arrow(), rhs).map(|a| Box::new(a) as ArrayRef)
                        }).collect::<PolarsResult<_>>()?;
                        lhs.copy_with_chunks(chunks, false, false)

                    }
                }
            }
            _ => polars_bail!(ComputeError: "Cannot apply operation on arrays of different lengths"),
        };
        ca.rename(lhs.name());
        Ok(ca.into_decimal_unchecked(self.precision(), self.scale()))

    }

}

fn reversed<Kernel>(lhs_val: i128, lhs_dtype: &ArrowDataType, rhs: &PrimitiveArray<i128>, op: Kernel) -> PolarsResult<PrimitiveArray<i128>>
where Kernel: Fn(&PrimitiveArray<i128>, i128, &ArrowDataType) -> PolarsResult<PrimitiveArray<i128>>,
{
    op(rhs, lhs_val, lhs_dtype)
}

impl Add for &DecimalChunked
{
    type Output = PolarsResult<DecimalChunked>;

    fn add(self, rhs: Self) -> Self::Output {
        self.arithmetic_helper(rhs, decimal::add, decimal::add_scalar, |lhs_val, lhs_dtype, rhs| reversed(lhs_val, lhs_dtype, rhs, decimal::add_scalar))
    }
}
