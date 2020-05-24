use crate::error::{PolarsError, Result};
use crate::series::chunked_array::{ChunkOps, ChunkedArray};
use arrow::array::{ArrayRef, BooleanArray, PrimitiveArray};
use arrow::datatypes::ArrowNumericType;
use arrow::{compute, datatypes};
use std::sync::Arc;

impl<T> ChunkedArray<T>
where
    T: ArrowNumericType,
{
    fn comparison(
        &self,
        rhs: &ChunkedArray<T>,
        operator: impl Fn(&PrimitiveArray<T>, &PrimitiveArray<T>) -> arrow::error::Result<BooleanArray>,
    ) -> Result<ChunkedArray<datatypes::BooleanType>> {
        let opt = self.optional_rechunk(rhs)?;
        let left = match &opt {
            Some(a) => a,
            None => self,
        };

        let chunks_res = left
            .downcast_chunks()
            .iter()
            .zip(rhs.downcast_chunks())
            .map(|(left, right)| operator(left, right))
            .collect::<std::result::Result<Vec<_>, arrow::error::ArrowError>>();

        let chunks_res = chunks_res.map(|chunks| {
            chunks
                .into_iter()
                .map(|arr| Arc::new(arr) as ArrayRef)
                .collect()
        });

        match chunks_res {
            Ok(chunks) => Ok(ChunkedArray::new_from_chunks("", chunks)),
            Err(e) => Err(PolarsError::ArrowError(e)),
        }
    }

    fn eq(&self, rhs: &ChunkedArray<T>) -> Result<ChunkedArray<datatypes::BooleanType>> {
        self.comparison(rhs, compute::eq)
    }

    fn neq(&self, rhs: &ChunkedArray<T>) -> Result<ChunkedArray<datatypes::BooleanType>> {
        self.comparison(rhs, compute::neq)
    }

    fn gt(&self, rhs: &ChunkedArray<T>) -> Result<ChunkedArray<datatypes::BooleanType>> {
        self.comparison(rhs, compute::gt)
    }

    fn gt_eq(&self, rhs: &ChunkedArray<T>) -> Result<ChunkedArray<datatypes::BooleanType>> {
        self.comparison(rhs, compute::gt_eq)
    }

    fn lt(&self, rhs: &ChunkedArray<T>) -> Result<ChunkedArray<datatypes::BooleanType>> {
        self.comparison(rhs, compute::lt)
    }

    fn lt_eq(&self, rhs: &ChunkedArray<T>) -> Result<ChunkedArray<datatypes::BooleanType>> {
        self.comparison(rhs, compute::lt_eq)
    }
}
