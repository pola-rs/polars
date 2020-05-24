use crate::series::iterator::ChunkIterator;
use crate::{
    datatypes,
    error::{PolarsError, Result},
    series::chunked_array::{ChunkOps, ChunkedArray},
};
use arrow::array::{ArrayRef, BooleanArray, PrimitiveArray, StringArray};
use arrow::compute;
use arrow::datatypes::ArrowNumericType;
use std::sync::Arc;

pub trait CmpOps<T> {
    fn eq(&self, rhs: &ChunkedArray<T>) -> Result<ChunkedArray<datatypes::BooleanType>>;

    fn neq(&self, rhs: &ChunkedArray<T>) -> Result<ChunkedArray<datatypes::BooleanType>>;

    fn gt(&self, rhs: &ChunkedArray<T>) -> Result<ChunkedArray<datatypes::BooleanType>>;

    fn gt_eq(&self, rhs: &ChunkedArray<T>) -> Result<ChunkedArray<datatypes::BooleanType>>;

    fn lt(&self, rhs: &ChunkedArray<T>) -> Result<ChunkedArray<datatypes::BooleanType>>;

    fn lt_eq(&self, rhs: &ChunkedArray<T>) -> Result<ChunkedArray<datatypes::BooleanType>>;
}

impl<T> ChunkedArray<T>
where
    T: ArrowNumericType,
{
    /// First ensure that the chunks of lhs and rhs match and then iterates over the chunks and applies
    /// the comparison operator.
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

        // TODO: Fix unnecessary second iter over chunk res
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
}

impl<T> CmpOps<T> for ChunkedArray<T>
where
    T: ArrowNumericType,
{
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

/// Auxiliary trait for CmpOps trait
pub trait BoundedToUtf8 {}
impl BoundedToUtf8 for datatypes::Utf8Type {}

impl ChunkedArray<datatypes::Utf8Type> {
    fn comparison<T: BoundedToUtf8>(
        &self,
        rhs: &ChunkedArray<T>,
        operator: impl Fn(&StringArray, &StringArray) -> arrow::error::Result<BooleanArray>,
    ) -> Result<ChunkedArray<datatypes::BooleanType>> {
        let opt = self.optional_rechunk(rhs)?;
        let left = match &opt {
            Some(a) => a,
            None => self,
        };

        let chunks = left
            .chunks
            .iter()
            .zip(&rhs.chunks)
            .map(|(left, right)| {
                let left = left
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .expect("could not downcast one of the chunks");
                let right = right
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .expect("could not downcast one of the chunks");
                let arr_res = operator(left, right);
                let arr = match arr_res {
                    Ok(arr) => arr,
                    Err(e) => return Err(PolarsError::ArrowError(e)),
                };
                Ok(Arc::new(arr) as ArrayRef)
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(ChunkedArray::new_from_chunks("", chunks))
    }
}

impl<T> CmpOps<T> for ChunkedArray<datatypes::Utf8Type>
where
    T: BoundedToUtf8,
{
    fn eq(&self, rhs: &ChunkedArray<T>) -> Result<ChunkedArray<datatypes::BooleanType>> {
        self.comparison(rhs, compute::eq_utf8)
    }

    fn neq(&self, rhs: &ChunkedArray<T>) -> Result<ChunkedArray<datatypes::BooleanType>> {
        self.comparison(rhs, compute::neq_utf8)
    }

    fn gt(&self, rhs: &ChunkedArray<T>) -> Result<ChunkedArray<datatypes::BooleanType>> {
        self.comparison(rhs, compute::gt_utf8)
    }

    fn gt_eq(&self, rhs: &ChunkedArray<T>) -> Result<ChunkedArray<datatypes::BooleanType>> {
        self.comparison(rhs, compute::gt_eq_utf8)
    }

    fn lt(&self, rhs: &ChunkedArray<T>) -> Result<ChunkedArray<datatypes::BooleanType>> {
        self.comparison(rhs, compute::lt_utf8)
    }

    fn lt_eq(&self, rhs: &ChunkedArray<T>) -> Result<ChunkedArray<datatypes::BooleanType>> {
        self.comparison(rhs, compute::lt_eq_utf8)
    }
}

mod test {
    use super::*;

    #[test]
    fn utf8_cmp() {
        let a = ChunkedArray::<datatypes::Utf8Type>::new_from_slice("a", &["hello", "world"]);
        let b = ChunkedArray::<datatypes::Utf8Type>::new_from_slice("a", &["hello", "world"]);
        let sum_true = a.eq(&b).unwrap().iter().fold(0, |acc, opt| match opt {
            Some(b) => acc + b as i32,
            None => acc,
        });

        assert_eq!(2, sum_true)
    }
}
