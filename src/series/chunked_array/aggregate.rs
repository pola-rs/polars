use crate::datatypes::PolarNumericType;
use crate::series::chunked_array::{iterator::ChunkIterator, ChunkedArray, SeriesOps};
use crate::{datatypes, datatypes::BooleanChunked};
use arrow::compute;
use arrow::datatypes::ArrowNumericType;
use num::traits::Zero;

pub trait Agg<T> {
    fn sum(&self) -> Option<T>;
}

impl<T> Agg<T::Native> for ChunkedArray<T>
where
    T: PolarNumericType,
    T::Native: std::ops::Add<Output = T::Native>,
{
    fn sum(&self) -> Option<T::Native> {
        self.downcast_chunks()
            .iter()
            .map(|&a| compute::sum(a))
            .fold(None, |acc, v| match v {
                Some(v) => match acc {
                    None => Some(v),
                    Some(acc) => Some(acc + v),
                },
                None => acc,
            })
    }
}

impl Agg<u64> for BooleanChunked {
    fn sum(&self) -> Option<u64> {
        if self.len() == 0 {
            return None;
        }
        let sum = self.iter().fold(0, |acc: u64, x| match x {
            Some(v) => acc + v as u64,
            None => acc,
        });
        Some(sum)
    }
}
