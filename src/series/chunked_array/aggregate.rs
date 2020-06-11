use crate::datatypes::PolarNumericType;
use crate::series::chunked_array::{iterator::ChunkIterator, ChunkedArray, SeriesOps};
use crate::{datatypes, datatypes::BooleanChunked};
use arrow::compute;
use arrow::datatypes::ArrowNumericType;
use num::traits::Zero;

pub trait Agg<T> {
    fn sum(&self) -> Option<T>;
    fn min(&self) -> Option<T>;
    fn max(&self) -> Option<T>;
}

impl<T> Agg<T::Native> for ChunkedArray<T>
where
    T: PolarNumericType,
    T::Native: std::ops::Add<Output = T::Native> + std::cmp::PartialOrd,
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

    fn min(&self) -> Option<T::Native> {
        self.downcast_chunks()
            .iter()
            .filter_map(|&a| compute::min(a))
            .fold_first(|acc, v| if acc < v { acc } else { v })
    }

    fn max(&self) -> Option<T::Native> {
        self.downcast_chunks()
            .iter()
            .filter_map(|&a| compute::max(a))
            .fold_first(|acc, v| if acc > v { acc } else { v })
    }
}

fn min_max_helper(ca: &BooleanChunked, min: bool) -> Option<u64> {
    let min_max = ca.iter().fold(0, |acc: u64, x| match x {
        Some(v) => {
            let v = v as u64;
            if min {
                if acc < v {
                    acc
                } else {
                    v
                }
            } else {
                if acc > v {
                    acc
                } else {
                    v
                }
            }
        }
        None => acc,
    });
    Some(min_max)
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

    fn min(&self) -> Option<u64> {
        if self.len() == 0 {
            return None;
        }
        min_max_helper(self, true)
    }

    fn max(&self) -> Option<u64> {
        if self.len() == 0 {
            return None;
        }
        min_max_helper(self, false)
    }
}
