use crate::datatypes::BooleanChunked;
use crate::series::chunked_array::{iterator::ChunkIterator, ChunkedArray, SeriesOps};
use crate::{datatypes::PolarNumericType, prelude::*};
use arrow::compute;

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
    /// Returns `None` if the array is empty or only contains null values.
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

    /// Returns the minimum value in the array, according to the natural order.
    /// Returns an option because the array is nullable.
    fn min(&self) -> Option<T::Native> {
        self.downcast_chunks()
            .iter()
            .filter_map(|&a| compute::min(a))
            .fold_first(|acc, v| if acc < v { acc } else { v })
    }

    /// Returns the maximum value in the array, according to the natural order.
    /// Returns an option because the array is nullable.
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

/// Booleans are casted to 1 or 0.
impl Agg<u64> for BooleanChunked {
    /// Returns `None` if the array is empty or only contains null values.
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
