//! Implementations of the ChunkAgg trait.
use crate::chunked_array::ChunkedArray;
use crate::datatypes::BooleanChunked;
use crate::{datatypes::PolarsNumericType, prelude::*};
use arrow::compute;
use num::{Num, NumCast, ToPrimitive};
use std::cmp::PartialOrd;
use std::ops::{Add, Div};

impl<T> ChunkAgg<T::Native> for ChunkedArray<T>
where
    T: PolarsNumericType,
    T::Native: Add<Output = T::Native> + PartialOrd + Div<Output = T::Native> + Num + NumCast,
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

    fn mean(&self) -> Option<T::Native> {
        let len = (self.len() - self.null_count()) as f64;
        self.sum()
            .map(|v| NumCast::from(v.to_f64().unwrap() / len).unwrap())
    }

    fn median(&self) -> Option<T::Native> {
        let null_count = self.null_count();
        self.sort(false)
            .slice((self.len() - null_count) / 2 + null_count, 1)
            .unwrap()
            .into_iter()
            .next()
            .unwrap()
    }
}

fn min_max_helper(ca: &BooleanChunked, min: bool) -> Option<u8> {
    let min_max = ca.into_iter().fold(0, |acc: u8, x| match x {
        Some(v) => {
            let v = v as u8;
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
impl ChunkAgg<u8> for BooleanChunked {
    /// Returns `None` if the array is empty or only contains null values.
    fn sum(&self) -> Option<u8> {
        if self.len() == 0 {
            return None;
        }
        let sum = self.into_iter().fold(0, |acc: u8, x| match x {
            Some(v) => acc + v as u8,
            None => acc,
        });
        Some(sum)
    }

    fn min(&self) -> Option<u8> {
        if self.len() == 0 {
            return None;
        }
        min_max_helper(self, true)
    }

    fn max(&self) -> Option<u8> {
        if self.len() == 0 {
            return None;
        }
        min_max_helper(self, false)
    }

    fn mean(&self) -> Option<u8> {
        let len = self.len() - self.null_count();
        self.sum().map(|v| (v as usize / len) as u8)
    }

    fn median(&self) -> Option<u8> {
        let null_count = self.null_count();
        let opt_v = self
            .sort(false)
            .slice((self.len() - null_count) / 2 + null_count, 1)
            .unwrap()
            .into_iter()
            .next()
            .unwrap();
        opt_v.map(|v| v as u8)
    }
}

#[cfg(test)]
mod test {
    use crate::prelude::*;

    #[test]
    fn test_median() {
        let ca = UInt32Chunked::new_from_opt_slice(
            "a",
            &[Some(2), Some(1), None, Some(3), Some(5), None, Some(4)],
        );
        assert_eq!(ca.median(), Some(3));
        let ca = UInt32Chunked::new_from_opt_slice(
            "a",
            &[
                None,
                Some(7),
                Some(6),
                Some(2),
                Some(1),
                None,
                Some(3),
                Some(5),
                None,
                Some(4),
            ],
        );
        assert_eq!(ca.median(), Some(4));
    }
}
