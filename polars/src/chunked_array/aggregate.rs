//! Implementations of the ChunkAgg trait.
use crate::chunked_array::ChunkedArray;
use crate::datatypes::BooleanChunked;
use crate::{datatypes::PolarsNumericType, prelude::*};
use arrow::compute;
use num::{Num, NumCast, ToPrimitive};
use std::cmp::{Ordering, PartialOrd};
use std::ops::{Add, Div};

macro_rules! cmp_float_with_nans {
    ($a:expr, $b:expr, $precision:ty) => {{
        let a: $precision = NumCast::from($a).unwrap();
        let b: $precision = NumCast::from($b).unwrap();
        match (a.is_nan(), b.is_nan()) {
            (true, true) => Ordering::Equal,
            (true, false) => Ordering::Greater,
            (false, true) => Ordering::Less,
            (false, false) => a.partial_cmp(&b).unwrap(),
        }
    }};
}

macro_rules! agg_float_with_nans {
    ($self:ident, $agg_method:ident, $precision:ty) => {{
        if $self.null_count() == 0 {
            $self
                .into_no_null_iter()
                .$agg_method(|&a, &b| cmp_float_with_nans!(a, b, $precision))
        } else {
            $self
                .into_iter()
                .filter(|opt| opt.is_some())
                .map(|opt| opt.unwrap())
                .$agg_method(|&a, &b| cmp_float_with_nans!(a, b, $precision))
        }
    }};
}

macro_rules! impl_quantile {
    ($self:expr, $quantile:expr) => {{
        let null_count = $self.null_count();
        let opt = $self
            .sort(false)
            .slice(
                ((($self.len() - null_count) as f64) * $quantile + null_count as f64) as usize,
                1,
            )
            .unwrap()
            .into_iter()
            .next()
            .unwrap();
        opt
    }};
}

impl<T> ChunkAgg<T::Native> for ChunkedArray<T>
where
    T: PolarsNumericType,
    T::Native: Add<Output = T::Native>
        + PartialOrd
        + Div<Output = T::Native>
        + Num
        + NumCast
        + ToPrimitive,
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
        match T::get_data_type() {
            ArrowDataType::Float32 => agg_float_with_nans!(self, min_by, f32),
            ArrowDataType::Float64 => agg_float_with_nans!(self, min_by, f64),
            _ => self
                .downcast_chunks()
                .iter()
                .filter_map(|&a| compute::min(a))
                .fold_first(|acc, v| if acc > v { acc } else { v }),
        }
    }

    fn max(&self) -> Option<T::Native> {
        match T::get_data_type() {
            ArrowDataType::Float32 => agg_float_with_nans!(self, max_by, f32),
            ArrowDataType::Float64 => agg_float_with_nans!(self, max_by, f64),
            _ => self
                .downcast_chunks()
                .iter()
                .filter_map(|&a| compute::max(a))
                .fold_first(|acc, v| if acc > v { acc } else { v }),
        }
    }

    fn mean(&self) -> Option<T::Native> {
        let len = (self.len() - self.null_count()) as f64;
        self.sum()
            .map(|v| NumCast::from(v.to_f64().unwrap() / len).unwrap())
    }

    fn median(&self) -> Option<T::Native> {
        self.quantile(0.5).unwrap()
    }

    fn quantile(&self, quantile: f64) -> Result<Option<T::Native>> {
        if quantile < 0.0 || quantile > 1.0 {
            Err(PolarsError::ValueError(
                "quantile should be between 0.0 and 1.0".into(),
            ))
        } else {
            let opt = impl_quantile!(self, quantile);
            Ok(opt)
        }
    }
}

fn min_max_helper(ca: &BooleanChunked, min: bool) -> Option<u32> {
    let min_max = ca.into_iter().fold(0, |acc: u32, x| match x {
        Some(v) => {
            let v = v as u32;
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
impl ChunkAgg<u32> for BooleanChunked {
    /// Returns `None` if the array is empty or only contains null values.
    fn sum(&self) -> Option<u32> {
        if self.len() == 0 {
            return None;
        }
        let sum = self.into_iter().fold(0, |acc: u32, x| match x {
            Some(v) => acc + v as u32,
            None => acc,
        });
        Some(sum)
    }

    fn min(&self) -> Option<u32> {
        if self.len() == 0 {
            return None;
        }
        min_max_helper(self, true)
    }

    fn max(&self) -> Option<u32> {
        if self.len() == 0 {
            return None;
        }
        min_max_helper(self, false)
    }

    fn mean(&self) -> Option<u32> {
        let len = self.len() - self.null_count();
        self.sum().map(|v| (v as usize / len) as u32)
    }

    fn median(&self) -> Option<u32> {
        self.quantile(0.5).unwrap()
    }

    fn quantile(&self, quantile: f64) -> Result<Option<u32>> {
        if quantile < 0.0 || quantile > 1.0 {
            Err(PolarsError::ValueError(
                "quantile should be between 0.0 and 1.0".into(),
            ))
        } else {
            let opt = impl_quantile!(self, quantile);
            Ok(opt.map(|v| v as u32))
        }
    }
}

#[cfg(test)]
mod test {
    use crate::prelude::*;

    #[test]
    fn test_agg_float() {
        let ca1 = Float32Chunked::new_from_slice("a", &[1.0, f32::NAN]);
        let ca2 = Float32Chunked::new_from_slice("b", &[f32::NAN, 1.0]);
        assert_eq!(ca1.min(), ca2.min());
        let ca1 = Float64Chunked::new_from_slice("a", &[1.0, f64::NAN]);
        let ca2 = Float64Chunked::new_from_slice("b", &[f64::NAN, 1.0]);
        assert_eq!(ca1.min(), ca2.min());
        println!("{:?}", (ca1.min(), ca2.min()))
    }

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
