//! Implementations of the ChunkAgg trait.
use crate::chunked_array::builder::get_list_builder;
use crate::chunked_array::ChunkedArray;
use crate::datatypes::BooleanChunked;
use crate::{datatypes::PolarsNumericType, prelude::*, utils::CustomIterTools};
use arrow::compute;
use num::{Num, NumCast, ToPrimitive, Zero};
use std::cmp::PartialOrd;

/// Aggregations that return Series of unit length. Those can be used in broadcasting operations.
pub trait ChunkAggSeries {
    /// Get the sum of the ChunkedArray as a new Series of length 1.
    fn sum_as_series(&self) -> Series {
        unimplemented!()
    }
    /// Get the max of the ChunkedArray as a new Series of length 1.
    fn max_as_series(&self) -> Series {
        unimplemented!()
    }
    /// Get the min of the ChunkedArray as a new Series of length 1.
    fn min_as_series(&self) -> Series {
        unimplemented!()
    }
    /// Get the mean of the ChunkedArray as a new Series of length 1.
    fn mean_as_series(&self) -> Series {
        unimplemented!()
    }
    /// Get the median of the ChunkedArray as a new Series of length 1.
    fn median_as_series(&self) -> Series {
        unimplemented!()
    }
    /// Get the quantile of the ChunkedArray as a new Series of length 1.
    fn quantile_as_series(&self, _quantile: f64) -> Result<Series> {
        unimplemented!()
    }
}

pub trait VarAggSeries {
    /// Get the variance of the ChunkedArray as a new Series of length 1.
    fn var_as_series(&self) -> Series;
    /// Get the standard deviation of the ChunkedArray as a new Series of length 1.
    fn std_as_series(&self) -> Series;
}

macro_rules! agg_float_with_nans {
    ($self:ident, $agg_method:ident, $precision:ty) => {{
        if $self.null_count() == 0 {
            $self
                .into_no_null_iter()
                .map(|a| -> $precision { NumCast::from(a).unwrap() })
                .fold_first_(|a, b| a.$agg_method(b))
                .map(|a| NumCast::from(a).unwrap())
        } else {
            $self
                .into_iter()
                .filter(|opt| opt.is_some())
                .map(|opt| opt.unwrap())
                .map(|a| -> $precision { NumCast::from(a).unwrap() })
                .fold_first_(|a, b| a.$agg_method(b))
                .map(|a| NumCast::from(a).unwrap())
        }
    }};
}

macro_rules! impl_quantile {
    ($self:expr, $quantile:expr) => {{
        let null_count = $self.null_count();
        let opt = ChunkSort::sort($self, false)
            .slice(
                ((($self.len() - null_count) as f64) * $quantile + null_count as f64) as i64,
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
    T::Native: PartialOrd + Num + NumCast + Zero,
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
        match T::get_dtype() {
            DataType::Float32 => agg_float_with_nans!(self, min, f32),
            DataType::Float64 => agg_float_with_nans!(self, min, f64),
            _ => self
                .downcast_chunks()
                .iter()
                .filter_map(|&a| compute::min(a))
                .fold_first_(|acc, v| if acc < v { acc } else { v }),
        }
    }

    fn max(&self) -> Option<T::Native> {
        match T::get_dtype() {
            DataType::Float32 => agg_float_with_nans!(self, max, f32),
            DataType::Float64 => agg_float_with_nans!(self, max, f64),
            _ => self
                .downcast_chunks()
                .iter()
                .filter_map(|&a| compute::max(a))
                .fold_first_(|acc, v| if acc > v { acc } else { v }),
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
        if !(0.0..=1.0).contains(&quantile) {
            Err(PolarsError::ValueError(
                "quantile should be between 0.0 and 1.0".into(),
            ))
        } else {
            let opt = impl_quantile!(self, quantile);
            Ok(opt)
        }
    }
}

macro_rules! impl_var {
    ($self:expr, $ty: ty) => {{
        let mean = $self.mean()?;
        let ca = $self - mean;
        let squared = &ca * &ca;
        let opt_v = squared.sum();
        let div = ($self.len() - 1) as $ty;
        opt_v.map(|v| v / div)
    }};
}

impl<T> ChunkVar<f64> for ChunkedArray<T>
where
    T: PolarsIntegerType,
    T::Native: PartialOrd + Num + NumCast,
{
    fn var(&self) -> Option<f64> {
        let ca = self.cast::<Float64Type>().ok()?;
        impl_var!(&ca, f64)
    }
    fn std(&self) -> Option<f64> {
        self.var().map(|var| var.sqrt())
    }
}

impl ChunkVar<f32> for Float32Chunked {
    fn var(&self) -> Option<f32> {
        impl_var!(self, f32)
    }
    fn std(&self) -> Option<f32> {
        self.var().map(|var| var.sqrt())
    }
}

impl ChunkVar<f64> for Float64Chunked {
    fn var(&self) -> Option<f64> {
        impl_var!(self, f64)
    }
    fn std(&self) -> Option<f64> {
        self.var().map(|var| var.sqrt())
    }
}

impl ChunkVar<String> for Utf8Chunked {}
impl ChunkVar<Series> for ListChunked {}
impl ChunkVar<u32> for CategoricalChunked {}
#[cfg(feature = "object")]
impl<T> ChunkVar<Series> for ObjectChunked<T> {}
impl ChunkVar<bool> for BooleanChunked {}

fn min_max_helper(ca: &BooleanChunked, min: bool) -> u32 {
    ca.into_iter().fold(0, |acc: u32, x| match x {
        Some(v) => {
            let v = v as u32;
            if min {
                if acc < v {
                    acc
                } else {
                    v
                }
            } else if acc > v {
                acc
            } else {
                v
            }
        }
        None => acc,
    })
}

/// Booleans are casted to 1 or 0.
impl ChunkAgg<u32> for BooleanChunked {
    /// Returns `None` if the array is empty or only contains null values.
    fn sum(&self) -> Option<u32> {
        if self.is_empty() {
            return None;
        }
        let sum = self.into_iter().fold(0, |acc: u32, x| match x {
            Some(v) => acc + v as u32,
            None => acc,
        });
        Some(sum)
    }

    fn min(&self) -> Option<u32> {
        if self.is_empty() {
            return None;
        }
        Some(min_max_helper(self, true))
    }

    fn max(&self) -> Option<u32> {
        if self.is_empty() {
            return None;
        }
        Some(min_max_helper(self, false))
    }

    fn mean(&self) -> Option<u32> {
        let len = self.len() - self.null_count();
        self.sum().map(|v| (v as usize / len) as u32)
    }

    fn median(&self) -> Option<u32> {
        self.quantile(0.5).unwrap()
    }

    fn quantile(&self, quantile: f64) -> Result<Option<u32>> {
        if !(0.0..=1.0).contains(&quantile) {
            Err(PolarsError::ValueError(
                "quantile should be between 0.0 and 1.0".into(),
            ))
        } else {
            let opt = impl_quantile!(self, quantile);
            Ok(opt.map(|v| v as u32))
        }
    }
}

// Needs the same trait bounds as the implementation of ChunkedArray<T> of dyn Series
impl<T> ChunkAggSeries for ChunkedArray<T>
where
    T: PolarsNumericType,
    T::Native: PartialOrd + Num + NumCast,
    ChunkedArray<T>: IntoSeries,
{
    fn sum_as_series(&self) -> Series {
        let v = self.sum();
        let mut ca: ChunkedArray<T> = [v].iter().copied().collect();
        ca.rename(self.name());
        ca.into_series()
    }
    fn max_as_series(&self) -> Series {
        let v = self.max();
        let mut ca: ChunkedArray<T> = [v].iter().copied().collect();
        ca.rename(self.name());
        ca.into_series()
    }
    fn min_as_series(&self) -> Series {
        let v = self.min();
        let mut ca: ChunkedArray<T> = [v].iter().copied().collect();
        ca.rename(self.name());
        ca.into_series()
    }
    fn mean_as_series(&self) -> Series {
        let s = self.sum_as_series();
        let mut out = s.cast::<Float64Type>().unwrap() / (self.len() - self.null_count()) as f64;
        out.rename(self.name());
        out
    }
    fn median_as_series(&self) -> Series {
        let v = self.median();
        let mut ca: ChunkedArray<T> = [v].iter().copied().collect();
        ca.rename(self.name());
        ca.into_series()
    }
    fn quantile_as_series(&self, quantile: f64) -> Result<Series> {
        let v = self.quantile(quantile)?;
        let mut ca: ChunkedArray<T> = [v].iter().copied().collect();
        ca.rename(self.name());
        Ok(ca.into_series())
    }
}

macro_rules! impl_as_series {
    ($self:expr, $agg:ident, $ty: ty) => {{
        let v = $self.$agg();
        let mut ca: $ty = [v].iter().copied().collect();
        ca.rename($self.name());
        ca.into_series()
    }};
}

impl<T> VarAggSeries for ChunkedArray<T>
where
    T: PolarsIntegerType,
    T::Native: PartialOrd + Num + NumCast,
{
    fn var_as_series(&self) -> Series {
        impl_as_series!(self, var, Float64Chunked)
    }

    fn std_as_series(&self) -> Series {
        impl_as_series!(self, std, Float64Chunked)
    }
}

impl VarAggSeries for Float32Chunked {
    fn var_as_series(&self) -> Series {
        impl_as_series!(self, var, Float32Chunked)
    }

    fn std_as_series(&self) -> Series {
        impl_as_series!(self, std, Float32Chunked)
    }
}

impl VarAggSeries for Float64Chunked {
    fn var_as_series(&self) -> Series {
        impl_as_series!(self, var, Float64Chunked)
    }

    fn std_as_series(&self) -> Series {
        impl_as_series!(self, std, Float64Chunked)
    }
}

impl VarAggSeries for BooleanChunked {
    fn var_as_series(&self) -> Series {
        Self::full_null(self.name(), 1).into_series()
    }

    fn std_as_series(&self) -> Series {
        Self::full_null(self.name(), 1).into_series()
    }
}
impl VarAggSeries for CategoricalChunked {
    fn var_as_series(&self) -> Series {
        self.cast::<UInt32Type>().unwrap().var_as_series()
    }

    fn std_as_series(&self) -> Series {
        self.cast::<UInt32Type>().unwrap().std_as_series()
    }
}
impl VarAggSeries for ListChunked {
    fn var_as_series(&self) -> Series {
        Self::full_null(self.name(), 1).into_series()
    }

    fn std_as_series(&self) -> Series {
        Self::full_null(self.name(), 1).into_series()
    }
}
#[cfg(feature = "object")]
impl<T> VarAggSeries for ObjectChunked<T> {
    fn var_as_series(&self) -> Series {
        unimplemented!()
    }

    fn std_as_series(&self) -> Series {
        unimplemented!()
    }
}
impl VarAggSeries for Utf8Chunked {
    fn var_as_series(&self) -> Series {
        Self::full_null(self.name(), 1).into_series()
    }

    fn std_as_series(&self) -> Series {
        Self::full_null(self.name(), 1).into_series()
    }
}

impl ChunkAggSeries for BooleanChunked {
    fn sum_as_series(&self) -> Series {
        let v = ChunkAgg::sum(self);
        let mut ca: UInt32Chunked = [v].iter().copied().collect();
        ca.rename(self.name());
        ca.into_series()
    }
    fn max_as_series(&self) -> Series {
        let v = ChunkAgg::max(self);
        let mut ca: UInt32Chunked = [v].iter().copied().collect();
        ca.rename(self.name());
        ca.into_series()
    }
    fn min_as_series(&self) -> Series {
        let v = ChunkAgg::min(self);
        let mut ca: UInt32Chunked = [v].iter().copied().collect();
        ca.rename(self.name());
        ca.into_series()
    }
    fn mean_as_series(&self) -> Series {
        let v = ChunkAgg::mean(self);
        let mut ca: UInt32Chunked = [v].iter().copied().collect();
        ca.rename(self.name());
        ca.into_series()
    }
    fn median_as_series(&self) -> Series {
        let v = ChunkAgg::median(self);
        let mut ca: UInt32Chunked = [v].iter().copied().collect();
        ca.rename(self.name());
        ca.into_series()
    }
    fn quantile_as_series(&self, quantile: f64) -> Result<Series> {
        let v = ChunkAgg::quantile(self, quantile)?;
        let mut ca: UInt32Chunked = [v].iter().copied().collect();
        ca.rename(self.name());
        Ok(ca.into_series())
    }
}

macro_rules! one_null_utf8 {
    ($self:ident) => {{
        let mut builder = Utf8ChunkedBuilder::new($self.name(), 1, 0);
        builder.append_null();
        builder.finish().into_series()
    }};
}

impl ChunkAggSeries for Utf8Chunked {
    fn sum_as_series(&self) -> Series {
        one_null_utf8!(self)
    }
    fn max_as_series(&self) -> Series {
        one_null_utf8!(self)
    }
    fn min_as_series(&self) -> Series {
        one_null_utf8!(self)
    }
    fn mean_as_series(&self) -> Series {
        one_null_utf8!(self)
    }
    fn median_as_series(&self) -> Series {
        one_null_utf8!(self)
    }
    fn quantile_as_series(&self, _quantile: f64) -> Result<Series> {
        Ok(one_null_utf8!(self))
    }
}

impl ChunkAggSeries for CategoricalChunked {}

macro_rules! one_null_list {
    ($self:ident) => {{
        let mut builder = get_list_builder(&DataType::Null, 0, 1, $self.name());
        builder.append_opt_series(None);
        builder.finish().into_series()
    }};
}

impl ChunkAggSeries for ListChunked {
    fn sum_as_series(&self) -> Series {
        one_null_list!(self)
    }
    fn max_as_series(&self) -> Series {
        one_null_list!(self)
    }
    fn min_as_series(&self) -> Series {
        one_null_list!(self)
    }
    fn mean_as_series(&self) -> Series {
        one_null_list!(self)
    }
    fn median_as_series(&self) -> Series {
        one_null_list!(self)
    }
    fn quantile_as_series(&self, _quantile: f64) -> Result<Series> {
        Ok(one_null_list!(self))
    }
}

#[cfg(feature = "object")]
impl<T> ChunkAggSeries for ObjectChunked<T> {}

impl<T> ArgAgg for ChunkedArray<T>
where
    T: PolarsNumericType,
{
    fn arg_min(&self) -> Option<usize> {
        self.into_iter()
            .enumerate()
            .reduce(|acc, (idx, val)| if acc.1 > val { (idx, val) } else { acc })
            .map(|tpl| tpl.0)
    }
    fn arg_max(&self) -> Option<usize> {
        self.into_iter()
            .enumerate()
            .reduce(|acc, (idx, val)| if acc.1 < val { (idx, val) } else { acc })
            .map(|tpl| tpl.0)
    }
}

impl ArgAgg for BooleanChunked {}
impl ArgAgg for CategoricalChunked {}
impl ArgAgg for Utf8Chunked {}
impl ArgAgg for ListChunked {}

#[cfg(feature = "object")]
impl<T> ArgAgg for ObjectChunked<T> {}

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
