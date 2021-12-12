//! Implementations of the ChunkAgg trait.
use crate::chunked_array::builder::get_list_builder;
use crate::chunked_array::ChunkedArray;
use crate::datatypes::BooleanChunked;
use crate::{datatypes::PolarsNumericType, prelude::*, utils::CustomIterTools};
use arrow::compute;
use arrow::types::simd::Simd;
use num::{NumCast, ToPrimitive};
use std::ops::Add;

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum QuantileInterpolOptions {
    Nearest,
    Lower,
    Higher,
    Midpoint,
    Linear,
}

impl Default for QuantileInterpolOptions {
    fn default() -> Self {
        QuantileInterpolOptions::Nearest
    }
}

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
    fn quantile_as_series(
        &self,
        _quantile: f64,
        _interpol: QuantileInterpolOptions,
    ) -> Result<Series> {
        unimplemented!()
    }
}

pub trait VarAggSeries {
    /// Get the variance of the ChunkedArray as a new Series of length 1.
    fn var_as_series(&self) -> Series;
    /// Get the standard deviation of the ChunkedArray as a new Series of length 1.
    fn std_as_series(&self) -> Series;
}

impl<T> ChunkAgg<T::Native> for ChunkedArray<T>
where
    T: PolarsNumericType,
    <T::Native as Simd>::Simd: Add<Output = <T::Native as Simd>::Simd>
        + compute::aggregate::Sum<T::Native>
        + compute::aggregate::SimdOrd<T::Native>,
{
    fn sum(&self) -> Option<T::Native> {
        self.downcast_iter()
            .map(compute::aggregate::sum_primitive)
            .fold(None, |acc, v| match v {
                Some(v) => match acc {
                    None => Some(v),
                    Some(acc) => Some(acc + v),
                },
                None => acc,
            })
    }

    fn min(&self) -> Option<T::Native> {
        self.downcast_iter()
            .filter_map(compute::aggregate::min_primitive)
            .fold_first_(|acc, v| if acc < v { acc } else { v })
    }

    fn max(&self) -> Option<T::Native> {
        self.downcast_iter()
            .filter_map(compute::aggregate::max_primitive)
            .fold_first_(|acc, v| if acc > v { acc } else { v })
    }

    fn mean(&self) -> Option<f64> {
        let len = (self.len() - self.null_count()) as f64;
        self.sum().map(|v| v.to_f64().unwrap() / len)
    }

    fn median(&self) -> Option<f64> {
        let null_count = self.null_count();
        let value_len = self.len() - null_count;
        match value_len {
            0 => None,
            _ => {
                let sorted = ChunkSort::sort(self, false);

                // After sorting the nulls are at the start of the array.
                let mid = value_len / 2 + null_count;
                if value_len % 2 == 0 {
                    NumCast::from(sorted.get(mid - 1).unwrap() + sorted.get(mid).unwrap())
                        .map(|v: f64| v / 2.0)
                } else {
                    sorted.get(mid).map(|v| NumCast::from(v).unwrap())
                }
            }
        }
    }

    fn quantile(
        &self,
        quantile: f64,
        interpol: QuantileInterpolOptions,
    ) -> Result<Option<T::Native>> {
        if !(0.0..=1.0).contains(&quantile) {
            return Err(PolarsError::ValueError(
                "quantile should be between 0.0 and 1.0".into(),
            ));
        }

        let null_count = self.null_count();
        let length = self.len();

        if null_count == length {
            return Ok(None);
        }

        let mut idx = match interpol {
            QuantileInterpolOptions::Nearest => {
                (((length - null_count) as f64) * quantile + null_count as f64) as i64
            }
            QuantileInterpolOptions::Lower
            | QuantileInterpolOptions::Midpoint
            | QuantileInterpolOptions::Linear => {
                (((length - null_count) as f64 - 1.0) * quantile + null_count as f64) as i64
            }
            QuantileInterpolOptions::Higher => {
                (((length - null_count) as f64 - 1.0) * quantile + null_count as f64).ceil() as i64
            }
        };

        if idx >= length as i64 {
            idx = (length - 1) as i64;
        } else if idx <= 0i64 {
            idx = 0;
        }

        let opt = match interpol {
            QuantileInterpolOptions::Midpoint => {
                let top_idx = (((length - null_count) as f64 - 1.0) * quantile + null_count as f64)
                    .ceil() as i64;
                if top_idx == idx {
                    ChunkSort::sort(self, false)
                        .slice(idx, 1)
                        .into_iter()
                        .next()
                        .flatten()
                } else {
                    let bounds: Vec<Option<T::Native>> = ChunkSort::sort(self, false)
                        .slice(idx, 2)
                        .into_iter()
                        .collect();

                    let length: T::Native = NumCast::from(bounds.len()).unwrap();

                    Some((bounds[0].unwrap() + bounds[1].unwrap()) / length)
                }
            }
            QuantileInterpolOptions::Linear => {
                let float_idx = ((length - null_count) as f64 - 1.0) * quantile + null_count as f64;
                let top_idx = f64::ceil(float_idx) as i64;

                if top_idx == idx {
                    ChunkSort::sort(self, false)
                        .slice(idx, 1)
                        .into_iter()
                        .next()
                        .flatten()
                } else {
                    let bounds: Vec<Option<T::Native>> = ChunkSort::sort(self, false)
                        .slice(idx, 2)
                        .into_iter()
                        .collect();

                    if bounds[0] == bounds[1] {
                        Some(bounds[0].unwrap())
                    } else {
                        let proportion: T::Native = NumCast::from(float_idx - idx as f64).unwrap();
                        Some(
                            proportion * (bounds[1].unwrap() - bounds[0].unwrap())
                                + bounds[0].unwrap(),
                        )
                    }
                }
            }
            _ => ChunkSort::sort(self, false)
                .slice(idx, 1)
                .into_iter()
                .next()
                .flatten(),
        };

        Ok(opt)
    }
}

impl<T> ChunkVar<f64> for ChunkedArray<T>
where
    T: PolarsIntegerType,
    <T::Native as Simd>::Simd: Add<Output = <T::Native as Simd>::Simd>
        + compute::aggregate::Sum<T::Native>
        + compute::aggregate::SimdOrd<T::Native>,
{
    fn var(&self) -> Option<f64> {
        let mean = self.mean()?;
        let squared = self.apply_cast_numeric::<_, Float64Type>(|value| {
            (value.to_f64().unwrap() - mean).powf(2.0)
        });
        // Note, this is similar behavior to numpy if DDOF=1.
        // in statistics DDOF often = 1.
        // this last step is similar to mean, only now instead of 1/n it is 1/(n-1)
        squared
            .sum()
            .map(|sum| sum / (self.len() - self.null_count() - 1) as f64)
    }
    fn std(&self) -> Option<f64> {
        self.var().map(|var| var.sqrt())
    }
}

impl ChunkVar<f32> for Float32Chunked {
    fn var(&self) -> Option<f32> {
        let mean = self.mean()? as f32;
        let squared = self.apply(|value| (value - mean).powf(2.0));
        squared
            .sum()
            .map(|sum| sum / (self.len() - self.null_count() - 1) as f32)
    }
    fn std(&self) -> Option<f32> {
        self.var().map(|var| var.sqrt())
    }
}

impl ChunkVar<f64> for Float64Chunked {
    fn var(&self) -> Option<f64> {
        let mean = self.mean()?;
        let squared = self.apply(|value| (value - mean).powf(2.0));
        squared
            .sum()
            .map(|sum| sum / (self.len() - self.null_count() - 1) as f64)
    }
    fn std(&self) -> Option<f64> {
        self.var().map(|var| var.sqrt())
    }
}

impl ChunkVar<String> for Utf8Chunked {}
impl ChunkVar<Series> for ListChunked {}
#[cfg(feature = "dtype-categorical")]
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
}

// Needs the same trait bounds as the implementation of ChunkedArray<T> of dyn Series
impl<T> ChunkAggSeries for ChunkedArray<T>
where
    T: PolarsNumericType,
    <T::Native as Simd>::Simd: Add<Output = <T::Native as Simd>::Simd>
        + compute::aggregate::Sum<T::Native>
        + compute::aggregate::SimdOrd<T::Native>,
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
        let val = [self.mean()];
        Series::new(self.name(), val)
    }
    fn median_as_series(&self) -> Series {
        let val = [self.median()];
        Series::new(self.name(), val)
    }
    fn quantile_as_series(
        &self,
        quantile: f64,
        interpol: QuantileInterpolOptions,
    ) -> Result<Series> {
        let v = self.quantile(quantile, interpol)?;
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
    <T::Native as Simd>::Simd: Add<Output = <T::Native as Simd>::Simd>
        + compute::aggregate::Sum<T::Native>
        + compute::aggregate::SimdOrd<T::Native>,
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
#[cfg(feature = "dtype-categorical")]
impl VarAggSeries for CategoricalChunked {
    fn var_as_series(&self) -> Series {
        unimplemented!()
    }

    fn std_as_series(&self) -> Series {
        unimplemented!()
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
        BooleanChunked::full_null(self.name(), 1).into_series()
    }
    fn median_as_series(&self) -> Series {
        BooleanChunked::full_null(self.name(), 1).into_series()
    }
    fn quantile_as_series(
        &self,
        _quantile: f64,
        _interpol: QuantileInterpolOptions,
    ) -> Result<Series> {
        Ok(BooleanChunked::full_null(self.name(), 1).into_series())
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
    fn quantile_as_series(
        &self,
        _quantile: f64,
        _interpol: QuantileInterpolOptions,
    ) -> Result<Series> {
        Ok(one_null_utf8!(self))
    }
}

#[cfg(feature = "dtype-categorical")]
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
    fn quantile_as_series(
        &self,
        _quantile: f64,
        _interpol: QuantileInterpolOptions,
    ) -> Result<Series> {
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
#[cfg(feature = "dtype-categorical")]
impl ArgAgg for CategoricalChunked {}
impl ArgAgg for Utf8Chunked {}
impl ArgAgg for ListChunked {}

#[cfg(feature = "object")]
impl<T> ArgAgg for ObjectChunked<T> {}

#[cfg(test)]
mod test {
    use crate::prelude::*;

    #[test]
    fn test_var() {
        // validated with numpy
        // Note that numpy as an argument ddof wich influences results. The default is ddof=0
        // we chose ddof=1, which is standard in statistics
        let ca1 = Int32Chunked::new("", &[5, 8, 9, 5, 0]);
        let ca2 = Int32Chunked::new(
            "",
            &[
                Some(5),
                None,
                Some(8),
                Some(9),
                None,
                Some(5),
                Some(0),
                None,
            ],
        );
        for ca in &[ca1, ca2] {
            let out = ca.var();
            assert_eq!(out, Some(12.3));
            let out = ca.std().unwrap();
            assert!((3.5071355833500366 - out).abs() < 0.000000001);
        }
    }

    #[test]
    fn test_agg_float() {
        let ca1 = Float32Chunked::new("a", &[1.0, f32::NAN]);
        let ca2 = Float32Chunked::new("b", &[f32::NAN, 1.0]);
        assert_eq!(ca1.min(), ca2.min());
        let ca1 = Float64Chunked::new("a", &[1.0, f64::NAN]);
        let ca2 = Float64Chunked::new_from_slice("b", &[f64::NAN, 1.0]);
        assert_eq!(ca1.min(), ca2.min());
        println!("{:?}", (ca1.min(), ca2.min()))
    }

    #[test]
    fn test_median() {
        let ca = UInt32Chunked::new(
            "a",
            &[Some(2), Some(1), None, Some(3), Some(5), None, Some(4)],
        );
        assert_eq!(ca.median(), Some(3.0));
        let ca = UInt32Chunked::new(
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
        assert_eq!(ca.median(), Some(4.0));

        let ca = Float32Chunked::new_from_slice(
            "",
            &[
                0.166189,
                0.166559,
                0.168517,
                0.169393,
                0.175272,
                0.23316699999999999,
                0.238787,
                0.266562,
                0.26903,
                0.285792,
                0.292801,
                0.29342899999999994,
                0.30170600000000003,
                0.308534,
                0.331489,
                0.346095,
                0.36764399999999997,
                0.36993899999999996,
                0.37207399999999996,
                0.41014000000000006,
                0.415789,
                0.421781,
                0.4277250000000001,
                0.46536299999999997,
                0.500208,
                2.6217269999999995,
                2.803311,
                3.868526,
            ],
        );
        assert!((ca.median().unwrap() - 0.3200115).abs() < 0.0001)
    }

    #[test]
    fn test_mean() {
        let ca = Float32Chunked::new("", &[Some(1.0), Some(2.0), None]);
        assert_eq!(ca.mean().unwrap(), 1.5);
        // all mean_as_series are cast to f64.
        assert_eq!(ca.mean_as_series().f64().unwrap().get(0).unwrap(), 1.5);
        // all null values case
        let ca = Float32Chunked::full_null("", 3);
        assert_eq!(ca.mean(), None);
        assert_eq!(ca.mean_as_series().f64().unwrap().get(0), None);
    }

    #[test]
    fn test_quantile_all_null() {
        let test_f32 = Float32Chunked::new_from_opt_slice("", &[None, None, None]);
        let test_i32 = Int32Chunked::new_from_opt_slice("", &[None, None, None]);
        let test_f64 = Float64Chunked::new_from_opt_slice("", &[None, None, None]);
        let test_i64 = Int64Chunked::new_from_opt_slice("", &[None, None, None]);

        let interpol_options = vec![
            QuantileInterpolOptions::Nearest,
            QuantileInterpolOptions::Lower,
            QuantileInterpolOptions::Higher,
            QuantileInterpolOptions::Midpoint,
            QuantileInterpolOptions::Linear,
        ];

        for interpol in interpol_options {
            assert_eq!(test_f32.quantile(0.9, interpol).unwrap(), None);
            assert_eq!(test_i32.quantile(0.9, interpol).unwrap(), None);
            assert_eq!(test_f64.quantile(0.9, interpol).unwrap(), None);
            assert_eq!(test_i64.quantile(0.9, interpol).unwrap(), None);
        }
    }

    #[test]
    fn test_quantile_single_value() {
        let test_f32 = Float32Chunked::new_from_opt_slice("", &[Some(1.0)]);
        let test_i32 = Int32Chunked::new_from_opt_slice("", &[Some(1)]);
        let test_f64 = Float64Chunked::new_from_opt_slice("", &[Some(1.0)]);
        let test_i64 = Int64Chunked::new_from_opt_slice("", &[Some(1)]);

        let interpol_options = vec![
            QuantileInterpolOptions::Nearest,
            QuantileInterpolOptions::Lower,
            QuantileInterpolOptions::Higher,
            QuantileInterpolOptions::Midpoint,
            QuantileInterpolOptions::Linear,
        ];

        for interpol in interpol_options {
            assert_eq!(test_f32.quantile(0.5, interpol).unwrap(), Some(1.0));
            assert_eq!(test_i32.quantile(0.5, interpol).unwrap(), Some(1));
            assert_eq!(test_f64.quantile(0.5, interpol).unwrap(), Some(1.0));
            assert_eq!(test_i64.quantile(0.5, interpol).unwrap(), Some(1));
        }
    }

    #[test]
    fn test_quantile_min_max() {
        let test_f32 =
            Float32Chunked::new_from_opt_slice("", &[None, Some(1f32), Some(5f32), Some(1f32)]);
        let test_i32 =
            Int32Chunked::new_from_opt_slice("", &[None, Some(1i32), Some(5i32), Some(1i32)]);
        let test_f64 =
            Float64Chunked::new_from_opt_slice("", &[None, Some(1f64), Some(5f64), Some(1f64)]);
        let test_i64 =
            Int64Chunked::new_from_opt_slice("", &[None, Some(1i64), Some(5i64), Some(1i64)]);

        let interpol_options = vec![
            QuantileInterpolOptions::Nearest,
            QuantileInterpolOptions::Lower,
            QuantileInterpolOptions::Higher,
            QuantileInterpolOptions::Midpoint,
            QuantileInterpolOptions::Linear,
        ];

        for interpol in interpol_options {
            assert_eq!(test_f32.quantile(0.0, interpol).unwrap(), test_f32.min());
            assert_eq!(test_f32.quantile(1.0, interpol).unwrap(), test_f32.max());

            assert_eq!(test_i32.quantile(0.0, interpol).unwrap(), test_i32.min());
            assert_eq!(test_i32.quantile(1.0, interpol).unwrap(), test_i32.max());

            assert_eq!(test_f64.quantile(0.0, interpol).unwrap(), test_f64.min());
            assert_eq!(test_f64.quantile(1.0, interpol).unwrap(), test_f64.max());
            assert_eq!(test_f64.quantile(0.5, interpol).unwrap(), test_f64.median());

            assert_eq!(test_i64.quantile(0.0, interpol).unwrap(), test_i64.min());
            assert_eq!(test_i64.quantile(1.0, interpol).unwrap(), test_i64.max());
        }
    }

    #[test]
    fn test_quantile() {
        let ca = UInt32Chunked::new(
            "a",
            &[Some(2), Some(1), None, Some(3), Some(5), None, Some(4)],
        );

        assert_eq!(
            ca.quantile(0.1, QuantileInterpolOptions::Nearest).unwrap(),
            Some(1)
        );
        assert_eq!(
            ca.quantile(0.9, QuantileInterpolOptions::Nearest).unwrap(),
            Some(5)
        );
        assert_eq!(
            ca.quantile(0.6, QuantileInterpolOptions::Nearest).unwrap(),
            Some(4)
        );

        assert_eq!(
            ca.quantile(0.1, QuantileInterpolOptions::Lower).unwrap(),
            Some(1)
        );
        assert_eq!(
            ca.quantile(0.9, QuantileInterpolOptions::Lower).unwrap(),
            Some(4)
        );
        assert_eq!(
            ca.quantile(0.6, QuantileInterpolOptions::Lower).unwrap(),
            Some(3)
        );

        assert_eq!(
            ca.quantile(0.1, QuantileInterpolOptions::Higher).unwrap(),
            Some(2)
        );
        assert_eq!(
            ca.quantile(0.9, QuantileInterpolOptions::Higher).unwrap(),
            Some(5)
        );
        assert_eq!(
            ca.quantile(0.6, QuantileInterpolOptions::Higher).unwrap(),
            Some(4)
        );

        assert_eq!(
            ca.quantile(0.1, QuantileInterpolOptions::Midpoint).unwrap(),
            Some(1)
        );
        assert_eq!(
            ca.quantile(0.9, QuantileInterpolOptions::Midpoint).unwrap(),
            Some(4)
        );
        assert_eq!(
            ca.quantile(0.6, QuantileInterpolOptions::Midpoint).unwrap(),
            Some(3)
        );

        assert_eq!(
            ca.quantile(0.1, QuantileInterpolOptions::Linear).unwrap(),
            Some(1)
        );
        assert_eq!(
            ca.quantile(0.9, QuantileInterpolOptions::Linear).unwrap(),
            Some(4)
        );
        assert_eq!(
            ca.quantile(0.6, QuantileInterpolOptions::Linear).unwrap(),
            Some(3)
        );

        let ca = UInt32Chunked::new(
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

        assert_eq!(
            ca.quantile(0.1, QuantileInterpolOptions::Nearest).unwrap(),
            Some(1)
        );
        assert_eq!(
            ca.quantile(0.9, QuantileInterpolOptions::Nearest).unwrap(),
            Some(7)
        );
        assert_eq!(
            ca.quantile(0.6, QuantileInterpolOptions::Nearest).unwrap(),
            Some(5)
        );

        assert_eq!(
            ca.quantile(0.1, QuantileInterpolOptions::Lower).unwrap(),
            Some(1)
        );
        assert_eq!(
            ca.quantile(0.9, QuantileInterpolOptions::Lower).unwrap(),
            Some(6)
        );
        assert_eq!(
            ca.quantile(0.6, QuantileInterpolOptions::Lower).unwrap(),
            Some(4)
        );

        assert_eq!(
            ca.quantile(0.1, QuantileInterpolOptions::Higher).unwrap(),
            Some(2)
        );
        assert_eq!(
            ca.quantile(0.9, QuantileInterpolOptions::Higher).unwrap(),
            Some(7)
        );
        assert_eq!(
            ca.quantile(0.6, QuantileInterpolOptions::Higher).unwrap(),
            Some(5)
        );

        assert_eq!(
            ca.quantile(0.1, QuantileInterpolOptions::Midpoint).unwrap(),
            Some(1)
        );
        assert_eq!(
            ca.quantile(0.9, QuantileInterpolOptions::Midpoint).unwrap(),
            Some(6)
        );
        assert_eq!(
            ca.quantile(0.6, QuantileInterpolOptions::Midpoint).unwrap(),
            Some(4)
        );

        assert_eq!(
            ca.quantile(0.1, QuantileInterpolOptions::Linear).unwrap(),
            Some(1)
        );
        assert_eq!(
            ca.quantile(0.9, QuantileInterpolOptions::Linear).unwrap(),
            Some(6)
        );
        assert_eq!(
            ca.quantile(0.6, QuantileInterpolOptions::Linear).unwrap(),
            Some(4)
        );

        let ca = Float32Chunked::new_from_slice(
            "",
            &[
                0.166189,
                0.166559,
                0.168517,
                0.169393,
                0.175272,
                0.23316699999999999,
                0.238787,
                0.266562,
                0.26903,
                0.285792,
                0.292801,
                0.29342899999999994,
                0.30170600000000003,
                0.308534,
                0.331489,
                0.346095,
                0.36764399999999997,
                0.36993899999999996,
                0.37207399999999996,
                0.41014000000000006,
                0.415789,
                0.421781,
                0.4277250000000001,
                0.46536299999999997,
                0.500208,
                2.6217269999999995,
                2.803311,
                3.868526,
            ],
        );

        assert!(
            (ca.quantile(0.1, QuantileInterpolOptions::Nearest)
                .unwrap()
                .unwrap()
                - 0.168517)
                .abs()
                < 0.00001
        );
        assert!(
            (ca.quantile(0.9, QuantileInterpolOptions::Nearest)
                .unwrap()
                .unwrap()
                - 2.621727)
                .abs()
                < 0.00001
        );
        assert!(
            (ca.quantile(0.6, QuantileInterpolOptions::Nearest)
                .unwrap()
                .unwrap()
                - 0.367644)
                .abs()
                < 0.00001
        );

        assert!(
            (ca.quantile(0.1, QuantileInterpolOptions::Lower)
                .unwrap()
                .unwrap()
                - 0.168517)
                .abs()
                < 0.00001
        );
        assert!(
            (ca.quantile(0.9, QuantileInterpolOptions::Lower)
                .unwrap()
                .unwrap()
                - 0.500208)
                .abs()
                < 0.00001
        );
        assert!(
            (ca.quantile(0.6, QuantileInterpolOptions::Lower)
                .unwrap()
                .unwrap()
                - 0.367644)
                .abs()
                < 0.00001
        );

        assert!(
            (ca.quantile(0.1, QuantileInterpolOptions::Higher)
                .unwrap()
                .unwrap()
                - 0.169393)
                .abs()
                < 0.00001
        );
        assert!(
            (ca.quantile(0.9, QuantileInterpolOptions::Higher)
                .unwrap()
                .unwrap()
                - 2.621727)
                .abs()
                < 0.00001
        );
        assert!(
            (ca.quantile(0.6, QuantileInterpolOptions::Higher)
                .unwrap()
                .unwrap()
                - 0.369939)
                .abs()
                < 0.00001
        );

        assert!(
            (ca.quantile(0.1, QuantileInterpolOptions::Midpoint)
                .unwrap()
                .unwrap()
                - 0.168955)
                .abs()
                < 0.0001
        );
        assert!(
            (ca.quantile(0.9, QuantileInterpolOptions::Midpoint)
                .unwrap()
                .unwrap()
                - 1.560967)
                .abs()
                < 0.0001
        );
        assert!(
            (ca.quantile(0.6, QuantileInterpolOptions::Midpoint)
                .unwrap()
                .unwrap()
                - 0.368791)
                .abs()
                < 0.0001
        );

        assert!(
            (ca.quantile(0.1, QuantileInterpolOptions::Linear)
                .unwrap()
                .unwrap()
                - 0.169130)
                .abs()
                < 0.0001
        );
        assert!(
            (ca.quantile(0.9, QuantileInterpolOptions::Linear)
                .unwrap()
                .unwrap()
                - 1.136664)
                .abs()
                < 0.0001
        );
        assert!(
            (ca.quantile(0.6, QuantileInterpolOptions::Linear)
                .unwrap()
                .unwrap()
                - 0.368103)
                .abs()
                < 0.0001
        );
    }
}
