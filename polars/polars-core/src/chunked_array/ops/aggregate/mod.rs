//! Implementations of the ChunkAgg trait.
mod quantile;
mod var;

use std::cmp::Ordering;
use std::ops::Add;

use arrow::compute;
use arrow::types::simd::Simd;
use arrow::types::NativeType;
use num::{Float, ToPrimitive};
use polars_arrow::kernels::rolling::{compare_fn_nan_max, compare_fn_nan_min};
pub use quantile::*;
pub use var::*;

use crate::chunked_array::ChunkedArray;
use crate::datatypes::{BooleanChunked, PolarsNumericType};
use crate::prelude::*;
use crate::series::IsSorted;
use crate::utils::CustomIterTools;

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
    /// Get the product of the ChunkedArray as a new Series of length 1.
    fn prod_as_series(&self) -> Series {
        unimplemented!()
    }
}

fn sum_float_unaligned_slice<T: NumericNative>(values: &[T]) -> Option<T> {
    Some(values.iter().copied().sum())
}

fn sum_float_unaligned<T: NumericNative>(array: &PrimitiveArray<T>) -> Option<T> {
    if array.len() == 0 {
        return Some(T::zero());
    }
    if array.null_count() == array.len() {
        return None;
    }
    Some(array.into_iter().flatten().copied().sum())
}

/// Floating point arithmetic is non-associative.
/// The simd chunks are determined by memory location
/// e.g.
///
/// |HEAD|  - | SIMD | - |TAIL|
///
/// The SIMD chunks have a certain alignment and depending of the start of the buffer
/// head and tail may have different sizes, making a sum non-deterministic for the same
/// values but different memory locations
fn stable_sum<T: NumericNative + NativeType>(array: &PrimitiveArray<T>) -> Option<T>
where
    T: NumericNative + NativeType,
    <T as Simd>::Simd: Add<Output = <T as Simd>::Simd>
        + compute::aggregate::Sum<T>
        + compute::aggregate::SimdOrd<T>,
{
    if T::is_float() {
        use arrow::types::simd::NativeSimd;
        let values = array.values().as_slice();
        let (a, _, _) = <T as Simd>::Simd::align(values);
        // we only choose SIMD path if buffer is aligned to SIMD
        if a.is_empty() {
            compute::aggregate::sum_primitive(array)
        } else if array.null_count() == 0 {
            sum_float_unaligned_slice(values)
        } else {
            sum_float_unaligned(array)
        }
    } else {
        compute::aggregate::sum_primitive(array)
    }
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
            .map(stable_sum)
            .fold(None, |acc, v| match v {
                Some(v) => match acc {
                    None => Some(v),
                    Some(acc) => Some(acc + v),
                },
                None => acc,
            })
    }

    fn min(&self) -> Option<T::Native> {
        match self.is_sorted_flag2() {
            IsSorted::Ascending => {
                self.first_non_null().and_then(|idx| {
                    // Safety:
                    // first_non_null returns in bound index
                    unsafe { self.get_unchecked(idx) }
                })
            }
            IsSorted::Descending => {
                self.last_non_null().and_then(|idx| {
                    // Safety:
                    // last returns in bound index
                    unsafe { self.get_unchecked(idx) }
                })
            }
            IsSorted::Not => self
                .downcast_iter()
                .filter_map(compute::aggregate::min_primitive)
                .fold_first_(|acc, v| {
                    if matches!(compare_fn_nan_max(&acc, &v), Ordering::Less) {
                        acc
                    } else {
                        v
                    }
                }),
        }
    }

    fn max(&self) -> Option<T::Native> {
        match self.is_sorted_flag2() {
            IsSorted::Ascending => {
                self.last_non_null().and_then(|idx| {
                    // Safety:
                    // first_non_null returns in bound index
                    unsafe { self.get_unchecked(idx) }
                })
            }
            IsSorted::Descending => {
                self.first_non_null().and_then(|idx| {
                    // Safety:
                    // last returns in bound index
                    unsafe { self.get_unchecked(idx) }
                })
            }
            IsSorted::Not => self
                .downcast_iter()
                .filter_map(compute::aggregate::max_primitive)
                .fold_first_(|acc, v| {
                    if matches!(compare_fn_nan_min(&acc, &v), Ordering::Greater) {
                        acc
                    } else {
                        v
                    }
                }),
        }
    }

    fn mean(&self) -> Option<f64> {
        match self.dtype() {
            DataType::Float64 => {
                let len = (self.len() - self.null_count()) as f64;
                self.sum().map(|v| v.to_f64().unwrap() / len)
            }
            _ => {
                let null_count = self.null_count();
                let len = self.len();
                if null_count == len {
                    None
                } else {
                    let mut acc = 0.0;
                    let len = (len - null_count) as f64;

                    for arr in self.downcast_iter() {
                        if arr.null_count() > 0 {
                            for v in arr.into_iter().flatten() {
                                // safety
                                // all these types can be coerced to f64
                                unsafe {
                                    let val = v.to_f64().unwrap_unchecked();
                                    acc += val
                                }
                            }
                        } else {
                            for v in arr.values().as_slice() {
                                // safety
                                // all these types can be coerced to f64
                                unsafe {
                                    let val = v.to_f64().unwrap_unchecked();
                                    acc += val
                                }
                            }
                        }
                    }
                    Some(acc / len)
                }
            }
        }
    }
}

/// Booleans are casted to 1 or 0.
impl ChunkAgg<IdxSize> for BooleanChunked {
    /// Returns `None` if the array is empty or only contains null values.
    fn sum(&self) -> Option<IdxSize> {
        if self.is_empty() {
            None
        } else {
            Some(
                self.downcast_iter()
                    .map(|arr| match arr.validity() {
                        Some(validity) => {
                            (arr.len() - (validity & arr.values()).unset_bits()) as IdxSize
                        }
                        None => (arr.len() - arr.values().unset_bits()) as IdxSize,
                    })
                    .sum(),
            )
        }
    }

    fn min(&self) -> Option<IdxSize> {
        if self.is_empty() {
            return None;
        }
        if self.all() {
            Some(1)
        } else {
            Some(0)
        }
    }

    fn max(&self) -> Option<IdxSize> {
        if self.is_empty() {
            return None;
        }
        if self.any() {
            Some(1)
        } else {
            Some(0)
        }
    }
    fn mean(&self) -> Option<f64> {
        self.sum()
            .map(|sum| sum as f64 / (self.len() - self.null_count()) as f64)
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

    fn prod_as_series(&self) -> Series {
        let mut prod = None;
        for opt_v in self.into_iter() {
            match (prod, opt_v) {
                (_, None) => return Self::full_null(self.name(), 1).into_series(),
                (None, Some(v)) => prod = Some(v),
                (Some(p), Some(v)) => prod = Some(p * v),
            }
        }
        Self::from_slice_options(self.name(), &[prod]).into_series()
    }
}

macro_rules! impl_as_series {
    ($self:expr, $agg:ident, $ty: ty) => {{
        let v = $self.$agg();
        let mut ca: $ty = [v].iter().copied().collect();
        ca.rename($self.name());
        ca.into_series()
    }};
    ($self:expr, $agg:ident, $arg:expr, $ty: ty) => {{
        let v = $self.$agg($arg);
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
    fn var_as_series(&self, ddof: u8) -> Series {
        impl_as_series!(self, var, ddof, Float64Chunked)
    }

    fn std_as_series(&self, ddof: u8) -> Series {
        impl_as_series!(self, std, ddof, Float64Chunked)
    }
}

impl VarAggSeries for Float32Chunked {
    fn var_as_series(&self, ddof: u8) -> Series {
        impl_as_series!(self, var, ddof, Float32Chunked)
    }

    fn std_as_series(&self, ddof: u8) -> Series {
        impl_as_series!(self, std, ddof, Float32Chunked)
    }
}

impl VarAggSeries for Float64Chunked {
    fn var_as_series(&self, ddof: u8) -> Series {
        impl_as_series!(self, var, ddof, Float64Chunked)
    }

    fn std_as_series(&self, ddof: u8) -> Series {
        impl_as_series!(self, std, ddof, Float64Chunked)
    }
}

macro_rules! impl_quantile_as_series {
    ($self:expr, $agg:ident, $ty: ty, $qtl:expr, $opt:expr) => {{
        let v = $self.$agg($qtl, $opt)?;
        let mut ca: $ty = [v].iter().copied().collect();
        ca.rename($self.name());
        Ok(ca.into_series())
    }};
}

impl<T> QuantileAggSeries for ChunkedArray<T>
where
    T: PolarsIntegerType,
    T::Native: Ord,
    <T::Native as Simd>::Simd: Add<Output = <T::Native as Simd>::Simd>
        + compute::aggregate::Sum<T::Native>
        + compute::aggregate::SimdOrd<T::Native>,
{
    fn quantile_as_series(
        &self,
        quantile: f64,
        interpol: QuantileInterpolOptions,
    ) -> PolarsResult<Series> {
        impl_quantile_as_series!(self, quantile, Float64Chunked, quantile, interpol)
    }

    fn median_as_series(&self) -> Series {
        impl_as_series!(self, median, Float64Chunked)
    }
}

impl QuantileAggSeries for Float32Chunked {
    fn quantile_as_series(
        &self,
        quantile: f64,
        interpol: QuantileInterpolOptions,
    ) -> PolarsResult<Series> {
        impl_quantile_as_series!(self, quantile, Float32Chunked, quantile, interpol)
    }

    fn median_as_series(&self) -> Series {
        impl_as_series!(self, median, Float32Chunked)
    }
}

impl QuantileAggSeries for Float64Chunked {
    fn quantile_as_series(
        &self,
        quantile: f64,
        interpol: QuantileInterpolOptions,
    ) -> PolarsResult<Series> {
        impl_quantile_as_series!(self, quantile, Float64Chunked, quantile, interpol)
    }

    fn median_as_series(&self) -> Series {
        impl_as_series!(self, median, Float64Chunked)
    }
}

impl ChunkAggSeries for BooleanChunked {
    fn sum_as_series(&self) -> Series {
        let v = ChunkAgg::sum(self);
        let mut ca: IdxCa = [v].iter().copied().collect();
        ca.rename(self.name());
        ca.into_series()
    }
    fn max_as_series(&self) -> Series {
        let v = ChunkAgg::max(self);
        let mut ca: IdxCa = [v].iter().copied().collect();
        ca.rename(self.name());
        ca.into_series()
    }
    fn min_as_series(&self) -> Series {
        let v = ChunkAgg::min(self);
        let mut ca: IdxCa = [v].iter().copied().collect();
        ca.rename(self.name());
        ca.into_series()
    }
}

impl Utf8Chunked {
    pub(crate) fn max_str(&self) -> Option<&str> {
        match self.is_sorted_flag2() {
            IsSorted::Ascending => self.get(self.len() - 1),
            IsSorted::Descending => self.get(0),
            IsSorted::Not => self
                .downcast_iter()
                .filter_map(compute::aggregate::max_string)
                .fold_first_(|acc, v| if acc > v { acc } else { v }),
        }
    }
    pub(crate) fn min_str(&self) -> Option<&str> {
        match self.is_sorted_flag2() {
            IsSorted::Ascending => self.get(0),
            IsSorted::Descending => self.get(self.len() - 1),
            IsSorted::Not => self
                .downcast_iter()
                .filter_map(compute::aggregate::min_string)
                .fold_first_(|acc, v| if acc < v { acc } else { v }),
        }
    }
}

impl ChunkAggSeries for Utf8Chunked {
    fn sum_as_series(&self) -> Series {
        Utf8Chunked::full_null(self.name(), 1).into_series()
    }
    fn max_as_series(&self) -> Series {
        Series::new(self.name(), &[self.max_str()])
    }
    fn min_as_series(&self) -> Series {
        Series::new(self.name(), &[self.min_str()])
    }
}

#[cfg(feature = "dtype-binary")]
impl ChunkAggSeries for BinaryChunked {
    fn sum_as_series(&self) -> Series {
        BinaryChunked::full_null(self.name(), 1).into_series()
    }
    fn max_as_series(&self) -> Series {
        Series::new(
            self.name(),
            &[self
                .downcast_iter()
                .filter_map(compute::aggregate::max_binary)
                .fold_first_(|acc, v| if acc > v { acc } else { v })],
        )
    }
    fn min_as_series(&self) -> Series {
        Series::new(
            self.name(),
            &[self
                .downcast_iter()
                .filter_map(compute::aggregate::min_binary)
                .fold_first_(|acc, v| if acc < v { acc } else { v })],
        )
    }
}

impl ChunkAggSeries for ListChunked {
    fn sum_as_series(&self) -> Series {
        ListChunked::full_null_with_dtype(self.name(), 1, &self.inner_dtype()).into_series()
    }
    fn max_as_series(&self) -> Series {
        ListChunked::full_null_with_dtype(self.name(), 1, &self.inner_dtype()).into_series()
    }
    fn min_as_series(&self) -> Series {
        ListChunked::full_null_with_dtype(self.name(), 1, &self.inner_dtype()).into_series()
    }
}

#[cfg(feature = "object")]
impl<T: PolarsObject> ChunkAggSeries for ObjectChunked<T> {}

#[cfg(test)]
mod test {
    use polars_arrow::prelude::QuantileInterpolOptions;

    use crate::prelude::*;

    #[test]
    fn test_var() {
        // validated with numpy
        // Note that numpy as an argument ddof which influences results. The default is ddof=0
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
            let out = ca.var(1);
            assert_eq!(out, Some(12.3));
            let out = ca.std(1).unwrap();
            assert!((3.5071355833500366 - out).abs() < 0.000000001);
        }
    }

    #[test]
    fn test_agg_float() {
        let ca1 = Float32Chunked::new("a", &[1.0, f32::NAN]);
        let ca2 = Float32Chunked::new("b", &[f32::NAN, 1.0]);
        assert_eq!(ca1.min(), ca2.min());
        let ca1 = Float64Chunked::new("a", &[1.0, f64::NAN]);
        let ca2 = Float64Chunked::from_slice("b", &[f64::NAN, 1.0]);
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

        let ca = Float32Chunked::from_slice(
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
        assert_eq!(
            ca.into_series()
                .mean_as_series()
                .f32()
                .unwrap()
                .get(0)
                .unwrap(),
            1.5
        );
        // all null values case
        let ca = Float32Chunked::full_null("", 3);
        assert_eq!(ca.mean(), None);
        assert_eq!(
            ca.into_series().mean_as_series().f32().unwrap().get(0),
            None
        );
    }

    #[test]
    fn test_quantile_all_null() {
        let test_f32 = Float32Chunked::from_slice_options("", &[None, None, None]);
        let test_i32 = Int32Chunked::from_slice_options("", &[None, None, None]);
        let test_f64 = Float64Chunked::from_slice_options("", &[None, None, None]);
        let test_i64 = Int64Chunked::from_slice_options("", &[None, None, None]);

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
        let test_f32 = Float32Chunked::from_slice_options("", &[Some(1.0)]);
        let test_i32 = Int32Chunked::from_slice_options("", &[Some(1)]);
        let test_f64 = Float64Chunked::from_slice_options("", &[Some(1.0)]);
        let test_i64 = Int64Chunked::from_slice_options("", &[Some(1)]);

        let interpol_options = vec![
            QuantileInterpolOptions::Nearest,
            QuantileInterpolOptions::Lower,
            QuantileInterpolOptions::Higher,
            QuantileInterpolOptions::Midpoint,
            QuantileInterpolOptions::Linear,
        ];

        for interpol in interpol_options {
            assert_eq!(test_f32.quantile(0.5, interpol).unwrap(), Some(1.0));
            assert_eq!(test_i32.quantile(0.5, interpol).unwrap(), Some(1.0));
            assert_eq!(test_f64.quantile(0.5, interpol).unwrap(), Some(1.0));
            assert_eq!(test_i64.quantile(0.5, interpol).unwrap(), Some(1.0));
        }
    }

    #[test]
    fn test_quantile_min_max() {
        let test_f32 =
            Float32Chunked::from_slice_options("", &[None, Some(1f32), Some(5f32), Some(1f32)]);
        let test_i32 =
            Int32Chunked::from_slice_options("", &[None, Some(1i32), Some(5i32), Some(1i32)]);
        let test_f64 =
            Float64Chunked::from_slice_options("", &[None, Some(1f64), Some(5f64), Some(1f64)]);
        let test_i64 =
            Int64Chunked::from_slice_options("", &[None, Some(1i64), Some(5i64), Some(1i64)]);

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

            assert_eq!(
                test_i32.quantile(0.0, interpol).unwrap().unwrap(),
                test_i32.min().unwrap() as f64
            );
            assert_eq!(
                test_i32.quantile(1.0, interpol).unwrap().unwrap(),
                test_i32.max().unwrap() as f64
            );

            assert_eq!(test_f64.quantile(0.0, interpol).unwrap(), test_f64.min());
            assert_eq!(test_f64.quantile(1.0, interpol).unwrap(), test_f64.max());
            assert_eq!(test_f64.quantile(0.5, interpol).unwrap(), test_f64.median());

            assert_eq!(
                test_i64.quantile(0.0, interpol).unwrap().unwrap(),
                test_i64.min().unwrap() as f64
            );
            assert_eq!(
                test_i64.quantile(1.0, interpol).unwrap().unwrap(),
                test_i64.max().unwrap() as f64
            );
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
            Some(1.0)
        );
        assert_eq!(
            ca.quantile(0.9, QuantileInterpolOptions::Nearest).unwrap(),
            Some(5.0)
        );
        assert_eq!(
            ca.quantile(0.6, QuantileInterpolOptions::Nearest).unwrap(),
            Some(4.0)
        );

        assert_eq!(
            ca.quantile(0.1, QuantileInterpolOptions::Lower).unwrap(),
            Some(1.0)
        );
        assert_eq!(
            ca.quantile(0.9, QuantileInterpolOptions::Lower).unwrap(),
            Some(4.0)
        );
        assert_eq!(
            ca.quantile(0.6, QuantileInterpolOptions::Lower).unwrap(),
            Some(3.0)
        );

        assert_eq!(
            ca.quantile(0.1, QuantileInterpolOptions::Higher).unwrap(),
            Some(2.0)
        );
        assert_eq!(
            ca.quantile(0.9, QuantileInterpolOptions::Higher).unwrap(),
            Some(5.0)
        );
        assert_eq!(
            ca.quantile(0.6, QuantileInterpolOptions::Higher).unwrap(),
            Some(4.0)
        );

        assert_eq!(
            ca.quantile(0.1, QuantileInterpolOptions::Midpoint).unwrap(),
            Some(1.5)
        );
        assert_eq!(
            ca.quantile(0.9, QuantileInterpolOptions::Midpoint).unwrap(),
            Some(4.5)
        );
        assert_eq!(
            ca.quantile(0.6, QuantileInterpolOptions::Midpoint).unwrap(),
            Some(3.5)
        );

        assert_eq!(
            ca.quantile(0.1, QuantileInterpolOptions::Linear).unwrap(),
            Some(1.4)
        );
        assert_eq!(
            ca.quantile(0.9, QuantileInterpolOptions::Linear).unwrap(),
            Some(4.6)
        );
        assert!(
            (ca.quantile(0.6, QuantileInterpolOptions::Linear)
                .unwrap()
                .unwrap()
                - 3.4)
                .abs()
                < 0.0000001
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
            Some(1.0)
        );
        assert_eq!(
            ca.quantile(0.9, QuantileInterpolOptions::Nearest).unwrap(),
            Some(7.0)
        );
        assert_eq!(
            ca.quantile(0.6, QuantileInterpolOptions::Nearest).unwrap(),
            Some(5.0)
        );

        assert_eq!(
            ca.quantile(0.1, QuantileInterpolOptions::Lower).unwrap(),
            Some(1.0)
        );
        assert_eq!(
            ca.quantile(0.9, QuantileInterpolOptions::Lower).unwrap(),
            Some(6.0)
        );
        assert_eq!(
            ca.quantile(0.6, QuantileInterpolOptions::Lower).unwrap(),
            Some(4.0)
        );

        assert_eq!(
            ca.quantile(0.1, QuantileInterpolOptions::Higher).unwrap(),
            Some(2.0)
        );
        assert_eq!(
            ca.quantile(0.9, QuantileInterpolOptions::Higher).unwrap(),
            Some(7.0)
        );
        assert_eq!(
            ca.quantile(0.6, QuantileInterpolOptions::Higher).unwrap(),
            Some(5.0)
        );

        assert_eq!(
            ca.quantile(0.1, QuantileInterpolOptions::Midpoint).unwrap(),
            Some(1.5)
        );
        assert_eq!(
            ca.quantile(0.9, QuantileInterpolOptions::Midpoint).unwrap(),
            Some(6.5)
        );
        assert_eq!(
            ca.quantile(0.6, QuantileInterpolOptions::Midpoint).unwrap(),
            Some(4.5)
        );

        assert_eq!(
            ca.quantile(0.1, QuantileInterpolOptions::Linear).unwrap(),
            Some(1.6)
        );
        assert_eq!(
            ca.quantile(0.9, QuantileInterpolOptions::Linear).unwrap(),
            Some(6.4)
        );
        assert_eq!(
            ca.quantile(0.6, QuantileInterpolOptions::Linear).unwrap(),
            Some(4.6)
        );

        let ca = Float32Chunked::from_slice(
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

    #[test]
    fn test_median_floats() {
        let a = Series::new("a", &[1.0f64, 2.0, 3.0]);
        let expected = Series::new("a", [2.0f64]);
        assert!(a.median_as_series().series_equal_missing(&expected));
        assert_eq!(a.median(), Some(2.0f64))
    }
}
