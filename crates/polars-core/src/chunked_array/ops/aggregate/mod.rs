//! Implementations of the ChunkAgg trait.
mod quantile;
mod var;

use std::ops::Add;

use arrow::compute;
use arrow::types::simd::Simd;
use arrow::types::NativeType;
use num_traits::{Float, One, ToPrimitive, Zero};
use polars_compute::min_max::MinMaxKernel;
use polars_utils::min_max::MinMax;
pub use quantile::*;
pub use var::*;

use crate::chunked_array::ChunkedArray;
use crate::datatypes::{BooleanChunked, PolarsNumericType};
use crate::prelude::*;
use crate::series::implementations::SeriesWrap;
use crate::series::IsSorted;

pub mod float_sum;

/// Aggregations that return [`Series`] of unit length. Those can be used in broadcasting operations.
pub trait ChunkAggSeries {
    /// Get the sum of the [`ChunkedArray`] as a new [`Series`] of length 1.
    fn sum_as_series(&self) -> Series {
        unimplemented!()
    }
    /// Get the max of the [`ChunkedArray`] as a new [`Series`] of length 1.
    fn max_as_series(&self) -> Series {
        unimplemented!()
    }
    /// Get the min of the [`ChunkedArray`] as a new [`Series`] of length 1.
    fn min_as_series(&self) -> Series {
        unimplemented!()
    }
    /// Get the product of the [`ChunkedArray`] as a new [`Series`] of length 1.
    fn prod_as_series(&self) -> Series {
        unimplemented!()
    }
}

fn sum<T: NumericNative + NativeType>(array: &PrimitiveArray<T>) -> T
where
    T: NumericNative + NativeType,
    <T as Simd>::Simd: Add<Output = <T as Simd>::Simd> + compute::aggregate::Sum<T>,
{
    if array.null_count() == array.len() {
        return T::default();
    }

    if T::is_float() {
        let values = array.values().as_slice();
        let validity = array.validity().filter(|_| array.null_count() > 0);
        if T::is_f32() {
            let f32_values = unsafe { std::mem::transmute::<&[T], &[f32]>(values) };
            let sum: f32 = if let Some(bitmap) = validity {
                float_sum::f32::sum_with_validity(f32_values, bitmap) as f32 // TODO: f64?
            } else {
                float_sum::f32::sum(f32_values) as f32 // TODO: f64?
            };
            unsafe { std::mem::transmute_copy::<f32, T>(&sum) }
        } else if T::is_f64() {
            let f64_values = unsafe { std::mem::transmute::<&[T], &[f64]>(values) };
            let sum: f64 = if let Some(bitmap) = validity {
                float_sum::f64::sum_with_validity(f64_values, bitmap)
            } else {
                float_sum::f64::sum(f64_values)
            };
            unsafe { std::mem::transmute_copy::<f64, T>(&sum) }
        } else {
            unreachable!("only supported float types are f32 and f64");
        }
    } else {
        compute::aggregate::sum_primitive(array).unwrap_or(T::zero())
    }
}

impl<T> ChunkAgg<T::Native> for ChunkedArray<T>
where
    T: PolarsNumericType,
    PrimitiveArray<T::Native>: for<'a> MinMaxKernel<Scalar<'a> = T::Native>,
    <T::Native as Simd>::Simd:
        Add<Output = <T::Native as Simd>::Simd> + compute::aggregate::Sum<T::Native>,
{
    fn sum(&self) -> Option<T::Native> {
        Some(
            self.downcast_iter()
                .map(sum)
                .fold(T::Native::zero(), |acc, v| acc + v),
        )
    }

    fn min(&self) -> Option<T::Native> {
        if self.is_empty() {
            return None;
        }
        match self.is_sorted_flag() {
            IsSorted::Ascending => {
                self.first_non_null().and_then(|idx| {
                    // SAFETY: first_non_null returns in bound index.
                    unsafe { self.get_unchecked(idx) }
                })
            },
            IsSorted::Descending => {
                self.last_non_null().and_then(|idx| {
                    // SAFETY: last returns in bound index.
                    unsafe { self.get_unchecked(idx) }
                })
            },
            IsSorted::Not => self
                .downcast_iter()
                .filter_map(MinMaxKernel::min_ignore_nan_kernel)
                .reduce(MinMax::min_ignore_nan),
        }
    }

    fn max(&self) -> Option<T::Native> {
        if self.is_empty() {
            return None;
        }
        match self.is_sorted_flag() {
            IsSorted::Ascending => {
                self.last_non_null().and_then(|idx| {
                    // SAFETY:
                    // last_non_null returns in bound index
                    unsafe { self.get_unchecked(idx) }
                })
            },
            IsSorted::Descending => {
                self.first_non_null().and_then(|idx| {
                    // SAFETY:
                    // first_non_null returns in bound index
                    unsafe { self.get_unchecked(idx) }
                })
            },
            IsSorted::Not => self
                .downcast_iter()
                .filter_map(MinMaxKernel::max_ignore_nan_kernel)
                .reduce(MinMax::max_ignore_nan),
        }
    }

    fn mean(&self) -> Option<f64> {
        if self.is_empty() || self.null_count() == self.len() {
            return None;
        }
        match self.dtype() {
            DataType::Float64 => {
                let len = (self.len() - self.null_count()) as f64;
                self.sum().map(|v| v.to_f64().unwrap() / len)
            },
            _ => {
                let null_count = self.null_count();
                let len = self.len();
                match null_count {
                    nc if nc == len => None,
                    #[cfg(feature = "simd")]
                    _ => {
                        // TODO: investigate if we need a stable mean
                        // because of SIMD memory locations and associativity.
                        // similar to sum
                        let mut sum = 0.0;
                        for arr in self.downcast_iter() {
                            sum += arrow::legacy::kernels::agg_mean::sum_as_f64(arr);
                        }
                        Some(sum / (len - null_count) as f64)
                    },
                    #[cfg(not(feature = "simd"))]
                    _ => {
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
                    },
                }
            },
        }
    }
}

/// Booleans are casted to 1 or 0.
impl BooleanChunked {
    pub fn sum(&self) -> Option<IdxSize> {
        Some(if self.is_empty() {
            0
        } else {
            self.downcast_iter()
                .map(|arr| match arr.validity() {
                    Some(validity) => {
                        (arr.len() - (validity & arr.values()).unset_bits()) as IdxSize
                    },
                    None => (arr.len() - arr.values().unset_bits()) as IdxSize,
                })
                .sum()
        })
    }

    pub fn min(&self) -> Option<bool> {
        let nc = self.null_count();
        let len = self.len();
        if self.is_empty() || nc == len {
            return None;
        }
        if nc == 0 {
            if self.all() {
                Some(true)
            } else {
                Some(false)
            }
        } else {
            // we can unwrap as we already checked empty and all null above
            if (self.sum().unwrap() + nc as IdxSize) == len as IdxSize {
                Some(true)
            } else {
                Some(false)
            }
        }
    }

    pub fn max(&self) -> Option<bool> {
        if self.is_empty() || self.null_count() == self.len() {
            return None;
        }
        if self.any() {
            Some(true)
        } else {
            Some(false)
        }
    }
    pub fn mean(&self) -> Option<f64> {
        if self.is_empty() || self.null_count() == self.len() {
            return None;
        }
        self.sum()
            .map(|sum| sum as f64 / (self.len() - self.null_count()) as f64)
    }
}

// Needs the same trait bounds as the implementation of ChunkedArray<T> of dyn Series.
impl<T> ChunkAggSeries for ChunkedArray<T>
where
    T: PolarsNumericType,
    PrimitiveArray<T::Native>: for<'a> MinMaxKernel<Scalar<'a> = T::Native>,
    <T::Native as Simd>::Simd:
        Add<Output = <T::Native as Simd>::Simd> + compute::aggregate::Sum<T::Native>,
    ChunkedArray<T>: IntoSeries,
{
    fn sum_as_series(&self) -> Series {
        let v = self.sum();
        let mut ca: ChunkedArray<T> = [v].iter().copied().collect();
        ca.rename(self.name());
        ca.into_series()
    }

    fn max_as_series(&self) -> Series {
        let v = ChunkAgg::max(self);
        let mut ca: ChunkedArray<T> = [v].iter().copied().collect();
        ca.rename(self.name());
        ca.into_series()
    }

    fn min_as_series(&self) -> Series {
        let v = ChunkAgg::min(self);
        let mut ca: ChunkedArray<T> = [v].iter().copied().collect();
        ca.rename(self.name());
        ca.into_series()
    }

    fn prod_as_series(&self) -> Series {
        let mut prod = T::Native::one();
        for opt_v in self.into_iter().flatten() {
            prod = prod * opt_v;
        }
        Self::from_slice_options(self.name(), &[Some(prod)]).into_series()
    }
}

fn as_series<T>(name: &str, v: Option<T::Native>) -> Series
where
    T: PolarsNumericType,
    SeriesWrap<ChunkedArray<T>>: SeriesTrait,
{
    let mut ca: ChunkedArray<T> = [v].into_iter().collect();
    ca.rename(name);
    ca.into_series()
}

impl<T> VarAggSeries for ChunkedArray<T>
where
    T: PolarsIntegerType,
    ChunkedArray<T>: ChunkVar,
{
    fn var_as_series(&self, ddof: u8) -> Series {
        as_series::<Float64Type>(self.name(), self.var(ddof))
    }

    fn std_as_series(&self, ddof: u8) -> Series {
        as_series::<Float64Type>(self.name(), self.std(ddof))
    }
}

impl VarAggSeries for Float32Chunked {
    fn var_as_series(&self, ddof: u8) -> Series {
        as_series::<Float32Type>(self.name(), self.var(ddof).map(|x| x as f32))
    }

    fn std_as_series(&self, ddof: u8) -> Series {
        as_series::<Float32Type>(self.name(), self.std(ddof).map(|x| x as f32))
    }
}

impl VarAggSeries for Float64Chunked {
    fn var_as_series(&self, ddof: u8) -> Series {
        as_series::<Float64Type>(self.name(), self.var(ddof))
    }

    fn std_as_series(&self, ddof: u8) -> Series {
        as_series::<Float64Type>(self.name(), self.std(ddof))
    }
}

impl<T> QuantileAggSeries for ChunkedArray<T>
where
    T: PolarsIntegerType,
    T::Native: Ord,
    <T::Native as Simd>::Simd:
        Add<Output = <T::Native as Simd>::Simd> + compute::aggregate::Sum<T::Native>,
{
    fn quantile_as_series(
        &self,
        quantile: f64,
        interpol: QuantileInterpolOptions,
    ) -> PolarsResult<Series> {
        Ok(as_series::<Float64Type>(
            self.name(),
            self.quantile(quantile, interpol)?,
        ))
    }

    fn median_as_series(&self) -> Series {
        as_series::<Float64Type>(self.name(), self.median())
    }
}

impl QuantileAggSeries for Float32Chunked {
    fn quantile_as_series(
        &self,
        quantile: f64,
        interpol: QuantileInterpolOptions,
    ) -> PolarsResult<Series> {
        Ok(as_series::<Float32Type>(
            self.name(),
            self.quantile(quantile, interpol)?,
        ))
    }

    fn median_as_series(&self) -> Series {
        as_series::<Float32Type>(self.name(), self.median())
    }
}

impl QuantileAggSeries for Float64Chunked {
    fn quantile_as_series(
        &self,
        quantile: f64,
        interpol: QuantileInterpolOptions,
    ) -> PolarsResult<Series> {
        Ok(as_series::<Float64Type>(
            self.name(),
            self.quantile(quantile, interpol)?,
        ))
    }

    fn median_as_series(&self) -> Series {
        as_series::<Float64Type>(self.name(), self.median())
    }
}

impl ChunkAggSeries for BooleanChunked {
    fn sum_as_series(&self) -> Series {
        let v = self.sum();
        Series::new(self.name(), [v])
    }
    fn max_as_series(&self) -> Series {
        let v = self.max();
        Series::new(self.name(), [v])
    }
    fn min_as_series(&self) -> Series {
        let v = self.min();
        Series::new(self.name(), [v])
    }
}

impl Utf8Chunked {
    pub(crate) fn max_str(&self) -> Option<&str> {
        if self.is_empty() {
            return None;
        }
        match self.is_sorted_flag() {
            IsSorted::Ascending => {
                self.last_non_null().and_then(|idx| {
                    // SAFETY: last_non_null returns in bound index
                    unsafe { self.get_unchecked(idx) }
                })
            },
            IsSorted::Descending => {
                self.first_non_null().and_then(|idx| {
                    // SAFETY: first_non_null returns in bound index
                    unsafe { self.get_unchecked(idx) }
                })
            },
            IsSorted::Not => self
                .downcast_iter()
                .filter_map(MinMaxKernel::max_ignore_nan_kernel)
                .reduce(MinMax::max_ignore_nan),
        }
    }
    pub(crate) fn min_str(&self) -> Option<&str> {
        if self.is_empty() {
            return None;
        }
        match self.is_sorted_flag() {
            IsSorted::Ascending => {
                self.first_non_null().and_then(|idx| {
                    // SAFETY: first_non_null returns in bound index
                    unsafe { self.get_unchecked(idx) }
                })
            },
            IsSorted::Descending => {
                self.last_non_null().and_then(|idx| {
                    // SAFETY: last_non_null returns in bound index
                    unsafe { self.get_unchecked(idx) }
                })
            },
            IsSorted::Not => self
                .downcast_iter()
                .filter_map(MinMaxKernel::min_ignore_nan_kernel)
                .reduce(MinMax::min_ignore_nan),
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

impl BinaryChunked {
    pub(crate) fn max_binary(&self) -> Option<&[u8]> {
        if self.is_empty() {
            return None;
        }
        match self.is_sorted_flag() {
            IsSorted::Ascending => {
                self.last_non_null().and_then(|idx| {
                    // SAFETY: last_non_null returns in bound index.
                    unsafe { self.get_unchecked(idx) }
                })
            },
            IsSorted::Descending => {
                self.first_non_null().and_then(|idx| {
                    // SAFETY: first_non_null returns in bound index.
                    unsafe { self.get_unchecked(idx) }
                })
            },
            IsSorted::Not => self
                .downcast_iter()
                .filter_map(MinMaxKernel::max_ignore_nan_kernel)
                .reduce(MinMax::max_ignore_nan),
        }
    }

    pub(crate) fn min_binary(&self) -> Option<&[u8]> {
        if self.is_empty() {
            return None;
        }
        match self.is_sorted_flag() {
            IsSorted::Ascending => {
                self.first_non_null().and_then(|idx| {
                    // SAFETY: first_non_null returns in bound index.
                    unsafe { self.get_unchecked(idx) }
                })
            },
            IsSorted::Descending => {
                self.last_non_null().and_then(|idx| {
                    // SAFETY: last_non_null returns in bound index.
                    unsafe { self.get_unchecked(idx) }
                })
            },
            IsSorted::Not => self
                .downcast_iter()
                .filter_map(MinMaxKernel::min_ignore_nan_kernel)
                .reduce(MinMax::min_ignore_nan),
        }
    }
}

impl ChunkAggSeries for BinaryChunked {
    fn sum_as_series(&self) -> Series {
        unimplemented!()
    }
    fn max_as_series(&self) -> Series {
        Series::new(self.name(), [self.max_binary()])
    }
    fn min_as_series(&self) -> Series {
        Series::new(self.name(), [self.min_binary()])
    }
}

#[cfg(feature = "object")]
impl<T: PolarsObject> ChunkAggSeries for ObjectChunked<T> {}

#[cfg(test)]
mod test {
    use arrow::legacy::prelude::QuantileInterpolOptions;

    use crate::prelude::*;

    #[test]
    fn test_var() {
        // Validated with numpy. Note that numpy uses ddof as an argument which
        // influences results. The default ddof=0, we chose ddof=1, which is
        // standard in statistics.
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
                0.166189, 0.166559, 0.168517, 0.169393, 0.175272, 0.233167, 0.238787, 0.266562,
                0.26903, 0.285792, 0.292801, 0.293429, 0.301706, 0.308534, 0.331489, 0.346095,
                0.367644, 0.369939, 0.372074, 0.41014, 0.415789, 0.421781, 0.427725, 0.465363,
                0.500208, 2.621727, 2.803311, 3.868526,
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
            Some(3.0)
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
            Some(2.0)
        );
        assert_eq!(
            ca.quantile(0.9, QuantileInterpolOptions::Nearest).unwrap(),
            Some(6.0)
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
    }
}
