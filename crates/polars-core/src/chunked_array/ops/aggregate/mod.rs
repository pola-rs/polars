//! Implementations of the ChunkAgg trait.
mod quantile;
mod var;

use std::ops::Add;

use arrow::compute;
use arrow::types::simd::Simd;
use arrow::types::NativeType;
use num_traits::{Float, One, ToPrimitive, Zero};
use polars_compute::float_sum;
use polars_compute::min_max::MinMaxKernel;
use polars_utils::min_max::MinMax;
pub use quantile::*;
pub use var::*;

use super::float_sorted_arg_max::{
    float_arg_max_sorted_ascending, float_arg_max_sorted_descending,
};
use crate::chunked_array::metadata::MetadataEnv;
use crate::chunked_array::ChunkedArray;
use crate::datatypes::{BooleanChunked, PolarsNumericType};
use crate::prelude::*;
use crate::series::IsSorted;

/// Aggregations that return [`Series`] of unit length. Those can be used in broadcasting operations.
pub trait ChunkAggSeries {
    /// Get the sum of the [`ChunkedArray`] as a new [`Series`] of length 1.
    fn sum_reduce(&self) -> Scalar {
        unimplemented!()
    }
    /// Get the max of the [`ChunkedArray`] as a new [`Series`] of length 1.
    fn max_reduce(&self) -> Scalar {
        unimplemented!()
    }
    /// Get the min of the [`ChunkedArray`] as a new [`Series`] of length 1.
    fn min_reduce(&self) -> Scalar {
        unimplemented!()
    }
    /// Get the product of the [`ChunkedArray`] as a new [`Series`] of length 1.
    fn prod_reduce(&self) -> Scalar {
        unimplemented!()
    }
}

fn sum<T>(array: &PrimitiveArray<T>) -> T
where
    T: NumericNative + NativeType,
    <T as Simd>::Simd: Add<Output = <T as Simd>::Simd> + compute::aggregate::Sum<T>,
{
    if array.null_count() == array.len() {
        return T::default();
    }

    if T::is_float() {
        unsafe {
            if T::is_f32() {
                let f32_arr =
                    std::mem::transmute::<&PrimitiveArray<T>, &PrimitiveArray<f32>>(array);
                let sum = float_sum::sum_arr_as_f32(f32_arr);
                std::mem::transmute_copy::<f32, T>(&sum)
            } else if T::is_f64() {
                let f64_arr =
                    std::mem::transmute::<&PrimitiveArray<T>, &PrimitiveArray<f64>>(array);
                let sum = float_sum::sum_arr_as_f64(f64_arr);
                std::mem::transmute_copy::<f64, T>(&sum)
            } else {
                unreachable!("only supported float types are f32 and f64");
            }
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
        if self.null_count() == self.len() {
            return None;
        }

        // There is at least one non-null value.

        let result = match self.is_sorted_flag() {
            IsSorted::Ascending => {
                let idx = self.first_non_null().unwrap();
                unsafe { self.get_unchecked(idx) }
            },
            IsSorted::Descending => {
                let idx = self.last_non_null().unwrap();
                unsafe { self.get_unchecked(idx) }
            },
            IsSorted::Not => self
                .downcast_iter()
                .filter_map(MinMaxKernel::min_ignore_nan_kernel)
                .reduce(MinMax::min_ignore_nan),
        };

        if MetadataEnv::experimental_enabled() {
            self.interior_mut_metadata().set_min_value(result);
        }

        result
    }

    fn max(&self) -> Option<T::Native> {
        if self.null_count() == self.len() {
            return None;
        }
        // There is at least one non-null value.

        let result = match self.is_sorted_flag() {
            IsSorted::Ascending => {
                let idx = if T::get_dtype().is_float() {
                    float_arg_max_sorted_ascending(self)
                } else {
                    self.last_non_null().unwrap()
                };

                unsafe { self.get_unchecked(idx) }
            },
            IsSorted::Descending => {
                let idx = if T::get_dtype().is_float() {
                    float_arg_max_sorted_descending(self)
                } else {
                    self.first_non_null().unwrap()
                };

                unsafe { self.get_unchecked(idx) }
            },
            IsSorted::Not => self
                .downcast_iter()
                .filter_map(MinMaxKernel::max_ignore_nan_kernel)
                .reduce(MinMax::max_ignore_nan),
        };

        if MetadataEnv::experimental_enabled() {
            self.interior_mut_metadata().set_max_value(result);
        }

        result
    }

    fn min_max(&self) -> Option<(T::Native, T::Native)> {
        if self.null_count() == self.len() {
            return None;
        }
        // There is at least one non-null value.

        let result = match self.is_sorted_flag() {
            IsSorted::Ascending => {
                let min = unsafe { self.get_unchecked(self.first_non_null().unwrap()) };
                let max = {
                    let idx = if T::get_dtype().is_float() {
                        float_arg_max_sorted_ascending(self)
                    } else {
                        self.last_non_null().unwrap()
                    };

                    unsafe { self.get_unchecked(idx) }
                };
                min.zip(max)
            },
            IsSorted::Descending => {
                let min = unsafe { self.get_unchecked(self.last_non_null().unwrap()) };
                let max = {
                    let idx = if T::get_dtype().is_float() {
                        float_arg_max_sorted_descending(self)
                    } else {
                        self.first_non_null().unwrap()
                    };

                    unsafe { self.get_unchecked(idx) }
                };

                min.zip(max)
            },
            IsSorted::Not => self
                .downcast_iter()
                .filter_map(MinMaxKernel::min_max_ignore_nan_kernel)
                .reduce(|(min1, max1), (min2, max2)| {
                    (
                        MinMax::min_ignore_nan(min1, min2),
                        MinMax::max_ignore_nan(max1, max2),
                    )
                }),
        };

        if MetadataEnv::experimental_enabled() {
            let (min, max) = match result {
                Some((min, max)) => (Some(min), Some(max)),
                None => (None, None),
            };

            let mut md = self.interior_mut_metadata();

            md.set_min_value(min);
            md.set_max_value(max);
        }

        result
    }

    fn mean(&self) -> Option<f64> {
        if self.null_count() == self.len() {
            return None;
        }

        let len = (self.len() - self.null_count()) as f64;
        let sum: f64 = self.downcast_iter().map(float_sum::sum_arr_as_f64).sum();
        Some(sum / len)
    }
}

/// Booleans are cast to 1 or 0.
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
    fn sum_reduce(&self) -> Scalar {
        let v: Option<T::Native> = self.sum();
        Scalar::new(T::get_dtype(), v.into())
    }

    fn max_reduce(&self) -> Scalar {
        let v = ChunkAgg::max(self);
        Scalar::new(T::get_dtype(), v.into())
    }

    fn min_reduce(&self) -> Scalar {
        let v = ChunkAgg::min(self);
        Scalar::new(T::get_dtype(), v.into())
    }

    fn prod_reduce(&self) -> Scalar {
        let mut prod = T::Native::one();

        for arr in self.downcast_iter() {
            for v in arr.into_iter().flatten() {
                prod = prod * *v
            }
        }
        Scalar::new(T::get_dtype(), prod.into())
    }
}

impl<T> VarAggSeries for ChunkedArray<T>
where
    T: PolarsIntegerType,
    ChunkedArray<T>: ChunkVar,
{
    fn var_reduce(&self, ddof: u8) -> Scalar {
        let v = self.var(ddof);
        Scalar::new(DataType::Float64, v.into())
    }

    fn std_reduce(&self, ddof: u8) -> Scalar {
        let v = self.std(ddof);
        Scalar::new(DataType::Float64, v.into())
    }
}

impl VarAggSeries for Float32Chunked {
    fn var_reduce(&self, ddof: u8) -> Scalar {
        let v = self.var(ddof).map(|v| v as f32);
        Scalar::new(DataType::Float32, v.into())
    }

    fn std_reduce(&self, ddof: u8) -> Scalar {
        let v = self.std(ddof).map(|v| v as f32);
        Scalar::new(DataType::Float32, v.into())
    }
}

impl VarAggSeries for Float64Chunked {
    fn var_reduce(&self, ddof: u8) -> Scalar {
        let v = self.var(ddof);
        Scalar::new(DataType::Float64, v.into())
    }

    fn std_reduce(&self, ddof: u8) -> Scalar {
        let v = self.std(ddof);
        Scalar::new(DataType::Float64, v.into())
    }
}

impl<T> QuantileAggSeries for ChunkedArray<T>
where
    T: PolarsIntegerType,
    T::Native: Ord,
    <T::Native as Simd>::Simd:
        Add<Output = <T::Native as Simd>::Simd> + compute::aggregate::Sum<T::Native>,
{
    fn quantile_reduce(
        &self,
        quantile: f64,
        interpol: QuantileInterpolOptions,
    ) -> PolarsResult<Scalar> {
        let v = self.quantile(quantile, interpol)?;
        Ok(Scalar::new(DataType::Float64, v.into()))
    }

    fn median_reduce(&self) -> Scalar {
        let v = self.median();
        Scalar::new(DataType::Float64, v.into())
    }
}

impl QuantileAggSeries for Float32Chunked {
    fn quantile_reduce(
        &self,
        quantile: f64,
        interpol: QuantileInterpolOptions,
    ) -> PolarsResult<Scalar> {
        let v = self.quantile(quantile, interpol)?;
        Ok(Scalar::new(DataType::Float32, v.into()))
    }

    fn median_reduce(&self) -> Scalar {
        let v = self.median();
        Scalar::new(DataType::Float32, v.into())
    }
}

impl QuantileAggSeries for Float64Chunked {
    fn quantile_reduce(
        &self,
        quantile: f64,
        interpol: QuantileInterpolOptions,
    ) -> PolarsResult<Scalar> {
        let v = self.quantile(quantile, interpol)?;
        Ok(Scalar::new(DataType::Float64, v.into()))
    }

    fn median_reduce(&self) -> Scalar {
        let v = self.median();
        Scalar::new(DataType::Float64, v.into())
    }
}

impl ChunkAggSeries for BooleanChunked {
    fn sum_reduce(&self) -> Scalar {
        let v = self.sum();
        Scalar::new(IDX_DTYPE, v.into())
    }
    fn max_reduce(&self) -> Scalar {
        let v = self.max();
        Scalar::new(DataType::Boolean, v.into())
    }
    fn min_reduce(&self) -> Scalar {
        let v = self.min();
        Scalar::new(DataType::Boolean, v.into())
    }
}

impl StringChunked {
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

impl ChunkAggSeries for StringChunked {
    fn max_reduce(&self) -> Scalar {
        let av: AnyValue = self.max_str().into();
        Scalar::new(DataType::String, av.into_static().unwrap())
    }
    fn min_reduce(&self) -> Scalar {
        let av: AnyValue = self.min_str().into();
        Scalar::new(DataType::String, av.into_static().unwrap())
    }
}

#[cfg(feature = "dtype-categorical")]
impl CategoricalChunked {
    fn min_categorical(&self) -> Option<&str> {
        if self.is_empty() || self.null_count() == self.len() {
            return None;
        }
        if self.uses_lexical_ordering() {
            // Fast path where all categories are used
            if self._can_fast_unique() {
                self.get_rev_map().get_categories().min_ignore_nan_kernel()
            } else {
                let rev_map = self.get_rev_map();
                // SAFETY:
                // Indices are in bounds
                self.physical()
                    .iter()
                    .flat_map(|opt_el: Option<u32>| {
                        opt_el.map(|el| unsafe { rev_map.get_unchecked(el) })
                    })
                    .min()
            }
        } else {
            // SAFETY:
            // Indices are in bounds
            self.physical()
                .min()
                .map(|el| unsafe { self.get_rev_map().get_unchecked(el) })
        }
    }

    fn max_categorical(&self) -> Option<&str> {
        if self.is_empty() || self.null_count() == self.len() {
            return None;
        }
        if self.uses_lexical_ordering() {
            // Fast path where all categories are used
            if self._can_fast_unique() {
                self.get_rev_map().get_categories().max_ignore_nan_kernel()
            } else {
                let rev_map = self.get_rev_map();
                // SAFETY:
                // Indices are in bounds
                self.physical()
                    .iter()
                    .flat_map(|opt_el: Option<u32>| {
                        opt_el.map(|el| unsafe { rev_map.get_unchecked(el) })
                    })
                    .max()
            }
        } else {
            // SAFETY:
            // Indices are in bounds
            self.physical()
                .max()
                .map(|el| unsafe { self.get_rev_map().get_unchecked(el) })
        }
    }
}

#[cfg(feature = "dtype-categorical")]
impl ChunkAggSeries for CategoricalChunked {
    fn min_reduce(&self) -> Scalar {
        let av: AnyValue = self.min_categorical().into();
        Scalar::new(DataType::String, av.into_static().unwrap())
    }
    fn max_reduce(&self) -> Scalar {
        let av: AnyValue = self.max_categorical().into();
        Scalar::new(DataType::String, av.into_static().unwrap())
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
    fn sum_reduce(&self) -> Scalar {
        unimplemented!()
    }
    fn max_reduce(&self) -> Scalar {
        let av: AnyValue = self.max_binary().into();
        Scalar::new(self.dtype().clone(), av.into_static().unwrap())
    }
    fn min_reduce(&self) -> Scalar {
        let av: AnyValue = self.min_binary().into();
        Scalar::new(self.dtype().clone(), av.into_static().unwrap())
    }
}

#[cfg(feature = "object")]
impl<T: PolarsObject> ChunkAggSeries for ObjectChunked<T> {}

#[cfg(test)]
mod test {
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
                .mean_reduce()
                .value()
                .extract::<f32>()
                .unwrap(),
            1.5
        );
        // all null values case
        let ca = Float32Chunked::full_null("", 3);
        assert_eq!(ca.mean(), None);
        assert_eq!(
            ca.into_series().mean_reduce().value().extract::<f32>(),
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
