#[cfg(feature = "rolling_window")]
mod floats;
#[cfg(feature = "rolling_window")]
mod ints;

use crate::prelude::*;
use polars_arrow::export::arrow;
use polars_core::prelude::*;
use arrow::array::{Array, PrimitiveArray, ArrayRef};
use arrow::bitmap::MutableBitmap;
use polars_core::export::num::{Float, Zero};
use polars_arrow::bit_util::unset_bit_raw;
use polars_arrow::data_types::IsFloat;
use polars_arrow::prelude::QuantileInterpolOptions;
use polars_arrow::{kernels::rolling, trusted_len::PushUnchecked};
use std::convert::TryFrom;
use std::ops::SubAssign;
use crate::series::WrapFloat;

#[derive(Clone)]
pub struct RollingOptions {
    /// The length of the window.
    pub window_size: Duration,
    /// Amount of elements in the window that should be filled before computing a result.
    pub min_periods: usize,
    /// An optional slice with the same length as the window that will be multiplied
    ///              elementwise with the values in the window.
    pub weights: Option<Vec<f64>>,
    /// Set the labels at the center of the window.
    pub center: bool,
}

impl Default for RollingOptions {
    fn default() -> Self {
        RollingOptions {
            window_size: Duration::parse("3i"),
            min_periods: 1,
            weights: None,
            center: false,
        }
    }
}

#[cfg(feature = "rolling_window")]
impl Into<RollingOptionsFixedWindow> for RollingOptions {
    fn into(self) -> RollingOptionsFixedWindow {
        let options = self;
        let window_size = options.window_size;
        assert!(window_size.parsed_int, "should be fixed integer window size at this point");

        RollingOptionsFixedWindow {
            window_size: window_size.nanoseconds() as usize,
            min_periods: options.min_periods,
            weights: options.weights,
            center: options.center
        }
    }
}


#[cfg(feature = "rolling_window")]
pub trait RollingAgg {
    /// Apply a rolling mean (moving mean) over the values in this array.
    /// A window of length `window_size` will traverse the array. The values that fill this window
    /// will (optionally) be multiplied with the weights given by the `weights` vector. The resulting
    /// values will be aggregated to their mean.
    fn rolling_mean(&self, options: RollingOptions) -> Result<Series>;

    /// Apply a rolling sum (moving sum) over the values in this array.
    /// A window of length `window_size` will traverse the array. The values that fill this window
    /// will (optionally) be multiplied with the weights given by the `weights` vector. The resulting
    /// values will be aggregated to their sum.
    fn rolling_sum(&self, options: RollingOptions) -> Result<Series>;


    /// Apply a rolling min (moving min) over the values in this array.
    /// A window of length `window_size` will traverse the array. The values that fill this window
    /// will (optionally) be multiplied with the weights given by the `weights` vector. The resulting
    /// values will be aggregated to their min.
    fn rolling_min(&self, options: RollingOptions) -> Result<Series>;

    /// Apply a rolling max (moving max) over the values in this array.
    /// A window of length `window_size` will traverse the array. The values that fill this window
    /// will (optionally) be multiplied with the weights given by the `weights` vector. The resulting
    /// values will be aggregated to their max.
    fn rolling_max(&self, options: RollingOptions) -> Result<Series>;

    /// Apply a rolling median (moving median) over the values in this array.
    /// A window of length `window_size` will traverse the array. The values that fill this window
    /// will (optionally) be weighted according to the `weights` vector.
    fn rolling_median(&self, options: RollingOptions) -> Result<Series>;

    /// Apply a rolling quantile (moving quantile) over the values in this array.
    /// A window of length `window_size` will traverse the array. The values that fill this window
    /// will (optionally) be weighted according to the `weights` vector.
    fn rolling_quantile(
        &self,
        quantile: f64,
        interpolation: QuantileInterpolOptions,
        options: RollingOptions,
    ) -> Result<Series>;

    /// Apply a rolling var (moving var) over the values in this array.
    /// A window of length `window_size` will traverse the array. The values that fill this window
    /// will (optionally) be multiplied with the weights given by the `weights` vector. The resulting
    /// values will be aggregated to their var.
    fn rolling_var(&self, options: RollingOptions) -> Result<Series>;

    /// Apply a rolling std (moving std) over the values in this array.
    /// A window of length `window_size` will traverse the array. The values that fill this window
    /// will (optionally) be multiplied with the weights given by the `weights` vector. The resulting
    /// values will be aggregated to their std.
    fn rolling_std(&self, options: RollingOptions) -> Result<Series>;
}

    /// utility
#[cfg(feature = "rolling_window")]
fn check_input(window_size: usize, min_periods: usize) -> Result<()> {
    if min_periods > window_size {
        Err(PolarsError::ComputeError(
            "`windows_size` should be >= `min_periods`".into(),
        ))
    } else {
        Ok(())
    }
}

/// utility
#[cfg(feature = "rolling_window")]
fn window_edges(idx: usize, len: usize, window_size: usize, center: bool) -> (usize, usize) {
    let (start, end) = if center {
        let right_window = (window_size + 1) / 2;
        (
            idx.saturating_sub(window_size - right_window),
            std::cmp::min(len, idx + right_window),
        )
    } else {
        (idx.saturating_sub(window_size - 1), idx + 1)
    };

    (start, end - start)
}

#[cfg(feature = "rolling_window")]
fn rolling_agg<T, F1, F2>(
    ca: &ChunkedArray<T>,
    options: RollingOptionsFixedWindow,
    rolling_agg_fn: F1,
    rolling_agg_fn_nulls: F2,
) -> Result<Series>
    where
        T: PolarsNumericType,
        F1: FnOnce(&[T::Native], usize, usize, bool, Option<&[f64]>) -> ArrayRef,
        F2: FnOnce(&PrimitiveArray<T::Native>, usize, usize, bool, Option<&[f64]>) -> ArrayRef,
{
    check_input(options.window_size, options.min_periods)?;
    let ca = ca.rechunk();

    let arr = ca.downcast_iter().next().unwrap();
    let arr = match ca.has_validity() {
        false => rolling_agg_fn(
            arr.values().as_slice(),
            options.window_size,
            options.min_periods,
            options.center,
            options.weights.as_deref(),
        ),
        _ => rolling_agg_fn_nulls(
            arr,
            options.window_size,
            options.min_periods,
            options.center,
            options.weights.as_deref(),
        ),
    };
    Series::try_from((ca.name(), arr))
}


#[cfg(all(test, feature = "rolling_window"))]
mod test {
    use crate::prelude::*;
    use polars_core::prelude::*;

    #[test]
    fn test_rolling() {
        let ca = Int32Chunked::new("foo", &[1, 2, 3, 2, 1]);
        let a = ca
            .rolling_sum(RollingOptions {
                window_size: 2,
                min_periods: 1,
                ..Default::default()
            })
            .unwrap();
        let a = a.i32().unwrap();
        assert_eq!(
            Vec::from(a),
            [1, 3, 5, 5, 3]
                .iter()
                .copied()
                .map(Some)
                .collect::<Vec<_>>()
        );
        let a = ca
            .rolling_min(RollingOptions {
                window_size: 2,
                min_periods: 1,
                ..Default::default()
            })
            .unwrap();
        let a = a.i32().unwrap();
        assert_eq!(
            Vec::from(a),
            [1, 1, 2, 2, 1]
                .iter()
                .copied()
                .map(Some)
                .collect::<Vec<_>>()
        );
        let a = ca
            .rolling_max(RollingOptions {
                window_size: 2,
                weights: Some(vec![1., 1.]),
                min_periods: 1,
                center: false,
            })
            .unwrap();

        let a = a.f64().unwrap();
        assert_eq!(
            Vec::from(a),
            [1., 2., 3., 3., 2.]
                .iter()
                .copied()
                .map(Some)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_rolling_min_periods() {
        let ca = Int32Chunked::from_slice("foo", &[1, 2, 3, 2, 1]);
        let a = ca
            .rolling_max(RollingOptions {
                window_size: 2,
                min_periods: 2,
                ..Default::default()
            })
            .unwrap();
        let a = a.i32().unwrap();
        assert_eq!(Vec::from(a), &[None, Some(2), Some(3), Some(3), Some(2)]);
    }

    #[test]
    fn test_rolling_mean() {
        let ca = Float64Chunked::new(
            "foo",
            &[
                Some(0.0),
                Some(1.0),
                Some(2.0),
                None,
                None,
                Some(5.0),
                Some(6.0),
            ],
        );

        // check err on wrong input
        assert!(ca
            .rolling_mean(RollingOptions {
                window_size: 1,
                min_periods: 2,
                ..Default::default()
            })
            .is_err());

        // validate that we divide by the proper window length. (same as pandas)
        let a = ca
            .rolling_mean(RollingOptions {
                window_size: 3,
                min_periods: 1,
                center: false,
                weights: None,
            })
            .unwrap();
        let a = a.f64().unwrap();
        assert_eq!(
            Vec::from(a),
            &[
                Some(0.0),
                Some(0.5),
                Some(1.0),
                Some(1.5),
                Some(2.0),
                Some(5.0),
                Some(5.5)
            ]
        );

        // check centered rolling window
        let a = ca
            .rolling_mean(RollingOptions {
                window_size: 3,
                min_periods: 1,
                center: true,
                weights: None,
            })
            .unwrap();
        let a = a.f64().unwrap();
        assert_eq!(
            Vec::from(a),
            &[
                Some(0.5),
                Some(1.0),
                Some(1.5),
                Some(2.0),
                Some(5.0),
                Some(5.5),
                Some(5.5)
            ]
        );

        // integers
        let ca = Int32Chunked::from_slice("", &[1, 8, 6, 2, 16, 10]);
        let out = ca
            .into_series()
            .rolling_mean(RollingOptions {
                window_size: 2,
                weights: None,
                min_periods: 2,
                center: false,
            })
            .unwrap();

        let out = out.f64().unwrap();
        assert_eq!(
            Vec::from(out),
            &[None, Some(4.5), Some(7.0), Some(4.0), Some(9.0), Some(13.0),]
        );
    }

    #[test]
    fn test_rolling_apply() {
        let ca = Float64Chunked::new(
            "foo",
            &[
                Some(0.0),
                Some(1.0),
                Some(2.0),
                None,
                None,
                Some(5.0),
                Some(6.0),
            ],
        );

        let out = ca
            .rolling_apply(
                &|s| s.sum_as_series(),
                RollingOptions {
                    window_size: 3,
                    min_periods: 3,
                    ..Default::default()
                },
            )
            .unwrap();

        let out = out.f64().unwrap();

        assert_eq!(
            Vec::from(out),
            &[
                None,
                None,
                Some(3.0),
                Some(3.0),
                Some(2.0),
                Some(5.0),
                Some(11.0)
            ]
        );
    }

    #[test]
    fn test_rolling_var() {
        let ca = Float64Chunked::new(
            "foo",
            &[
                Some(0.0),
                Some(1.0),
                Some(2.0),
                None,
                None,
                Some(5.0),
                Some(6.0),
            ],
        );
        // window larger than array
        assert_eq!(
            ca.rolling_var(RollingOptions {
                window_size: 10,
                min_periods: 10,
                ..Default::default()
            })
                .unwrap()
                .null_count(),
            ca.len()
        );

        let options = RollingOptions {
            window_size: 3,
            min_periods: 3,
            ..Default::default()
        };
        let out = ca
            .rolling_var(options.clone())
            .unwrap()
            .cast(&DataType::Int32)
            .unwrap();
        let out = out.i32().unwrap();
        assert_eq!(
            Vec::from(out),
            &[None, None, Some(1), None, None, None, None,]
        );

        let ca = Float64Chunked::from_slice("", &[0.0, 2.0, 8.0, 3.0, 12.0, 1.0]);
        let out = ca
            .rolling_var(options)
            .unwrap()
            .cast(&DataType::Int32)
            .unwrap();
        let out = out.i32().unwrap();

        assert_eq!(
            Vec::from(out),
            &[None, None, Some(17), Some(10), Some(20), Some(34),]
        );

        // check centered rolling window
        let out = ca
            .rolling_var(RollingOptions {
                window_size: 4,
                min_periods: 3,
                center: true,
                weights: None,
            })
            .unwrap()
            .round(2)
            .unwrap();
        let out = out.f64().unwrap();

        assert_eq!(
            Vec::from(out),
            &[
                None,
                Some(17.33),
                Some(11.58),
                Some(21.58),
                Some(24.67),
                Some(34.33)
            ]
        );
    }

    #[test]
    fn test_median_quantile_types() {
        let ca = Int32Chunked::new("foo", &[1, 2, 3, 2, 1]);
        let rol_med = ca
            .rolling_median(RollingOptions {
                window_size: 2,
                min_periods: 1,
                ..Default::default()
            })
            .unwrap();

        let rol_med_weighted = ca
            .rolling_median(RollingOptions {
                window_size: 2,
                min_periods: 1,
                weights: Some(vec![1.0, 2.0]),
                ..Default::default()
            })
            .unwrap();

        let rol_quantile = ca
            .rolling_quantile(
                0.3,
                QuantileInterpolOptions::Linear,
                RollingOptions {
                    window_size: 2,
                    min_periods: 1,
                    ..Default::default()
                },
            )
            .unwrap();

        let rol_quantile_weighted = ca
            .rolling_quantile(
                0.3,
                QuantileInterpolOptions::Linear,
                RollingOptions {
                    window_size: 2,
                    min_periods: 1,
                    weights: Some(vec![1.0, 2.0]),
                    ..Default::default()
                },
            )
            .unwrap();

        assert_eq!(*rol_med.dtype(), DataType::Float64);
        assert_eq!(*rol_med_weighted.dtype(), DataType::Float64);
        assert_eq!(*rol_quantile.dtype(), DataType::Float64);
        assert_eq!(*rol_quantile_weighted.dtype(), DataType::Float64);

        let ca = Float32Chunked::new("foo", &[1.0, 2.0, 3.0, 2.0, 1.0]);
        let rol_med = ca
            .rolling_median(RollingOptions {
                window_size: 2,
                min_periods: 1,
                ..Default::default()
            })
            .unwrap();

        let rol_med_weighted = ca
            .rolling_median(RollingOptions {
                window_size: 2,
                min_periods: 1,
                weights: Some(vec![1.0, 2.0]),
                ..Default::default()
            })
            .unwrap();

        let rol_quantile = ca
            .rolling_quantile(
                0.3,
                QuantileInterpolOptions::Linear,
                RollingOptions {
                    window_size: 2,
                    min_periods: 1,
                    ..Default::default()
                },
            )
            .unwrap();

        let rol_quantile_weighted = ca
            .rolling_quantile(
                0.3,
                QuantileInterpolOptions::Linear,
                RollingOptions {
                    window_size: 2,
                    min_periods: 1,
                    weights: Some(vec![1.0, 2.0]),
                    ..Default::default()
                },
            )
            .unwrap();

        assert_eq!(*rol_med.dtype(), DataType::Float32);
        assert_eq!(*rol_med_weighted.dtype(), DataType::Float32);
        assert_eq!(*rol_quantile.dtype(), DataType::Float32);
        assert_eq!(*rol_quantile_weighted.dtype(), DataType::Float32);

        let ca = Float64Chunked::new("foo", &[1.0, 2.0, 3.0, 2.0, 1.0]);
        let rol_med = ca
            .rolling_median(RollingOptions {
                window_size: 2,
                min_periods: 1,
                ..Default::default()
            })
            .unwrap();

        let rol_med_weighted = ca
            .rolling_median(RollingOptions {
                window_size: 2,
                min_periods: 1,
                weights: Some(vec![1.0, 2.0]),
                ..Default::default()
            })
            .unwrap();

        let rol_quantile = ca
            .rolling_quantile(
                0.3,
                QuantileInterpolOptions::Linear,
                RollingOptions {
                    window_size: 2,
                    min_periods: 1,
                    ..Default::default()
                },
            )
            .unwrap();

        let rol_quantile_weighted = ca
            .rolling_quantile(
                0.3,
                QuantileInterpolOptions::Linear,
                RollingOptions {
                    window_size: 2,
                    min_periods: 1,
                    weights: Some(vec![1.0, 2.0]),
                    ..Default::default()
                },
            )
            .unwrap();

        assert_eq!(*rol_med.dtype(), DataType::Float64);
        assert_eq!(*rol_med_weighted.dtype(), DataType::Float64);
        assert_eq!(*rol_quantile.dtype(), DataType::Float64);
        assert_eq!(*rol_quantile_weighted.dtype(), DataType::Float64);
    }
}
