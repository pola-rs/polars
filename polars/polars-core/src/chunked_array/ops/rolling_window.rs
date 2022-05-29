#[derive(Clone)]
pub struct RollingOptions {
    /// The length of the window.
    pub window_size: usize,
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
            window_size: 3,
            min_periods: 1,
            weights: None,
            center: false,
        }
    }
}

#[cfg(feature = "rolling_window")]
mod inner_mod {
    use crate::prelude::*;
    use arrow::array::{Array, PrimitiveArray};
    use arrow::bitmap::MutableBitmap;
    use num::{Float, Zero};
    use polars_arrow::bit_util::unset_bit_raw;
    use polars_arrow::data_types::IsFloat;
    use polars_arrow::prelude::QuantileInterpolOptions;
    use polars_arrow::{kernels::rolling, trusted_len::PushUnchecked};
    use std::convert::TryFrom;
    use std::ops::SubAssign;

    fn rolling_agg<T, F1, F2>(
        ca: &ChunkedArray<T>,
        options: RollingOptions,
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

    macro_rules! rolling_impl {
        ($ca:ty) => {
            impl $ca {
                /// Apply a rolling mean (moving mean) over the values in this array.
                /// A window of length `window_size` will traverse the array. The values that fill this window
                /// will (optionally) be multiplied with the weights given by the `weights` vector. The resulting
                /// values will be aggregated to their mean.
                pub fn rolling_mean(&self, options: RollingOptions) -> Result<Series> {
                    rolling_agg(
                        self,
                        options,
                        rolling::no_nulls::rolling_mean,
                        rolling::nulls::rolling_mean,
                    )
                }

                /// Apply a rolling sum (moving sum) over the values in this array.
                /// A window of length `window_size` will traverse the array. The values that fill this window
                /// will (optionally) be multiplied with the weights given by the `weights` vector. The resulting
                /// values will be aggregated to their sum.
                pub fn rolling_sum(&self, options: RollingOptions) -> Result<Series> {
                    rolling_agg(
                        self,
                        options,
                        rolling::no_nulls::rolling_sum,
                        rolling::nulls::rolling_sum,
                    )
                }

                /// Apply a rolling min (moving min) over the values in this array.
                /// A window of length `window_size` will traverse the array. The values that fill this window
                /// will (optionally) be multiplied with the weights given by the `weights` vector. The resulting
                /// values will be aggregated to their min.
                pub fn rolling_min(&self, options: RollingOptions) -> Result<Series> {
                    rolling_agg(
                        self,
                        options,
                        rolling::no_nulls::rolling_min,
                        rolling::nulls::rolling_min,
                    )
                }

                /// Apply a rolling max (moving max) over the values in this array.
                /// A window of length `window_size` will traverse the array. The values that fill this window
                /// will (optionally) be multiplied with the weights given by the `weights` vector. The resulting
                /// values will be aggregated to their max.
                pub fn rolling_max(&self, options: RollingOptions) -> Result<Series> {
                    rolling_agg(
                        self,
                        options,
                        rolling::no_nulls::rolling_max,
                        rolling::nulls::rolling_max,
                    )
                }

                /// Apply a rolling median (moving median) over the values in this array.
                /// A window of length `window_size` will traverse the array. The values that fill this window
                /// will (optionally) be weighted according to the `weights` vector.
                pub fn rolling_median(&self, options: RollingOptions) -> Result<Series> {
                    rolling_agg(
                        self,
                        options,
                        rolling::no_nulls::rolling_median,
                        rolling::nulls::rolling_median,
                    )
                }

                /// Apply a rolling quantile (moving quantile) over the values in this array.
                /// A window of length `window_size` will traverse the array. The values that fill this window
                /// will (optionally) be weighted according to the `weights` vector.
                pub fn rolling_quantile(
                    &self,
                    quantile: f64,
                    interpolation: QuantileInterpolOptions,
                    options: RollingOptions,
                ) -> Result<Series> {
                    check_input(options.window_size, options.min_periods)?;
                    let ca = self.rechunk();

                    let arr = ca.downcast_iter().next().unwrap();
                    let arr = match self.has_validity() {
                        false => rolling::no_nulls::rolling_quantile(
                            arr.values(),
                            quantile,
                            interpolation,
                            options.window_size,
                            options.min_periods,
                            options.center,
                            options.weights.as_deref(),
                        ),
                        _ => rolling::nulls::rolling_quantile(
                            arr,
                            quantile,
                            interpolation,
                            options.window_size,
                            options.min_periods,
                            options.center,
                            options.weights.as_deref(),
                        ),
                    };
                    Series::try_from((self.name(), arr))
                }
            }
        };
    }

    rolling_impl!(Float32Chunked);
    rolling_impl!(Float64Chunked);

    impl<T> ChunkedArray<T>
    where
        T: PolarsIntegerType,
        T::Native: IsFloat + SubAssign,
    {
        /// Apply a rolling sum (moving sum) over the values in this array.
        /// A window of length `window_size` will traverse the array. The values that fill this window
        /// will (optionally) be multiplied with the weights given by the `weights` vector. The resulting
        /// values will be aggregated to their sum.
        pub fn rolling_sum(&self, options: RollingOptions) -> Result<Series> {
            if options.weights.is_some() {
                return self.cast(&DataType::Float64)?.rolling_sum(options);
            }
            rolling_agg(
                self,
                options,
                rolling::no_nulls::rolling_sum,
                rolling::nulls::rolling_sum,
            )
        }

        /// Apply a rolling median (moving median) over the values in this array.
        /// A window of length `window_size` will traverse the array. The values that fill this window
        /// will (optionally) be weighted according to the `weights` vector.
        pub fn rolling_median(&self, options: RollingOptions) -> Result<Series> {
            self.cast(&DataType::Float64)?.rolling_median(options)
        }

        /// Apply a rolling quantile (moving quantile) over the values in this array.
        /// A window of length `window_size` will traverse the array. The values that fill this window
        /// will (optionally) be weighted according to the `weights` vector.
        pub fn rolling_quantile(
            &self,
            quantile: f64,
            interpolation: QuantileInterpolOptions,
            options: RollingOptions,
        ) -> Result<Series> {
            self.cast(&DataType::Float64)?
                .rolling_quantile(quantile, interpolation, options)
        }

        /// Apply a rolling min (moving min) over the values in this array.
        /// A window of length `window_size` will traverse the array. The values that fill this window
        /// will (optionally) be multiplied with the weights given by the `weights` vector. The resulting
        /// values will be aggregated to their min.
        pub fn rolling_min(&self, options: RollingOptions) -> Result<Series> {
            if options.weights.is_some() {
                return self.cast(&DataType::Float64)?.rolling_min(options);
            }
            rolling_agg(
                self,
                options,
                rolling::no_nulls::rolling_min,
                rolling::nulls::rolling_min,
            )
        }

        /// Apply a rolling max (moving max) over the values in this array.
        /// A window of length `window_size` will traverse the array. The values that fill this window
        /// will (optionally) be multiplied with the weights given by the `weights` vector. The resulting
        /// values will be aggregated to their max.
        pub fn rolling_max(&self, options: RollingOptions) -> Result<Series> {
            if options.weights.is_some() {
                return self.cast(&DataType::Float64)?.rolling_max(options);
            }
            rolling_agg(
                self,
                options,
                rolling::no_nulls::rolling_max,
                rolling::nulls::rolling_max,
            )
        }
    }

    /// utility
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

    impl<T> ChunkRollApply for ChunkedArray<T>
    where
        T: PolarsNumericType,
        Self: IntoSeries,
    {
        /// Apply a rolling custom function. This is pretty slow because of dynamic dispatch.
        fn rolling_apply(
            &self,
            f: &dyn Fn(&Series) -> Series,
            options: RollingOptions,
        ) -> Result<Series> {
            check_input(options.window_size, options.min_periods)?;

            let ca = self.rechunk();
            if options.weights.is_some()
                && !matches!(self.dtype(), DataType::Float64 | DataType::Float32)
            {
                let s = self.cast(&DataType::Float64)?;
                return s.rolling_apply(f, options);
            }

            if options.window_size >= self.len() {
                return Ok(Self::full_null(self.name(), self.len()).into_series());
            }

            let len = self.len();
            let arr = ca.downcast_iter().next().unwrap();
            let series_container =
                ChunkedArray::<T>::from_slice("", &[T::Native::zero()]).into_series();
            let array_ptr = &series_container.array_ref(0);
            let ptr = Arc::as_ptr(array_ptr) as *mut dyn Array as *mut PrimitiveArray<T::Native>;
            let mut builder = PrimitiveChunkedBuilder::<T>::new(self.name(), self.len());

            if let Some(weights) = options.weights {
                let weights_series = Float64Chunked::new("weights", &weights).into_series();

                let weights_series = weights_series.cast(self.dtype()).unwrap();

                for idx in 0..len {
                    let (start, size) = window_edges(idx, len, options.window_size, options.center);

                    if size < options.min_periods {
                        builder.append_null();
                    } else {
                        let arr_window = arr.slice(start, size);

                        // Safety.
                        // ptr is not dropped as we are in scope
                        // We are also the only owner of the contents of the Arc
                        // we do this to reduce heap allocs.
                        unsafe {
                            *ptr = arr_window;
                        }

                        let s = if size == options.window_size {
                            f(&series_container.multiply(&weights_series).unwrap())
                        } else {
                            let weights_cutoff: Series = match self.dtype() {
                                DataType::Float64 => weights_series
                                    .f64()
                                    .unwrap()
                                    .into_iter()
                                    .take(series_container.len())
                                    .collect(),
                                _ => weights_series // Float32 case
                                    .f32()
                                    .unwrap()
                                    .into_iter()
                                    .take(series_container.len())
                                    .collect(),
                            };
                            f(&series_container.multiply(&weights_cutoff).unwrap())
                        };

                        let out = self.unpack_series_matching_type(&s)?;
                        builder.append_option(out.get(0));
                    }
                }

                Ok(builder.finish().into_series())
            } else {
                for idx in 0..len {
                    let (start, size) = window_edges(idx, len, options.window_size, options.center);

                    if size < options.min_periods {
                        builder.append_null();
                    } else {
                        let arr_window = arr.slice(start, size);

                        // Safety.
                        // ptr is not dropped as we are in scope
                        // We are also the only owner of the contents of the Arc
                        // we do this to reduce heap allocs.
                        unsafe {
                            *ptr = arr_window;
                        }

                        let s = f(&series_container);
                        let out = self.unpack_series_matching_type(&s)?;
                        builder.append_option(out.get(0));
                    }
                }

                Ok(builder.finish().into_series())
            }
        }
    }

    impl<T> ChunkedArray<T>
    where
        ChunkedArray<T>: IntoSeries,
        T: PolarsFloatType,
        T::Native: Float + IsFloat + SubAssign,
    {
        /// Apply a rolling custom function. This is pretty slow because of dynamic dispatch.
        pub fn rolling_apply_float<F>(&self, window_size: usize, f: F) -> Result<Self>
        where
            F: Fn(&ChunkedArray<T>) -> Option<T::Native>,
        {
            if window_size >= self.len() {
                return Ok(Self::full_null(self.name(), self.len()));
            }
            let ca = self.rechunk();
            let arr = ca.downcast_iter().next().unwrap();

            let arr_container = ChunkedArray::<T>::from_slice("", &[T::Native::zero()]);
            let array_ptr = &arr_container.chunks()[0];
            let ptr = Arc::as_ptr(array_ptr) as *mut dyn Array as *mut PrimitiveArray<T::Native>;

            let mut validity = MutableBitmap::with_capacity(ca.len());
            validity.extend_constant(window_size - 1, false);
            validity.extend_constant(ca.len() - (window_size - 1), true);
            let validity_ptr = validity.as_slice().as_ptr() as *mut u8;

            let mut values = Vec::with_capacity(ca.len());
            values.extend(std::iter::repeat(T::Native::default()).take(window_size - 1));

            for offset in 0..self.len() + 1 - window_size {
                let arr_window = arr.slice(offset, window_size);

                // Safety.
                // ptr is not dropped as we are in scope
                // We are also the only owner of the contents of the Arc
                // we do this to reduce heap allocs.
                unsafe {
                    *ptr = arr_window;
                }

                let out = f(&arr_container);
                match out {
                    Some(v) => unsafe { values.push_unchecked(v) },
                    None => unsafe { unset_bit_raw(validity_ptr, offset + window_size - 1) },
                }
            }
            let arr = PrimitiveArray::from_data(
                T::get_dtype().to_arrow(),
                values.into(),
                Some(validity.into()),
            );
            Ok(Self::from_chunks(self.name(), vec![Arc::new(arr)]))
        }

        /// Apply a rolling var (moving var) over the values in this array.
        /// A window of length `window_size` will traverse the array. The values that fill this window
        /// will (optionally) be multiplied with the weights given by the `weights` vector. The resulting
        /// values will be aggregated to their var.
        pub fn rolling_var(&self, options: RollingOptions) -> Result<Series> {
            check_input(options.window_size, options.min_periods)?;
            let ca = self.rechunk();
            if options.weights.is_some()
                && !matches!(self.dtype(), DataType::Float64 | DataType::Float32)
            {
                let s = ca.cast(&DataType::Float64).unwrap();
                return s.f64().unwrap().rolling_var(options);
            }

            let arr = ca.downcast_iter().next().unwrap();
            let arr = match self.has_validity() {
                false => rolling::no_nulls::rolling_var(
                    arr.values(),
                    options.window_size,
                    options.min_periods,
                    options.center,
                    options.weights.as_deref(),
                ),
                _ => rolling::nulls::rolling_var(
                    arr,
                    options.window_size,
                    options.min_periods,
                    options.center,
                    options.weights.as_deref(),
                ),
            };
            Series::try_from((self.name(), arr))
        }
        /// Apply a rolling std (moving std) over the values in this array.
        /// A window of length `window_size` will traverse the array. The values that fill this window
        /// will (optionally) be multiplied with the weights given by the `weights` vector. The resulting
        /// values will be aggregated to their std.
        pub fn rolling_std(&self, options: RollingOptions) -> Result<Series> {
            let s = self.rolling_var(options)?;
            // Safety:
            // We are still guarded by the type system.
            let out = match self.dtype() {
                DataType::Float32 => s.f32().unwrap().pow_f32(0.5).into_series(),
                _ => s.f64().unwrap().pow_f64(0.5).into_series(), //Float64 case
            };
            Ok(out)
        }
    }
}

#[cfg(feature = "rolling_window")]
pub use inner_mod::*;

#[cfg(all(test, feature = "rolling_window"))]
mod test {
    use crate::prelude::*;

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
