use std::hash::{Hash, Hasher};

use polars_compute::rolling::RollingFnParams;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
#[cfg_attr(feature = "rolling_window", derive(PartialEq))]
pub struct RollingOptionsFixedWindow {
    /// The length of the window.
    pub window_size: usize,
    /// Amount of elements in the window that should be filled before computing a result.
    pub min_periods: usize,
    /// An optional slice with the same length as the window that will be multiplied
    ///              elementwise with the values in the window.
    pub weights: Option<Vec<f64>>,
    /// Set the labels at the center of the window.
    pub center: bool,
    /// Optional parameters for the rolling
    #[cfg_attr(any(feature = "serde", feature = "dsl-schema"), serde(default))]
    pub fn_params: Option<RollingFnParams>,
}

impl Hash for RollingOptionsFixedWindow {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.window_size.hash(state);
        self.min_periods.hash(state);
        self.center.hash(state);
        self.weights.is_some().hash(state);
    }
}

impl Default for RollingOptionsFixedWindow {
    fn default() -> Self {
        RollingOptionsFixedWindow {
            window_size: 3,
            min_periods: 1,
            weights: None,
            center: false,
            fn_params: None,
        }
    }
}

#[cfg(feature = "rolling_window")]
mod inner_mod {
    use num_traits::Zero;

    use crate::chunked_array::cast::CastOptions;
    use crate::prelude::*;

    /// utility
    fn check_input(window_size: usize, min_periods: usize) -> PolarsResult<()> {
        polars_ensure!(
            min_periods <= window_size,
            ComputeError: "`window_size`: {} should be >= `min_periods`: {}",
            window_size, min_periods
        );
        Ok(())
    }

    /// utility
    fn window_edges(idx: usize, len: usize, window_size: usize, center: bool) -> (usize, usize) {
        let (start, end) = if center {
            let right_window = window_size.div_ceil(2);
            (
                idx.saturating_sub(window_size - right_window),
                len.min(idx + right_window),
            )
        } else {
            (idx.saturating_sub(window_size - 1), idx + 1)
        };

        (start, end - start)
    }

    impl<T: PolarsNumericType> ChunkRollApply for ChunkedArray<T> {
        /// Apply a rolling custom function. This is pretty slow because of dynamic dispatch.
        fn rolling_map(
            &self,
            f: &dyn Fn(&Series) -> PolarsResult<Series>,
            mut options: RollingOptionsFixedWindow,
        ) -> PolarsResult<Series> {
            check_input(options.window_size, options.min_periods)?;

            let ca = self.rechunk();
            if options.weights.is_some()
                && !matches!(self.dtype(), DataType::Float64 | DataType::Float32)
            {
                let s = self.cast_with_options(&DataType::Float64, CastOptions::NonStrict)?;
                return s.rolling_map(f, options);
            }

            options.window_size = std::cmp::min(self.len(), options.window_size);

            let len = self.len();
            let arr = ca.downcast_as_array();
            let mut ca = ChunkedArray::<T>::from_slice(PlSmallStr::EMPTY, &[T::Native::zero()]);
            let ptr = ca.chunks[0].as_mut() as *mut dyn Array as *mut PrimitiveArray<T::Native>;
            let mut series_container = ca.into_series();

            let mut builder = PrimitiveChunkedBuilder::<T>::new(self.name().clone(), self.len());

            if let Some(weights) = options.weights {
                let weights_series =
                    Float64Chunked::new(PlSmallStr::from_static("weights"), &weights).into_series();

                let weights_series = weights_series.cast(self.dtype()).unwrap();

                for idx in 0..len {
                    let (start, size) = window_edges(idx, len, options.window_size, options.center);

                    if size < options.min_periods {
                        builder.append_null();
                    } else {
                        // SAFETY:
                        // we are in bounds
                        let arr_window = unsafe { arr.slice_typed_unchecked(start, size) };

                        // ensure we still meet window size criteria after removing null values
                        if size - arr_window.null_count() < options.min_periods {
                            builder.append_null();
                            continue;
                        }

                        // SAFETY.
                        // ptr is not dropped as we are in scope
                        // We are also the only owner of the contents of the Arc
                        // we do this to reduce heap allocs.
                        unsafe {
                            *ptr = arr_window;
                        }
                        // reset flags as we reuse this container
                        series_container.clear_flags();
                        // ensure the length is correct
                        series_container._get_inner_mut().compute_len();
                        let s = if size == options.window_size {
                            f(&series_container.multiply(&weights_series).unwrap())?
                        } else {
                            // Determine which side to slice weights from
                            let weights_cutoff: Series = match self.dtype() {
                                DataType::Float64 => {
                                    let ws = weights_series.f64().unwrap();
                                    if start == 0 {
                                        ws.slice(
                                            (ws.len() - series_container.len()) as i64,
                                            series_container.len(),
                                        )
                                        .into_series()
                                    } else {
                                        ws.slice(0, series_container.len()).into_series()
                                    }
                                },
                                _ => {
                                    let ws = weights_series.f32().unwrap();
                                    if start == 0 {
                                        ws.slice(
                                            (ws.len() - series_container.len()) as i64,
                                            series_container.len(),
                                        )
                                        .into_series()
                                    } else {
                                        ws.slice(0, series_container.len()).into_series()
                                    }
                                },
                            };
                            f(&series_container.multiply(&weights_cutoff).unwrap())?
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
                        // SAFETY:
                        // we are in bounds
                        let arr_window = unsafe { arr.slice_typed_unchecked(start, size) };

                        // ensure we still meet window size criteria after removing null values
                        if size - arr_window.null_count() < options.min_periods {
                            builder.append_null();
                            continue;
                        }

                        // SAFETY.
                        // ptr is not dropped as we are in scope
                        // We are also the only owner of the contents of the Arc
                        // we do this to reduce heap allocs.
                        unsafe {
                            *ptr = arr_window;
                        }
                        // reset flags as we reuse this container
                        series_container.clear_flags();
                        // ensure the length is correct
                        series_container._get_inner_mut().compute_len();
                        let s = f(&series_container)?;
                        let out = self.unpack_series_matching_type(&s)?;
                        builder.append_option(out.get(0));
                    }
                }

                Ok(builder.finish().into_series())
            }
        }
    }
}
