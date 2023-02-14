#[derive(Clone)]
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
}

impl Default for RollingOptionsFixedWindow {
    fn default() -> Self {
        RollingOptionsFixedWindow {
            window_size: 3,
            min_periods: 1,
            weights: None,
            center: false,
        }
    }
}

#[cfg(feature = "rolling_window")]
mod inner_mod {
    use std::ops::SubAssign;

    use arrow::array::{Array, PrimitiveArray};
    use arrow::bitmap::MutableBitmap;
    use num::{Float, Zero};
    use polars_arrow::bit_util::unset_bit_raw;
    use polars_arrow::data_types::IsFloat;
    use polars_arrow::trusted_len::PushUnchecked;

    use crate::prelude::*;

    /// utility
    fn check_input(window_size: usize, min_periods: usize) -> PolarsResult<()> {
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
            options: RollingOptionsFixedWindow,
        ) -> PolarsResult<Series> {
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
            let mut series_container =
                ChunkedArray::<T>::from_slice("", &[T::Native::zero()]).into_series();
            let array_ptr = series_container.array_ref(0);
            let ptr = array_ptr.as_ref() as *const dyn Array as *mut dyn Array
                as *mut PrimitiveArray<T::Native>;
            let mut builder = PrimitiveChunkedBuilder::<T>::new(self.name(), self.len());

            if let Some(weights) = options.weights {
                let weights_series = Float64Chunked::new("weights", &weights).into_series();

                let weights_series = weights_series.cast(self.dtype()).unwrap();

                for idx in 0..len {
                    let (start, size) = window_edges(idx, len, options.window_size, options.center);

                    if size < options.min_periods {
                        builder.append_null();
                    } else {
                        // safety:
                        // we are in bounds
                        let arr_window = unsafe { arr.slice_typed_unchecked(start, size) };

                        // Safety.
                        // ptr is not dropped as we are in scope
                        // We are also the only owner of the contents of the Arc
                        // we do this to reduce heap allocs.
                        unsafe {
                            *ptr = arr_window;
                        }
                        // ensure the length is correct
                        series_container._get_inner_mut().compute_len();

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
                        // safety:
                        // we are in bounds
                        let arr_window = unsafe { arr.slice_typed_unchecked(start, size) };

                        // Safety.
                        // ptr is not dropped as we are in scope
                        // We are also the only owner of the contents of the Arc
                        // we do this to reduce heap allocs.
                        unsafe {
                            *ptr = arr_window;
                        }
                        // ensure the length is correct
                        series_container._get_inner_mut().compute_len();

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
        T::Native: Float + IsFloat + SubAssign + num::pow::Pow<T::Native, Output = T::Native>,
    {
        /// Apply a rolling custom function. This is pretty slow because of dynamic dispatch.
        pub fn rolling_apply_float<F>(&self, window_size: usize, mut f: F) -> PolarsResult<Self>
        where
            F: FnMut(&mut ChunkedArray<T>) -> Option<T::Native>,
        {
            if window_size > self.len() {
                return Ok(Self::full_null(self.name(), self.len()));
            }
            let ca = self.rechunk();
            let arr = ca.downcast_iter().next().unwrap();

            // we create a temporary dummy ChunkedArray
            // this will be a container where we swap the window contents every iteration
            // doing so will save a lot of heap allocations.
            let mut heap_container = ChunkedArray::<T>::from_slice("", &[T::Native::zero()]);
            let array_ptr = &heap_container.chunks()[0];
            let ptr = array_ptr.as_ref() as *const dyn Array as *mut dyn Array
                as *mut PrimitiveArray<T::Native>;

            let mut validity = MutableBitmap::with_capacity(ca.len());
            validity.extend_constant(window_size - 1, false);
            validity.extend_constant(ca.len() - (window_size - 1), true);
            let validity_ptr = validity.as_slice().as_ptr() as *mut u8;

            let mut values = Vec::with_capacity(ca.len());
            values.extend(std::iter::repeat(T::Native::default()).take(window_size - 1));

            for offset in 0..self.len() + 1 - window_size {
                debug_assert!(offset + window_size <= arr.len());
                let arr_window = unsafe { arr.slice_typed_unchecked(offset, window_size) };
                // the lengths are cached, so we must update them
                heap_container.length = arr_window.len() as IdxSize;

                // Safety.
                // ptr is not dropped as we are in scope
                // We are also the only owner of the contents of the Arc
                // we do this to reduce heap allocs.
                unsafe {
                    *ptr = arr_window;
                }

                let out = f(&mut heap_container);
                match out {
                    Some(v) => {
                        // Safety: we have pre-allocated
                        unsafe { values.push_unchecked(v) }
                    }
                    None => {
                        // safety: we allocated enough for both the `values` vec
                        // and the `validity_ptr`
                        unsafe {
                            values.push_unchecked(T::Native::default());
                            unset_bit_raw(validity_ptr, offset + window_size - 1);
                        }
                    }
                }
            }
            let arr = PrimitiveArray::new(
                T::get_dtype().to_arrow(),
                values.into(),
                Some(validity.into()),
            );
            unsafe { Ok(Self::from_chunks(self.name(), vec![Box::new(arr)])) }
        }
    }
}

#[cfg(feature = "rolling_window")]
pub use inner_mod::*;
