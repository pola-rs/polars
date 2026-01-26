use arrow::bitmap::Bitmap;
use arrow::types::NativeType;
use polars_utils::IdxSize;
use polars_utils::min_max::MinMaxPolicy;

use super::RollingFnParams;
use super::arg_min_max::ArgMinMaxWindow;
use super::no_nulls::RollingAggWindowNoNulls;
use super::nulls::RollingAggWindowNulls;

/// Min/max window implemented on top of ArgMinMaxWindow (arg-based).
pub struct MinMaxWindow<'a, T, P> {
    inner: ArgMinMaxWindow<'a, T, P>,
    current_agg: Option<T>,
}

impl<'a, T: NativeType, P: MinMaxPolicy> RollingAggWindowNulls<T> for MinMaxWindow<'a, T, P> {
    type This<'b> = MinMaxWindow<'b, T, P>;

    fn new<'b>(
        slice: &'b [T],
        validity: &'b Bitmap,
        start: usize,
        end: usize,
        params: Option<RollingFnParams>,
        window_size: Option<usize>,
    ) -> Self::This<'b> {
        assert!(params.is_none());
        assert!(start <= slice.len() && end <= slice.len() && start <= end);

        let inner = <ArgMinMaxWindow<'b, T, P> as RollingAggWindowNulls<T, IdxSize>>::new(
            slice,
            validity,
            start,
            end,
            None,
            window_size,
        );

        MinMaxWindow {
            inner,
            current_agg: None,
        }
    }

    unsafe fn update(&mut self, start: usize, end: usize) {
        unsafe { RollingAggWindowNulls::<T, IdxSize>::update(&mut self.inner, start, end) };
        let rel = RollingAggWindowNulls::<T, IdxSize>::get_agg(&self.inner, start);
        self.current_agg = if let Some(rel) = rel {
            let abs = start + rel as usize;
            unsafe { Some(*self.inner.values.get_unchecked(abs)) }
        } else {
            None
        };
    }

    fn get_agg(&self, _idx: usize) -> Option<T> {
        self.current_agg
    }

    fn is_valid(&self, min_periods: usize) -> bool {
        RollingAggWindowNulls::<T, IdxSize>::is_valid(&self.inner, min_periods)
    }

    fn slice_len(&self) -> usize {
        RollingAggWindowNulls::<T, IdxSize>::slice_len(&self.inner)
    }
}

impl<'a, T: NativeType, P: MinMaxPolicy> RollingAggWindowNoNulls<T> for MinMaxWindow<'a, T, P> {
    type This<'b> = MinMaxWindow<'b, T, P>;

    fn new<'b>(
        slice: &'b [T],
        start: usize,
        end: usize,
        params: Option<RollingFnParams>,
        window_size: Option<usize>,
    ) -> Self::This<'b> {
        assert!(params.is_none());
        assert!(start <= slice.len() && end <= slice.len() && start <= end);

        let inner = <ArgMinMaxWindow<'b, T, P> as RollingAggWindowNoNulls<T, IdxSize>>::new(
            slice,
            start,
            end,
            None,
            window_size,
        );

        MinMaxWindow {
            inner,
            current_agg: None,
        }
    }

    unsafe fn update(&mut self, start: usize, end: usize) {
        unsafe {
            RollingAggWindowNoNulls::<T, IdxSize>::update(&mut self.inner, start, end);
        };
        let rel = RollingAggWindowNoNulls::<T, IdxSize>::get_agg(&self.inner, start);
        self.current_agg = if let Some(rel) = rel {
            let abs = start + rel as usize;
            unsafe { Some(*self.inner.values.get_unchecked(abs)) }
        } else {
            None
        };
    }

    fn get_agg(&self, _idx: usize) -> Option<T> {
        self.current_agg
    }

    fn slice_len(&self) -> usize {
        RollingAggWindowNoNulls::<T, IdxSize>::slice_len(&self.inner)
    }
}
