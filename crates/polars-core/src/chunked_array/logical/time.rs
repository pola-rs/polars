use arrow::compute::cast::CastOptionsImpl;

use super::*;
use crate::prelude::*;

pub type TimeChunked = Logical<TimeType, Int64Type>;

impl From<Int64Chunked> for TimeChunked {
    fn from(ca: Int64Chunked) -> Self {
        TimeChunked::new_logical(ca)
    }
}

impl Int64Chunked {
    pub fn into_time(mut self) -> TimeChunked {
        let mut null_count = 0;

        let chunks = std::mem::take(&mut self.chunks)
            .into_iter()
            .map(|chunk| {
                let casted = arrow::compute::cast::cast(
                    chunk.as_ref(),
                    &TimeType::get_dtype().to_arrow(CompatLevel::newest()),
                    CastOptionsImpl::default(),
                )
                .unwrap();
                null_count += casted.null_count();
                casted
            })
            .collect::<Vec<Box<dyn Array>>>();

        let null_count = null_count as IdxSize;

        // @TODO: We throw away metadata here. That is mostly not needed.
        // SAFETY: We calculated the null_count again. And we are taking the rest from the previous
        // Int64Chunked.
        let int64chunked =
            unsafe { Self::new_with_dims(self.field.clone(), chunks, self.length, null_count) };

        TimeChunked::new_logical(int64chunked)
    }
}

impl LogicalType for TimeChunked {
    fn dtype(&self) -> &'static DataType {
        &DataType::Time
    }

    #[cfg(feature = "dtype-time")]
    fn get_any_value(&self, i: usize) -> PolarsResult<AnyValue<'_>> {
        self.0.get_any_value(i).map(|av| av.as_time())
    }
    unsafe fn get_any_value_unchecked(&self, i: usize) -> AnyValue<'_> {
        self.0.get_any_value_unchecked(i).as_time()
    }

    fn cast_with_options(
        &self,
        dtype: &DataType,
        cast_options: CastOptions,
    ) -> PolarsResult<Series> {
        use DataType::*;
        match dtype {
            Time => Ok(self.clone().into_series()),
            #[cfg(feature = "dtype-duration")]
            Duration(tu) => {
                let out = self
                    .0
                    .cast_with_options(&DataType::Duration(TimeUnit::Nanoseconds), cast_options);
                if !matches!(tu, TimeUnit::Nanoseconds) {
                    out?.cast_with_options(dtype, cast_options)
                } else {
                    out
                }
            },
            #[cfg(feature = "dtype-datetime")]
            Datetime(_, _) => {
                polars_bail!(
                    InvalidOperation:
                    "casting from {:?} to {:?} not supported; consider using `dt.combine`",
                    self.dtype(), dtype
                )
            },
            dt if dt.is_numeric() => self.0.cast_with_options(dtype, cast_options),
            _ => {
                polars_bail!(
                    InvalidOperation:
                    "casting from {:?} to {:?} not supported",
                    self.dtype(), dtype
                )
            },
        }
    }
}
