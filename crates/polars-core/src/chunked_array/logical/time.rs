use polars_compute::cast::CastOptionsImpl;

use super::*;
use crate::prelude::*;

pub type TimeChunked = Logical<TimeType, Int64Type>;

impl Int64Chunked {
    pub fn into_time(mut self) -> TimeChunked {
        let mut null_count = 0;

        // Invalid time values are replaced with `null` during the arrow cast. We utilize the
        // validity coming from there to create the new TimeChunked.
        let chunks = std::mem::take(&mut self.chunks)
            .into_iter()
            .map(|chunk| {
                // A time holds a day's worth of nanoseconds, and an `i64` outside that range names
                // none: the cast is that range check, and the chunk it answers is the chunk itself
                // when every value fell inside it.
                let casted = polars_compute::cast::cast(
                    &*chunk,
                    &DataType::Int64,
                    &DataType::Time,
                    CastOptionsImpl::default(),
                )
                .unwrap();
                null_count += casted.null_count();
                casted
            })
            .collect::<Vec<PlArrayRef>>();

        debug_assert!(null_count >= self.null_count);

        // SAFETY: We calculated the null_count again. And we are taking the rest from the previous
        // Int64Chunked.
        let mut ca =
            unsafe { Self::new_with_dims(self.field.clone(), chunks, self.length, null_count) };
        if null_count == self.null_count {
            ca.set_sorted_flag(self.is_sorted_flag());
        }

        // SAFETY: no invalid states.
        unsafe { TimeChunked::new_logical(ca, DataType::Time) }
    }
}

impl LogicalType for TimeChunked {
    fn dtype(&self) -> &'static DataType {
        &DataType::Time
    }

    #[cfg(feature = "dtype-time")]
    fn get_any_value(&self, i: usize) -> PolarsResult<AnyValue<'_>> {
        self.phys.get_any_value(i).map(|av| av.as_time())
    }
    unsafe fn get_any_value_unchecked(&self, i: usize) -> AnyValue<'_> {
        self.phys.get_any_value_unchecked(i).as_time()
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
                    .phys
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
            dt if dt.is_primitive_numeric() => self.phys.cast_with_options(dtype, cast_options),
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
