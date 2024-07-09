use super::*;
use crate::prelude::*;

pub type TimeChunked = Logical<TimeType, Int64Type>;

impl From<Int64Chunked> for TimeChunked {
    fn from(ca: Int64Chunked) -> Self {
        TimeChunked::new_logical(ca)
    }
}

impl Int64Chunked {
    pub fn into_time(self) -> TimeChunked {
        TimeChunked::new_logical(self)
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
