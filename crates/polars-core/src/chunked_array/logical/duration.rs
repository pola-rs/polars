use super::*;
use crate::prelude::*;

pub type DurationChunked = Logical<DurationType, Int64Type>;

impl Int64Chunked {
    pub fn into_duration(self, timeunit: TimeUnit) -> DurationChunked {
        let mut dt = DurationChunked::new_logical(self);
        dt.2 = Some(DataType::Duration(timeunit));
        dt
    }
}

impl LogicalType for DurationChunked {
    fn dtype(&self) -> &DataType {
        self.2.as_ref().unwrap()
    }

    fn get_any_value(&self, i: usize) -> PolarsResult<AnyValue<'_>> {
        self.0
            .get_any_value(i)
            .map(|av| av.as_duration(self.time_unit()))
    }
    unsafe fn get_any_value_unchecked(&self, i: usize) -> AnyValue<'_> {
        self.0
            .get_any_value_unchecked(i)
            .as_duration(self.time_unit())
    }

    fn cast_with_options(
        &self,
        dtype: &DataType,
        cast_options: CastOptions,
    ) -> PolarsResult<Series> {
        use DataType::*;
        use TimeUnit::*;
        match dtype {
            Duration(tu) => {
                let to_unit = *tu;
                let out = match (self.time_unit(), to_unit) {
                    (Milliseconds, Microseconds) => self.0.as_ref() * 1_000i64,
                    (Milliseconds, Nanoseconds) => self.0.as_ref() * 1_000_000i64,
                    (Microseconds, Milliseconds) => {
                        self.0.as_ref().wrapping_trunc_div_scalar(1_000i64)
                    },
                    (Microseconds, Nanoseconds) => self.0.as_ref() * 1_000i64,
                    (Nanoseconds, Milliseconds) => {
                        self.0.as_ref().wrapping_trunc_div_scalar(1_000_000i64)
                    },
                    (Nanoseconds, Microseconds) => {
                        self.0.as_ref().wrapping_trunc_div_scalar(1_000i64)
                    },
                    _ => return Ok(self.clone().into_series()),
                };
                Ok(out.into_duration(to_unit).into_series())
            },
            dt if dt.is_numeric() => self.0.cast_with_options(dtype, cast_options),
            dt => {
                polars_bail!(
                    InvalidOperation:
                    "casting from {:?} to {:?} not supported",
                    self.dtype(), dt
                )
            },
        }
    }
}
