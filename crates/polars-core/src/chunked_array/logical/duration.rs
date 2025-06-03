use super::*;
use crate::prelude::*;

pub type DurationChunked = Logical<DurationType, Int64Type>;

impl Int64Chunked {
    pub fn into_duration(self, timeunit: TimeUnit) -> DurationChunked {
        // SAFETY: no invalid states.
        unsafe { DurationChunked::new_logical(self, DataType::Duration(timeunit)) }
    }
}

impl LogicalType for DurationChunked {
    fn dtype(&self) -> &DataType {
        &self.dtype
    }

    fn get_any_value(&self, i: usize) -> PolarsResult<AnyValue<'_>> {
        self.phys
            .get_any_value(i)
            .map(|av| av.as_duration(self.time_unit()))
    }
    unsafe fn get_any_value_unchecked(&self, i: usize) -> AnyValue<'_> {
        self.phys
            .get_any_value_unchecked(i)
            .as_duration(self.time_unit())
    }

    fn cast_with_options(
        &self,
        dtype: &DataType,
        cast_options: CastOptions,
    ) -> PolarsResult<Series> {
        use DataType::*;

        use crate::datatypes::time_unit::TimeUnit::*;
        match dtype {
            Duration(tu) => {
                let to_unit = *tu;
                let (divisor, multiplier) = match (self.time_unit(), to_unit) {
                    (Milliseconds, Microseconds) => (None, Some(1_000i64)),
                    (Milliseconds, Nanoseconds) => (None, Some(1_000_000i64)),
                    (Microseconds, Milliseconds) => (Some(1_000i64), None),
                    (Microseconds, Nanoseconds) => (None, Some(1_000i64)),
                    (Nanoseconds, Milliseconds) => (Some(1_000_000i64), None),
                    (Nanoseconds, Microseconds) => (Some(1_000i64), None),
                    _ => return Ok(self.clone().into_series()),
                };

                let out = match (divisor, multiplier) {
                    (None, None) | (Some(_), Some(_)) => unreachable!(),
                    (_, Some(multiplier)) => self.phys.as_ref().checked_mul_scalar(multiplier),
                    (Some(divisor), _) => self.phys.as_ref().wrapping_trunc_div_scalar(divisor),
                };
                Ok(out.into_duration(to_unit).into_series())
            },
            dt if dt.is_primitive_numeric() => self.phys.cast_with_options(dtype, cast_options),
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
