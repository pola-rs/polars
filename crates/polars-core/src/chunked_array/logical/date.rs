use super::*;
use crate::prelude::*;
pub type DateChunked = Logical<DateType, Int32Type>;

impl Int32Chunked {
    pub fn into_date(self) -> DateChunked {
        // SAFETY: no invalid states.
        unsafe { DateChunked::new_logical(self, DataType::Date) }
    }
}

impl LogicalType for DateChunked {
    fn dtype(&self) -> &DataType {
        &DataType::Date
    }

    fn get_any_value(&self, i: usize) -> PolarsResult<AnyValue<'_>> {
        self.phys.get_any_value(i).map(|av| av.as_date())
    }

    unsafe fn get_any_value_unchecked(&self, i: usize) -> AnyValue<'_> {
        self.phys.get_any_value_unchecked(i).as_date()
    }

    fn cast_with_options(
        &self,
        dtype: &DataType,
        cast_options: CastOptions,
    ) -> PolarsResult<Series> {
        use DataType::*;
        match dtype {
            Date => Ok(self.clone().into_series()),
            #[cfg(feature = "dtype-datetime")]
            Datetime(tu, tz) => {
                let casted = self.phys.cast_with_options(dtype, cast_options)?;
                let casted = casted.datetime().unwrap();
                let conversion = match tu {
                    TimeUnit::Nanoseconds => NS_IN_DAY,
                    TimeUnit::Microseconds => US_IN_DAY,
                    TimeUnit::Milliseconds => MS_IN_DAY,
                };
                Ok(casted
                    .physical()
                    .checked_mul_scalar(conversion)
                    .into_datetime(*tu, tz.clone())
                    .into_series())
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
