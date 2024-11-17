use polars_core::prelude::*;

use crate::prelude::*;

#[cfg(feature = "dtype-datetime")]
pub trait PolarsReplaceDatetime: DatetimeMethods {
    #[allow(clippy::too_many_arguments)]
    fn replace(
        &self,
        year: &Int32Chunked,
        month: &Int8Chunked,
        day: &Int8Chunked,
        hour: &Int8Chunked,
        minute: &Int8Chunked,
        second: &Int8Chunked,
        microsecond: &Int32Chunked,
        ambiguous: &StringChunked,
    ) -> PolarsResult<Self>
    where
        Self: Sized;
}

#[cfg(feature = "dtype-date")]
pub trait PolarsReplaceDate: DateMethods {
    fn replace(
        &self,
        year: &Int32Chunked,
        month: &Int8Chunked,
        day: &Int8Chunked,
    ) -> PolarsResult<Self>
    where
        Self: Sized;
}

#[cfg(feature = "dtype-datetime")]
impl PolarsReplaceDatetime for DatetimeChunked {
    #[allow(clippy::too_many_arguments)]
    fn replace(
        &self,
        year: &Int32Chunked,
        month: &Int8Chunked,
        day: &Int8Chunked,
        hour: &Int8Chunked,
        minute: &Int8Chunked,
        second: &Int8Chunked,
        microsecond: &Int32Chunked,
        ambiguous: &StringChunked,
    ) -> PolarsResult<Self> {
        let n = self.len();

        // For each argument, we must check if:
        // 1. No value was supplied (None)       --> Use existing year from Series
        // 2. Value was supplied and is a Scalar --> Create full Series of value
        // 3. Value was supplied and is Series   --> Update all elements with the non-null values
        let year = if year.len() == 1 {
            // SAFETY: array has one value.
            if let Some(value) = unsafe { year.get_unchecked(0) } {
                if n == 1 {
                    year
                } else {
                    &Int32Chunked::full("".into(), value, n)
                }
            } else {
                &self.year()
            }
        } else {
            &year.zip_with(&year.is_not_null(), &self.year())?
        };
        let month = if month.len() == 1 {
            // SAFETY: array has one value.
            if let Some(value) = unsafe { month.get_unchecked(0) } {
                if n == 1 {
                    month
                } else {
                    &Int8Chunked::full("".into(), value, n)
                }
            } else {
                &self.month()
            }
        } else {
            &month.zip_with(&month.is_not_null(), &self.month())?
        };
        let day = if day.len() == 1 {
            // SAFETY: array has one value.
            if let Some(value) = unsafe { day.get_unchecked(0) } {
                if n == 1 {
                    day
                } else {
                    &Int8Chunked::full("".into(), value, n)
                }
            } else {
                &self.day()
            }
        } else {
            &day.zip_with(&day.is_not_null(), &self.day())?
        };
        let hour = if hour.len() == 1 {
            // SAFETY: array has one value.
            if let Some(value) = unsafe { hour.get_unchecked(0) } {
                if n == 1 {
                    hour
                } else {
                    &Int8Chunked::full("".into(), value, n)
                }
            } else {
                &self.hour()
            }
        } else {
            &hour.zip_with(&hour.is_not_null(), &self.hour())?
        };
        let minute = if minute.len() == 1 {
            // SAFETY: array has one value.
            if let Some(value) = unsafe { minute.get_unchecked(0) } {
                if n == 1 {
                    minute
                } else {
                    &Int8Chunked::full("".into(), value, n)
                }
            } else {
                &self.minute()
            }
        } else {
            &minute.zip_with(&minute.is_not_null(), &self.minute())?
        };
        let second = if second.len() == 1 {
            // SAFETY: array has one value.
            if let Some(value) = unsafe { second.get_unchecked(0) } {
                if n == 1 {
                    second
                } else {
                    &Int8Chunked::full("".into(), value, n)
                }
            } else {
                &self.second()
            }
        } else {
            &second.zip_with(&second.is_not_null(), &self.second())?
        };
        let microsecond = if microsecond.len() == 1 {
            // SAFETY: array has one value.
            if let Some(value) = unsafe { microsecond.get_unchecked(0) } {
                if n == 1 {
                    microsecond
                } else {
                    &Int32Chunked::full("".into(), value, n)
                }
            } else {
                &(self.nanosecond() / 1000)
            }
        } else {
            &microsecond.zip_with(&microsecond.is_not_null(), &(self.nanosecond() / 1000))?
        };

        let out = DatetimeChunked::from_parts(
            year,
            month,
            day,
            hour,
            minute,
            second,
            microsecond,
            ambiguous,
            &self.time_unit(),
            self.time_zone().as_deref(),
            self.name().clone(),
        )?;
        Ok(out)
    }
}

#[cfg(feature = "dtype-date")]
impl PolarsReplaceDate for DateChunked {
    fn replace(
        &self,
        year: &Int32Chunked,
        month: &Int8Chunked,
        day: &Int8Chunked,
    ) -> PolarsResult<Self> {
        let n = self.len();

        let year = if year.len() == 1 {
            // SAFETY: array has one value.
            if let Some(value) = unsafe { year.get_unchecked(0) } {
                if n == 1 {
                    year
                } else {
                    &Int32Chunked::full("".into(), value, n)
                }
            } else {
                &self.year()
            }
        } else {
            &year.zip_with(&year.is_not_null(), &self.year())?
        };
        let month = if month.len() == 1 {
            // SAFETY: array has one value.
            if let Some(value) = unsafe { month.get_unchecked(0) } {
                if n == 1 {
                    month
                } else {
                    &Int8Chunked::full("".into(), value, n)
                }
            } else {
                &self.month()
            }
        } else {
            &month.zip_with(&month.is_not_null(), &self.month())?
        };
        let day = if day.len() == 1 {
            // SAFETY: array has one value.
            if let Some(value) = unsafe { day.get_unchecked(0) } {
                if n == 1 {
                    day
                } else {
                    &Int8Chunked::full("".into(), value, n)
                }
            } else {
                &self.day()
            }
        } else {
            &day.zip_with(&day.is_not_null(), &self.day())?
        };

        let out = DateChunked::from_parts(year, month, day, self.name().clone())?;
        Ok(out)
    }
}
