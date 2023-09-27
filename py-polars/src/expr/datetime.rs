use polars::prelude::*;
use pyo3::prelude::*;

use crate::conversion::Wrap;
use crate::PyExpr;

#[pymethods]
impl PyExpr {
    fn dt_to_string(&self, format: &str) -> Self {
        self.inner.clone().dt().to_string(format).into()
    }

    fn dt_offset_by(&self, by: PyExpr) -> Self {
        self.inner.clone().dt().offset_by(by.inner).into()
    }

    fn dt_epoch_seconds(&self) -> Self {
        self.clone()
            .inner
            .map(
                |s| {
                    s.timestamp(TimeUnit::Milliseconds)
                        .map(|ca| Some((ca / 1000).into_series()))
                },
                GetOutput::from_type(DataType::Int64),
            )
            .into()
    }

    fn dt_with_time_unit(&self, time_unit: Wrap<TimeUnit>) -> Self {
        self.inner.clone().dt().with_time_unit(time_unit.0).into()
    }

    #[cfg(feature = "timezones")]
    fn dt_convert_time_zone(&self, time_zone: TimeZone) -> Self {
        self.inner.clone().dt().convert_time_zone(time_zone).into()
    }

    fn dt_cast_time_unit(&self, time_unit: Wrap<TimeUnit>) -> Self {
        self.inner.clone().dt().cast_time_unit(time_unit.0).into()
    }

    #[cfg(feature = "timezones")]
    #[pyo3(signature = (time_zone, ambiguous))]
    fn dt_replace_time_zone(&self, time_zone: Option<String>, ambiguous: Self) -> Self {
        self.inner
            .clone()
            .dt()
            .replace_time_zone(time_zone, ambiguous.inner)
            .into()
    }

    fn dt_truncate(&self, every: Self, offset: String, ambiguous: Self) -> Self {
        self.inner
            .clone()
            .dt()
            .truncate(every.inner, offset, ambiguous.inner)
            .into()
    }

    fn dt_month_start(&self) -> Self {
        self.inner.clone().dt().month_start().into()
    }

    fn dt_month_end(&self) -> Self {
        self.inner.clone().dt().month_end().into()
    }

    #[cfg(feature = "timezones")]
    fn dt_base_utc_offset(&self) -> Self {
        self.inner.clone().dt().base_utc_offset().into()
    }
    #[cfg(feature = "timezones")]
    fn dt_dst_offset(&self) -> Self {
        self.inner.clone().dt().dst_offset().into()
    }

    fn dt_round(&self, every: &str, offset: &str, ambiguous: Self) -> Self {
        self.inner
            .clone()
            .dt()
            .round(every, offset, ambiguous.inner)
            .into()
    }

    fn dt_combine(&self, time: Self, time_unit: Wrap<TimeUnit>) -> Self {
        self.inner
            .clone()
            .dt()
            .combine(time.inner, time_unit.0)
            .into()
    }

    fn dt_year(&self) -> Self {
        self.clone().inner.dt().year().into()
    }
    fn dt_is_leap_year(&self) -> Self {
        self.clone().inner.dt().is_leap_year().into()
    }
    fn dt_iso_year(&self) -> Self {
        self.clone().inner.dt().iso_year().into()
    }
    fn dt_quarter(&self) -> Self {
        self.clone().inner.dt().quarter().into()
    }
    fn dt_month(&self) -> Self {
        self.clone().inner.dt().month().into()
    }
    fn dt_week(&self) -> Self {
        self.clone().inner.dt().week().into()
    }
    fn dt_weekday(&self) -> Self {
        self.clone().inner.dt().weekday().into()
    }
    fn dt_day(&self) -> Self {
        self.clone().inner.dt().day().into()
    }
    fn dt_ordinal_day(&self) -> Self {
        self.clone().inner.dt().ordinal_day().into()
    }
    fn dt_time(&self) -> Self {
        self.clone().inner.dt().time().into()
    }
    fn dt_date(&self) -> Self {
        self.clone().inner.dt().date().into()
    }
    fn dt_datetime(&self) -> Self {
        self.clone().inner.dt().datetime().into()
    }
    fn dt_hour(&self) -> Self {
        self.clone().inner.dt().hour().into()
    }
    fn dt_minute(&self) -> Self {
        self.clone().inner.dt().minute().into()
    }
    fn dt_second(&self) -> Self {
        self.clone().inner.dt().second().into()
    }
    fn dt_millisecond(&self) -> Self {
        self.clone().inner.dt().millisecond().into()
    }
    fn dt_microsecond(&self) -> Self {
        self.clone().inner.dt().microsecond().into()
    }
    fn dt_nanosecond(&self) -> Self {
        self.clone().inner.dt().nanosecond().into()
    }
    fn dt_timestamp(&self, time_unit: Wrap<TimeUnit>) -> Self {
        self.inner.clone().dt().timestamp(time_unit.0).into()
    }

    fn duration_days(&self) -> Self {
        self.inner
            .clone()
            .map(
                |s| Ok(Some(s.duration()?.days().into_series())),
                GetOutput::from_type(DataType::Int64),
            )
            .into()
    }
    fn duration_hours(&self) -> Self {
        self.inner
            .clone()
            .map(
                |s| Ok(Some(s.duration()?.hours().into_series())),
                GetOutput::from_type(DataType::Int64),
            )
            .into()
    }
    fn duration_minutes(&self) -> Self {
        self.inner
            .clone()
            .map(
                |s| Ok(Some(s.duration()?.minutes().into_series())),
                GetOutput::from_type(DataType::Int64),
            )
            .into()
    }
    fn duration_seconds(&self) -> Self {
        self.inner
            .clone()
            .map(
                |s| Ok(Some(s.duration()?.seconds().into_series())),
                GetOutput::from_type(DataType::Int64),
            )
            .into()
    }
    fn duration_milliseconds(&self) -> Self {
        self.inner
            .clone()
            .map(
                |s| Ok(Some(s.duration()?.milliseconds().into_series())),
                GetOutput::from_type(DataType::Int64),
            )
            .into()
    }
    fn duration_microseconds(&self) -> Self {
        self.inner
            .clone()
            .map(
                |s| Ok(Some(s.duration()?.microseconds().into_series())),
                GetOutput::from_type(DataType::Int64),
            )
            .into()
    }
    fn duration_nanoseconds(&self) -> Self {
        self.inner
            .clone()
            .map(
                |s| Ok(Some(s.duration()?.nanoseconds().into_series())),
                GetOutput::from_type(DataType::Int64),
            )
            .into()
    }
}
