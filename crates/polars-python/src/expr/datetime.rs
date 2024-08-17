use polars::prelude::*;
use pyo3::prelude::*;

use crate::conversion::Wrap;
use crate::PyExpr;

#[pymethods]
impl PyExpr {
    fn dt_add_business_days(
        &self,
        n: PyExpr,
        week_mask: [bool; 7],
        holidays: Vec<i32>,
        roll: Wrap<Roll>,
    ) -> Self {
        self.inner
            .clone()
            .dt()
            .add_business_days(n.inner, week_mask, holidays, roll.0)
            .into()
    }

    fn dt_to_string(&self, format: &str) -> Self {
        self.inner.clone().dt().to_string(format).into()
    }

    fn dt_offset_by(&self, by: PyExpr) -> Self {
        self.inner.clone().dt().offset_by(by.inner).into()
    }

    fn dt_epoch_seconds(&self) -> Self {
        self.inner
            .clone()
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
    #[pyo3(signature = (time_zone, ambiguous, non_existent))]
    fn dt_replace_time_zone(
        &self,
        time_zone: Option<String>,
        ambiguous: Self,
        non_existent: Wrap<NonExistent>,
    ) -> Self {
        self.inner
            .clone()
            .dt()
            .replace_time_zone(time_zone, ambiguous.inner, non_existent.0)
            .into()
    }

    fn dt_truncate(&self, every: Self) -> Self {
        self.inner.clone().dt().truncate(every.inner).into()
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

    fn dt_round(&self, every: Self) -> Self {
        self.inner.clone().dt().round(every.inner).into()
    }

    fn dt_combine(&self, time: Self, time_unit: Wrap<TimeUnit>) -> Self {
        self.inner
            .clone()
            .dt()
            .combine(time.inner, time_unit.0)
            .into()
    }
    fn dt_millennium(&self) -> Self {
        self.inner.clone().dt().millennium().into()
    }
    fn dt_century(&self) -> Self {
        self.inner.clone().dt().century().into()
    }
    fn dt_year(&self) -> Self {
        self.inner.clone().dt().year().into()
    }
    fn dt_is_leap_year(&self) -> Self {
        self.inner.clone().dt().is_leap_year().into()
    }
    fn dt_iso_year(&self) -> Self {
        self.inner.clone().dt().iso_year().into()
    }
    fn dt_quarter(&self) -> Self {
        self.inner.clone().dt().quarter().into()
    }
    fn dt_month(&self) -> Self {
        self.inner.clone().dt().month().into()
    }
    fn dt_week(&self) -> Self {
        self.inner.clone().dt().week().into()
    }
    fn dt_weekday(&self) -> Self {
        self.inner.clone().dt().weekday().into()
    }
    fn dt_day(&self) -> Self {
        self.inner.clone().dt().day().into()
    }
    fn dt_ordinal_day(&self) -> Self {
        self.inner.clone().dt().ordinal_day().into()
    }
    fn dt_time(&self) -> Self {
        self.inner.clone().dt().time().into()
    }
    fn dt_date(&self) -> Self {
        self.inner.clone().dt().date().into()
    }
    fn dt_datetime(&self) -> Self {
        self.inner.clone().dt().datetime().into()
    }
    fn dt_hour(&self) -> Self {
        self.inner.clone().dt().hour().into()
    }
    fn dt_minute(&self) -> Self {
        self.inner.clone().dt().minute().into()
    }
    fn dt_second(&self) -> Self {
        self.inner.clone().dt().second().into()
    }
    fn dt_millisecond(&self) -> Self {
        self.inner.clone().dt().millisecond().into()
    }
    fn dt_microsecond(&self) -> Self {
        self.inner.clone().dt().microsecond().into()
    }
    fn dt_nanosecond(&self) -> Self {
        self.inner.clone().dt().nanosecond().into()
    }
    fn dt_timestamp(&self, time_unit: Wrap<TimeUnit>) -> Self {
        self.inner.clone().dt().timestamp(time_unit.0).into()
    }
    fn dt_total_days(&self) -> Self {
        self.inner.clone().dt().total_days().into()
    }
    fn dt_total_hours(&self) -> Self {
        self.inner.clone().dt().total_hours().into()
    }
    fn dt_total_minutes(&self) -> Self {
        self.inner.clone().dt().total_minutes().into()
    }
    fn dt_total_seconds(&self) -> Self {
        self.inner.clone().dt().total_seconds().into()
    }
    fn dt_total_milliseconds(&self) -> Self {
        self.inner.clone().dt().total_milliseconds().into()
    }
    fn dt_total_microseconds(&self) -> Self {
        self.inner.clone().dt().total_microseconds().into()
    }
    fn dt_total_nanoseconds(&self) -> Self {
        self.inner.clone().dt().total_nanoseconds().into()
    }
}
