use polars_time::prelude::TemporalMethods;

use super::*;
use crate::prelude::function_expr::TemporalFunction;

/// Specialized expressions for [`Series`] with dates/datetimes.
pub struct DateLikeNameSpace(pub(crate) Expr);

impl DateLikeNameSpace {
    /// Convert from Date/Time/Datetime into Utf8 with the given format.
    /// See [chrono strftime/strptime](https://docs.rs/chrono/0.4.19/chrono/format/strftime/index.html).
    pub fn to_string(self, format: &str) -> Expr {
        let format = format.to_string();
        let function = move |s: Series| TemporalMethods::to_string(&s, &format).map(Some);
        self.0
            .map(function, GetOutput::from_type(DataType::Utf8))
            .with_fmt("to_string")
    }

    /// Convert from Date/Time/Datetime into Utf8 with the given format.
    /// See [chrono strftime/strptime](https://docs.rs/chrono/0.4.19/chrono/format/strftime/index.html).
    ///
    /// Alias for `to_string`.
    pub fn strftime(self, format: &str) -> Expr {
        self.to_string(format)
    }

    /// Change the underlying [`TimeUnit`]. And update the data accordingly.
    pub fn cast_time_unit(self, tu: TimeUnit) -> Expr {
        self.0.map(
            move |s| match s.dtype() {
                DataType::Datetime(_, _) => {
                    let ca = s.datetime().unwrap();
                    Ok(Some(ca.cast_time_unit(tu).into_series()))
                }
                #[cfg(feature = "dtype-duration")]
                DataType::Duration(_) => {
                    let ca = s.duration().unwrap();
                    Ok(Some(ca.cast_time_unit(tu).into_series()))
                }
                dt => polars_bail!(ComputeError: "dtype `{}` has no time unit", dt),
            },
            GetOutput::map_dtype(move |dtype| match dtype {
                DataType::Duration(_) => DataType::Duration(tu),
                DataType::Datetime(_, tz) => DataType::Datetime(tu, tz.clone()),
                _ => panic!("expected duration or datetime"),
            }),
        )
    }

    /// Change the underlying [`TimeUnit`] of the [`Series`]. This does not modify the data.
    pub fn with_time_unit(self, tu: TimeUnit) -> Expr {
        self.0.map(
            move |s| match s.dtype() {
                DataType::Datetime(_, _) => {
                    let mut ca = s.datetime().unwrap().clone();
                    ca.set_time_unit(tu);
                    Ok(Some(ca.into_series()))
                }
                #[cfg(feature = "dtype-duration")]
                DataType::Duration(_) => {
                    let mut ca = s.duration().unwrap().clone();
                    ca.set_time_unit(tu);
                    Ok(Some(ca.into_series()))
                }
                dt => polars_bail!(ComputeError: "dtype `{}` has no time unit", dt),
            },
            GetOutput::same_type(),
        )
    }

    /// Change the underlying [`TimeZone`] of the [`Series`]. This does not modify the data.
    #[cfg(feature = "timezones")]
    pub fn convert_time_zone(self, time_zone: TimeZone) -> Expr {
        let time_zone_clone = time_zone.clone();
        self.0.map(
            move |s| match s.dtype() {
                DataType::Datetime(_, Some(_)) => {
                    let mut ca = s.datetime().unwrap().clone();
                    ca.set_time_zone(time_zone.clone())?;
                    Ok(Some(ca.into_series()))
                }
                _ => polars_bail!(
                    ComputeError:
                    "cannot call `convert_time_zone` on tz-naive; set a time zone first \
                    with `replace_time_zone`"
                ),
            },
            GetOutput::map_dtype(move |dtype| match dtype {
                DataType::Datetime(tu, _) => DataType::Datetime(*tu, Some(time_zone_clone.clone())),
                _ => panic!("expected datetime"),
            }),
        )
    }

    /// Localize tz-naive Datetime Series to tz-aware Datetime Series.
    //
    // This method takes a naive Datetime Series and makes this time zone aware.
    // It does not move the time to another time zone.
    #[cfg(feature = "timezones")]
    #[deprecated(note = "use replace_time_zone")]
    pub fn tz_localize(self, tz: TimeZone) -> Expr {
        self.0
            .map_private(FunctionExpr::TemporalExpr(TemporalFunction::TzLocalize(tz)))
    }

    /// Get the year of a Date/Datetime
    pub fn year(self) -> Expr {
        self.0
            .map_private(FunctionExpr::TemporalExpr(TemporalFunction::Year))
    }

    // Compute whether the year of a Date/Datetime is a leap year.
    pub fn is_leap_year(self) -> Expr {
        self.0
            .map_private(FunctionExpr::TemporalExpr(TemporalFunction::IsLeapYear))
    }

    /// Get the iso-year of a Date/Datetime.
    /// This may not correspond with a calendar year.
    pub fn iso_year(self) -> Expr {
        self.0
            .map_private(FunctionExpr::TemporalExpr(TemporalFunction::IsoYear))
    }

    /// Get the month of a Date/Datetime
    pub fn month(self) -> Expr {
        self.0
            .map_private(FunctionExpr::TemporalExpr(TemporalFunction::Month))
    }

    /// Extract quarter from underlying NaiveDateTime representation.
    /// Quarters range from 1 to 4.
    pub fn quarter(self) -> Expr {
        self.0
            .map_private(FunctionExpr::TemporalExpr(TemporalFunction::Quarter))
    }

    /// Extract the week from the underlying Date representation.
    /// Can be performed on Date and Datetime

    /// Returns the ISO week number starting from 1.
    /// The return value ranges from 1 to 53. (The last week of year differs by years.)
    pub fn week(self) -> Expr {
        self.0
            .map_private(FunctionExpr::TemporalExpr(TemporalFunction::Week))
    }

    /// Extract the week day from the underlying Date representation.
    /// Can be performed on Date and Datetime.

    /// Returns the weekday number where monday = 0 and sunday = 6
    pub fn weekday(self) -> Expr {
        self.0
            .map_private(FunctionExpr::TemporalExpr(TemporalFunction::WeekDay))
    }

    /// Get the month of a Date/Datetime
    pub fn day(self) -> Expr {
        self.0
            .map_private(FunctionExpr::TemporalExpr(TemporalFunction::Day))
    }

    /// Get the ordinal_day of a Date/Datetime
    pub fn ordinal_day(self) -> Expr {
        self.0
            .map_private(FunctionExpr::TemporalExpr(TemporalFunction::OrdinalDay))
    }

    /// Get the (local) time of a Date/Datetime/Time
    pub fn time(self) -> Expr {
        self.0
            .map_private(FunctionExpr::TemporalExpr(TemporalFunction::Time))
    }

    /// Get the (local) date of a Date/Datetime
    pub fn date(self) -> Expr {
        self.0
            .map_private(FunctionExpr::TemporalExpr(TemporalFunction::Date))
    }

    /// Get the (local) datetime of a Datetime
    pub fn datetime(self) -> Expr {
        self.0
            .map_private(FunctionExpr::TemporalExpr(TemporalFunction::Datetime))
    }

    /// Get the hour of a Datetime/Time64
    pub fn hour(self) -> Expr {
        self.0
            .map_private(FunctionExpr::TemporalExpr(TemporalFunction::Hour))
    }

    /// Get the minute of a Datetime/Time64
    pub fn minute(self) -> Expr {
        self.0
            .map_private(FunctionExpr::TemporalExpr(TemporalFunction::Minute))
    }

    /// Get the second of a Datetime/Time64
    pub fn second(self) -> Expr {
        self.0
            .map_private(FunctionExpr::TemporalExpr(TemporalFunction::Second))
    }

    /// Get the millisecond of a Time64 (scaled from nanosecs)
    pub fn millisecond(self) -> Expr {
        self.0
            .map_private(FunctionExpr::TemporalExpr(TemporalFunction::Millisecond))
    }

    /// Get the microsecond of a Time64 (scaled from nanosecs)
    pub fn microsecond(self) -> Expr {
        self.0
            .map_private(FunctionExpr::TemporalExpr(TemporalFunction::Microsecond))
    }

    /// Get the nanosecond part of a Time64
    pub fn nanosecond(self) -> Expr {
        self.0
            .map_private(FunctionExpr::TemporalExpr(TemporalFunction::Nanosecond))
    }

    pub fn timestamp(self, tu: TimeUnit) -> Expr {
        self.0
            .map_private(FunctionExpr::TemporalExpr(TemporalFunction::TimeStamp(tu)))
    }

    pub fn truncate<S: AsRef<str>>(self, every: S, offset: S) -> Expr {
        let every = every.as_ref().into();
        let offset = offset.as_ref().into();
        self.0
            .map_private(FunctionExpr::TemporalExpr(TemporalFunction::Truncate(
                every, offset,
            )))
    }

    // roll backward to the first day of the month
    #[cfg(feature = "date_offset")]
    pub fn month_start(self) -> Expr {
        self.0
            .map_private(FunctionExpr::TemporalExpr(TemporalFunction::MonthStart))
    }

    // roll forward to the last day of the month
    #[cfg(feature = "date_offset")]
    pub fn month_end(self) -> Expr {
        self.0
            .map_private(FunctionExpr::TemporalExpr(TemporalFunction::MonthEnd))
    }

    pub fn round<S: AsRef<str>>(self, every: S, offset: S) -> Expr {
        let every = every.as_ref().into();
        let offset = offset.as_ref().into();
        self.0
            .map_private(FunctionExpr::TemporalExpr(TemporalFunction::Round(
                every, offset,
            )))
    }

    /// Offset this `Date/Datetime` by a given offset [`Duration`].
    /// This will take leap years/ months into account.
    #[cfg(feature = "date_offset")]
    pub fn offset_by(self, by: Duration) -> Expr {
        self.0.map_private(FunctionExpr::DateOffset(by))
    }

    #[cfg(feature = "timezones")]
    pub fn replace_time_zone(
        self,
        time_zone: Option<TimeZone>,
        use_earliest: Option<bool>,
    ) -> Expr {
        self.0
            .map_private(FunctionExpr::TemporalExpr(TemporalFunction::CastTimezone(
                time_zone,
                use_earliest,
            )))
    }

    pub fn combine(self, time: Expr, tu: TimeUnit) -> Expr {
        self.0.map_many_private(
            FunctionExpr::TemporalExpr(TemporalFunction::Combine(tu)),
            &[time],
            false,
        )
    }
}
