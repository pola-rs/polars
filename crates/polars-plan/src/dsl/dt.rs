use super::*;

/// Specialized expressions for [`Series`] with dates/datetimes.
pub struct DateLikeNameSpace(pub(crate) Expr);

impl DateLikeNameSpace {
    /// Convert from Date/Time/Datetime into String with the given format.
    /// See [chrono strftime/strptime](https://docs.rs/chrono/0.4.19/chrono/format/strftime/index.html).
    pub fn to_string(self, format: &str) -> Expr {
        let format = format.to_string();
        self.0
            .map_private(FunctionExpr::TemporalExpr(TemporalFunction::ToString(
                format,
            )))
    }

    /// Convert from Date/Time/Datetime into String with the given format.
    /// See [chrono strftime/strptime](https://docs.rs/chrono/0.4.19/chrono/format/strftime/index.html).
    ///
    /// Alias for `to_string`.
    pub fn strftime(self, format: &str) -> Expr {
        self.to_string(format)
    }

    /// Change the underlying [`TimeUnit`]. And update the data accordingly.
    pub fn cast_time_unit(self, tu: TimeUnit) -> Expr {
        self.0
            .map_private(FunctionExpr::TemporalExpr(TemporalFunction::CastTimeUnit(
                tu,
            )))
    }

    /// Change the underlying [`TimeUnit`] of the [`Series`]. This does not modify the data.
    pub fn with_time_unit(self, tu: TimeUnit) -> Expr {
        self.0
            .map_private(FunctionExpr::TemporalExpr(TemporalFunction::WithTimeUnit(
                tu,
            )))
    }

    /// Change the underlying [`TimeZone`] of the [`Series`]. This does not modify the data.
    #[cfg(feature = "timezones")]
    pub fn convert_time_zone(self, time_zone: TimeZone) -> Expr {
        self.0.map_private(FunctionExpr::TemporalExpr(
            TemporalFunction::ConvertTimeZone(time_zone),
        ))
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

    /// Extract the ISO week day from the underlying Date representation.
    /// Can be performed on Date and Datetime.

    /// Returns the weekday number where monday = 1 and sunday = 7
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

    pub fn truncate(self, every: Expr, offset: String) -> Expr {
        self.0.map_many_private(
            FunctionExpr::TemporalExpr(TemporalFunction::Truncate(offset)),
            &[every],
            false,
            false,
        )
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

    // Get the base offset from UTC
    #[cfg(feature = "timezones")]
    pub fn base_utc_offset(self) -> Expr {
        self.0
            .map_private(FunctionExpr::TemporalExpr(TemporalFunction::BaseUtcOffset))
    }

    // Get the additional offset from UTC currently in effect (usually due to daylight saving time)
    #[cfg(feature = "timezones")]
    pub fn dst_offset(self) -> Expr {
        self.0
            .map_private(FunctionExpr::TemporalExpr(TemporalFunction::DSTOffset))
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
    pub fn offset_by(self, by: Expr) -> Expr {
        self.0
            .map_many_private(FunctionExpr::DateOffset, &[by], false, false)
    }

    #[cfg(feature = "timezones")]
    pub fn replace_time_zone(self, time_zone: Option<TimeZone>, ambiguous: Expr) -> Expr {
        self.0.map_many_private(
            FunctionExpr::TemporalExpr(TemporalFunction::ReplaceTimeZone(time_zone)),
            &[ambiguous],
            false,
            false,
        )
    }

    pub fn combine(self, time: Expr, tu: TimeUnit) -> Expr {
        self.0.map_many_private(
            FunctionExpr::TemporalExpr(TemporalFunction::Combine(tu)),
            &[time],
            false,
            false,
        )
    }
}
