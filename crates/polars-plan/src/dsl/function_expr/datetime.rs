use super::*;

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
#[derive(Clone, PartialEq, Debug, Eq, Hash)]
pub enum TemporalFunction {
    Millennium,
    Century,
    Year,
    IsLeapYear,
    IsoYear,
    Quarter,
    Month,
    DaysInMonth,
    Week,
    WeekDay,
    Day,
    OrdinalDay,
    Time,
    Date,
    Datetime,
    #[cfg(feature = "dtype-duration")]
    Duration(TimeUnit),
    Hour,
    Minute,
    Second,
    Millisecond,
    Microsecond,
    Nanosecond,
    #[cfg(feature = "dtype-duration")]
    TotalDays {
        fractional: bool,
    },
    #[cfg(feature = "dtype-duration")]
    TotalHours {
        fractional: bool,
    },
    #[cfg(feature = "dtype-duration")]
    TotalMinutes {
        fractional: bool,
    },
    #[cfg(feature = "dtype-duration")]
    TotalSeconds {
        fractional: bool,
    },
    #[cfg(feature = "dtype-duration")]
    TotalMilliseconds {
        fractional: bool,
    },
    #[cfg(feature = "dtype-duration")]
    TotalMicroseconds {
        fractional: bool,
    },
    #[cfg(feature = "dtype-duration")]
    TotalNanoseconds {
        fractional: bool,
    },
    ToString(String),
    CastTimeUnit(TimeUnit),
    WithTimeUnit(TimeUnit),
    #[cfg(feature = "timezones")]
    ConvertTimeZone(TimeZone),
    TimeStamp(TimeUnit),
    Truncate,
    #[cfg(feature = "offset_by")]
    OffsetBy,
    #[cfg(feature = "month_start")]
    MonthStart,
    #[cfg(feature = "month_end")]
    MonthEnd,
    #[cfg(feature = "timezones")]
    BaseUtcOffset,
    #[cfg(feature = "timezones")]
    DSTOffset,
    Round,
    Replace,
    #[cfg(feature = "timezones")]
    ReplaceTimeZone(Option<TimeZone>, NonExistent),
    Combine(TimeUnit),
    DatetimeFunction {
        time_unit: TimeUnit,
        time_zone: Option<TimeZone>,
    },
}

impl Display for TemporalFunction {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        use TemporalFunction::*;
        let s = match self {
            Millennium => "millennium",
            Century => "century",
            Year => "year",
            IsLeapYear => "is_leap_year",
            IsoYear => "iso_year",
            Quarter => "quarter",
            Month => "month",
            DaysInMonth => "days_in_month",
            Week => "week",
            WeekDay => "weekday",
            Day => "day",
            OrdinalDay => "ordinal_day",
            Time => "time",
            Date => "date",
            Datetime => "datetime",
            #[cfg(feature = "dtype-duration")]
            Duration(_) => "duration",
            Hour => "hour",
            Minute => "minute",
            Second => "second",
            Millisecond => "millisecond",
            Microsecond => "microsecond",
            Nanosecond => "nanosecond",
            #[cfg(feature = "dtype-duration")]
            TotalDays { .. } => "total_days",
            #[cfg(feature = "dtype-duration")]
            TotalHours { .. } => "total_hours",
            #[cfg(feature = "dtype-duration")]
            TotalMinutes { .. } => "total_minutes",
            #[cfg(feature = "dtype-duration")]
            TotalSeconds { .. } => "total_seconds",
            #[cfg(feature = "dtype-duration")]
            TotalMilliseconds { .. } => "total_milliseconds",
            #[cfg(feature = "dtype-duration")]
            TotalMicroseconds { .. } => "total_microseconds",
            #[cfg(feature = "dtype-duration")]
            TotalNanoseconds { .. } => "total_nanoseconds",
            ToString(_) => "to_string",
            #[cfg(feature = "timezones")]
            ConvertTimeZone(_) => "convert_time_zone",
            CastTimeUnit(_) => "cast_time_unit",
            WithTimeUnit(_) => "with_time_unit",
            TimeStamp(tu) => return write!(f, "dt.timestamp({tu})"),
            Truncate => "truncate",
            #[cfg(feature = "offset_by")]
            OffsetBy => "offset_by",
            #[cfg(feature = "month_start")]
            MonthStart => "month_start",
            #[cfg(feature = "month_end")]
            MonthEnd => "month_end",
            #[cfg(feature = "timezones")]
            BaseUtcOffset => "base_utc_offset",
            #[cfg(feature = "timezones")]
            DSTOffset => "dst_offset",
            Round => "round",
            Replace => "replace",
            #[cfg(feature = "timezones")]
            ReplaceTimeZone(_, _) => "replace_time_zone",
            DatetimeFunction { .. } => return write!(f, "dt.datetime"),
            Combine(_) => "combine",
        };
        write!(f, "dt.{s}")
    }
}
