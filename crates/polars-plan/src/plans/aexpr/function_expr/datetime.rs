use super::*;

#[cfg_attr(feature = "ir_serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, PartialEq, Debug, Eq, Hash)]
pub enum IRTemporalFunction {
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

impl IRTemporalFunction {
    pub(super) fn get_field(&self, mapper: FieldsMapper) -> PolarsResult<Field> {
        use IRTemporalFunction::*;
        match self {
            Millennium | Century | Year | IsoYear => mapper.with_dtype(DataType::Int32),
            OrdinalDay => mapper.with_dtype(DataType::Int16),
            Month | DaysInMonth | Quarter | Week | WeekDay | Day | Hour | Minute | Second => {
                mapper.with_dtype(DataType::Int8)
            },
            Millisecond | Microsecond | Nanosecond => mapper.with_dtype(DataType::Int32),
            #[cfg(feature = "dtype-duration")]
            TotalDays { fractional }
            | TotalHours { fractional }
            | TotalMinutes { fractional }
            | TotalSeconds { fractional }
            | TotalMilliseconds { fractional }
            | TotalMicroseconds { fractional }
            | TotalNanoseconds { fractional } => {
                if *fractional {
                    mapper.with_dtype(DataType::Float64)
                } else {
                    mapper.with_dtype(DataType::Int64)
                }
            },
            ToString(_) => mapper.with_dtype(DataType::String),
            WithTimeUnit(tu) | CastTimeUnit(tu) => mapper.try_map_dtype(|dt| match dt {
                DataType::Duration(_) => Ok(DataType::Duration(*tu)),
                DataType::Datetime(_, tz) => Ok(DataType::Datetime(*tu, tz.clone())),
                dtype => polars_bail!(ComputeError: "expected duration or datetime, got {dtype}"),
            }),
            #[cfg(feature = "timezones")]
            ConvertTimeZone(tz) => mapper.try_map_dtype(|dt| match dt {
                DataType::Datetime(tu, _) => Ok(DataType::Datetime(*tu, Some(tz.clone()))),
                dtype => polars_bail!(ComputeError: "expected Datetime, got {dtype}"),
            }),
            TimeStamp(_) => mapper.with_dtype(DataType::Int64),
            IsLeapYear => mapper.with_dtype(DataType::Boolean),
            Time => mapper.with_dtype(DataType::Time),
            #[cfg(feature = "dtype-duration")]
            Duration(tu) => mapper.with_dtype(DataType::Duration(*tu)),
            Date => mapper.with_dtype(DataType::Date),
            Datetime => mapper.try_map_dtype(|dt| match dt {
                DataType::Datetime(tu, _) => Ok(DataType::Datetime(*tu, None)),
                dtype => polars_bail!(ComputeError: "expected Datetime, got {dtype}"),
            }),
            Truncate => mapper.with_same_dtype(),
            #[cfg(feature = "offset_by")]
            OffsetBy => mapper.with_same_dtype(),
            #[cfg(feature = "month_start")]
            MonthStart => mapper.with_same_dtype(),
            #[cfg(feature = "month_end")]
            MonthEnd => mapper.with_same_dtype(),
            #[cfg(feature = "timezones")]
            BaseUtcOffset => mapper.with_dtype(DataType::Duration(TimeUnit::Milliseconds)),
            #[cfg(feature = "timezones")]
            DSTOffset => mapper.with_dtype(DataType::Duration(TimeUnit::Milliseconds)),
            Round => mapper.with_same_dtype(),
            Replace => mapper.with_same_dtype(),
            #[cfg(feature = "timezones")]
            ReplaceTimeZone(tz, _non_existent) => mapper.map_datetime_dtype_timezone(tz.as_ref()),
            DatetimeFunction {
                time_unit,
                time_zone,
            } => Ok(Field::new(
                PlSmallStr::from_static("datetime"),
                DataType::Datetime(*time_unit, time_zone.clone()),
            )),
            Combine(tu) => mapper.try_map_dtype(|dt| match dt {
                DataType::Datetime(_, tz) => Ok(DataType::Datetime(*tu, tz.clone())),
                DataType::Date => Ok(DataType::Datetime(*tu, None)),
                dtype => {
                    polars_bail!(ComputeError: "expected Date or Datetime, got {dtype}")
                },
            }),
        }
    }

    pub fn function_options(&self) -> FunctionOptions {
        use IRTemporalFunction as T;
        match self {
            T::Millennium
            | T::Century
            | T::Year
            | T::IsLeapYear
            | T::IsoYear
            | T::Quarter
            | T::Month
            | T::DaysInMonth
            | T::Week
            | T::WeekDay
            | T::Day
            | T::OrdinalDay
            | T::Time
            | T::Date
            | T::Datetime
            | T::Hour
            | T::Minute
            | T::Second
            | T::Millisecond
            | T::Microsecond
            | T::Nanosecond
            | T::ToString(_)
            | T::TimeStamp(_)
            | T::CastTimeUnit(_)
            | T::WithTimeUnit(_) => FunctionOptions::elementwise(),
            #[cfg(feature = "dtype-duration")]
            T::TotalDays { .. }
            | T::TotalHours { .. }
            | T::TotalMinutes { .. }
            | T::TotalSeconds { .. }
            | T::TotalMilliseconds { .. }
            | T::TotalMicroseconds { .. }
            | T::TotalNanoseconds { .. } => FunctionOptions::elementwise(),
            #[cfg(feature = "timezones")]
            T::ConvertTimeZone(_) => FunctionOptions::elementwise(),
            #[cfg(feature = "month_start")]
            T::MonthStart => FunctionOptions::elementwise(),
            #[cfg(feature = "month_end")]
            T::MonthEnd => FunctionOptions::elementwise(),
            #[cfg(feature = "timezones")]
            T::BaseUtcOffset | T::DSTOffset => FunctionOptions::elementwise(),
            T::Truncate => FunctionOptions::elementwise(),
            #[cfg(feature = "offset_by")]
            T::OffsetBy => FunctionOptions::elementwise(),
            T::Round => FunctionOptions::elementwise(),
            T::Replace => FunctionOptions::elementwise(),
            #[cfg(feature = "dtype-duration")]
            T::Duration(_) => FunctionOptions::elementwise(),
            #[cfg(feature = "timezones")]
            T::ReplaceTimeZone(_, _) => FunctionOptions::elementwise(),
            T::Combine(_) => FunctionOptions::elementwise(),
            T::DatetimeFunction { .. } => {
                FunctionOptions::elementwise().with_flags(|f| f | FunctionFlags::ALLOW_RENAME)
            },
        }
    }
}

impl Display for IRTemporalFunction {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        use IRTemporalFunction::*;
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
