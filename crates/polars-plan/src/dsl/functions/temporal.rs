use chrono::{Datelike, Timelike};

use super::*;

macro_rules! impl_unit_setter {
    ($fn_name:ident($field:ident)) => {
        #[doc = concat!("Set the ", stringify!($field))]
        pub fn $fn_name(mut self, n: Expr) -> Self {
            self.$field = n.into();
            self
        }
    };
}

/// Arguments used by `datetime` in order to produce an [`Expr`] of Datetime
///
/// Construct a [`DatetimeArgs`] with `DatetimeArgs::new(y, m, d)`. This will set the other time units to `lit(0)`. You
/// can then set the other fields with the `with_*` methods, or use `with_hms` to set `hour`, `minute`, and `second` all
/// at once.
///
/// # Examples
/// ```
/// use polars_plan::prelude::*;
/// // construct a DatetimeArgs set to July 20, 1969 at 20:17
/// let args = DatetimeArgs::new(lit(1969), lit(7), lit(20)).with_hms(lit(20), lit(17), lit(0));
/// // or
/// let args = DatetimeArgs::new(lit(1969), lit(7), lit(20)).with_hour(lit(20)).with_minute(lit(17));
///
/// // construct a DatetimeArgs using existing columns
/// let args = DatetimeArgs::new(lit(2023), col("month"), col("day"));
/// ```
#[derive(Debug, Clone)]
pub struct DatetimeArgs {
    pub year: Expr,
    pub month: Expr,
    pub day: Expr,
    pub hour: Expr,
    pub minute: Expr,
    pub second: Expr,
    pub microsecond: Expr,
    pub time_unit: TimeUnit,
    pub time_zone: Option<TimeZone>,
    pub ambiguous: Expr,
}

impl Default for DatetimeArgs {
    fn default() -> Self {
        Self {
            year: lit(1970),
            month: lit(1),
            day: lit(1),
            hour: lit(0),
            minute: lit(0),
            second: lit(0),
            microsecond: lit(0),
            time_unit: TimeUnit::Microseconds,
            time_zone: None,
            ambiguous: lit(String::from("raise")),
        }
    }
}

impl DatetimeArgs {
    /// Construct a new `DatetimeArgs` set to `year`, `month`, `day`
    ///
    /// Other fields default to `lit(0)`. Use the `with_*` methods to set them.
    pub fn new(year: Expr, month: Expr, day: Expr) -> Self {
        Self {
            year,
            month,
            day,
            ..Default::default()
        }
    }

    /// Set `hour`, `minute`, and `second`
    ///
    /// Equivalent to
    /// ```ignore
    /// self.with_hour(hour)
    ///     .with_minute(minute)
    ///     .with_second(second)
    /// ```
    pub fn with_hms(self, hour: Expr, minute: Expr, second: Expr) -> Self {
        Self {
            hour,
            minute,
            second,
            ..self
        }
    }

    impl_unit_setter!(with_year(year));
    impl_unit_setter!(with_month(month));
    impl_unit_setter!(with_day(day));
    impl_unit_setter!(with_hour(hour));
    impl_unit_setter!(with_minute(minute));
    impl_unit_setter!(with_second(second));
    impl_unit_setter!(with_microsecond(microsecond));

    pub fn with_time_unit(self, time_unit: TimeUnit) -> Self {
        Self { time_unit, ..self }
    }
    #[cfg(feature = "timezones")]
    pub fn with_time_zone(self, time_zone: Option<TimeZone>) -> Self {
        Self { time_zone, ..self }
    }
    #[cfg(feature = "timezones")]
    pub fn with_ambiguous(self, ambiguous: Expr) -> Self {
        Self { ambiguous, ..self }
    }

    fn all_literal(&self) -> bool {
        use Expr::*;
        [
            &self.year,
            &self.month,
            &self.day,
            &self.hour,
            &self.minute,
            &self.second,
            &self.microsecond,
        ]
        .iter()
        .all(|e| matches!(e, Literal(_)))
    }

    fn as_literal(&self) -> Option<Expr> {
        if self.time_zone.is_some() || !self.all_literal() {
            return None;
        };
        let Expr::Literal(lv) = &self.year else {
            unreachable!()
        };
        let year = lv.to_any_value()?.extract()?;
        let Expr::Literal(lv) = &self.month else {
            unreachable!()
        };
        let month = lv.to_any_value()?.extract()?;
        let Expr::Literal(lv) = &self.day else {
            unreachable!()
        };
        let day = lv.to_any_value()?.extract()?;
        let Expr::Literal(lv) = &self.hour else {
            unreachable!()
        };
        let hour = lv.to_any_value()?.extract()?;
        let Expr::Literal(lv) = &self.minute else {
            unreachable!()
        };
        let minute = lv.to_any_value()?.extract()?;
        let Expr::Literal(lv) = &self.second else {
            unreachable!()
        };
        let second = lv.to_any_value()?.extract()?;
        let Expr::Literal(lv) = &self.microsecond else {
            unreachable!()
        };
        let ms: u32 = lv.to_any_value()?.extract()?;

        let dt = chrono::NaiveDateTime::default()
            .with_year(year)?
            .with_month(month)?
            .with_day(day)?
            .with_hour(hour)?
            .with_minute(minute)?
            .with_second(second)?
            .with_nanosecond(ms * 1000)?;

        let ts = match self.time_unit {
            TimeUnit::Milliseconds => dt.and_utc().timestamp_millis(),
            TimeUnit::Microseconds => dt.and_utc().timestamp_micros(),
            TimeUnit::Nanoseconds => dt.and_utc().timestamp_nanos_opt()?,
        };

        Some(Expr::Literal(LiteralValue::DateTime(ts, self.time_unit, None)).alias("datetime"))
    }
}

/// Construct a column of `Datetime` from the provided [`DatetimeArgs`].
pub fn datetime(args: DatetimeArgs) -> Expr {
    if let Some(e) = args.as_literal() {
        return e;
    }

    let year = args.year;
    let month = args.month;
    let day = args.day;
    let hour = args.hour;
    let minute = args.minute;
    let second = args.second;
    let microsecond = args.microsecond;
    let time_unit = args.time_unit;
    let time_zone = args.time_zone;
    let ambiguous = args.ambiguous;

    let input = vec![
        year,
        month,
        day,
        hour,
        minute,
        second,
        microsecond,
        ambiguous,
    ];

    Expr::Function {
        input,
        function: FunctionExpr::TemporalExpr(TemporalFunction::DatetimeFunction {
            time_unit,
            time_zone,
        }),
        options: FunctionOptions {
            collect_groups: ApplyOptions::ElementWise,
            flags: FunctionFlags::default()
                | FunctionFlags::INPUT_WILDCARD_EXPANSION
                | FunctionFlags::ALLOW_RENAME,
            fmt_str: "datetime",
            ..Default::default()
        },
    }
}

/// Arguments used by `duration` in order to produce an [`Expr`] of [`Duration`]
///
/// To construct a [`DurationArgs`], use struct literal syntax with `..Default::default()` to leave unspecified fields at
/// their default value of `lit(0)`, as demonstrated below.
///
/// ```
/// # use polars_plan::prelude::*;
/// let args = DurationArgs {
///     days: lit(5),
///     hours: col("num_hours"),
///     minutes: col("num_minutes"),
///     ..Default::default()  // other fields are lit(0)
/// };
/// ```
/// If you prefer builder syntax, `with_*` methods are also available.
/// ```
/// # use polars_plan::prelude::*;
/// let args = DurationArgs::new().with_weeks(lit(42)).with_hours(lit(84));
/// ```
#[derive(Debug, Clone)]
pub struct DurationArgs {
    pub weeks: Expr,
    pub days: Expr,
    pub hours: Expr,
    pub minutes: Expr,
    pub seconds: Expr,
    pub milliseconds: Expr,
    pub microseconds: Expr,
    pub nanoseconds: Expr,
    pub time_unit: TimeUnit,
}

impl Default for DurationArgs {
    fn default() -> Self {
        Self {
            weeks: lit(0),
            days: lit(0),
            hours: lit(0),
            minutes: lit(0),
            seconds: lit(0),
            milliseconds: lit(0),
            microseconds: lit(0),
            nanoseconds: lit(0),
            time_unit: TimeUnit::Microseconds,
        }
    }
}

impl DurationArgs {
    /// Create a new [`DurationArgs`] with all fields set to `lit(0)`. Use the `with_*` methods to set the fields.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set `hours`, `minutes`, and `seconds`
    ///
    /// Equivalent to:
    ///
    /// ```ignore
    /// self.with_hours(hours)
    ///     .with_minutes(minutes)
    ///     .with_seconds(seconds)
    /// ```
    pub fn with_hms(self, hours: Expr, minutes: Expr, seconds: Expr) -> Self {
        Self {
            hours,
            minutes,
            seconds,
            ..self
        }
    }

    /// Set `milliseconds`, `microseconds`, and `nanoseconds`
    ///
    /// Equivalent to
    /// ```ignore
    /// self.with_milliseconds(milliseconds)
    ///     .with_microseconds(microseconds)
    ///     .with_nanoseconds(nanoseconds)
    /// ```
    pub fn with_fractional_seconds(
        self,
        milliseconds: Expr,
        microseconds: Expr,
        nanoseconds: Expr,
    ) -> Self {
        Self {
            milliseconds,
            microseconds,
            nanoseconds,
            ..self
        }
    }

    impl_unit_setter!(with_weeks(weeks));
    impl_unit_setter!(with_days(days));
    impl_unit_setter!(with_hours(hours));
    impl_unit_setter!(with_minutes(minutes));
    impl_unit_setter!(with_seconds(seconds));
    impl_unit_setter!(with_milliseconds(milliseconds));
    impl_unit_setter!(with_microseconds(microseconds));
    impl_unit_setter!(with_nanoseconds(nanoseconds));

    fn all_literal(&self) -> bool {
        use Expr::*;
        [
            &self.weeks,
            &self.days,
            &self.hours,
            &self.seconds,
            &self.minutes,
            &self.milliseconds,
            &self.microseconds,
            &self.nanoseconds,
        ]
        .iter()
        .all(|e| matches!(e, Literal(_)))
    }

    fn as_literal(&self) -> Option<Expr> {
        if !self.all_literal() {
            return None;
        };
        let Expr::Literal(lv) = &self.weeks else {
            unreachable!()
        };
        let weeks = lv.to_any_value()?.extract()?;
        let Expr::Literal(lv) = &self.days else {
            unreachable!()
        };
        let days = lv.to_any_value()?.extract()?;
        let Expr::Literal(lv) = &self.hours else {
            unreachable!()
        };
        let hours = lv.to_any_value()?.extract()?;
        let Expr::Literal(lv) = &self.seconds else {
            unreachable!()
        };
        let seconds = lv.to_any_value()?.extract()?;
        let Expr::Literal(lv) = &self.minutes else {
            unreachable!()
        };
        let minutes = lv.to_any_value()?.extract()?;
        let Expr::Literal(lv) = &self.milliseconds else {
            unreachable!()
        };
        let milliseconds = lv.to_any_value()?.extract()?;
        let Expr::Literal(lv) = &self.microseconds else {
            unreachable!()
        };
        let microseconds = lv.to_any_value()?.extract()?;
        let Expr::Literal(lv) = &self.nanoseconds else {
            unreachable!()
        };
        let nanoseconds = lv.to_any_value()?.extract()?;

        type D = chrono::Duration;
        let delta = D::weeks(weeks)
            + D::days(days)
            + D::hours(hours)
            + D::seconds(seconds)
            + D::minutes(minutes)
            + D::milliseconds(milliseconds)
            + D::microseconds(microseconds)
            + D::nanoseconds(nanoseconds);

        let d = match self.time_unit {
            TimeUnit::Milliseconds => delta.num_milliseconds(),
            TimeUnit::Microseconds => delta.num_microseconds()?,
            TimeUnit::Nanoseconds => delta.num_nanoseconds()?,
        };

        Some(Expr::Literal(LiteralValue::Duration(d, self.time_unit)).alias("duration"))
    }
}

/// Construct a column of [`Duration`] from the provided [`DurationArgs`]
pub fn duration(args: DurationArgs) -> Expr {
    if let Some(e) = args.as_literal() {
        return e;
    }
    Expr::Function {
        input: vec![
            args.weeks,
            args.days,
            args.hours,
            args.minutes,
            args.seconds,
            args.milliseconds,
            args.microseconds,
            args.nanoseconds,
        ],
        function: FunctionExpr::TemporalExpr(TemporalFunction::Duration(args.time_unit)),
        options: FunctionOptions {
            collect_groups: ApplyOptions::ElementWise,
            flags: FunctionFlags::default() | FunctionFlags::INPUT_WILDCARD_EXPANSION,
            ..Default::default()
        },
    }
}
