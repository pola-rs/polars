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
}

/// Construct a column of `Datetime` from the provided [`DatetimeArgs`].
#[cfg(feature = "temporal")]
pub fn datetime(args: DatetimeArgs) -> Expr {
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
            collect_groups: ApplyOptions::ApplyFlat,
            allow_rename: true,
            input_wildcard_expansion: true,
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
/// let args = DurationArgs {
///     days: lit(5),
///     hours: col("num_hours"),
///     minutes: col("num_minutes"),
///     ..Default::default()  // other fields are lit(0)
/// };
/// ```
/// If you prefer builder syntax, `with_*` methods are also available.
/// ```
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
}

/// Construct a column of [`Duration`] from the provided [`DurationArgs`]
#[cfg(feature = "temporal")]
pub fn duration(args: DurationArgs) -> Expr {
    let function = SpecialEq::new(Arc::new(move |s: &mut [Series]| {
        assert_eq!(s.len(), 8);
        if s.iter().any(|s| s.is_empty()) {
            return Ok(Some(Series::new_empty(
                s[0].name(),
                &DataType::Duration(TimeUnit::Nanoseconds),
            )));
        }

        let days = s[0].cast(&DataType::Int64).unwrap();
        let seconds = s[1].cast(&DataType::Int64).unwrap();
        let mut nanoseconds = s[2].cast(&DataType::Int64).unwrap();
        let microseconds = s[3].cast(&DataType::Int64).unwrap();
        let milliseconds = s[4].cast(&DataType::Int64).unwrap();
        let minutes = s[5].cast(&DataType::Int64).unwrap();
        let hours = s[6].cast(&DataType::Int64).unwrap();
        let weeks = s[7].cast(&DataType::Int64).unwrap();

        let max_len = s.iter().map(|s| s.len()).max().unwrap();

        let condition = |s: &Series| {
            // check if not literal 0 || full column
            (s.len() != max_len && s.get(0).unwrap() != AnyValue::Int64(0)) || s.len() == max_len
        };

        if nanoseconds.len() != max_len {
            nanoseconds = nanoseconds.new_from_index(0, max_len);
        }
        if condition(&microseconds) {
            nanoseconds = nanoseconds + (microseconds * 1_000);
        }
        if condition(&milliseconds) {
            nanoseconds = nanoseconds + (milliseconds * 1_000_000);
        }
        if condition(&seconds) {
            nanoseconds = nanoseconds + (seconds * NANOSECONDS);
        }
        if condition(&days) {
            nanoseconds = nanoseconds + (days * NANOSECONDS * SECONDS_IN_DAY);
        }
        if condition(&minutes) {
            nanoseconds = nanoseconds + minutes * NANOSECONDS * 60;
        }
        if condition(&hours) {
            nanoseconds = nanoseconds + hours * NANOSECONDS * 60 * 60;
        }
        if condition(&weeks) {
            nanoseconds = nanoseconds + weeks * NANOSECONDS * SECONDS_IN_DAY * 7;
        }

        nanoseconds
            .cast(&DataType::Duration(TimeUnit::Nanoseconds))
            .map(Some)
    }) as Arc<dyn SeriesUdf>);

    Expr::AnonymousFunction {
        input: vec![
            args.days,
            args.seconds,
            args.nanoseconds,
            args.microseconds,
            args.milliseconds,
            args.minutes,
            args.hours,
            args.weeks,
        ],
        function,
        output_type: GetOutput::from_type(DataType::Duration(TimeUnit::Nanoseconds)),
        options: FunctionOptions {
            collect_groups: ApplyOptions::ApplyFlat,
            input_wildcard_expansion: true,
            fmt_str: "duration",
            ..Default::default()
        },
    }
    .alias("duration")
}
