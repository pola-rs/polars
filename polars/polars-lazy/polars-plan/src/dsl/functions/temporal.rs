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

/// Arguments used by [`datetime`] in order to produce an `Expr` of `Datetime`
///
/// Construct a `DatetimeArgs` with `DatetimeArgs::new(y, m, d)`. This will set the other time units to `lit(0)`. You
/// can then set the other fields with the `with_*` methods, or use `with_hms` to set `hour`, `minute`, and `second` all
/// at once.
///
/// # Examples
/// ```
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
            hour: lit(0),
            minute: lit(0),
            second: lit(0),
            microsecond: lit(0),
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
}

/// Construct a column of `Datetime` from the provided [`DatetimeArgs`].
#[cfg(feature = "temporal")]
pub fn datetime(args: DatetimeArgs) -> Expr {
    use polars_core::export::chrono::NaiveDate;
    use polars_core::utils::CustomIterTools;

    let year = args.year;
    let month = args.month;
    let day = args.day;
    let hour = args.hour;
    let minute = args.minute;
    let second = args.second;
    let microsecond = args.microsecond;

    let function = SpecialEq::new(Arc::new(move |s: &mut [Series]| {
        assert_eq!(s.len(), 7);
        let max_len = s.iter().map(|s| s.len()).max().unwrap();
        let mut year = s[0].cast(&DataType::Int32)?;
        if year.len() < max_len {
            year = year.new_from_index(0, max_len)
        }
        let year = year.i32()?;
        let mut month = s[1].cast(&DataType::UInt32)?;
        if month.len() < max_len {
            month = month.new_from_index(0, max_len);
        }
        let month = month.u32()?;
        let mut day = s[2].cast(&DataType::UInt32)?;
        if day.len() < max_len {
            day = day.new_from_index(0, max_len);
        }
        let day = day.u32()?;
        let mut hour = s[3].cast(&DataType::UInt32)?;
        if hour.len() < max_len {
            hour = hour.new_from_index(0, max_len);
        }
        let hour = hour.u32()?;

        let mut minute = s[4].cast(&DataType::UInt32)?;
        if minute.len() < max_len {
            minute = minute.new_from_index(0, max_len);
        }
        let minute = minute.u32()?;

        let mut second = s[5].cast(&DataType::UInt32)?;
        if second.len() < max_len {
            second = second.new_from_index(0, max_len);
        }
        let second = second.u32()?;

        let mut microsecond = s[6].cast(&DataType::UInt32)?;
        if microsecond.len() < max_len {
            microsecond = microsecond.new_from_index(0, max_len);
        }
        let microsecond = microsecond.u32()?;

        let ca: Int64Chunked = year
            .into_iter()
            .zip(month.into_iter())
            .zip(day.into_iter())
            .zip(hour.into_iter())
            .zip(minute.into_iter())
            .zip(second.into_iter())
            .zip(microsecond.into_iter())
            .map(|((((((y, m), d), h), mnt), s), us)| {
                if let (Some(y), Some(m), Some(d), Some(h), Some(mnt), Some(s), Some(us)) =
                    (y, m, d, h, mnt, s, us)
                {
                    NaiveDate::from_ymd_opt(y, m, d)
                        .and_then(|nd| nd.and_hms_micro_opt(h, mnt, s, us))
                        .map(|ndt| ndt.timestamp_micros())
                } else {
                    None
                }
            })
            .collect_trusted();

        Ok(Some(
            ca.into_datetime(TimeUnit::Microseconds, None).into_series(),
        ))
    }) as Arc<dyn SeriesUdf>);

    Expr::AnonymousFunction {
        input: vec![year, month, day, hour, minute, second, microsecond],
        function,
        output_type: GetOutput::from_type(DataType::Datetime(TimeUnit::Microseconds, None)),
        options: FunctionOptions {
            collect_groups: ApplyOptions::ApplyFlat,
            input_wildcard_expansion: true,
            fmt_str: "datetime",
            ..Default::default()
        },
    }
    .alias("datetime")
}

/// Arguments used by [`duration`] in order to produce an `Expr` of `Duration`
///
/// To construct a `DurationArgs`, use struct literal syntax with `..Default::default()` to leave unspecified fields at
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
    /// Create a new `DurationArgs` with all fields set to `lit(0)`. Use the `with_*` methods to set the fields.
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

/// Construct a column of `Duration` from the provided [`DurationArgs`]
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
