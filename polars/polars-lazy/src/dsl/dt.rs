use super::*;
use polars_time::prelude::TemporalMethods;

/// Specialized expressions for [`Series`] with dates/datetimes.
pub struct DateLikeNameSpace(pub(crate) Expr);

impl DateLikeNameSpace {
    /// Format Date/datetime with a formatting rule
    /// See [chrono strftime/strptime](https://docs.rs/chrono/0.4.19/chrono/format/strftime/index.html).
    pub fn strftime(self, fmt: &str) -> Expr {
        let fmt = fmt.to_string();
        let function = move |s: Series| s.strftime(&fmt);
        self.0
            .map(function, GetOutput::from_type(DataType::Utf8))
            .with_fmt("strftime")
    }

    /// Change the underlying [`TimeUnit`]. And update the data accordingly.
    pub fn cast_time_unit(self, tu: TimeUnit) -> Expr {
        self.0.map(
            move |s| match s.dtype() {
                DataType::Datetime(_, _) => {
                    let ca = s.datetime().unwrap();
                    Ok(ca.cast_time_unit(tu).into_series())
                }
                #[cfg(feature = "dtype-duration")]
                DataType::Duration(_) => {
                    let ca = s.duration().unwrap();
                    Ok(ca.cast_time_unit(tu).into_series())
                }
                dt => Err(PolarsError::ComputeError(
                    format!("Series of dtype {:?} has got no time unit", dt).into(),
                )),
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
                    Ok(ca.into_series())
                }
                #[cfg(feature = "dtype-duration")]
                DataType::Duration(_) => {
                    let mut ca = s.duration().unwrap().clone();
                    ca.set_time_unit(tu);
                    Ok(ca.into_series())
                }
                dt => Err(PolarsError::ComputeError(
                    format!("Series of dtype {:?} has got no time unit", dt).into(),
                )),
            },
            GetOutput::same_type(),
        )
    }

    /// Get the year of a Date/Datetime
    pub fn year(self) -> Expr {
        let function = move |s: Series| s.year().map(|ca| ca.into_series());
        self.0
            .map(function, GetOutput::from_type(DataType::UInt32))
            .with_fmt("dt.year")
    }

    /// Get the month of a Date/Datetime
    pub fn month(self) -> Expr {
        let function = move |s: Series| s.month().map(|ca| ca.into_series());
        self.0
            .map(function, GetOutput::from_type(DataType::UInt32))
            .with_fmt("dt.month")
    }

    /// Extract quarter from underlying NaiveDateTime representation.
    /// Quarters range from 1 to 4.
    pub fn quarter(self) -> Expr {
        let function = move |s: Series| s.quarter().map(|ca| ca.into_series());
        self.0
            .map(function, GetOutput::from_type(DataType::UInt32))
            .with_fmt("dt.quarter")
    }

    /// Extract the week from the underlying Date representation.
    /// Can be performed on Date and Datetime

    /// Returns the ISO week number starting from 1.
    /// The return value ranges from 1 to 53. (The last week of year differs by years.)
    pub fn week(self) -> Expr {
        let function = move |s: Series| s.week().map(|ca| ca.into_series());
        self.0
            .map(function, GetOutput::from_type(DataType::UInt32))
            .with_fmt("dt.week")
    }

    /// Extract the week day from the underlying Date representation.
    /// Can be performed on Date and Datetime.

    /// Returns the weekday number where monday = 0 and sunday = 6
    pub fn weekday(self) -> Expr {
        let function = move |s: Series| s.weekday().map(|ca| ca.into_series());
        self.0
            .map(function, GetOutput::from_type(DataType::UInt32))
            .with_fmt("dt.weekday")
    }

    /// Get the month of a Date/Datetime
    pub fn day(self) -> Expr {
        let function = move |s: Series| s.day().map(|ca| ca.into_series());
        self.0
            .map(function, GetOutput::from_type(DataType::UInt32))
            .with_fmt("day")
    }
    /// Get the ordinal_day of a Date/Datetime
    pub fn ordinal_day(self) -> Expr {
        let function = move |s: Series| s.ordinal_day().map(|ca| ca.into_series());
        self.0
            .map(function, GetOutput::from_type(DataType::UInt32))
            .with_fmt("dt.ordinal_day")
    }
    /// Get the hour of a Datetime/Time64
    pub fn hour(self) -> Expr {
        let function = move |s: Series| s.hour().map(|ca| ca.into_series());
        self.0
            .map(function, GetOutput::from_type(DataType::UInt32))
            .with_fmt("dt.hour")
    }
    /// Get the minute of a Datetime/Time64
    pub fn minute(self) -> Expr {
        let function = move |s: Series| s.minute().map(|ca| ca.into_series());
        self.0
            .map(function, GetOutput::from_type(DataType::UInt32))
            .with_fmt("dt.minute")
    }

    /// Get the second of a Datetime/Time64
    pub fn second(self) -> Expr {
        let function = move |s: Series| s.second().map(|ca| ca.into_series());
        self.0
            .map(function, GetOutput::from_type(DataType::UInt32))
            .with_fmt("dt.second")
    }
    /// Get the nanosecond of a Time64
    pub fn nanosecond(self) -> Expr {
        let function = move |s: Series| s.nanosecond().map(|ca| ca.into_series());
        self.0
            .map(function, GetOutput::from_type(DataType::UInt32))
            .with_fmt("dt.nanosecond")
    }

    pub fn timestamp(self, tu: TimeUnit) -> Expr {
        self.0
            .map(
                move |s| s.timestamp(tu).map(|ca| ca.into_series()),
                GetOutput::from_type(DataType::Int64),
            )
            .with_fmt("dt.timestamp")
    }

    /// Offset this `Date/Datetime` by a given offset [`Duration`].
    /// This will take leap years/ months into account.
    #[cfg(feature = "date_offset")]
    pub fn offset_by(self, by: Duration) -> Expr {
        self.0
            .map_private(FunctionExpr::DateOffset(by), "dt.offset_by")
    }
}
