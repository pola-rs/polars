//! The `i64` space a temporal group-by computes its windows in.
//!
//! Both engines cast the index column to a `Datetime` before computing windows, and work in
//! that dtype's physical `i64` values: `Datetime` in its own time unit and zone, `Date` as
//! microseconds, integers reinterpreted as nanoseconds. [`IndexSpace`] holds that mapping in
//! one place for the in-memory engine and the streaming nodes.

use polars_arrow::legacy::time_zone::Tz;
use polars_core::prelude::*;

/// The `i64` space a temporal group-by computes its windows in: the unit and zone of the
/// `Datetime` the index is cast to.
#[derive(Clone)]
pub struct IndexSpace {
    pub time_unit: TimeUnit,
    time_zone: Option<TimeZone>,
    tz: Option<Tz>,
    index_dtype: DataType,
}

impl std::fmt::Debug for IndexSpace {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IndexSpace")
            .field("time_unit", &self.time_unit)
            .field("time_zone", &self.time_zone)
            .field("index_dtype", &self.index_dtype)
            .finish()
    }
}

impl IndexSpace {
    /// The space for a `group_by_dynamic` index of `index_dtype`, which must be a `Datetime`,
    /// `Date` or signed integer.
    ///
    /// A zone `chrono-tz` cannot parse, which needs `POLARS_IGNORE_TIMEZONE_PARSE_ERROR=1` or
    /// the `timezones` feature to be off, is kept in `time_zone` but ignored for the
    /// arithmetic: the engines compute such windows as if the index had no zone, and so does
    /// this.
    pub fn dynamic(index_dtype: &DataType) -> PolarsResult<Self> {
        Self::new(index_dtype, false)
    }

    /// The space for a `rolling` index of `index_dtype`, which may also be an unsigned integer.
    /// See [`Self::dynamic`] for the zone handling.
    pub fn rolling(index_dtype: &DataType) -> PolarsResult<Self> {
        Self::new(index_dtype, true)
    }

    fn new(index_dtype: &DataType, allow_unsigned: bool) -> PolarsResult<Self> {
        let DataType::Datetime(time_unit, time_zone) =
            window_datetime_dtype(index_dtype, allow_unsigned)?
        else {
            unreachable!()
        };
        #[cfg(feature = "timezones")]
        let tz = time_zone.as_ref().and_then(|tz| tz.parse::<Tz>().ok());
        #[cfg(not(feature = "timezones"))]
        let tz = None;
        Ok(Self {
            time_unit,
            time_zone,
            tz,
            index_dtype: index_dtype.clone(),
        })
    }

    /// The zone the arithmetic runs in, as `Duration` and `Window` take it. `None` when the
    /// index has no zone or one the engines ignore, see [`Self::dynamic`].
    pub fn tz(&self) -> Option<&Tz> {
        self.tz.as_ref()
    }

    /// The zone of the `Datetime` dtype of this space.
    pub fn time_zone(&self) -> Option<&TimeZone> {
        self.time_zone.as_ref()
    }

    /// The `Datetime` dtype of this space.
    pub fn window_dtype(&self) -> DataType {
        DataType::Datetime(self.time_unit, self.time_zone.clone())
    }

    /// `index`, a column of the dtype this space was made for, cast to [`Self::window_dtype`].
    pub fn cast_to_space(&self, index: &Column) -> PolarsResult<Column> {
        debug_assert_eq!(index.dtype(), &self.index_dtype);
        match &self.index_dtype {
            DataType::Datetime(_, _) => Ok(index.clone()),
            DataType::Int32 | DataType::UInt32 | DataType::UInt64 => {
                index.cast(&DataType::Int64)?.cast(&self.window_dtype())
            },
            _ => index.cast(&self.window_dtype()),
        }
    }

    /// `column`, in [`Self::window_dtype`], cast back to the dtype this space was made for.
    pub fn cast_from_space(&self, column: &Column) -> PolarsResult<Column> {
        debug_assert_eq!(column.dtype(), &self.window_dtype());
        match &self.index_dtype {
            DataType::Datetime(_, _) => Ok(column.clone()),
            dt if dt.is_integer() => column.cast(&DataType::Int64)?.cast(dt),
            dt => column.cast(dt),
        }
    }

    /// `column`, in [`Self::window_dtype`], cast to the dtype of a `group_by_dynamic` window
    /// boundary: the index dtype, except that boundaries of a `Date` index stay a `Datetime`
    /// because a window need not start on a day.
    pub fn cast_to_boundary(&self, column: &Column) -> PolarsResult<Column> {
        match &self.index_dtype {
            DataType::Date => Ok(column.clone()),
            _ => self.cast_from_space(column),
        }
    }
}

/// The `Datetime` dtype an index column is cast to before its windows are computed: `Date`
/// becomes microseconds since the epoch and integers are reinterpreted as nanoseconds.
fn window_datetime_dtype(index_dtype: &DataType, allow_unsigned: bool) -> PolarsResult<DataType> {
    use DataType::*;
    Ok(match index_dtype {
        Datetime(_, _) => index_dtype.clone(),
        Date => Datetime(TimeUnit::Microseconds, None),
        Int32 | Int64 => Datetime(TimeUnit::Nanoseconds, None),
        UInt32 | UInt64 if allow_unsigned => Datetime(TimeUnit::Nanoseconds, None),
        dt if allow_unsigned => polars_bail!(
            ComputeError:
            "expected any of the following dtypes: {{ Date, Datetime, Int32, Int64, UInt32, UInt64 }}, got {}",
            dt
        ),
        dt => polars_bail!(
            ComputeError:
            "expected any of the following dtypes: {{ Date, Datetime, Int32, Int64 }}, got {}",
            dt
        ),
    })
}
