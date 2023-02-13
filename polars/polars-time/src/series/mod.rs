mod _trait;
mod implementations;
use std::ops::Deref;
use std::sync::Arc;

use polars_core::prelude::*;
use polars_core::utils::Wrap;
pub use SeriesOpsTime;

pub use self::_trait::*;
use crate::chunkedarray::*;

type SeriesOpsRef = Arc<dyn SeriesOpsTime>;

pub trait IntoSeriesOps {
    fn to_ops(&self) -> SeriesOpsRef;
}

impl IntoSeriesOps for Series {
    fn to_ops(&self) -> SeriesOpsRef {
        match self.dtype() {
            DataType::Int8 => self.i8().unwrap().to_ops(),
            DataType::Int16 => self.i16().unwrap().to_ops(),
            DataType::Int32 => self.i32().unwrap().to_ops(),
            DataType::Int64 => self.i64().unwrap().to_ops(),
            DataType::UInt8 => self.u8().unwrap().to_ops(),
            DataType::UInt16 => self.u16().unwrap().to_ops(),
            DataType::UInt32 => self.u32().unwrap().to_ops(),
            DataType::UInt64 => self.u64().unwrap().to_ops(),
            DataType::Float32 => self.f32().unwrap().to_ops(),
            DataType::Float64 => self.f64().unwrap().to_ops(),
            #[cfg(feature = "dtype-categorical")]
            DataType::Categorical(_) => self.categorical().unwrap().to_ops(),
            DataType::Boolean => self.bool().unwrap().to_ops(),
            DataType::Utf8 => self.utf8().unwrap().to_ops(),
            #[cfg(feature = "dtype-date")]
            DataType::Date => self.date().unwrap().to_ops(),
            #[cfg(feature = "dtype-datetime")]
            DataType::Datetime(_, _) => self.datetime().unwrap().to_ops(),
            #[cfg(feature = "dtype-duration")]
            DataType::Duration(_) => self.duration().unwrap().to_ops(),
            #[cfg(feature = "dtype-time")]
            DataType::Time => self.time().unwrap().to_ops(),
            DataType::List(_) => self.list().unwrap().to_ops(),
            #[cfg(feature = "dtype-struct")]
            DataType::Struct(_) => self.struct_().unwrap().to_ops(),
            _ => unimplemented!(),
        }
    }
}

impl<T: PolarsIntegerType> IntoSeriesOps for &ChunkedArray<T>
where
    T::Native: NumericNative,
{
    fn to_ops(&self) -> SeriesOpsRef {
        Arc::new(WrapInt((*self).clone()))
    }
}

#[repr(transparent)]
pub(crate) struct WrapFloat<T>(pub T);

#[repr(transparent)]
pub(crate) struct WrapInt<T>(pub T);

impl IntoSeriesOps for Float32Chunked {
    fn to_ops(&self) -> SeriesOpsRef {
        Arc::new(WrapFloat(self.clone()))
    }
}

impl IntoSeriesOps for Float64Chunked {
    fn to_ops(&self) -> SeriesOpsRef {
        Arc::new(WrapFloat(self.clone()))
    }
}

macro_rules! into_ops_impl_wrapped {
    ($tp:ty) => {
        impl IntoSeriesOps for $tp {
            fn to_ops(&self) -> SeriesOpsRef {
                Arc::new(Wrap(self.clone()))
            }
        }
    };
}

into_ops_impl_wrapped!(Utf8Chunked);
into_ops_impl_wrapped!(BooleanChunked);
#[cfg(feature = "dtype-date")]
into_ops_impl_wrapped!(DateChunked);
#[cfg(feature = "dtype-time")]
into_ops_impl_wrapped!(TimeChunked);
#[cfg(feature = "dtype-duration")]
into_ops_impl_wrapped!(DurationChunked);
#[cfg(feature = "dtype-datetime")]
into_ops_impl_wrapped!(DatetimeChunked);
#[cfg(feature = "dtype-struct")]
into_ops_impl_wrapped!(StructChunked);
into_ops_impl_wrapped!(ListChunked);

#[cfg(feature = "dtype-categorical")]
into_ops_impl_wrapped!(CategoricalChunked);

#[cfg(feature = "object")]
impl<T: PolarsObject> IntoSeriesOps for ObjectChunked<T> {
    fn to_ops(&self) -> SeriesOpsRef {
        Arc::new(Wrap(self.clone()))
    }
}

pub trait AsSeries {
    fn as_series(&self) -> &Series;
}

impl AsSeries for Series {
    fn as_series(&self) -> &Series {
        self
    }
}

pub trait TemporalMethods: AsSeries {
    /// Extract hour from underlying NaiveDateTime representation.
    /// Returns the hour number from 0 to 23.
    fn hour(&self) -> PolarsResult<UInt32Chunked> {
        let s = self.as_series();
        match s.dtype() {
            #[cfg(feature = "dtype-datetime")]
            DataType::Datetime(_, _) => s.datetime().map(|ca| ca.hour()),
            #[cfg(feature = "dtype-time")]
            DataType::Time => s.time().map(|ca| ca.hour()),
            _ => Err(PolarsError::InvalidOperation(
                format!("operation not supported on dtype {:?}", s.dtype()).into(),
            )),
        }
    }

    /// Extract minute from underlying NaiveDateTime representation.
    /// Returns the minute number from 0 to 59.
    fn minute(&self) -> PolarsResult<UInt32Chunked> {
        let s = self.as_series();
        match s.dtype() {
            #[cfg(feature = "dtype-datetime")]
            DataType::Datetime(_, _) => s.datetime().map(|ca| ca.minute()),
            #[cfg(feature = "dtype-time")]
            DataType::Time => s.time().map(|ca| ca.minute()),
            _ => Err(PolarsError::InvalidOperation(
                format!("operation not supported on dtype {:?}", s.dtype()).into(),
            )),
        }
    }

    /// Extract second from underlying NaiveDateTime representation.
    /// Returns the second number from 0 to 59.
    fn second(&self) -> PolarsResult<UInt32Chunked> {
        let s = self.as_series();
        match s.dtype() {
            #[cfg(feature = "dtype-datetime")]
            DataType::Datetime(_, _) => s.datetime().map(|ca| ca.second()),
            #[cfg(feature = "dtype-time")]
            DataType::Time => s.time().map(|ca| ca.second()),
            _ => Err(PolarsError::InvalidOperation(
                format!("operation not supported on dtype {:?}", s.dtype()).into(),
            )),
        }
    }

    /// Returns the number of nanoseconds since the whole non-leap second.
    /// The range from 1,000,000,000 to 1,999,999,999 represents the leap second.
    fn nanosecond(&self) -> PolarsResult<UInt32Chunked> {
        let s = self.as_series();
        match s.dtype() {
            #[cfg(feature = "dtype-datetime")]
            DataType::Datetime(_, _) => s.datetime().map(|ca| ca.nanosecond()),
            #[cfg(feature = "dtype-time")]
            DataType::Time => s.time().map(|ca| ca.nanosecond()),
            _ => Err(PolarsError::InvalidOperation(
                format!("operation not supported on dtype {:?}", s.dtype()).into(),
            )),
        }
    }

    /// Extract day from underlying NaiveDateTime representation.
    /// Returns the day of month starting from 1.
    ///
    /// The return value ranges from 1 to 31. (The last day of month differs by months.)
    fn day(&self) -> PolarsResult<UInt32Chunked> {
        let s = self.as_series();
        match s.dtype() {
            #[cfg(feature = "dtype-date")]
            DataType::Date => s.date().map(|ca| ca.day()),
            #[cfg(feature = "dtype-datetime")]
            DataType::Datetime(_, _) => s.datetime().map(|ca| ca.day()),
            _ => Err(PolarsError::InvalidOperation(
                format!("operation not supported on dtype {:?}", s.dtype()).into(),
            )),
        }
    }
    /// Returns the weekday number where monday = 0 and sunday = 6
    fn weekday(&self) -> PolarsResult<UInt32Chunked> {
        let s = self.as_series();
        match s.dtype() {
            #[cfg(feature = "dtype-date")]
            DataType::Date => s.date().map(|ca| ca.weekday()),
            #[cfg(feature = "dtype-datetime")]
            DataType::Datetime(_, _) => s.datetime().map(|ca| ca.weekday()),
            _ => Err(PolarsError::InvalidOperation(
                format!("operation not supported on dtype {:?}", s.dtype()).into(),
            )),
        }
    }

    /// Returns the ISO week number starting from 1.
    /// The return value ranges from 1 to 53. (The last week of year differs by years.)
    fn week(&self) -> PolarsResult<UInt32Chunked> {
        let s = self.as_series();
        match s.dtype() {
            #[cfg(feature = "dtype-date")]
            DataType::Date => s.date().map(|ca| ca.week()),
            #[cfg(feature = "dtype-datetime")]
            DataType::Datetime(_, _) => s.datetime().map(|ca| ca.week()),
            _ => Err(PolarsError::InvalidOperation(
                format!("operation not supported on dtype {:?}", s.dtype()).into(),
            )),
        }
    }

    /// Returns the day of year starting from 1.
    ///
    /// The return value ranges from 1 to 366. (The last day of year differs by years.)
    fn ordinal_day(&self) -> PolarsResult<UInt32Chunked> {
        let s = self.as_series();
        match s.dtype() {
            #[cfg(feature = "dtype-date")]
            DataType::Date => s.date().map(|ca| ca.ordinal()),
            #[cfg(feature = "dtype-datetime")]
            DataType::Datetime(_, _) => s.datetime().map(|ca| ca.ordinal()),
            _ => Err(PolarsError::InvalidOperation(
                format!("operation not supported on dtype {:?}", s.dtype()).into(),
            )),
        }
    }

    /// Extract year from underlying NaiveDateTime representation.
    /// Returns the year number in the calendar date.
    fn year(&self) -> PolarsResult<Int32Chunked> {
        let s = self.as_series();
        match s.dtype() {
            #[cfg(feature = "dtype-date")]
            DataType::Date => s.date().map(|ca| ca.year()),
            #[cfg(feature = "dtype-datetime")]
            DataType::Datetime(_, _) => s.datetime().map(|ca| ca.year()),
            _ => Err(PolarsError::InvalidOperation(
                format!("operation not supported on dtype {:?}", s.dtype()).into(),
            )),
        }
    }

    fn iso_year(&self) -> PolarsResult<Int32Chunked> {
        let s = self.as_series();
        match s.dtype() {
            #[cfg(feature = "dtype-date")]
            DataType::Date => s.date().map(|ca| ca.iso_year()),
            #[cfg(feature = "dtype-datetime")]
            DataType::Datetime(_, _) => s.datetime().map(|ca| ca.iso_year()),
            _ => Err(PolarsError::InvalidOperation(
                format!("operation not supported on dtype {:?}", s.dtype()).into(),
            )),
        }
    }

    /// Extract ordinal year from underlying NaiveDateTime representation.
    /// Returns the year number in the calendar date.
    fn ordinal_year(&self) -> PolarsResult<Int32Chunked> {
        let s = self.as_series();
        match s.dtype() {
            #[cfg(feature = "dtype-date")]
            DataType::Date => s.date().map(|ca| ca.year()),
            #[cfg(feature = "dtype-datetime")]
            DataType::Datetime(_, _) => s.datetime().map(|ca| ca.year()),
            _ => Err(PolarsError::InvalidOperation(
                format!("operation not supported on dtype {:?}", s.dtype()).into(),
            )),
        }
    }

    /// Extract quarter from underlying NaiveDateTime representation.
    /// Quarters range from 1 to 4.
    fn quarter(&self) -> PolarsResult<UInt32Chunked> {
        let s = self.as_series();
        match s.dtype() {
            #[cfg(feature = "dtype-date")]
            DataType::Date => s.date().map(|ca| ca.quarter()),
            #[cfg(feature = "dtype-datetime")]
            DataType::Datetime(_, _) => s.datetime().map(|ca| ca.quarter()),
            _ => Err(PolarsError::InvalidOperation(
                format!("operation not supported on dtype {:?}", s.dtype()).into(),
            )),
        }
    }

    /// Extract month from underlying NaiveDateTime representation.
    /// Returns the month number starting from 1.
    ///
    /// The return value ranges from 1 to 12.
    fn month(&self) -> PolarsResult<UInt32Chunked> {
        let s = self.as_series();
        match s.dtype() {
            #[cfg(feature = "dtype-date")]
            DataType::Date => s.date().map(|ca| ca.month()),
            #[cfg(feature = "dtype-datetime")]
            DataType::Datetime(_, _) => s.datetime().map(|ca| ca.month()),
            _ => Err(PolarsError::InvalidOperation(
                format!("operation not supported on dtype {:?}", s.dtype()).into(),
            )),
        }
    }

    /// Format Date/Datetimewith a `fmt` rule. See [chrono strftime/strptime](https://docs.rs/chrono/0.4.19/chrono/format/strftime/index.html).
    fn strftime(&self, fmt: &str) -> PolarsResult<Series> {
        let s = self.as_series();
        match s.dtype() {
            #[cfg(feature = "dtype-date")]
            DataType::Date => s.date().map(|ca| ca.strftime(fmt).into_series()),
            #[cfg(feature = "dtype-datetime")]
            DataType::Datetime(_, _) => {
                s.datetime().map(|ca| Ok(ca.strftime(fmt)?.into_series()))?
            }
            #[cfg(feature = "dtype-time")]
            DataType::Time => s.time().map(|ca| ca.strftime(fmt).into_series()),
            _ => Err(PolarsError::InvalidOperation(
                format!("operation not supported on dtype {:?}", s.dtype()).into(),
            )),
        }
    }

    #[cfg(all(feature = "dtype-date", feature = "dtype-datetime"))]
    /// Convert date(time) object to timestamp in [`TimeUnit`].
    fn timestamp(&self, tu: TimeUnit) -> PolarsResult<Int64Chunked> {
        let s = self.as_series();
        if matches!(s.dtype(), DataType::Time) {
            Err(PolarsError::ComputeError(
                "Cannot compute timestamp of a series with dtype 'Time'".into(),
            ))
        } else {
            s.cast(&DataType::Datetime(tu, None))
                .map(|s| s.datetime().unwrap().deref().clone())
        }
    }
}

impl<T: ?Sized + AsSeries> TemporalMethods for T {}
