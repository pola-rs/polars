use std::borrow::Cow;

#[cfg(feature = "dtype-duration")]
use chrono::Duration as ChronoDuration;
#[cfg(feature = "dtype-date")]
use chrono::NaiveDate;
#[cfg(feature = "dtype-datetime")]
use chrono::NaiveDateTime;
#[cfg(feature = "dtype-time")]
use chrono::NaiveTime;

use crate::chunked_array::builder::{get_list_builder, AnonymousListBuilder};
use crate::prelude::*;

pub trait NamedFrom<T, Phantom: ?Sized> {
    /// Initialize by name and values.
    fn new(name: &str, _: T) -> Self;
}

pub trait NamedFromOwned<T> {
    /// Initialize by name and values.
    fn from_vec(name: &str, _: T) -> Self;
}

macro_rules! impl_named_from_owned {
    ($type:ty, $polars_type:ident) => {
        impl NamedFromOwned<$type> for Series {
            fn from_vec(name: &str, v: $type) -> Self {
                ChunkedArray::<$polars_type>::from_vec(name, v).into_series()
            }
        }
    };
}

#[cfg(feature = "dtype-i8")]
impl_named_from_owned!(Vec<i8>, Int8Type);
#[cfg(feature = "dtype-i16")]
impl_named_from_owned!(Vec<i16>, Int16Type);
impl_named_from_owned!(Vec<i32>, Int32Type);
impl_named_from_owned!(Vec<i64>, Int64Type);
#[cfg(feature = "dtype-u8")]
impl_named_from_owned!(Vec<u8>, UInt8Type);
#[cfg(feature = "dtype-u16")]
impl_named_from_owned!(Vec<u16>, UInt16Type);
impl_named_from_owned!(Vec<u32>, UInt32Type);
impl_named_from_owned!(Vec<u64>, UInt64Type);
impl_named_from_owned!(Vec<f32>, Float32Type);
impl_named_from_owned!(Vec<f64>, Float64Type);

macro_rules! impl_named_from {
    ($type:ty, $polars_type:ident, $method:ident) => {
        impl<T: AsRef<$type>> NamedFrom<T, $type> for Series {
            fn new(name: &str, v: T) -> Self {
                ChunkedArray::<$polars_type>::$method(name, v.as_ref()).into_series()
            }
        }
        impl<T: AsRef<$type>> NamedFrom<T, $type> for ChunkedArray<$polars_type> {
            fn new(name: &str, v: T) -> Self {
                ChunkedArray::<$polars_type>::$method(name, v.as_ref())
            }
        }
    };
}

impl_named_from!([String], Utf8Type, from_slice);
#[cfg(feature = "dtype-binary")]
impl_named_from!([Vec<u8>], BinaryType, from_slice);
impl_named_from!([bool], BooleanType, from_slice);
#[cfg(feature = "dtype-u8")]
impl_named_from!([u8], UInt8Type, from_slice);
#[cfg(feature = "dtype-u16")]
impl_named_from!([u16], UInt16Type, from_slice);
impl_named_from!([u32], UInt32Type, from_slice);
impl_named_from!([u64], UInt64Type, from_slice);
#[cfg(feature = "dtype-i8")]
impl_named_from!([i8], Int8Type, from_slice);
#[cfg(feature = "dtype-i16")]
impl_named_from!([i16], Int16Type, from_slice);
impl_named_from!([i32], Int32Type, from_slice);
impl_named_from!([i64], Int64Type, from_slice);
impl_named_from!([f32], Float32Type, from_slice);
impl_named_from!([f64], Float64Type, from_slice);
impl_named_from!([Option<String>], Utf8Type, from_slice_options);
#[cfg(feature = "dtype-binary")]
impl_named_from!([Option<Vec<u8>>], BinaryType, from_slice_options);
impl_named_from!([Option<bool>], BooleanType, from_slice_options);
#[cfg(feature = "dtype-u8")]
impl_named_from!([Option<u8>], UInt8Type, from_slice_options);
#[cfg(feature = "dtype-u16")]
impl_named_from!([Option<u16>], UInt16Type, from_slice_options);
impl_named_from!([Option<u32>], UInt32Type, from_slice_options);
impl_named_from!([Option<u64>], UInt64Type, from_slice_options);
#[cfg(feature = "dtype-i8")]
impl_named_from!([Option<i8>], Int8Type, from_slice_options);
#[cfg(feature = "dtype-i16")]
impl_named_from!([Option<i16>], Int16Type, from_slice_options);
impl_named_from!([Option<i32>], Int32Type, from_slice_options);
impl_named_from!([Option<i64>], Int64Type, from_slice_options);
impl_named_from!([Option<f32>], Float32Type, from_slice_options);
impl_named_from!([Option<f64>], Float64Type, from_slice_options);

macro_rules! impl_named_from_range {
    ($range:ty, $polars_type:ident) => {
        impl NamedFrom<$range, $polars_type> for ChunkedArray<$polars_type> {
            fn new(name: &str, range: $range) -> Self {
                let values = range.collect::<Vec<_>>();
                ChunkedArray::<$polars_type>::from_vec(name, values)
            }
        }

        impl NamedFrom<$range, $polars_type> for Series {
            fn new(name: &str, range: $range) -> Self {
                ChunkedArray::new(name, range).into_series()
            }
        }
    };
}
impl_named_from_range!(std::ops::Range<i64>, Int64Type);
impl_named_from_range!(std::ops::Range<i32>, Int32Type);
impl_named_from_range!(std::ops::Range<u64>, UInt64Type);
impl_named_from_range!(std::ops::Range<u32>, UInt32Type);

impl<T: AsRef<[Series]>> NamedFrom<T, ListType> for Series {
    fn new(name: &str, s: T) -> Self {
        let series_slice = s.as_ref();
        let list_cap = series_slice.len();

        let dt = series_slice[0].dtype();

        // inner type is also list so we need the anonymous builder
        if let DataType::List(_) = dt {
            let mut builder = AnonymousListBuilder::new(name, list_cap, Some(dt.clone()));
            for s in series_slice {
                builder.append_series(s)
            }
            builder.finish().into_series()
        } else {
            let values_cap = series_slice.iter().fold(0, |acc, s| acc + s.len());

            let mut builder = get_list_builder(dt, values_cap, list_cap, name).unwrap();
            for series in series_slice {
                builder.append_series(series)
            }
            builder.finish().into_series()
        }
    }
}

impl<T: AsRef<[Option<Series>]>> NamedFrom<T, [Option<Series>]> for Series {
    fn new(name: &str, s: T) -> Self {
        let series_slice = s.as_ref();
        let values_cap = series_slice.iter().fold(0, |acc, opt_s| {
            acc + opt_s.as_ref().map(|s| s.len()).unwrap_or(0)
        });

        let dt = series_slice
            .iter()
            .filter_map(|opt| opt.as_ref())
            .next()
            .expect("cannot create List Series from a slice of nulls")
            .dtype();

        let mut builder = get_list_builder(dt, values_cap, series_slice.len(), name).unwrap();
        for series in series_slice {
            builder.append_opt_series(series.as_ref())
        }
        builder.finish().into_series()
    }
}
impl<'a, T: AsRef<[&'a str]>> NamedFrom<T, [&'a str]> for Series {
    fn new(name: &str, v: T) -> Self {
        Utf8Chunked::from_slice(name, v.as_ref()).into_series()
    }
}

impl NamedFrom<&Series, str> for Series {
    fn new(name: &str, s: &Series) -> Self {
        let mut s = s.clone();
        s.rename(name);
        s
    }
}

impl<'a, T: AsRef<[&'a str]>> NamedFrom<T, [&'a str]> for Utf8Chunked {
    fn new(name: &str, v: T) -> Self {
        Utf8Chunked::from_slice(name, v.as_ref())
    }
}

impl<'a, T: AsRef<[Option<&'a str>]>> NamedFrom<T, [Option<&'a str>]> for Series {
    fn new(name: &str, v: T) -> Self {
        Utf8Chunked::from_slice_options(name, v.as_ref()).into_series()
    }
}

impl<'a, T: AsRef<[Option<&'a str>]>> NamedFrom<T, [Option<&'a str>]> for Utf8Chunked {
    fn new(name: &str, v: T) -> Self {
        Utf8Chunked::from_slice_options(name, v.as_ref())
    }
}

impl<'a, T: AsRef<[Cow<'a, str>]>> NamedFrom<T, [Cow<'a, str>]> for Series {
    fn new(name: &str, v: T) -> Self {
        Utf8Chunked::from_iter_values(name, v.as_ref().iter().map(|value| value.as_ref()))
            .into_series()
    }
}

impl<'a, T: AsRef<[Cow<'a, str>]>> NamedFrom<T, [Cow<'a, str>]> for Utf8Chunked {
    fn new(name: &str, v: T) -> Self {
        Utf8Chunked::from_iter_values(name, v.as_ref().iter().map(|value| value.as_ref()))
    }
}

impl<'a, T: AsRef<[Option<Cow<'a, str>>]>> NamedFrom<T, [Option<Cow<'a, str>>]> for Series {
    fn new(name: &str, v: T) -> Self {
        Utf8Chunked::new(name, v).into_series()
    }
}

impl<'a, T: AsRef<[Option<Cow<'a, str>>]>> NamedFrom<T, [Option<Cow<'a, str>>]> for Utf8Chunked {
    fn new(name: &str, v: T) -> Self {
        Utf8Chunked::from_iter_options(
            name,
            v.as_ref()
                .iter()
                .map(|opt| opt.as_ref().map(|value| value.as_ref())),
        )
    }
}

#[cfg(feature = "dtype-binary")]
impl<'a, T: AsRef<[&'a [u8]]>> NamedFrom<T, [&'a [u8]]> for Series {
    fn new(name: &str, v: T) -> Self {
        BinaryChunked::from_slice(name, v.as_ref()).into_series()
    }
}

#[cfg(feature = "dtype-binary")]
impl<'a, T: AsRef<[&'a [u8]]>> NamedFrom<T, [&'a [u8]]> for BinaryChunked {
    fn new(name: &str, v: T) -> Self {
        BinaryChunked::from_slice(name, v.as_ref())
    }
}

#[cfg(feature = "dtype-binary")]
impl<'a, T: AsRef<[Option<&'a [u8]>]>> NamedFrom<T, [Option<&'a [u8]>]> for Series {
    fn new(name: &str, v: T) -> Self {
        BinaryChunked::from_slice_options(name, v.as_ref()).into_series()
    }
}

#[cfg(feature = "dtype-binary")]
impl<'a, T: AsRef<[Option<&'a [u8]>]>> NamedFrom<T, [Option<&'a [u8]>]> for BinaryChunked {
    fn new(name: &str, v: T) -> Self {
        BinaryChunked::from_slice_options(name, v.as_ref())
    }
}

#[cfg(feature = "dtype-binary")]
impl<'a, T: AsRef<[Cow<'a, [u8]>]>> NamedFrom<T, [Cow<'a, [u8]>]> for Series {
    fn new(name: &str, v: T) -> Self {
        BinaryChunked::from_iter_values(name, v.as_ref().iter().map(|value| value.as_ref()))
            .into_series()
    }
}

#[cfg(feature = "dtype-binary")]
impl<'a, T: AsRef<[Cow<'a, [u8]>]>> NamedFrom<T, [Cow<'a, [u8]>]> for BinaryChunked {
    fn new(name: &str, v: T) -> Self {
        BinaryChunked::from_iter_values(name, v.as_ref().iter().map(|value| value.as_ref()))
    }
}

#[cfg(feature = "dtype-binary")]
impl<'a, T: AsRef<[Option<Cow<'a, [u8]>>]>> NamedFrom<T, [Option<Cow<'a, [u8]>>]> for Series {
    fn new(name: &str, v: T) -> Self {
        BinaryChunked::new(name, v).into_series()
    }
}

#[cfg(feature = "dtype-binary")]
impl<'a, T: AsRef<[Option<Cow<'a, [u8]>>]>> NamedFrom<T, [Option<Cow<'a, [u8]>>]>
    for BinaryChunked
{
    fn new(name: &str, v: T) -> Self {
        BinaryChunked::from_iter_options(
            name,
            v.as_ref()
                .iter()
                .map(|opt| opt.as_ref().map(|value| value.as_ref())),
        )
    }
}

#[cfg(feature = "dtype-date")]
impl<T: AsRef<[NaiveDate]>> NamedFrom<T, [NaiveDate]> for DateChunked {
    fn new(name: &str, v: T) -> Self {
        DateChunked::from_naive_date(name, v.as_ref().iter().copied())
    }
}

#[cfg(feature = "dtype-date")]
impl<T: AsRef<[NaiveDate]>> NamedFrom<T, [NaiveDate]> for Series {
    fn new(name: &str, v: T) -> Self {
        DateChunked::new(name, v).into_series()
    }
}

#[cfg(feature = "dtype-date")]
impl<T: AsRef<[Option<NaiveDate>]>> NamedFrom<T, [Option<NaiveDate>]> for DateChunked {
    fn new(name: &str, v: T) -> Self {
        DateChunked::from_naive_date_options(name, v.as_ref().iter().copied())
    }
}

#[cfg(feature = "dtype-date")]
impl<T: AsRef<[Option<NaiveDate>]>> NamedFrom<T, [Option<NaiveDate>]> for Series {
    fn new(name: &str, v: T) -> Self {
        DateChunked::new(name, v).into_series()
    }
}

#[cfg(feature = "dtype-datetime")]
impl<T: AsRef<[NaiveDateTime]>> NamedFrom<T, [NaiveDateTime]> for DatetimeChunked {
    fn new(name: &str, v: T) -> Self {
        DatetimeChunked::from_naive_datetime(
            name,
            v.as_ref().iter().copied(),
            TimeUnit::Milliseconds,
        )
    }
}

#[cfg(feature = "dtype-datetime")]
impl<T: AsRef<[NaiveDateTime]>> NamedFrom<T, [NaiveDateTime]> for Series {
    fn new(name: &str, v: T) -> Self {
        DatetimeChunked::new(name, v).into_series()
    }
}

#[cfg(feature = "dtype-datetime")]
impl<T: AsRef<[Option<NaiveDateTime>]>> NamedFrom<T, [Option<NaiveDateTime>]> for DatetimeChunked {
    fn new(name: &str, v: T) -> Self {
        DatetimeChunked::from_naive_datetime_options(
            name,
            v.as_ref().iter().copied(),
            TimeUnit::Milliseconds,
        )
    }
}

#[cfg(feature = "dtype-datetime")]
impl<T: AsRef<[Option<NaiveDateTime>]>> NamedFrom<T, [Option<NaiveDateTime>]> for Series {
    fn new(name: &str, v: T) -> Self {
        DatetimeChunked::new(name, v).into_series()
    }
}

#[cfg(feature = "dtype-duration")]
impl<T: AsRef<[ChronoDuration]>> NamedFrom<T, [ChronoDuration]> for DurationChunked {
    fn new(name: &str, v: T) -> Self {
        DurationChunked::from_duration(name, v.as_ref().iter().copied(), TimeUnit::Nanoseconds)
    }
}

#[cfg(feature = "dtype-duration")]
impl<T: AsRef<[ChronoDuration]>> NamedFrom<T, [ChronoDuration]> for Series {
    fn new(name: &str, v: T) -> Self {
        DurationChunked::new(name, v).into_series()
    }
}

#[cfg(feature = "dtype-duration")]
impl<T: AsRef<[Option<ChronoDuration>]>> NamedFrom<T, [Option<ChronoDuration>]>
    for DurationChunked
{
    fn new(name: &str, v: T) -> Self {
        DurationChunked::from_duration_options(
            name,
            v.as_ref().iter().copied(),
            TimeUnit::Nanoseconds,
        )
    }
}

#[cfg(feature = "dtype-duration")]
impl<T: AsRef<[Option<ChronoDuration>]>> NamedFrom<T, [Option<ChronoDuration>]> for Series {
    fn new(name: &str, v: T) -> Self {
        DurationChunked::new(name, v).into_series()
    }
}

#[cfg(feature = "dtype-time")]
impl<T: AsRef<[NaiveTime]>> NamedFrom<T, [NaiveTime]> for TimeChunked {
    fn new(name: &str, v: T) -> Self {
        TimeChunked::from_naive_time(name, v.as_ref().iter().copied())
    }
}

#[cfg(feature = "dtype-time")]
impl<T: AsRef<[NaiveTime]>> NamedFrom<T, [NaiveTime]> for Series {
    fn new(name: &str, v: T) -> Self {
        TimeChunked::new(name, v).into_series()
    }
}

#[cfg(feature = "dtype-time")]
impl<T: AsRef<[Option<NaiveTime>]>> NamedFrom<T, [Option<NaiveTime>]> for TimeChunked {
    fn new(name: &str, v: T) -> Self {
        TimeChunked::from_naive_time_options(name, v.as_ref().iter().copied())
    }
}

#[cfg(feature = "dtype-time")]
impl<T: AsRef<[Option<NaiveTime>]>> NamedFrom<T, [Option<NaiveTime>]> for Series {
    fn new(name: &str, v: T) -> Self {
        TimeChunked::new(name, v).into_series()
    }
}

#[cfg(feature = "object")]
impl<T: PolarsObject> NamedFrom<&[T], &[T]> for ObjectChunked<T> {
    fn new(name: &str, v: &[T]) -> Self {
        ObjectChunked::from_slice(name, v)
    }
}

#[cfg(feature = "object")]
impl<T: PolarsObject, S: AsRef<[Option<T>]>> NamedFrom<S, [Option<T>]> for ObjectChunked<T> {
    fn new(name: &str, v: S) -> Self {
        ObjectChunked::from_slice_options(name, v.as_ref())
    }
}

impl<T: PolarsNumericType> ChunkedArray<T> {
    /// Specialization that prevents an allocation
    /// prefer this over ChunkedArray::new when you have a `Vec<T::Native>` and no null values.
    pub fn new_vec(name: &str, v: Vec<T::Native>) -> Self {
        ChunkedArray::from_vec(name, v)
    }
}

/// For any [`ChunkedArray`] and [`Series`]
impl<T: IntoSeries> NamedFrom<T, T> for Series {
    fn new(name: &str, t: T) -> Self {
        let mut s = t.into_series();
        s.rename(name);
        s
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[cfg(all(
        feature = "dtype-datetime",
        feature = "dtype-duration",
        feature = "dtype-date",
        feature = "dtype-time"
    ))]
    #[test]
    fn test_temporal_df_construction() {
        // check if we can construct.
        let _df = df![
            "date" => [NaiveDate::from_ymd_opt(2021, 1, 1).unwrap()],
            "datetime" => [NaiveDate::from_ymd_opt(2021, 1, 1).unwrap().and_hms_opt(0, 0, 0).unwrap()],
            "optional_date" => [Some(NaiveDate::from_ymd_opt(2021, 1, 1).unwrap())],
            "optional_datetime" => [Some(NaiveDate::from_ymd_opt(2021, 1, 1).unwrap().and_hms_opt(0, 0, 0).unwrap())],
            "time" => [NaiveTime::from_hms_opt(23, 23, 23).unwrap()],
            "optional_time" => [Some(NaiveTime::from_hms_opt(23, 23, 23).unwrap())],
            "duration" => [ChronoDuration::from_std(std::time::Duration::from_secs(10)).unwrap()],
            "optional_duration" => [Some(ChronoDuration::from_std(std::time::Duration::from_secs(10)).unwrap())],
        ].unwrap();
    }
}
