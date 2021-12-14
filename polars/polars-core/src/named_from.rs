use crate::chunked_array::builder::get_list_builder;
use crate::prelude::*;
use arrow::buffer::MutableBuffer;
use std::borrow::Cow;

pub trait NamedFrom<T, Phantom: ?Sized> {
    /// Initialize by name and values.
    fn new(name: &str, _: T) -> Self;
}

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

impl_named_from!([String], Utf8Type, new_from_slice);
impl_named_from!([bool], BooleanType, new_from_slice);
#[cfg(feature = "dtype-u8")]
impl_named_from!([u8], UInt8Type, new_from_slice);
#[cfg(feature = "dtype-u16")]
impl_named_from!([u16], UInt16Type, new_from_slice);
impl_named_from!([u32], UInt32Type, new_from_slice);
impl_named_from!([u64], UInt64Type, new_from_slice);
#[cfg(feature = "dtype-i8")]
impl_named_from!([i8], Int8Type, new_from_slice);
#[cfg(feature = "dtype-i16")]
impl_named_from!([i16], Int16Type, new_from_slice);
impl_named_from!([i32], Int32Type, new_from_slice);
impl_named_from!([i64], Int64Type, new_from_slice);
impl_named_from!([f32], Float32Type, new_from_slice);
impl_named_from!([f64], Float64Type, new_from_slice);
impl_named_from!([Option<String>], Utf8Type, new_from_opt_slice);
impl_named_from!([Option<bool>], BooleanType, new_from_opt_slice);
#[cfg(feature = "dtype-u8")]
impl_named_from!([Option<u8>], UInt8Type, new_from_opt_slice);
#[cfg(feature = "dtype-u16")]
impl_named_from!([Option<u16>], UInt16Type, new_from_opt_slice);
impl_named_from!([Option<u32>], UInt32Type, new_from_opt_slice);
impl_named_from!([Option<u64>], UInt64Type, new_from_opt_slice);
#[cfg(feature = "dtype-i8")]
impl_named_from!([Option<i8>], Int8Type, new_from_opt_slice);
#[cfg(feature = "dtype-i16")]
impl_named_from!([Option<i16>], Int16Type, new_from_opt_slice);
impl_named_from!([Option<i32>], Int32Type, new_from_opt_slice);
impl_named_from!([Option<i64>], Int64Type, new_from_opt_slice);
impl_named_from!([Option<f32>], Float32Type, new_from_opt_slice);
impl_named_from!([Option<f64>], Float64Type, new_from_opt_slice);

impl<T: AsRef<[Series]>> NamedFrom<T, ListType> for Series {
    fn new(name: &str, s: T) -> Self {
        let series_slice = s.as_ref();
        let values_cap = series_slice.iter().fold(0, |acc, s| acc + s.len());

        let dt = series_slice[0].dtype();
        let mut builder = get_list_builder(dt, values_cap, series_slice.len(), name);
        for series in series_slice {
            builder.append_series(series)
        }
        builder.finish().into_series()
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

        let mut builder = get_list_builder(dt, values_cap, series_slice.len(), name);
        for series in series_slice {
            builder.append_opt_series(series.as_ref())
        }
        builder.finish().into_series()
    }
}
impl<'a, T: AsRef<[&'a str]>> NamedFrom<T, [&'a str]> for Series {
    fn new(name: &str, v: T) -> Self {
        Utf8Chunked::new_from_slice(name, v.as_ref()).into_series()
    }
}

impl<'a, T: AsRef<[&'a str]>> NamedFrom<T, [&'a str]> for Utf8Chunked {
    fn new(name: &str, v: T) -> Self {
        Utf8Chunked::new_from_slice(name, v.as_ref())
    }
}

impl<'a, T: AsRef<[Option<&'a str>]>> NamedFrom<T, [Option<&'a str>]> for Series {
    fn new(name: &str, v: T) -> Self {
        Utf8Chunked::new_from_opt_slice(name, v.as_ref()).into_series()
    }
}

impl<'a, T: AsRef<[Option<&'a str>]>> NamedFrom<T, [Option<&'a str>]> for Utf8Chunked {
    fn new(name: &str, v: T) -> Self {
        Utf8Chunked::new_from_opt_slice(name, v.as_ref())
    }
}

impl<'a, T: AsRef<[Cow<'a, str>]>> NamedFrom<T, [Cow<'a, str>]> for Series {
    fn new(name: &str, v: T) -> Self {
        Utf8Chunked::new_from_iter(name, v.as_ref().iter().map(|value| value.as_ref()))
            .into_series()
    }
}

impl<'a, T: AsRef<[Cow<'a, str>]>> NamedFrom<T, [Cow<'a, str>]> for Utf8Chunked {
    fn new(name: &str, v: T) -> Self {
        Utf8Chunked::new_from_iter(name, v.as_ref().iter().map(|value| value.as_ref()))
    }
}

impl<'a, T: AsRef<[Option<Cow<'a, str>>]>> NamedFrom<T, [Option<Cow<'a, str>>]> for Series {
    fn new(name: &str, v: T) -> Self {
        Utf8Chunked::new(name, v).into_series()
    }
}

impl<'a, T: AsRef<[Option<Cow<'a, str>>]>> NamedFrom<T, [Option<Cow<'a, str>>]> for Utf8Chunked {
    fn new(name: &str, v: T) -> Self {
        Utf8Chunked::new_from_opt_iter(
            name,
            v.as_ref()
                .iter()
                .map(|opt| opt.as_ref().map(|value| value.as_ref())),
        )
    }
}

#[cfg(feature = "object")]
impl<T: PolarsObject> NamedFrom<&[T], &[T]> for ObjectChunked<T> {
    fn new(name: &str, v: &[T]) -> Self {
        ObjectChunked::new_from_slice(name, v)
    }
}

#[cfg(feature = "object")]
impl<T: PolarsObject, S: AsRef<[Option<T>]>> NamedFrom<S, [Option<T>]> for ObjectChunked<T> {
    fn new(name: &str, v: S) -> Self {
        ObjectChunked::new_from_opt_slice(name, v.as_ref())
    }
}

impl<T: PolarsNumericType> ChunkedArray<T> {
    /// Specialization that prevents an allocation
    /// prefer this over ChunkedArray::new when you have a `Vec<T::Native>` and no null values.
    pub fn new_vec(name: &str, v: Vec<T::Native>) -> Self {
        let buf = MutableBuffer::from_vec(v);
        ChunkedArray::new_from_aligned_vec(name, buf)
    }
}
