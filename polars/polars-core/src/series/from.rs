use crate::chunked_array::builder::get_list_builder;
use crate::chunked_array::cast::cast_chunks;
use crate::prelude::*;
use arrow::compute::cast;
use std::convert::TryFrom;

pub trait NamedFrom<T, Phantom: ?Sized> {
    /// Initialize by name and values.
    fn new(name: &str, _: T) -> Self;
}
//
macro_rules! impl_named_from {
    ($type:ty, $series_var:ident, $method:ident) => {
        impl<T: AsRef<$type>> NamedFrom<T, $type> for Series {
            fn new(name: &str, v: T) -> Self {
                ChunkedArray::<$series_var>::$method(name, v.as_ref()).into_series()
            }
        }
    };
}

impl<'a, T: AsRef<[&'a str]>> NamedFrom<T, [&'a str]> for Series {
    fn new(name: &str, v: T) -> Self {
        Utf8Chunked::new_from_slice(name, v.as_ref()).into_series()
    }
}
impl<'a, T: AsRef<[Option<&'a str>]>> NamedFrom<T, [Option<&'a str>]> for Series {
    fn new(name: &str, v: T) -> Self {
        Utf8Chunked::new_from_opt_slice(name, v.as_ref()).into_series()
    }
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

// TODO: add types
impl std::convert::TryFrom<(&str, Vec<ArrayRef>)> for Series {
    type Error = PolarsError;

    fn try_from(name_arr: (&str, Vec<ArrayRef>)) -> Result<Self> {
        let (name, chunks) = name_arr;

        let mut chunks_iter = chunks.iter();
        let data_type: &ArrowDataType = chunks_iter
            .next()
            .ok_or_else(|| PolarsError::NoData("Expected at least on ArrayRef".into()))?
            .data_type();

        for chunk in chunks_iter {
            if chunk.data_type() != data_type {
                return Err(PolarsError::InvalidOperation(
                    "Cannot create series from multiple arrays with different types".into(),
                ));
            }
        }

        match data_type {
            ArrowDataType::LargeUtf8 => {
                Ok(Utf8Chunked::new_from_chunks(name, chunks).into_series())
            }
            ArrowDataType::Utf8 => {
                let chunks = cast_chunks(&chunks, &DataType::Utf8).unwrap();
                Ok(Utf8Chunked::new_from_chunks(name, chunks).into_series())
            }
            ArrowDataType::List(fld) => {
                let chunks = chunks
                    .iter()
                    .map(|arr| {
                        cast::cast(arr.as_ref(), &ArrowDataType::LargeList(fld.clone()))
                            .unwrap()
                            .into()
                    })
                    .collect();
                Ok(ListChunked::new_from_chunks(name, chunks).into_series())
            }
            ArrowDataType::Boolean => {
                Ok(BooleanChunked::new_from_chunks(name, chunks).into_series())
            }
            #[cfg(feature = "dtype-u8")]
            ArrowDataType::UInt8 => Ok(UInt8Chunked::new_from_chunks(name, chunks).into_series()),
            #[cfg(feature = "dtype-u16")]
            ArrowDataType::UInt16 => Ok(UInt16Chunked::new_from_chunks(name, chunks).into_series()),
            ArrowDataType::UInt32 => Ok(UInt32Chunked::new_from_chunks(name, chunks).into_series()),
            ArrowDataType::UInt64 => Ok(UInt64Chunked::new_from_chunks(name, chunks).into_series()),
            #[cfg(feature = "dtype-i8")]
            ArrowDataType::Int8 => Ok(Int8Chunked::new_from_chunks(name, chunks).into_series()),
            #[cfg(feature = "dtype-i16")]
            ArrowDataType::Int16 => Ok(Int16Chunked::new_from_chunks(name, chunks).into_series()),
            ArrowDataType::Int32 => Ok(Int32Chunked::new_from_chunks(name, chunks).into_series()),
            ArrowDataType::Int64 => Ok(Int64Chunked::new_from_chunks(name, chunks).into_series()),
            ArrowDataType::Float32 => {
                Ok(Float32Chunked::new_from_chunks(name, chunks).into_series())
            }
            ArrowDataType::Float64 => {
                Ok(Float64Chunked::new_from_chunks(name, chunks).into_series())
            }
            #[cfg(feature = "dtype-date")]
            ArrowDataType::Date32 => {
                let chunks = cast_chunks(&chunks, &DataType::Int32).unwrap();
                Ok(Int32Chunked::new_from_chunks(name, chunks)
                    .into_date()
                    .into_series())
            }
            #[cfg(feature = "dtype-datetime")]
            ArrowDataType::Date64 => {
                let chunks = cast_chunks(&chunks, &DataType::Int64).unwrap();
                Ok(Int64Chunked::new_from_chunks(name, chunks)
                    .into_date()
                    .into_series())
            }
            #[cfg(feature = "dtype-datetime")]
            ArrowDataType::Timestamp(tu, tz) => {
                let s = if tz.is_none() || tz == &Some("".to_string()) {
                    let chunks = cast_chunks(&chunks, &DataType::Int64).unwrap();
                    Int64Chunked::new_from_chunks(name, chunks)
                        .into_date()
                        .into_series()
                } else {
                    return Err(PolarsError::InvalidOperation(
                        "Cannot create polars series timestamp with timezone".into(),
                    ));
                };
                Ok(match tu {
                    TimeUnit::Second => &s / 1000,
                    TimeUnit::Millisecond => s,
                    TimeUnit::Microsecond => &s * 1000,
                    TimeUnit::Nanosecond => &s * 1000000,
                })
            }
            #[cfg(feature = "dtype-time")]
            ArrowDataType::Time64(tu) | ArrowDataType::Time32(tu) => {
                let chunks = cast_chunks(&chunks, &DataType::Int64).unwrap();
                let s = Int64Chunked::new_from_chunks(name, chunks)
                    .into_time()
                    .into_series();
                Ok(match tu {
                    TimeUnit::Second => &s * 1000000000,
                    TimeUnit::Millisecond => &s * 1000000,
                    TimeUnit::Microsecond => &s * 1000,
                    TimeUnit::Nanosecond => s,
                })
            }
            ArrowDataType::LargeList(_) => {
                Ok(ListChunked::new_from_chunks(name, chunks).into_series())
            }
            ArrowDataType::Null => {
                // we don't support null types yet so we use a small digit type filled with nulls
                let len = chunks.iter().fold(0, |acc, array| acc + array.len());
                #[cfg(feature = "dtype-i8")]
                return Ok(Int8Chunked::full_null(name, len).into_series());
                #[cfg(not(feature = "dtype-i8"))]
                Ok(UInt32Chunked::full_null(name, len).into_series())
            }
            #[cfg(feature = "dtype-categorical")]
            ArrowDataType::Dictionary(key_type, value_type) => {
                use crate::chunked_array::categorical::CategoricalChunkedBuilder;
                use arrow::compute::cast::cast;
                let chunks = chunks.iter().map(|arr| &**arr).collect::<Vec<_>>();
                let arr = arrow::compute::concat::concatenate(&chunks)?;

                let (keys, values) = match (&**key_type, &**value_type) {
                    (ArrowDataType::Int8, ArrowDataType::LargeUtf8) => {
                        let arr = arr.as_any().downcast_ref::<DictionaryArray<i8>>().unwrap();
                        let keys = arr.keys();
                        let keys = cast(keys, &ArrowDataType::UInt32)
                            .unwrap()
                            .as_any()
                            .downcast_ref::<PrimitiveArray<u32>>()
                            .unwrap()
                            .clone();
                        let values = arr.values();
                        let values = values.as_any().downcast_ref::<LargeStringArray>().unwrap();
                        (keys, values.clone())
                    }
                    (ArrowDataType::Int16, ArrowDataType::LargeUtf8) => {
                        let arr = arr.as_any().downcast_ref::<DictionaryArray<i16>>().unwrap();
                        let keys = arr.keys();
                        let keys = cast(keys, &ArrowDataType::UInt32)
                            .unwrap()
                            .as_any()
                            .downcast_ref::<PrimitiveArray<u32>>()
                            .unwrap()
                            .clone();
                        let values = arr.values();
                        let values = values.as_any().downcast_ref::<LargeStringArray>().unwrap();
                        (keys, values.clone())
                    }
                    (ArrowDataType::Int32, ArrowDataType::LargeUtf8) => {
                        let arr = arr.as_any().downcast_ref::<DictionaryArray<i32>>().unwrap();
                        let keys = arr.keys();
                        let keys = cast(keys, &ArrowDataType::UInt32)
                            .unwrap()
                            .as_any()
                            .downcast_ref::<PrimitiveArray<u32>>()
                            .unwrap()
                            .clone();
                        let values = arr.values();
                        let values = values.as_any().downcast_ref::<LargeStringArray>().unwrap();
                        (keys, values.clone())
                    }
                    (ArrowDataType::UInt32, ArrowDataType::LargeUtf8) => {
                        let arr = arr.as_any().downcast_ref::<DictionaryArray<u32>>().unwrap();
                        let keys = arr.keys();
                        let values = arr.values();
                        let values = values.as_any().downcast_ref::<LargeStringArray>().unwrap();
                        (keys.clone(), values.clone())
                    }
                    (ArrowDataType::Int8, ArrowDataType::Utf8) => {
                        let arr = arr.as_any().downcast_ref::<DictionaryArray<i8>>().unwrap();
                        let keys = arr.keys();
                        let keys = cast(keys, &ArrowDataType::UInt32)
                            .unwrap()
                            .as_any()
                            .downcast_ref::<PrimitiveArray<u32>>()
                            .unwrap()
                            .clone();
                        let values = arr.values();
                        let values = values.as_any().downcast_ref::<Utf8Array<i32>>().unwrap();
                        let values = cast(values, &ArrowDataType::LargeUtf8)
                            .unwrap()
                            .as_any()
                            .downcast_ref::<Utf8Array<i64>>()
                            .unwrap()
                            .clone();
                        (keys, values)
                    }
                    (ArrowDataType::Int16, ArrowDataType::Utf8) => {
                        let arr = arr.as_any().downcast_ref::<DictionaryArray<i16>>().unwrap();
                        let keys = arr.keys();
                        let keys = cast(keys, &ArrowDataType::UInt32)
                            .unwrap()
                            .as_any()
                            .downcast_ref::<PrimitiveArray<u32>>()
                            .unwrap()
                            .clone();
                        let values = arr.values();
                        let values = values.as_any().downcast_ref::<Utf8Array<i32>>().unwrap();
                        let values = cast(values, &ArrowDataType::LargeUtf8)
                            .unwrap()
                            .as_any()
                            .downcast_ref::<Utf8Array<i64>>()
                            .unwrap()
                            .clone();
                        (keys, values)
                    }
                    (ArrowDataType::Int32, ArrowDataType::Utf8) => {
                        let arr = arr.as_any().downcast_ref::<DictionaryArray<i32>>().unwrap();
                        let keys = arr.keys();
                        let keys = cast(keys, &ArrowDataType::UInt32)
                            .unwrap()
                            .as_any()
                            .downcast_ref::<PrimitiveArray<u32>>()
                            .unwrap()
                            .clone();
                        let values = arr.values();
                        let values = values.as_any().downcast_ref::<Utf8Array<i32>>().unwrap();
                        let values = cast(values, &ArrowDataType::LargeUtf8)
                            .unwrap()
                            .as_any()
                            .downcast_ref::<Utf8Array<i64>>()
                            .unwrap()
                            .clone();
                        (keys, values)
                    }
                    (k, v) => {
                        return Err(PolarsError::InvalidOperation(
                            format!(
                            "Cannot create polars series dictionary type of key:  {:?} value: {:?}",
                            k, v
                        )
                            .into(),
                        ))
                    }
                };

                let mut builder = CategoricalChunkedBuilder::new(name, keys.len());
                let iter = keys
                    .into_iter()
                    .map(|opt_key| opt_key.map(|k| unsafe { values.value_unchecked(*k as usize) }));
                builder.from_iter(iter);
                Ok(builder.finish().into())
            }
            dt => Err(PolarsError::InvalidOperation(
                format!("Cannot create polars series from {:?} type", dt).into(),
            )),
        }
    }
}

impl TryFrom<(&str, ArrayRef)> for Series {
    type Error = PolarsError;

    fn try_from(name_arr: (&str, ArrayRef)) -> Result<Self> {
        let (name, arr) = name_arr;
        Series::try_from((name, vec![arr]))
    }
}

impl TryFrom<(&str, Box<dyn Array>)> for Series {
    type Error = PolarsError;

    fn try_from(name_arr: (&str, Box<dyn Array>)) -> Result<Self> {
        let (name, arr) = name_arr;
        Series::try_from((name, vec![arr.into()]))
    }
}

pub trait IntoSeries {
    fn is_series() -> bool {
        false
    }

    fn into_series(self) -> Series
    where
        Self: Sized;
}

impl<T> From<ChunkedArray<T>> for Series
where
    T: PolarsDataType,
    ChunkedArray<T>: IntoSeries,
{
    fn from(ca: ChunkedArray<T>) -> Self {
        ca.into_series()
    }
}

#[cfg(feature = "dtype-date")]
impl From<DateChunked> for Series {
    fn from(a: DateChunked) -> Self {
        a.into_series()
    }
}

#[cfg(feature = "dtype-datetime")]
impl From<DatetimeChunked> for Series {
    fn from(a: DatetimeChunked) -> Self {
        a.into_series()
    }
}

#[cfg(feature = "dtype-time")]
impl From<TimeChunked> for Series {
    fn from(a: TimeChunked) -> Self {
        a.into_series()
    }
}

impl IntoSeries for Arc<dyn SeriesTrait> {
    fn into_series(self) -> Series {
        Series(self)
    }
}

impl IntoSeries for Series {
    fn is_series() -> bool {
        true
    }

    fn into_series(self) -> Series {
        self
    }
}
