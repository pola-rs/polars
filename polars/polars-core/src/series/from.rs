use crate::chunked_array::cast::cast_chunks;
#[cfg(feature = "object")]
use crate::chunked_array::object::extension::polars_extension::PolarsExtension;
use crate::prelude::*;
use arrow::compute::cast::utf8_to_large_utf8;
use arrow::temporal_conversions::NANOSECONDS;
use polars_arrow::compute::cast::cast;
use std::convert::TryFrom;

fn convert_list_inner(arr: &ArrayRef, fld: &ArrowField) -> ArrayRef {
    // if inner type is Utf8, we need to convert that to large utf8
    match fld.data_type() {
        ArrowDataType::Utf8 => {
            let arr = arr.as_any().downcast_ref::<ListArray<i64>>().unwrap();
            let offsets = arr.offsets().iter().map(|x| *x as i64).collect();
            let values = arr.values();
            let values =
                utf8_to_large_utf8(values.as_any().downcast_ref::<Utf8Array<i32>>().unwrap());

            Arc::new(LargeListArray::from_data(
                ArrowDataType::LargeList(
                    ArrowField::new(fld.name(), ArrowDataType::LargeUtf8, true).into(),
                ),
                offsets,
                Arc::new(values),
                arr.validity().cloned(),
            ))
        }
        _ => arr.clone(),
    }
}

// TODO: add types
impl TryFrom<(&str, Vec<ArrayRef>)> for Series {
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
                        let arr: ArrayRef =
                            cast(arr.as_ref(), &ArrowDataType::LargeList(fld.clone()))
                                .unwrap()
                                .into();
                        convert_list_inner(&arr, fld)
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
                let ca = Int64Chunked::new_from_chunks(name, chunks);
                let ca = ca * 1_000_000;
                Ok(ca.into_date().into_series())
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
                    TimeUnit::Second => &s * NANOSECONDS,
                    TimeUnit::Millisecond => &s * 1_000_000,
                    TimeUnit::Microsecond => &s * 1_000,
                    TimeUnit::Nanosecond => s,
                })
            }
            #[cfg(feature = "dtype-time")]
            ArrowDataType::Time64(tu) | ArrowDataType::Time32(tu) => {
                let chunks = cast_chunks(&chunks, &DataType::Int64).unwrap();
                let s = Int64Chunked::new_from_chunks(name, chunks)
                    .into_time()
                    .into_series();
                Ok(match tu {
                    TimeUnit::Second => &s * NANOSECONDS,
                    TimeUnit::Millisecond => &s * 1_000_000,
                    TimeUnit::Microsecond => &s * 1_000,
                    TimeUnit::Nanosecond => s,
                })
            }
            ArrowDataType::LargeList(fld) => {
                let chunks = chunks
                    .iter()
                    .map(|arr| convert_list_inner(arr, fld))
                    .collect();
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
            #[cfg(not(feature = "dtype-categorical"))]
            ArrowDataType::Dictionary(_, _) => {
                panic!("activate dtype-categorical to convert dictionary arrays")
            }
            #[cfg(feature = "dtype-categorical")]
            ArrowDataType::Dictionary(key_type, value_type) => {
                use crate::chunked_array::categorical::CategoricalChunkedBuilder;
                use arrow::datatypes::IntegerType;
                let chunks = chunks.iter().map(|arr| &**arr).collect::<Vec<_>>();
                let arr = arrow::compute::concatenate::concatenate(&chunks)?;

                let (keys, values) = match (key_type, &**value_type) {
                    (IntegerType::Int8, ArrowDataType::LargeUtf8) => {
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
                    (IntegerType::Int16, ArrowDataType::LargeUtf8) => {
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
                    (IntegerType::Int32, ArrowDataType::LargeUtf8) => {
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
                    (IntegerType::UInt32, ArrowDataType::LargeUtf8) => {
                        let arr = arr.as_any().downcast_ref::<DictionaryArray<u32>>().unwrap();
                        let keys = arr.keys();
                        let values = arr.values();
                        let values = values.as_any().downcast_ref::<LargeStringArray>().unwrap();
                        (keys.clone(), values.clone())
                    }
                    (IntegerType::Int8, ArrowDataType::Utf8) => {
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
                    (IntegerType::Int16, ArrowDataType::Utf8) => {
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
                    (IntegerType::Int32, ArrowDataType::Utf8) => {
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
            #[cfg(not(feature = "dtype-u8"))]
            ArrowDataType::LargeBinary | ArrowDataType::Binary => {
                panic!("activate dtype-u8 to read binary data into polars List<u8>")
            }
            #[cfg(feature = "dtype-u8")]
            ArrowDataType::LargeBinary | ArrowDataType::Binary => {
                let chunks = chunks
                    .iter()
                    .map(|arr| {
                        let arr = cast(&**arr, &ArrowDataType::LargeBinary).unwrap();

                        let arr = arr.as_any().downcast_ref::<BinaryArray<i64>>().unwrap();
                        let values = arr.values().clone();
                        let offsets = arr.offsets().clone();
                        let validity = arr.validity().cloned();

                        let values = Arc::new(PrimitiveArray::from_data(
                            ArrowDataType::UInt8,
                            values,
                            None,
                        ));

                        let dtype = ListArray::<i64>::default_datatype(ArrowDataType::UInt8);
                        Arc::new(ListArray::<i64>::from_data(
                            dtype, offsets, values, validity,
                        )) as ArrayRef
                    })
                    .collect();
                Ok(ListChunked::new_from_chunks(name, chunks).into())
            }
            #[cfg(feature = "object")]
            ArrowDataType::Extension(s, _, Some(_)) if s == "POLARS_EXTENSION_TYPE" => {
                assert_eq!(chunks.len(), 1);
                let arr = chunks[0]
                    .as_any()
                    .downcast_ref::<FixedSizeBinaryArray>()
                    .unwrap();
                // Safety:
                // this is highly unsafe. it will dereference a raw ptr on the heap
                // make sure the ptr is allocated and from this pid
                // (the pid is checked before dereference)
                let s = unsafe {
                    let pe = PolarsExtension::new(arr.clone());
                    let s = pe.get_series();
                    pe.take_and_forget();
                    s
                };
                Ok(s)
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

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    #[cfg(feature = "dtype-u8")]
    fn test_binary_to_list() {
        let iter = std::iter::repeat(b"hello").take(2).map(Some);
        let a = Arc::new(iter.collect::<BinaryArray<i32>>()) as ArrayRef;

        let s = Series::try_from(("", a)).unwrap();
        assert_eq!(s.dtype(), &DataType::List(Box::new(DataType::UInt8)));
    }
}
