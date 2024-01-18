use std::convert::TryFrom;

use arrow::compute::cast::utf8_to_large_utf8;
use arrow::legacy::compute::cast::cast;
#[cfg(any(feature = "dtype-struct", feature = "dtype-categorical"))]
use arrow::legacy::kernels::concatenate::concatenate_owned_unchecked;
#[cfg(any(
    feature = "dtype-date",
    feature = "dtype-datetime",
    feature = "dtype-time",
    feature = "dtype-duration"
))]
use arrow::temporal_conversions::*;
use polars_error::feature_gated;

use crate::chunked_array::cast::cast_chunks;
#[cfg(feature = "object")]
use crate::chunked_array::object::extension::polars_extension::PolarsExtension;
#[cfg(feature = "object")]
use crate::chunked_array::object::extension::EXTENSION_NAME;
#[cfg(feature = "timezones")]
use crate::chunked_array::temporal::parse_fixed_offset;
#[cfg(feature = "timezones")]
use crate::chunked_array::temporal::validate_time_zone;
#[cfg(all(feature = "dtype-decimal", feature = "python"))]
use crate::config::decimal_is_active;
use crate::config::verbose;
use crate::prelude::*;

impl Series {
    /// Takes chunks and a polars datatype and constructs the Series
    /// This is faster than creating from chunks and an arrow datatype because there is no
    /// casting involved
    ///
    /// # Safety
    ///
    /// The caller must ensure that the given `dtype`'s physical type matches all the `ArrayRef` dtypes.
    pub unsafe fn from_chunks_and_dtype_unchecked(
        name: &str,
        chunks: Vec<ArrayRef>,
        dtype: &DataType,
    ) -> Self {
        use DataType::*;
        match dtype {
            #[cfg(feature = "dtype-i8")]
            Int8 => Int8Chunked::from_chunks(name, chunks).into_series(),
            #[cfg(feature = "dtype-i16")]
            Int16 => Int16Chunked::from_chunks(name, chunks).into_series(),
            Int32 => Int32Chunked::from_chunks(name, chunks).into_series(),
            Int64 => Int64Chunked::from_chunks(name, chunks).into_series(),
            #[cfg(feature = "dtype-u8")]
            UInt8 => UInt8Chunked::from_chunks(name, chunks).into_series(),
            #[cfg(feature = "dtype-u16")]
            UInt16 => UInt16Chunked::from_chunks(name, chunks).into_series(),
            UInt32 => UInt32Chunked::from_chunks(name, chunks).into_series(),
            UInt64 => UInt64Chunked::from_chunks(name, chunks).into_series(),
            #[cfg(feature = "dtype-date")]
            Date => Int32Chunked::from_chunks(name, chunks)
                .into_date()
                .into_series(),
            #[cfg(feature = "dtype-time")]
            Time => Int64Chunked::from_chunks(name, chunks)
                .into_time()
                .into_series(),
            #[cfg(feature = "dtype-duration")]
            Duration(tu) => Int64Chunked::from_chunks(name, chunks)
                .into_duration(*tu)
                .into_series(),
            #[cfg(feature = "dtype-datetime")]
            Datetime(tu, tz) => Int64Chunked::from_chunks(name, chunks)
                .into_datetime(*tu, tz.clone())
                .into_series(),
            #[cfg(feature = "dtype-decimal")]
            Decimal(precision, scale) => Int128Chunked::from_chunks(name, chunks)
                .into_decimal_unchecked(
                    *precision,
                    scale.unwrap_or_else(|| unreachable!("scale should be set")),
                )
                .into_series(),
            #[cfg(feature = "dtype-array")]
            Array(_, _) => {
                ArrayChunked::from_chunks_and_dtype_unchecked(name, chunks, dtype.clone())
                    .into_series()
            },
            List(_) => ListChunked::from_chunks_and_dtype_unchecked(name, chunks, dtype.clone())
                .into_series(),
            String => StringChunked::from_chunks(name, chunks).into_series(),
            Binary => BinaryChunked::from_chunks(name, chunks).into_series(),
            #[cfg(feature = "dtype-categorical")]
            Categorical(rev_map, ordering) => {
                let cats = UInt32Chunked::from_chunks(name, chunks);
                let mut ca = CategoricalChunked::from_cats_and_rev_map_unchecked(
                    cats,
                    rev_map.clone().unwrap(),
                    *ordering,
                );
                ca.set_fast_unique(false);
                ca.into_series()
            },
            Boolean => BooleanChunked::from_chunks(name, chunks).into_series(),
            Float32 => Float32Chunked::from_chunks(name, chunks).into_series(),
            Float64 => Float64Chunked::from_chunks(name, chunks).into_series(),
            #[cfg(feature = "dtype-struct")]
            Struct(_) => {
                Series::_try_from_arrow_unchecked(name, chunks, &dtype.to_arrow(true)).unwrap()
            },
            #[cfg(feature = "object")]
            Object(_, _) => {
                assert_eq!(chunks.len(), 1);
                let arr = chunks[0]
                    .as_any()
                    .downcast_ref::<FixedSizeBinaryArray>()
                    .unwrap();
                // Safety:
                // this is highly unsafe. it will dereference a raw ptr on the heap
                // make sure the ptr is allocated and from this pid
                // (the pid is checked before dereference)
                {
                    let pe = PolarsExtension::new(arr.clone());
                    let s = pe.get_series(name);
                    pe.take_and_forget();
                    s
                }
            },
            Null => new_null(name, &chunks),
            Unknown => panic!("uh oh, somehow we don't know the dtype?"),
            #[allow(unreachable_patterns)]
            _ => unreachable!(),
        }
    }

    /// Create a new Series without checking if the inner dtype of the chunks is correct
    ///
    /// # Safety
    /// The caller must ensure that the given `dtype` matches all the `ArrayRef` dtypes.
    pub unsafe fn _try_from_arrow_unchecked(
        name: &str,
        chunks: Vec<ArrayRef>,
        dtype: &ArrowDataType,
    ) -> PolarsResult<Self> {
        match dtype {
            ArrowDataType::LargeUtf8 => Ok(StringChunked::from_chunks(name, chunks).into_series()),
            ArrowDataType::Utf8 => {
                let chunks = cast_chunks(&chunks, &DataType::String, false).unwrap();
                Ok(StringChunked::from_chunks(name, chunks).into_series())
            },
            ArrowDataType::LargeBinary => {
                Ok(BinaryChunked::from_chunks(name, chunks).into_series())
            },
            ArrowDataType::Binary => {
                let chunks = cast_chunks(&chunks, &DataType::Binary, false).unwrap();
                Ok(BinaryChunked::from_chunks(name, chunks).into_series())
            },
            ArrowDataType::List(_) | ArrowDataType::LargeList(_) => {
                let (chunks, dtype) = to_physical_and_dtype(chunks);
                unsafe {
                    Ok(
                        ListChunked::from_chunks_and_dtype_unchecked(name, chunks, dtype)
                            .into_series(),
                    )
                }
            },
            #[cfg(feature = "dtype-array")]
            ArrowDataType::FixedSizeList(_, _) => {
                let (chunks, dtype) = to_physical_and_dtype(chunks);
                unsafe {
                    Ok(
                        ArrayChunked::from_chunks_and_dtype_unchecked(name, chunks, dtype)
                            .into_series(),
                    )
                }
            },
            ArrowDataType::Boolean => Ok(BooleanChunked::from_chunks(name, chunks).into_series()),
            #[cfg(feature = "dtype-u8")]
            ArrowDataType::UInt8 => Ok(UInt8Chunked::from_chunks(name, chunks).into_series()),
            #[cfg(feature = "dtype-u16")]
            ArrowDataType::UInt16 => Ok(UInt16Chunked::from_chunks(name, chunks).into_series()),
            ArrowDataType::UInt32 => Ok(UInt32Chunked::from_chunks(name, chunks).into_series()),
            ArrowDataType::UInt64 => Ok(UInt64Chunked::from_chunks(name, chunks).into_series()),
            #[cfg(feature = "dtype-i8")]
            ArrowDataType::Int8 => Ok(Int8Chunked::from_chunks(name, chunks).into_series()),
            #[cfg(feature = "dtype-i16")]
            ArrowDataType::Int16 => Ok(Int16Chunked::from_chunks(name, chunks).into_series()),
            ArrowDataType::Int32 => Ok(Int32Chunked::from_chunks(name, chunks).into_series()),
            ArrowDataType::Int64 => Ok(Int64Chunked::from_chunks(name, chunks).into_series()),
            ArrowDataType::Float16 => {
                let chunks = cast_chunks(&chunks, &DataType::Float32, false).unwrap();
                Ok(Float32Chunked::from_chunks(name, chunks).into_series())
            },
            ArrowDataType::Float32 => Ok(Float32Chunked::from_chunks(name, chunks).into_series()),
            ArrowDataType::Float64 => Ok(Float64Chunked::from_chunks(name, chunks).into_series()),
            #[cfg(feature = "dtype-date")]
            ArrowDataType::Date32 => {
                let chunks = cast_chunks(&chunks, &DataType::Int32, false).unwrap();
                Ok(Int32Chunked::from_chunks(name, chunks)
                    .into_date()
                    .into_series())
            },
            #[cfg(feature = "dtype-datetime")]
            ArrowDataType::Date64 => {
                let chunks = cast_chunks(&chunks, &DataType::Int64, false).unwrap();
                let ca = Int64Chunked::from_chunks(name, chunks);
                Ok(ca.into_datetime(TimeUnit::Milliseconds, None).into_series())
            },
            #[cfg(feature = "dtype-datetime")]
            ArrowDataType::Timestamp(tu, tz) => {
                let canonical_tz = DataType::canonical_timezone(tz);
                let tz = match canonical_tz.as_deref() {
                    #[cfg(feature = "timezones")]
                    Some(tz_str) => match validate_time_zone(tz_str) {
                        Ok(_) => canonical_tz,
                        Err(_) => Some(parse_fixed_offset(tz_str)?),
                    },
                    _ => canonical_tz,
                };
                let chunks = cast_chunks(&chunks, &DataType::Int64, false).unwrap();
                let s = Int64Chunked::from_chunks(name, chunks)
                    .into_datetime(tu.into(), tz)
                    .into_series();
                Ok(match tu {
                    ArrowTimeUnit::Second => &s * MILLISECONDS,
                    ArrowTimeUnit::Millisecond => s,
                    ArrowTimeUnit::Microsecond => s,
                    ArrowTimeUnit::Nanosecond => s,
                })
            },
            #[cfg(feature = "dtype-duration")]
            ArrowDataType::Duration(tu) => {
                let chunks = cast_chunks(&chunks, &DataType::Int64, false).unwrap();
                let s = Int64Chunked::from_chunks(name, chunks)
                    .into_duration(tu.into())
                    .into_series();
                Ok(match tu {
                    ArrowTimeUnit::Second => &s * MILLISECONDS,
                    ArrowTimeUnit::Millisecond => s,
                    ArrowTimeUnit::Microsecond => s,
                    ArrowTimeUnit::Nanosecond => s,
                })
            },
            #[cfg(feature = "dtype-time")]
            ArrowDataType::Time64(tu) | ArrowDataType::Time32(tu) => {
                let mut chunks = chunks;
                if matches!(dtype, ArrowDataType::Time32(_)) {
                    chunks = cast_chunks(&chunks, &DataType::Int32, false).unwrap();
                }
                let chunks = cast_chunks(&chunks, &DataType::Int64, false).unwrap();
                let s = Int64Chunked::from_chunks(name, chunks)
                    .into_time()
                    .into_series();
                Ok(match tu {
                    ArrowTimeUnit::Second => &s * NANOSECONDS,
                    ArrowTimeUnit::Millisecond => &s * 1_000_000,
                    ArrowTimeUnit::Microsecond => &s * 1_000,
                    ArrowTimeUnit::Nanosecond => s,
                })
            },
            ArrowDataType::Null => Ok(new_null(name, &chunks)),
            #[cfg(not(feature = "dtype-categorical"))]
            ArrowDataType::Dictionary(_, _, _) => {
                panic!("activate dtype-categorical to convert dictionary arrays")
            },
            #[cfg(feature = "dtype-categorical")]
            ArrowDataType::Dictionary(key_type, value_type, _) => {
                use arrow::datatypes::IntegerType;
                // don't spuriously call this; triggers a read on mmapped data
                let arr = if chunks.len() > 1 {
                    concatenate_owned_unchecked(&chunks)?
                } else {
                    chunks[0].clone()
                };

                if !matches!(
                    value_type.as_ref(),
                    ArrowDataType::Utf8 | ArrowDataType::LargeUtf8 | ArrowDataType::Null
                ) {
                    polars_bail!(
                        ComputeError: "only string-like values are supported in dictionaries"
                    );
                }

                macro_rules! unpack_keys_values {
                    ($dt:ty) => {{
                        let arr = arr.as_any().downcast_ref::<DictionaryArray<$dt>>().unwrap();
                        let keys = arr.keys();
                        let keys = cast(keys, &ArrowDataType::UInt32).unwrap();
                        let values = arr.values();
                        let values = cast(&**values, &ArrowDataType::LargeUtf8)?;
                        (keys, values)
                    }};
                }

                let (keys, values) = match key_type {
                    IntegerType::Int8 => {
                        unpack_keys_values!(i8)
                    },
                    IntegerType::UInt8 => {
                        unpack_keys_values!(u8)
                    },
                    IntegerType::Int16 => {
                        unpack_keys_values!(i16)
                    },
                    IntegerType::UInt16 => {
                        unpack_keys_values!(u16)
                    },
                    IntegerType::Int32 => {
                        unpack_keys_values!(i32)
                    },
                    IntegerType::UInt32 => {
                        unpack_keys_values!(u32)
                    },
                    IntegerType::Int64 => {
                        unpack_keys_values!(i64)
                    },
                    _ => polars_bail!(
                        ComputeError: "dictionaries with unsigned 64-bit keys are not supported"
                    ),
                };
                let keys = keys.as_any().downcast_ref::<PrimitiveArray<u32>>().unwrap();
                let values = values.as_any().downcast_ref::<Utf8Array<i64>>().unwrap();

                // Safety
                // the invariants of an Arrow Dictionary guarantee the keys are in bounds
                Ok(
                    CategoricalChunked::from_keys_and_values(
                        name,
                        keys,
                        values,
                        Default::default(),
                    )
                    .into_series(),
                )
            },
            #[cfg(feature = "object")]
            ArrowDataType::Extension(s, _, Some(_)) if s == EXTENSION_NAME => {
                assert_eq!(chunks.len(), 1);
                let arr = chunks[0]
                    .as_any()
                    .downcast_ref::<FixedSizeBinaryArray>()
                    .unwrap();
                // Safety:
                // this is highly unsafe. it will dereference a raw ptr on the heap
                // make sure the ptr is allocated and from this pid
                // (the pid is checked before dereference)
                let s = {
                    let pe = PolarsExtension::new(arr.clone());
                    let s = pe.get_series(name);
                    pe.take_and_forget();
                    s
                };
                Ok(s)
            },
            #[cfg(feature = "dtype-struct")]
            ArrowDataType::Struct(logical_fields) => {
                // We don't have to convert inner types, as that already
                // happens on `Field: Series` construction
                let arr = if chunks.len() > 1 {
                    // don't spuriously call this. This triggers a read on memmapped data
                    concatenate_owned_unchecked(&chunks).unwrap() as ArrayRef
                } else {
                    chunks[0].clone()
                };
                let mut struct_arr =
                    std::borrow::Cow::Borrowed(arr.as_any().downcast_ref::<StructArray>().unwrap());

                if let Some(validity) = struct_arr.validity() {
                    let new_values = struct_arr
                        .values()
                        .iter()
                        .map(|arr| match arr.data_type() {
                            ArrowDataType::Null => arr.clone(),
                            _ => match arr.validity() {
                                None => arr.with_validity(Some(validity.clone())),
                                Some(arr_validity) => {
                                    arr.with_validity(Some(arr_validity & validity))
                                },
                            },
                        })
                        .collect();

                    struct_arr = std::borrow::Cow::Owned(StructArray::new(
                        struct_arr.data_type().clone(),
                        new_values,
                        None,
                    ));
                }

                // ensure we maintain logical types if proved by the caller
                let dtype_fields = if logical_fields.is_empty() {
                    struct_arr.fields()
                } else {
                    logical_fields
                };

                let fields = struct_arr
                    .values()
                    .iter()
                    .zip(dtype_fields)
                    .map(|(arr, field)| {
                        Series::_try_from_arrow_unchecked(
                            &field.name,
                            vec![arr.clone()],
                            &field.data_type,
                        )
                    })
                    .collect::<PolarsResult<Vec<_>>>()?;
                Ok(StructChunked::new_unchecked(name, &fields).into_series())
            },
            ArrowDataType::FixedSizeBinary(_) => {
                if verbose() {
                    eprintln!(
                        "Polars does not support decimal types so the 'Series' are read as Float64"
                    );
                }
                let chunks = cast_chunks(&chunks, &DataType::Binary, true)?;
                Ok(BinaryChunked::from_chunks(name, chunks).into_series())
            },
            #[cfg(feature = "dtype-decimal")]
            ArrowDataType::Decimal(precision, scale)
            | ArrowDataType::Decimal256(precision, scale) => {
                #[cfg(feature = "python")]
                {
                    let (precision, scale) = (Some(*precision), *scale);
                    let chunks =
                        cast_chunks(&chunks, &DataType::Decimal(precision, Some(scale)), false)
                            .unwrap();
                    if decimal_is_active() {
                        Ok(Int128Chunked::from_chunks(name, chunks)
                            .into_decimal_unchecked(precision, scale)
                            .into_series())
                    } else {
                        if verbose() {
                            eprintln!(
                                "Activate beta decimal types to read as decimal. Current behavior casts to Float64."
                            );
                        }
                        Ok(Float64Chunked::from_chunks(
                            name,
                            cast_chunks(&chunks, &DataType::Float64, true).unwrap(),
                        )
                        .into_series())
                    }
                }

                #[cfg(not(feature = "python"))]
                {
                    let (precision, scale) = (Some(*precision), *scale);
                    let chunks =
                        cast_chunks(&chunks, &DataType::Decimal(precision, Some(scale)), false)
                            .unwrap();
                    // or DecimalChunked?
                    Ok(Int128Chunked::from_chunks(name, chunks)
                        .into_decimal_unchecked(precision, scale)
                        .into_series())
                }
            },
            #[allow(unreachable_patterns)]
            ArrowDataType::Decimal256(_, _) | ArrowDataType::Decimal(_, _) => {
                if verbose() {
                    eprintln!(
                        "Polars does not support decimal types so the 'Series' are read as Float64"
                    );
                }
                Ok(Float64Chunked::from_chunks(
                    name,
                    cast_chunks(&chunks, &DataType::Float64, true)?,
                )
                .into_series())
            },
            ArrowDataType::Map(_, _) => map_arrays_to_series(name, chunks),
            dt => polars_bail!(ComputeError: "cannot create series from {:?}", dt),
        }
    }
}

fn map_arrays_to_series(name: &str, chunks: Vec<ArrayRef>) -> PolarsResult<Series> {
    let chunks = chunks
        .iter()
        .map(|arr| {
            // we convert the map to the logical type: List<struct<key, value>>
            let arr = arr.as_any().downcast_ref::<MapArray>().unwrap();
            let inner = arr.field().clone();

            // map has i32 offsets
            let data_type = ListArray::<i32>::default_datatype(inner.data_type().clone());
            Box::new(ListArray::<i32>::new(
                data_type,
                arr.offsets().clone(),
                inner,
                arr.validity().cloned(),
            )) as ArrayRef
        })
        .collect::<Vec<_>>();
    Series::try_from((name, chunks))
}

fn convert<F: Fn(&dyn Array) -> ArrayRef>(arr: &[ArrayRef], f: F) -> Vec<ArrayRef> {
    arr.iter().map(|arr| f(&**arr)).collect()
}

/// Converts to physical types and bubbles up the correct [`DataType`].
unsafe fn to_physical_and_dtype(arrays: Vec<ArrayRef>) -> (Vec<ArrayRef>, DataType) {
    match arrays[0].data_type() {
        ArrowDataType::Utf8 => (
            convert(&arrays, |arr| {
                let arr = arr.as_any().downcast_ref::<Utf8Array<i32>>().unwrap();
                Box::from(utf8_to_large_utf8(arr))
            }),
            DataType::String,
        ),
        #[allow(unused_variables)]
        dt @ ArrowDataType::Dictionary(_, _, _) => {
            feature_gated!("dtype-categorical", {
                let s = unsafe {
                    let dt = dt.clone();
                    Series::_try_from_arrow_unchecked("", arrays, &dt)
                }
                .unwrap();
                (s.chunks().clone(), s.dtype().clone())
            })
        },
        ArrowDataType::List(field) => {
            let out = convert(&arrays, |arr| {
                cast(arr, &ArrowDataType::LargeList(field.clone())).unwrap()
            });
            to_physical_and_dtype(out)
        },
        #[cfg(feature = "dtype-array")]
        #[allow(unused_variables)]
        ArrowDataType::FixedSizeList(_, size) => {
            feature_gated!("dtype-array", {
                let values = arrays
                    .iter()
                    .map(|arr| {
                        let arr = arr.as_any().downcast_ref::<FixedSizeListArray>().unwrap();
                        arr.values().clone()
                    })
                    .collect::<Vec<_>>();

                let (converted_values, dtype) = to_physical_and_dtype(values);

                let arrays = arrays
                    .iter()
                    .zip(converted_values)
                    .map(|(arr, values)| {
                        let arr = arr.as_any().downcast_ref::<FixedSizeListArray>().unwrap();

                        let dtype =
                            FixedSizeListArray::default_datatype(values.data_type().clone(), *size);
                        Box::from(FixedSizeListArray::new(
                            dtype,
                            values,
                            arr.validity().cloned(),
                        )) as ArrayRef
                    })
                    .collect();
                (arrays, DataType::Array(Box::new(dtype), *size))
            })
        },
        ArrowDataType::FixedSizeBinary(_) | ArrowDataType::Binary => {
            let out = convert(&arrays, |arr| {
                cast(arr, &ArrowDataType::LargeBinary).unwrap()
            });
            to_physical_and_dtype(out)
        },
        ArrowDataType::LargeList(_) => {
            let values = arrays
                .iter()
                .map(|arr| {
                    let arr = arr.as_any().downcast_ref::<ListArray<i64>>().unwrap();
                    arr.values().clone()
                })
                .collect::<Vec<_>>();

            let (converted_values, dtype) = to_physical_and_dtype(values);

            let arrays = arrays
                .iter()
                .zip(converted_values)
                .map(|(arr, values)| {
                    let arr = arr.as_any().downcast_ref::<ListArray<i64>>().unwrap();

                    let dtype = ListArray::<i64>::default_datatype(values.data_type().clone());
                    Box::from(ListArray::<i64>::new(
                        dtype,
                        arr.offsets().clone(),
                        values,
                        arr.validity().cloned(),
                    )) as ArrayRef
                })
                .collect();
            (arrays, DataType::List(Box::new(dtype)))
        },
        ArrowDataType::Struct(_fields) => {
            feature_gated!("dtype-struct", {
                debug_assert_eq!(arrays.len(), 1);
                let arr = arrays[0].clone();
                let arr = arr.as_any().downcast_ref::<StructArray>().unwrap();
                let (values, dtypes): (Vec<_>, Vec<_>) = arr
                    .values()
                    .iter()
                    .map(|value| {
                        let mut out = to_physical_and_dtype(vec![value.clone()]);
                        (out.0.pop().unwrap(), out.1)
                    })
                    .unzip();

                let arrow_fields = values
                    .iter()
                    .zip(_fields.iter())
                    .map(|(arr, field)| ArrowField::new(&field.name, arr.data_type().clone(), true))
                    .collect();
                let arrow_array = Box::new(StructArray::new(
                    ArrowDataType::Struct(arrow_fields),
                    values,
                    arr.validity().cloned(),
                )) as ArrayRef;
                let polars_fields = _fields
                    .iter()
                    .zip(dtypes)
                    .map(|(field, dtype)| Field::new(&field.name, dtype))
                    .collect();
                (vec![arrow_array], DataType::Struct(polars_fields))
            })
        },
        // Use Series architecture to convert nested logical types to physical.
        dt @ (ArrowDataType::Duration(_)
        | ArrowDataType::Time32(_)
        | ArrowDataType::Time64(_)
        | ArrowDataType::Timestamp(_, _)
        | ArrowDataType::Date32
        | ArrowDataType::Decimal(_, _)
        | ArrowDataType::Date64) => {
            let dt = dt.clone();
            let mut s = Series::_try_from_arrow_unchecked("", arrays, &dt).unwrap();
            let dtype = s.dtype().clone();
            (std::mem::take(s.chunks_mut()), dtype)
        },
        dt => {
            let dtype = dt.into();
            (arrays, dtype)
        },
    }
}

impl TryFrom<(&str, Vec<ArrayRef>)> for Series {
    type Error = PolarsError;

    fn try_from(name_arr: (&str, Vec<ArrayRef>)) -> PolarsResult<Self> {
        let (name, chunks) = name_arr;

        let mut chunks_iter = chunks.iter();
        let data_type: ArrowDataType = chunks_iter
            .next()
            .ok_or_else(|| polars_err!(NoData: "expected at least one array-ref"))?
            .data_type()
            .clone();

        for chunk in chunks_iter {
            if chunk.data_type() != &data_type {
                polars_bail!(
                    ComputeError: "cannot create series from multiple arrays with different types"
                );
            }
        }
        // Safety:
        // dtype is checked
        unsafe { Series::_try_from_arrow_unchecked(name, chunks, &data_type) }
    }
}

impl TryFrom<(&str, ArrayRef)> for Series {
    type Error = PolarsError;

    fn try_from(name_arr: (&str, ArrayRef)) -> PolarsResult<Self> {
        let (name, arr) = name_arr;
        Series::try_from((name, vec![arr]))
    }
}

/// Used to convert a [`ChunkedArray`], `&dyn SeriesTrait` and [`Series`]
/// into a [`Series`].
/// # Safety
///
/// This trait is marked `unsafe` as the `is_series` return is used
/// to transmute to `Series`. This must always return `false` except
/// for `Series` structs.
pub unsafe trait IntoSeries {
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

#[cfg(feature = "dtype-duration")]
impl From<DurationChunked> for Series {
    fn from(a: DurationChunked) -> Self {
        a.into_series()
    }
}

#[cfg(feature = "dtype-time")]
impl From<TimeChunked> for Series {
    fn from(a: TimeChunked) -> Self {
        a.into_series()
    }
}

unsafe impl IntoSeries for Arc<dyn SeriesTrait> {
    fn into_series(self) -> Series {
        Series(self)
    }
}

unsafe impl IntoSeries for Series {
    fn is_series() -> bool {
        true
    }

    fn into_series(self) -> Series {
        self
    }
}

fn new_null(name: &str, chunks: &[ArrayRef]) -> Series {
    let len = chunks.iter().map(|arr| arr.len()).sum();
    Series::new_null(name, len)
}
