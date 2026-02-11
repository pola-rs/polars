use arrow::datatypes::{IntervalUnit, Metadata};
use arrow::offset::OffsetsBuffer;
#[cfg(any(
    feature = "dtype-date",
    feature = "dtype-datetime",
    feature = "dtype-time",
    feature = "dtype-duration"
))]
use arrow::temporal_conversions::*;
use arrow::types::months_days_ns;
use polars_compute::cast::cast_unchecked as cast;
#[cfg(feature = "dtype-decimal")]
use polars_compute::decimal::dec128_fits;
use polars_error::feature_gated;
use polars_utils::itertools::Itertools;

use crate::chunked_array::cast::{CastOptions, cast_chunks};
#[cfg(feature = "object")]
use crate::chunked_array::object::extension::polars_extension::PolarsExtension;
#[cfg(feature = "object")]
use crate::chunked_array::object::registry::get_object_builder;
use crate::config::check_allow_importing_interval_as_struct;
use crate::prelude::*;

impl Series {
    pub fn from_array<A: ParameterFreeDtypeStaticArray>(name: PlSmallStr, array: A) -> Self {
        unsafe {
            Self::from_chunks_and_dtype_unchecked(
                name,
                vec![Box::new(array)],
                &DataType::from_arrow_dtype(&A::get_dtype()),
            )
        }
    }

    pub fn from_chunk_and_dtype(
        name: PlSmallStr,
        chunk: ArrayRef,
        dtype: &DataType,
    ) -> PolarsResult<Self> {
        if &dtype.to_physical().to_arrow(CompatLevel::newest()) != chunk.dtype() {
            polars_bail!(
                InvalidOperation: "cannot create a series of type '{dtype}' of arrow chunk with type '{:?}'",
                chunk.dtype()
            );
        }

        // SAFETY: We check that the datatype matches.
        let series = unsafe { Self::from_chunks_and_dtype_unchecked(name, vec![chunk], dtype) };
        Ok(series)
    }

    /// Takes chunks and a polars datatype and constructs the Series.
    /// This is faster than creating from chunks and an arrow datatype because there is no
    /// casting involved.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the given `dtype`'s physical type matches all the `ArrayRef` dtypes.
    pub unsafe fn from_chunks_and_dtype_unchecked(
        name: PlSmallStr,
        chunks: Vec<ArrayRef>,
        dtype: &DataType,
    ) -> Self {
        use DataType::*;
        match dtype {
            Int8 => Int8Chunked::from_chunks(name, chunks).into_series(),
            Int16 => Int16Chunked::from_chunks(name, chunks).into_series(),
            Int32 => Int32Chunked::from_chunks(name, chunks).into_series(),
            Int64 => Int64Chunked::from_chunks(name, chunks).into_series(),
            UInt8 => UInt8Chunked::from_chunks(name, chunks).into_series(),
            UInt16 => UInt16Chunked::from_chunks(name, chunks).into_series(),
            UInt32 => UInt32Chunked::from_chunks(name, chunks).into_series(),
            UInt64 => UInt64Chunked::from_chunks(name, chunks).into_series(),
            #[cfg(feature = "dtype-i128")]
            Int128 => Int128Chunked::from_chunks(name, chunks).into_series(),
            #[cfg(feature = "dtype-u128")]
            UInt128 => UInt128Chunked::from_chunks(name, chunks).into_series(),
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
                .into_decimal_unchecked(*precision, *scale)
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
            dt @ (Categorical(_, _) | Enum(_, _)) => {
                with_match_categorical_physical_type!(dt.cat_physical().unwrap(), |$C| {
                    let phys = ChunkedArray::from_chunks(name, chunks);
                    CategoricalChunked::<$C>::from_cats_and_dtype_unchecked(phys, dt.clone()).into_series()
                })
            },
            Boolean => BooleanChunked::from_chunks(name, chunks).into_series(),
            #[cfg(feature = "dtype-f16")]
            Float16 => Float16Chunked::from_chunks(name, chunks).into_series(),
            Float32 => Float32Chunked::from_chunks(name, chunks).into_series(),
            Float64 => Float64Chunked::from_chunks(name, chunks).into_series(),
            BinaryOffset => BinaryOffsetChunked::from_chunks(name, chunks).into_series(),
            #[cfg(feature = "dtype-extension")]
            Extension(typ, storage) => ExtensionChunked::from_storage(
                typ.clone(),
                Series::from_chunks_and_dtype_unchecked(name, chunks, storage),
            )
            .into_series(),
            #[cfg(feature = "dtype-struct")]
            Struct(_) => {
                let mut ca =
                    StructChunked::from_chunks_and_dtype_unchecked(name, chunks, dtype.clone());
                StructChunked::propagate_nulls_mut(&mut ca);
                ca.into_series()
            },
            #[cfg(feature = "object")]
            Object(_) => {
                if let Some(arr) = chunks[0].as_any().downcast_ref::<FixedSizeBinaryArray>() {
                    assert_eq!(chunks.len(), 1);
                    // SAFETY:
                    // this is highly unsafe. it will dereference a raw ptr on the heap
                    // make sure the ptr is allocated and from this pid
                    // (the pid is checked before dereference)
                    {
                        let pe = PolarsExtension::new(arr.clone());
                        let s = pe.get_series(&name);
                        pe.take_and_forget();
                        s
                    }
                } else {
                    unsafe { get_object_builder(name, 0).from_chunks(chunks) }
                }
            },
            Null => new_null(name, &chunks),
            Unknown(_) => {
                panic!("dtype is unknown; consider supplying data-types for all operations")
            },
            #[allow(unreachable_patterns)]
            _ => unreachable!(),
        }
    }

    /// # Safety
    /// The caller must ensure that the given `dtype` matches all the `ArrayRef` dtypes.
    pub unsafe fn _try_from_arrow_unchecked(
        name: PlSmallStr,
        chunks: Vec<ArrayRef>,
        dtype: &ArrowDataType,
    ) -> PolarsResult<Self> {
        Self::_try_from_arrow_unchecked_with_md(name, chunks, dtype, None)
    }

    /// Create a new Series without checking if the inner dtype of the chunks is correct
    ///
    /// # Safety
    /// The caller must ensure that the given `dtype` matches all the `ArrayRef` dtypes.
    pub unsafe fn _try_from_arrow_unchecked_with_md(
        name: PlSmallStr,
        mut chunks: Vec<ArrayRef>,
        dtype: &ArrowDataType,
        md: Option<&Metadata>,
    ) -> PolarsResult<Self> {
        match dtype {
            ArrowDataType::Utf8View => Ok(StringChunked::from_chunks(name, chunks).into_series()),
            ArrowDataType::Utf8 | ArrowDataType::LargeUtf8 => {
                let chunks =
                    cast_chunks(&chunks, &DataType::String, CastOptions::NonStrict).unwrap();
                Ok(StringChunked::from_chunks(name, chunks).into_series())
            },
            ArrowDataType::BinaryView => Ok(BinaryChunked::from_chunks(name, chunks).into_series()),
            ArrowDataType::LargeBinary => {
                if let Some(md) = md {
                    if md.maintain_type() {
                        return Ok(BinaryOffsetChunked::from_chunks(name, chunks).into_series());
                    }
                }
                let chunks =
                    cast_chunks(&chunks, &DataType::Binary, CastOptions::NonStrict).unwrap();
                Ok(BinaryChunked::from_chunks(name, chunks).into_series())
            },
            ArrowDataType::Binary => {
                let chunks =
                    cast_chunks(&chunks, &DataType::Binary, CastOptions::NonStrict).unwrap();
                Ok(BinaryChunked::from_chunks(name, chunks).into_series())
            },
            ArrowDataType::List(_) | ArrowDataType::LargeList(_) => {
                let (chunks, dtype) = to_physical_and_dtype(chunks, md);
                unsafe {
                    Ok(
                        ListChunked::from_chunks_and_dtype_unchecked(name, chunks, dtype)
                            .into_series(),
                    )
                }
            },
            #[cfg(feature = "dtype-array")]
            ArrowDataType::FixedSizeList(_, _) => {
                let (chunks, dtype) = to_physical_and_dtype(chunks, md);
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
            ArrowDataType::UInt128 => feature_gated!(
                "dtype-u128",
                Ok(UInt128Chunked::from_chunks(name, chunks).into_series())
            ),
            #[cfg(feature = "dtype-i8")]
            ArrowDataType::Int8 => Ok(Int8Chunked::from_chunks(name, chunks).into_series()),
            #[cfg(feature = "dtype-i16")]
            ArrowDataType::Int16 => Ok(Int16Chunked::from_chunks(name, chunks).into_series()),
            ArrowDataType::Int32 => Ok(Int32Chunked::from_chunks(name, chunks).into_series()),
            ArrowDataType::Int64 => Ok(Int64Chunked::from_chunks(name, chunks).into_series()),
            ArrowDataType::Int128 => feature_gated!(
                "dtype-i128",
                Ok(Int128Chunked::from_chunks(name, chunks).into_series())
            ),
            #[cfg(feature = "dtype-f16")]
            ArrowDataType::Float16 => {
                let chunks =
                    cast_chunks(&chunks, &DataType::Float16, CastOptions::NonStrict).unwrap();
                Ok(Float16Chunked::from_chunks(name, chunks).into_series())
            },
            ArrowDataType::Float32 => Ok(Float32Chunked::from_chunks(name, chunks).into_series()),
            ArrowDataType::Float64 => Ok(Float64Chunked::from_chunks(name, chunks).into_series()),
            #[cfg(feature = "dtype-date")]
            ArrowDataType::Date32 => {
                let chunks =
                    cast_chunks(&chunks, &DataType::Int32, CastOptions::Overflowing).unwrap();
                Ok(Int32Chunked::from_chunks(name, chunks)
                    .into_date()
                    .into_series())
            },
            #[cfg(feature = "dtype-datetime")]
            ArrowDataType::Date64 => {
                let chunks =
                    cast_chunks(&chunks, &DataType::Int64, CastOptions::Overflowing).unwrap();
                let ca = Int64Chunked::from_chunks(name, chunks);
                Ok(ca.into_datetime(TimeUnit::Milliseconds, None).into_series())
            },
            #[cfg(feature = "dtype-datetime")]
            ArrowDataType::Timestamp(tu, tz) => {
                let tz = TimeZone::opt_try_new(tz.clone())?;
                let chunks =
                    cast_chunks(&chunks, &DataType::Int64, CastOptions::NonStrict).unwrap();
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
                let chunks =
                    cast_chunks(&chunks, &DataType::Int64, CastOptions::NonStrict).unwrap();
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
                    chunks =
                        cast_chunks(&chunks, &DataType::Int32, CastOptions::NonStrict).unwrap();
                }
                let chunks =
                    cast_chunks(&chunks, &DataType::Int64, CastOptions::NonStrict).unwrap();
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
            ArrowDataType::Decimal32(precision, scale) => {
                feature_gated!("dtype-decimal", {
                    polars_compute::decimal::dec128_verify_prec_scale(*precision, *scale)?;

                    let mut chunks = chunks;
                    for chunk in chunks.iter_mut() {
                        let old_chunk = chunk
                            .as_any_mut()
                            .downcast_mut::<PrimitiveArray<i32>>()
                            .unwrap();

                        // For now, we just cast the whole data to i128.
                        let (_, values, validity) = std::mem::take(old_chunk).into_inner();
                        *chunk = PrimitiveArray::new(
                            ArrowDataType::Int128,
                            values.iter().map(|&v| v as i128).collect(),
                            validity,
                        )
                        .to_boxed();
                    }

                    let s = Int128Chunked::from_chunks(name, chunks)
                        .into_decimal_unchecked(*precision, *scale)
                        .into_series();
                    Ok(s)
                })
            },
            ArrowDataType::Decimal64(precision, scale) => {
                feature_gated!("dtype-decimal", {
                    polars_compute::decimal::dec128_verify_prec_scale(*precision, *scale)?;

                    let mut chunks = chunks;
                    for chunk in chunks.iter_mut() {
                        let old_chunk = chunk
                            .as_any_mut()
                            .downcast_mut::<PrimitiveArray<i64>>()
                            .unwrap();

                        // For now, we just cast the whole data to i128.
                        let (_, values, validity) = std::mem::take(old_chunk).into_inner();
                        *chunk = PrimitiveArray::new(
                            ArrowDataType::Int128,
                            values.iter().map(|&v| v as i128).collect(),
                            validity,
                        )
                        .to_boxed();
                    }

                    let s = Int128Chunked::from_chunks(name, chunks)
                        .into_decimal_unchecked(*precision, *scale)
                        .into_series();
                    Ok(s)
                })
            },
            ArrowDataType::Decimal(precision, scale) => {
                feature_gated!("dtype-decimal", {
                    polars_compute::decimal::dec128_verify_prec_scale(*precision, *scale)?;

                    let mut chunks = chunks;
                    for chunk in chunks.iter_mut() {
                        *chunk = std::mem::take(
                            chunk
                                .as_any_mut()
                                .downcast_mut::<PrimitiveArray<i128>>()
                                .unwrap(),
                        )
                        .to(ArrowDataType::Int128)
                        .to_boxed();
                    }

                    let s = Int128Chunked::from_chunks(name, chunks)
                        .into_decimal_unchecked(*precision, *scale)
                        .into_series();
                    Ok(s)
                })
            },
            ArrowDataType::Decimal256(precision, scale) => {
                feature_gated!("dtype-decimal", {
                    use arrow::types::i256;

                    polars_compute::decimal::dec128_verify_prec_scale(*precision, *scale)?;

                    let mut chunks = chunks;
                    for chunk in chunks.iter_mut() {
                        let arr = std::mem::take(
                            chunk
                                .as_any_mut()
                                .downcast_mut::<PrimitiveArray<i256>>()
                                .unwrap(),
                        );
                        let arr_128: PrimitiveArray<i128> = arr.iter().map(|opt_v| {
                            if let Some(v) = opt_v {
                                let smaller: Option<i128> = (*v).try_into().ok();
                                let smaller = smaller.filter(|v| dec128_fits(*v, *precision));
                                smaller.ok_or_else(|| {
                                    polars_err!(ComputeError: "Decimal256 to Decimal128 conversion overflowed, Decimal256 is not (yet) supported in Polars")
                                }).map(Some)
                            } else {
                                Ok(None)
                            }
                        }).try_collect_arr_trusted()?;

                        *chunk = arr_128.to(ArrowDataType::Int128).to_boxed();
                    }

                    let s = Int128Chunked::from_chunks(name, chunks)
                        .into_decimal_unchecked(*precision, *scale)
                        .into_series();
                    Ok(s)
                })
            },
            ArrowDataType::Null => Ok(new_null(name, &chunks)),
            #[cfg(not(feature = "dtype-categorical"))]
            ArrowDataType::Dictionary(_, _, _) => {
                panic!("activate dtype-categorical to convert dictionary arrays")
            },
            #[cfg(feature = "dtype-categorical")]
            ArrowDataType::Dictionary(key_type, _, _) => {
                let polars_dtype = DataType::from_arrow(chunks[0].dtype(), md);

                let mut series_iter = chunks.into_iter().map(|arr| {
                    import_arrow_dictionary_array(name.clone(), arr, key_type, &polars_dtype)
                });

                let mut first = series_iter.next().unwrap()?;

                for s in series_iter {
                    first.append_owned(s?)?;
                }

                Ok(first)
            },
            #[cfg(feature = "object")]
            ArrowDataType::Extension(ext)
                if ext.name == POLARS_OBJECT_EXTENSION_NAME && ext.metadata.is_some() =>
            {
                assert_eq!(chunks.len(), 1);
                let arr = chunks[0]
                    .as_any()
                    .downcast_ref::<FixedSizeBinaryArray>()
                    .unwrap();
                // SAFETY:
                // this is highly unsafe. it will dereference a raw ptr on the heap
                // make sure the ptr is allocated and from this pid
                // (the pid is checked before dereference)
                let s = {
                    let pe = PolarsExtension::new(arr.clone());
                    let s = pe.get_series(&name);
                    pe.take_and_forget();
                    s
                };
                Ok(s)
            },
            #[cfg(feature = "dtype-extension")]
            ArrowDataType::Extension(ext) => {
                use crate::datatypes::extension::get_extension_type_or_storage;

                for chunk in &mut chunks {
                    debug_assert!(
                        chunk.dtype() == dtype,
                        "expected chunk dtype to be {:?}, got {:?}",
                        dtype,
                        chunk.dtype()
                    );
                    *chunk.dtype_mut() = ext.inner.clone();
                }
                let storage = Series::_try_from_arrow_unchecked_with_md(
                    name.clone(),
                    chunks,
                    &ext.inner,
                    md,
                )?;

                Ok(
                    match get_extension_type_or_storage(
                        &ext.name,
                        storage.dtype(),
                        ext.metadata.as_deref(),
                    ) {
                        Some(typ) => ExtensionChunked::from_storage(typ, storage).into_series(),
                        None => storage,
                    },
                )
            },

            #[cfg(feature = "dtype-struct")]
            ArrowDataType::Struct(_) => {
                let (chunks, dtype) = to_physical_and_dtype(chunks, md);

                unsafe {
                    let mut ca =
                        StructChunked::from_chunks_and_dtype_unchecked(name, chunks, dtype);
                    StructChunked::propagate_nulls_mut(&mut ca);
                    Ok(ca.into_series())
                }
            },
            ArrowDataType::FixedSizeBinary(_) => {
                let chunks = cast_chunks(&chunks, &DataType::Binary, CastOptions::NonStrict)?;
                Ok(BinaryChunked::from_chunks(name, chunks).into_series())
            },
            ArrowDataType::Map(field, _is_ordered) => {
                let struct_arrays = chunks
                    .iter()
                    .map(|arr| {
                        let arr = arr.as_any().downcast_ref::<MapArray>().unwrap();
                        arr.field().clone()
                    })
                    .collect::<Vec<_>>();

                let (phys_struct_arrays, dtype) =
                    to_physical_and_dtype(struct_arrays, field.metadata.as_deref());

                let chunks = chunks
                    .iter()
                    .zip(phys_struct_arrays)
                    .map(|(arr, values)| {
                        let arr = arr.as_any().downcast_ref::<MapArray>().unwrap();
                        let offsets: &OffsetsBuffer<i32> = arr.offsets();

                        let validity = values.validity().cloned();

                        Box::from(ListArray::<i64>::new(
                            ListArray::<i64>::default_datatype(values.dtype().clone()),
                            OffsetsBuffer::<i64>::from(offsets),
                            values,
                            validity,
                        )) as ArrayRef
                    })
                    .collect();

                unsafe {
                    let out = ListChunked::from_chunks_and_dtype_unchecked(
                        name,
                        chunks,
                        DataType::List(Box::new(dtype)),
                    );

                    Ok(out.into_series())
                }
            },
            ArrowDataType::Interval(IntervalUnit::MonthDayNano) => {
                check_allow_importing_interval_as_struct("month_day_nano_interval")?;

                feature_gated!("dtype-struct", {
                    let chunks = chunks
                        .into_iter()
                        .map(convert_month_day_nano_to_struct)
                        .collect::<PolarsResult<Vec<_>>>()?;

                    Ok(StructChunked::from_chunks_and_dtype_unchecked(
                        name,
                        chunks,
                        DataType::_month_days_ns_struct_type(),
                    )
                    .into_series())
                })
            },

            dt => polars_bail!(ComputeError: "cannot create series from {:?}", dt),
        }
    }
}

fn convert<F: Fn(&dyn Array) -> ArrayRef>(arr: &[ArrayRef], f: F) -> Vec<ArrayRef> {
    arr.iter().map(|arr| f(&**arr)).collect()
}

/// Converts to physical types and bubbles up the correct [`DataType`].
#[allow(clippy::only_used_in_recursion)]
unsafe fn to_physical_and_dtype(
    arrays: Vec<ArrayRef>,
    md: Option<&Metadata>,
) -> (Vec<ArrayRef>, DataType) {
    match arrays[0].dtype() {
        ArrowDataType::Utf8 | ArrowDataType::LargeUtf8 => {
            let chunks = cast_chunks(&arrays, &DataType::String, CastOptions::NonStrict).unwrap();
            (chunks, DataType::String)
        },
        ArrowDataType::Binary | ArrowDataType::LargeBinary | ArrowDataType::FixedSizeBinary(_) => {
            let chunks = cast_chunks(&arrays, &DataType::Binary, CastOptions::NonStrict).unwrap();
            (chunks, DataType::Binary)
        },
        #[allow(unused_variables)]
        dt @ ArrowDataType::Dictionary(_, _, _) => {
            feature_gated!("dtype-categorical", {
                let s = unsafe {
                    let dt = dt.clone();
                    Series::_try_from_arrow_unchecked_with_md(PlSmallStr::EMPTY, arrays, &dt, md)
                }
                .unwrap();
                (s.chunks().clone(), s.dtype().clone())
            })
        },
        dt @ ArrowDataType::Extension(_) => {
            feature_gated!("dtype-extension", {
                let s = unsafe {
                    let dt = dt.clone();
                    Series::_try_from_arrow_unchecked_with_md(PlSmallStr::EMPTY, arrays, &dt, md)
                }
                .unwrap();
                (s.chunks().clone(), s.dtype().clone())
            })
        },
        ArrowDataType::List(field) => {
            let out = convert(&arrays, |arr| {
                cast(arr, &ArrowDataType::LargeList(field.clone())).unwrap()
            });
            to_physical_and_dtype(out, md)
        },
        #[cfg(feature = "dtype-array")]
        ArrowDataType::FixedSizeList(field, size) => {
            let values = arrays
                .iter()
                .map(|arr| {
                    let arr = arr.as_any().downcast_ref::<FixedSizeListArray>().unwrap();
                    arr.values().clone()
                })
                .collect::<Vec<_>>();

            let (converted_values, dtype) =
                to_physical_and_dtype(values, field.metadata.as_deref());

            let arrays = arrays
                .iter()
                .zip(converted_values)
                .map(|(arr, values)| {
                    let arr = arr.as_any().downcast_ref::<FixedSizeListArray>().unwrap();

                    let dtype = FixedSizeListArray::default_datatype(values.dtype().clone(), *size);
                    Box::from(FixedSizeListArray::new(
                        dtype,
                        arr.len(),
                        values,
                        arr.validity().cloned(),
                    )) as ArrayRef
                })
                .collect();
            (arrays, DataType::Array(Box::new(dtype), *size))
        },
        ArrowDataType::LargeList(field) => {
            let values = arrays
                .iter()
                .map(|arr| {
                    let arr = arr.as_any().downcast_ref::<ListArray<i64>>().unwrap();
                    arr.values().clone()
                })
                .collect::<Vec<_>>();

            let (converted_values, dtype) =
                to_physical_and_dtype(values, field.metadata.as_deref());

            let arrays = arrays
                .iter()
                .zip(converted_values)
                .map(|(arr, values)| {
                    let arr = arr.as_any().downcast_ref::<ListArray<i64>>().unwrap();

                    let dtype = ListArray::<i64>::default_datatype(values.dtype().clone());
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
                let mut pl_fields = None;
                let arrays = arrays
                    .iter()
                    .map(|arr| {
                        let arr = arr.as_any().downcast_ref::<StructArray>().unwrap();
                        let (values, dtypes): (Vec<_>, Vec<_>) = arr
                            .values()
                            .iter()
                            .zip(_fields.iter())
                            .map(|(value, field)| {
                                let mut out = to_physical_and_dtype(
                                    vec![value.clone()],
                                    field.metadata.as_deref(),
                                );
                                (out.0.pop().unwrap(), out.1)
                            })
                            .unzip();

                        let arrow_fields = values
                            .iter()
                            .zip(_fields.iter())
                            .map(|(arr, field)| {
                                ArrowField::new(field.name.clone(), arr.dtype().clone(), true)
                            })
                            .collect();
                        let arrow_array = Box::new(StructArray::new(
                            ArrowDataType::Struct(arrow_fields),
                            arr.len(),
                            values,
                            arr.validity().cloned(),
                        )) as ArrayRef;

                        if pl_fields.is_none() {
                            pl_fields = Some(
                                _fields
                                    .iter()
                                    .zip(dtypes)
                                    .map(|(field, dtype)| Field::new(field.name.clone(), dtype))
                                    .collect_vec(),
                            )
                        }

                        arrow_array
                    })
                    .collect_vec();

                (arrays, DataType::Struct(pl_fields.unwrap()))
            })
        },
        // Use Series architecture to convert nested logical types to physical.
        dt @ (ArrowDataType::Duration(_)
        | ArrowDataType::Time32(_)
        | ArrowDataType::Time64(_)
        | ArrowDataType::Timestamp(_, _)
        | ArrowDataType::Date32
        | ArrowDataType::Decimal(_, _)
        | ArrowDataType::Date64
        | ArrowDataType::Map(_, _)) => {
            let dt = dt.clone();
            let mut s = Series::_try_from_arrow_unchecked(PlSmallStr::EMPTY, arrays, &dt).unwrap();
            let dtype = s.dtype().clone();
            (std::mem::take(s.chunks_mut()), dtype)
        },
        dt => {
            let dtype = DataType::from_arrow(dt, md);
            (arrays, dtype)
        },
    }
}

#[cfg(feature = "dtype-categorical")]
unsafe fn import_arrow_dictionary_array(
    name: PlSmallStr,
    arr: Box<dyn Array>,
    key_type: &arrow::datatypes::IntegerType,
    polars_dtype: &DataType,
) -> PolarsResult<Series> {
    use arrow::datatypes::IntegerType as I;

    if matches!(
        polars_dtype,
        DataType::Categorical(_, _) | DataType::Enum(_, _)
    ) {
        macro_rules! unpack_categorical_chunked {
            ($dt:ty) => {{
                let arr = arr.as_any().downcast_ref::<DictionaryArray<$dt>>().unwrap();
                let keys = arr.keys();
                let values = arr.values();
                let values = cast(&**values, &ArrowDataType::Utf8View)?;
                let values = values.as_any().downcast_ref::<Utf8ViewArray>().unwrap();
                with_match_categorical_physical_type!(polars_dtype.cat_physical().unwrap(), |$C| {
                    let ca = CategoricalChunked::<$C>::from_str_iter(
                        name,
                        polars_dtype.clone(),
                        keys.iter().map(|k| {
                            let k: usize = (*k?).try_into().ok()?;
                            values.get(k)
                        }),
                    )?;
                    Ok(ca.into_series())
                })
            }};
        }

        match key_type {
            I::Int8 => unpack_categorical_chunked!(i8),
            I::UInt8 => unpack_categorical_chunked!(u8),
            I::Int16 => unpack_categorical_chunked!(i16),
            I::UInt16 => unpack_categorical_chunked!(u16),
            I::Int32 => unpack_categorical_chunked!(i32),
            I::UInt32 => unpack_categorical_chunked!(u32),
            I::Int64 => unpack_categorical_chunked!(i64),
            I::UInt64 => unpack_categorical_chunked!(u64),
            _ => polars_bail!(
                ComputeError: "unsupported arrow key type: {key_type:?}"
            ),
        }
    } else {
        macro_rules! unpack_keys_values {
            ($dt:ty) => {{
                let arr = arr.as_any().downcast_ref::<DictionaryArray<$dt>>().unwrap();
                let keys = arr.keys();
                let keys = polars_compute::cast::primitive_to_primitive::<
                    $dt,
                    <IdxType as PolarsNumericType>::Native,
                >(keys, &IDX_DTYPE.to_arrow(CompatLevel::newest()));
                (keys, arr.values())
            }};
        }

        let (keys, values) = match key_type {
            I::Int8 => unpack_keys_values!(i8),
            I::UInt8 => unpack_keys_values!(u8),
            I::Int16 => unpack_keys_values!(i16),
            I::UInt16 => unpack_keys_values!(u16),
            I::Int32 => unpack_keys_values!(i32),
            I::UInt32 => unpack_keys_values!(u32),
            I::Int64 => unpack_keys_values!(i64),
            I::UInt64 => unpack_keys_values!(u64),
            _ => polars_bail!(
                ComputeError: "unsupported arrow key type: {key_type:?}"
            ),
        };

        let values = Series::_try_from_arrow_unchecked_with_md(
            name,
            vec![values.clone()],
            values.dtype(),
            None,
        )?;

        values.take(&IdxCa::from_chunks_and_dtype(
            PlSmallStr::EMPTY,
            vec![keys.to_boxed()],
            IDX_DTYPE,
        ))
    }
}

#[cfg(feature = "dtype-struct")]
fn convert_month_day_nano_to_struct(chunk: Box<dyn Array>) -> PolarsResult<Box<dyn Array>> {
    let arr: &PrimitiveArray<months_days_ns> = chunk.as_any().downcast_ref().unwrap();

    let values: &[months_days_ns] = arr.values();

    let (months_out, days_out, nanoseconds_out): (Vec<i32>, Vec<i32>, Vec<i64>) = values
        .iter()
        .map(|x| (x.months(), x.days(), x.ns()))
        .collect();

    let out = StructArray::new(
        DataType::_month_days_ns_struct_type()
            .to_physical()
            .to_arrow(CompatLevel::newest()),
        arr.len(),
        vec![
            PrimitiveArray::<i32>::from_vec(months_out).boxed(),
            PrimitiveArray::<i32>::from_vec(days_out).boxed(),
            PrimitiveArray::<i64>::from_vec(nanoseconds_out).boxed(),
        ],
        arr.validity().cloned(),
    );

    Ok(out.boxed())
}

fn check_types(chunks: &[ArrayRef]) -> PolarsResult<ArrowDataType> {
    let mut chunks_iter = chunks.iter();
    let dtype: ArrowDataType = chunks_iter
        .next()
        .ok_or_else(|| polars_err!(NoData: "expected at least one array-ref"))?
        .dtype()
        .clone();

    for chunk in chunks_iter {
        if chunk.dtype() != &dtype {
            polars_bail!(
                ComputeError: "cannot create series from multiple arrays with different types"
            );
        }
    }
    Ok(dtype)
}

impl Series {
    pub fn try_new<T>(
        name: PlSmallStr,
        data: T,
    ) -> Result<Self, <(PlSmallStr, T) as TryInto<Self>>::Error>
    where
        (PlSmallStr, T): TryInto<Self>,
    {
        // # TODO
        // * Remove the TryFrom<tuple> impls in favor of this
        <(PlSmallStr, T) as TryInto<Self>>::try_into((name, data))
    }
}

impl TryFrom<(PlSmallStr, Vec<ArrayRef>)> for Series {
    type Error = PolarsError;

    fn try_from(name_arr: (PlSmallStr, Vec<ArrayRef>)) -> PolarsResult<Self> {
        let (name, chunks) = name_arr;

        let dtype = check_types(&chunks)?;
        // SAFETY:
        // dtype is checked
        unsafe { Series::_try_from_arrow_unchecked(name, chunks, &dtype) }
    }
}

impl TryFrom<(PlSmallStr, ArrayRef)> for Series {
    type Error = PolarsError;

    fn try_from(name_arr: (PlSmallStr, ArrayRef)) -> PolarsResult<Self> {
        let (name, arr) = name_arr;
        Series::try_from((name, vec![arr]))
    }
}

impl TryFrom<(&ArrowField, Vec<ArrayRef>)> for Series {
    type Error = PolarsError;

    fn try_from(field_arr: (&ArrowField, Vec<ArrayRef>)) -> PolarsResult<Self> {
        let (field, chunks) = field_arr;
        let arrow_dt = field.dtype();
        let dtype = check_types(&chunks)?;
        let compatible = match (&dtype, arrow_dt) {
            // See #26174, we don't care about dictionary ordering.
            (
                ArrowDataType::Dictionary(int0, inner0, _ord0),
                ArrowDataType::Dictionary(int1, inner1, _ord1),
            ) => (int0, inner0) == (int1, inner1),
            (l, r) => l == r,
        };
        polars_ensure!(compatible, ComputeError: "Arrow Field dtype does not match the ArrayRef dtypes");

        // SAFETY:
        // dtype is checked
        unsafe {
            Series::_try_from_arrow_unchecked_with_md(
                field.name.clone(),
                chunks,
                &dtype,
                field.metadata.as_deref(),
            )
        }
    }
}

impl TryFrom<(&ArrowField, ArrayRef)> for Series {
    type Error = PolarsError;

    fn try_from(field_arr: (&ArrowField, ArrayRef)) -> PolarsResult<Self> {
        let (field, arr) = field_arr;
        Series::try_from((field, vec![arr]))
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

fn new_null(name: PlSmallStr, chunks: &[ArrayRef]) -> Series {
    let len = chunks.iter().map(|arr| arr.len()).sum();
    Series::new_null(name, len)
}
