//! Implementations of the ChunkCast Trait.
use std::convert::TryFrom;

use arrow::compute::cast::CastOptions;

#[cfg(feature = "dtype-categorical")]
use crate::chunked_array::categorical::CategoricalChunkedBuilder;
#[cfg(feature = "temporal")]
use crate::chunked_array::temporal::validate_is_number;
#[cfg(feature = "timezones")]
use crate::chunked_array::temporal::validate_time_zone;
use crate::prelude::DataType::Datetime;
use crate::prelude::*;

pub(crate) fn cast_chunks(
    chunks: &[ArrayRef],
    dtype: &DataType,
    checked: bool,
) -> PolarsResult<Vec<ArrayRef>> {
    let options = if checked {
        Default::default()
    } else {
        CastOptions {
            wrapped: true,
            partial: false,
        }
    };

    let arrow_dtype = dtype.to_arrow();
    chunks
        .iter()
        .map(|arr| arrow::compute::cast::cast(arr.as_ref(), &arrow_dtype, options))
        .collect::<PolarsResult<Vec<_>>>()
}

fn cast_impl_inner(
    name: &str,
    chunks: &[ArrayRef],
    dtype: &DataType,
    checked: bool,
) -> PolarsResult<Series> {
    let chunks = cast_chunks(chunks, &dtype.to_physical(), checked)?;
    let out = Series::try_from((name, chunks))?;
    use DataType::*;
    let out = match dtype {
        Date => out.into_date(),
        Datetime(tu, tz) => match tz {
            #[cfg(feature = "timezones")]
            Some(tz) => {
                validate_time_zone(tz)?;
                out.into_datetime(*tu, Some(tz.clone()))
            },
            _ => out.into_datetime(*tu, None),
        },
        Duration(tu) => out.into_duration(*tu),
        #[cfg(feature = "dtype-time")]
        Time => out.into_time(),
        _ => out,
    };

    Ok(out)
}

fn cast_impl(name: &str, chunks: &[ArrayRef], dtype: &DataType) -> PolarsResult<Series> {
    cast_impl_inner(name, chunks, dtype, true)
}

#[cfg(feature = "dtype-struct")]
fn cast_single_to_struct(
    name: &str,
    chunks: &[ArrayRef],
    fields: &[Field],
) -> PolarsResult<Series> {
    let mut new_fields = Vec::with_capacity(fields.len());
    // cast to first field dtype
    let mut fields = fields.iter();
    let fld = fields.next().unwrap();
    let s = cast_impl_inner(&fld.name, chunks, &fld.dtype, true)?;
    let length = s.len();
    new_fields.push(s);

    for fld in fields {
        new_fields.push(Series::full_null(&fld.name, length, &fld.dtype));
    }

    Ok(StructChunked::new_unchecked(name, &new_fields).into_series())
}

impl<T> ChunkedArray<T>
where
    T: PolarsNumericType,
{
    fn cast_impl(&self, data_type: &DataType, checked: bool) -> PolarsResult<Series> {
        if self.dtype() == data_type {
            // safety: chunks are correct dtype
            let mut out = unsafe {
                Series::from_chunks_and_dtype_unchecked(self.name(), self.chunks.clone(), data_type)
            };
            out.set_sorted_flag(self.is_sorted_flag());
            return Ok(out);
        }
        match data_type {
            #[cfg(feature = "dtype-categorical")]
            DataType::Categorical(_) => {
                polars_ensure!(
                    self.dtype() == &DataType::UInt32,
                    ComputeError: "cannot cast numeric types to 'Categorical'"
                );
                // SAFETY
                // we are guarded by the type system
                let ca = unsafe { &*(self as *const ChunkedArray<T> as *const UInt32Chunked) };
                CategoricalChunked::from_global_indices(ca.clone()).map(|ca| ca.into_series())
            },
            #[cfg(feature = "dtype-struct")]
            DataType::Struct(fields) => cast_single_to_struct(self.name(), &self.chunks, fields),
            _ => cast_impl_inner(self.name(), &self.chunks, data_type, checked).map(|mut s| {
                // maintain sorted if data types
                // - remain signed
                // - unsigned -> signed
                // this may still fail with overflow?
                let dtype = self.dtype();

                let to_signed = data_type.is_signed();
                let unsigned2unsigned = dtype.is_unsigned() && data_type.is_unsigned();
                let allowed = to_signed || unsigned2unsigned;

                if (allowed)
                    && (s.null_count() == self.null_count())
                    // physical to logicals
                    || (self.dtype().to_physical() == data_type.to_physical())
                {
                    let is_sorted = self.is_sorted_flag();
                    s.set_sorted_flag(is_sorted)
                }
                s
            }),
        }
    }
}

impl<T> ChunkCast for ChunkedArray<T>
where
    T: PolarsNumericType,
{
    fn cast(&self, data_type: &DataType) -> PolarsResult<Series> {
        self.cast_impl(data_type, true)
    }

    unsafe fn cast_unchecked(&self, data_type: &DataType) -> PolarsResult<Series> {
        match data_type {
            #[cfg(feature = "dtype-categorical")]
            DataType::Categorical(Some(rev_map)) => {
                if self.dtype() == &DataType::UInt32 {
                    // safety:
                    // we are guarded by the type system.
                    let ca = unsafe { &*(self as *const ChunkedArray<T> as *const UInt32Chunked) };
                    Ok(unsafe {
                        CategoricalChunked::from_cats_and_rev_map_unchecked(
                            ca.clone(),
                            rev_map.clone(),
                        )
                    }
                    .into_series())
                } else {
                    polars_bail!(ComputeError: "cannot cast numeric types to 'Categorical'");
                }
            },
            _ => self.cast_impl(data_type, false),
        }
    }
}

impl ChunkCast for Utf8Chunked {
    fn cast(&self, data_type: &DataType) -> PolarsResult<Series> {
        match data_type {
            #[cfg(feature = "dtype-categorical")]
            DataType::Categorical(_) => {
                let iter = self.into_iter();
                let mut builder = CategoricalChunkedBuilder::new(self.name(), self.len());
                builder.drain_iter(iter);
                let ca = builder.finish();
                Ok(ca.into_series())
            },
            #[cfg(feature = "dtype-struct")]
            DataType::Struct(fields) => cast_single_to_struct(self.name(), &self.chunks, fields),
            #[cfg(feature = "dtype-decimal")]
            DataType::Decimal(precision, scale) => match (precision, scale) {
                (precision, Some(scale)) => {
                    let chunks = self.downcast_iter().map(|arr| {
                        arrow::legacy::compute::cast::cast_utf8_to_decimal(arr, *precision, *scale)
                    });
                    Ok(Int128Chunked::from_chunk_iter(self.name(), chunks)
                        .into_decimal_unchecked(*precision, *scale)
                        .into_series())
                },
                (None, None) => self.to_decimal(100),
                _ => {
                    polars_bail!(ComputeError: "expected 'precision' or 'scale' when casting to Decimal")
                },
            },
            #[cfg(feature = "dtype-date")]
            DataType::Date if !validate_is_number(&self.chunks) => {
                let result = cast_chunks(&self.chunks, data_type, true)?;
                let out = Series::try_from((self.name(), result))?;
                Ok(out)
            },
            #[cfg(feature = "dtype-datetime")]
            DataType::Datetime(tu, tz) if !validate_is_number(&self.chunks) => {
                let out = match tz {
                    #[cfg(feature = "timezones")]
                    Some(tz) => {
                        validate_time_zone(tz)?;
                        let result = cast_chunks(
                            &self.chunks,
                            &Datetime(tu.to_owned(), Some(tz.clone())),
                            true,
                        )?;
                        Series::try_from((self.name(), result))
                    },
                    _ => {
                        let result =
                            cast_chunks(&self.chunks, &Datetime(tu.to_owned(), None), true)?;
                        Series::try_from((self.name(), result))
                    },
                };
                out
            },
            _ => cast_impl(self.name(), &self.chunks, data_type),
        }
    }

    unsafe fn cast_unchecked(&self, data_type: &DataType) -> PolarsResult<Series> {
        self.cast(data_type)
    }
}

unsafe fn binary_to_utf8_unchecked(from: &BinaryArray<i64>) -> Utf8Array<i64> {
    let values = from.values().clone();
    let offsets = from.offsets().clone();
    Utf8Array::<i64>::try_new_unchecked(
        ArrowDataType::LargeUtf8,
        offsets,
        values,
        from.validity().cloned(),
    )
    .unwrap()
}

impl BinaryChunked {
    /// # Safety
    /// Utf8 is not validated
    pub unsafe fn to_utf8(&self) -> Utf8Chunked {
        let chunks = self
            .downcast_iter()
            .map(|arr| Box::new(binary_to_utf8_unchecked(arr)) as ArrayRef)
            .collect();
        let field = Arc::new(Field::new(self.name(), DataType::Utf8));
        Utf8Chunked::from_chunks_and_metadata(chunks, field, self.bit_settings, true, true)
    }
}

impl Utf8Chunked {
    pub fn as_binary(&self) -> BinaryChunked {
        let chunks = self
            .downcast_iter()
            .map(|arr| {
                Box::new(arrow::compute::cast::utf8_to_binary(
                    arr,
                    ArrowDataType::LargeBinary,
                )) as ArrayRef
            })
            .collect();
        let field = Arc::new(Field::new(self.name(), DataType::Binary));
        unsafe {
            BinaryChunked::from_chunks_and_metadata(chunks, field, self.bit_settings, true, true)
        }
    }
}

impl ChunkCast for BinaryChunked {
    fn cast(&self, data_type: &DataType) -> PolarsResult<Series> {
        match data_type {
            #[cfg(feature = "dtype-struct")]
            DataType::Struct(fields) => cast_single_to_struct(self.name(), &self.chunks, fields),
            _ => cast_impl(self.name(), &self.chunks, data_type),
        }
    }

    unsafe fn cast_unchecked(&self, data_type: &DataType) -> PolarsResult<Series> {
        match data_type {
            DataType::Utf8 => unsafe { Ok(self.to_utf8().into_series()) },
            _ => self.cast(data_type),
        }
    }
}

fn boolean_to_utf8(ca: &BooleanChunked) -> Utf8Chunked {
    ca.into_iter()
        .map(|opt_b| match opt_b {
            Some(true) => Some("true"),
            Some(false) => Some("false"),
            None => None,
        })
        .collect()
}

impl ChunkCast for BooleanChunked {
    fn cast(&self, data_type: &DataType) -> PolarsResult<Series> {
        match data_type {
            DataType::Utf8 => {
                let mut ca = boolean_to_utf8(self);
                ca.rename(self.name());
                Ok(ca.into_series())
            },
            #[cfg(feature = "dtype-struct")]
            DataType::Struct(fields) => cast_single_to_struct(self.name(), &self.chunks, fields),
            _ => cast_impl(self.name(), &self.chunks, data_type),
        }
    }

    unsafe fn cast_unchecked(&self, data_type: &DataType) -> PolarsResult<Series> {
        self.cast(data_type)
    }
}

/// We cannot cast anything to or from List/LargeList
/// So this implementation casts the inner type
impl ChunkCast for ListChunked {
    fn cast(&self, data_type: &DataType) -> PolarsResult<Series> {
        use DataType::*;
        match data_type {
            List(child_type) => {
                match (self.inner_dtype(), &**child_type) {
                    #[cfg(feature = "dtype-categorical")]
                    (dt, Categorical(None)) if !matches!(dt, Utf8 | Null) => {
                        polars_bail!(ComputeError: "cannot cast List inner type: '{:?}' to Categorical", dt)
                    },
                    _ => {
                        // ensure the inner logical type bubbles up
                        let (arr, child_type) = cast_list(self, child_type)?;
                        // Safety: we just casted so the dtype matches.
                        // we must take this path to correct for physical types.
                        unsafe {
                            Ok(Series::from_chunks_and_dtype_unchecked(
                                self.name(),
                                vec![arr],
                                &List(Box::new(child_type)),
                            ))
                        }
                    },
                }
            },
            #[cfg(feature = "dtype-array")]
            Array(_, _) => {
                // TODO: bubble up logical types.
                let chunks = cast_chunks(self.chunks(), data_type, true)?;
                unsafe { Ok(ArrayChunked::from_chunks(self.name(), chunks).into_series()) }
            },
            _ => {
                polars_bail!(
                    ComputeError: "cannot cast List type (inner: '{:?}', to: '{:?}')",
                    self.inner_dtype(),
                    data_type,
                )
            },
        }
    }

    unsafe fn cast_unchecked(&self, data_type: &DataType) -> PolarsResult<Series> {
        use DataType::*;
        match data_type {
            List(child_type) => cast_list_unchecked(self, child_type),
            _ => self.cast(data_type),
        }
    }
}

/// We cannot cast anything to or from List/LargeList
/// So this implementation casts the inner type
#[cfg(feature = "dtype-array")]
impl ChunkCast for ArrayChunked {
    fn cast(&self, data_type: &DataType) -> PolarsResult<Series> {
        use DataType::*;
        match data_type {
            Array(child_type, width) => {
                match (self.inner_dtype(), &**child_type) {
                    #[cfg(feature = "dtype-categorical")]
                    (dt, Categorical(None)) if !matches!(dt, Utf8) => {
                        polars_bail!(ComputeError: "cannot cast fixed-size-list inner type: '{:?}' to Categorical", dt)
                    },
                    _ => {
                        // ensure the inner logical type bubbles up
                        let (arr, child_type) = cast_fixed_size_list(self, child_type)?;
                        // Safety: we just casted so the dtype matches.
                        // we must take this path to correct for physical types.
                        unsafe {
                            Ok(Series::from_chunks_and_dtype_unchecked(
                                self.name(),
                                vec![arr],
                                &Array(Box::new(child_type), *width),
                            ))
                        }
                    },
                }
            },
            List(_) => {
                // TODO! bubble up logical types
                let chunks = cast_chunks(self.chunks(), data_type, true)?;
                unsafe { Ok(ListChunked::from_chunks(self.name(), chunks).into_series()) }
            },
            _ => polars_bail!(ComputeError: "cannot cast list type"),
        }
    }

    unsafe fn cast_unchecked(&self, data_type: &DataType) -> PolarsResult<Series> {
        self.cast(data_type)
    }
}

// Returns inner data type. This is needed because a cast can instantiate the dtype inner
// values for instance with categoricals
fn cast_list(ca: &ListChunked, child_type: &DataType) -> PolarsResult<(ArrayRef, DataType)> {
    // We still rechunk because we must bubble up a single data-type
    // TODO!: consider a version that works on chunks and merges the data-types and arrays.
    let ca = ca.rechunk();
    let arr = ca.downcast_iter().next().unwrap();
    // safety: inner dtype is passed correctly
    let s = unsafe {
        Series::from_chunks_and_dtype_unchecked("", vec![arr.values().clone()], &ca.inner_dtype())
    };
    let new_inner = s.cast(child_type)?;

    let inner_dtype = new_inner.dtype().clone();
    debug_assert_eq!(&inner_dtype, child_type);

    let new_values = new_inner.array_ref(0).clone();

    let data_type = ListArray::<i64>::default_datatype(new_values.data_type().clone());
    let new_arr = ListArray::<i64>::new(
        data_type,
        arr.offsets().clone(),
        new_values,
        arr.validity().cloned(),
    );
    Ok((Box::new(new_arr), inner_dtype))
}

unsafe fn cast_list_unchecked(ca: &ListChunked, child_type: &DataType) -> PolarsResult<Series> {
    // TODO! add chunked, but this must correct for list offsets.
    let ca = ca.rechunk();
    let arr = ca.downcast_iter().next().unwrap();
    // safety: inner dtype is passed correctly
    let s = unsafe {
        Series::from_chunks_and_dtype_unchecked("", vec![arr.values().clone()], &ca.inner_dtype())
    };
    let new_inner = s.cast_unchecked(child_type)?;
    let new_values = new_inner.array_ref(0).clone();

    let data_type = ListArray::<i64>::default_datatype(new_values.data_type().clone());
    let new_arr = ListArray::<i64>::new(
        data_type,
        arr.offsets().clone(),
        new_values,
        arr.validity().cloned(),
    );
    Ok(ListChunked::from_chunks_and_dtype_unchecked(
        ca.name(),
        vec![Box::new(new_arr)],
        DataType::List(Box::new(child_type.clone())),
    )
    .into_series())
}

// Returns inner data type. This is needed because a cast can instantiate the dtype inner
// values for instance with categoricals
#[cfg(feature = "dtype-array")]
fn cast_fixed_size_list(
    ca: &ArrayChunked,
    child_type: &DataType,
) -> PolarsResult<(ArrayRef, DataType)> {
    let ca = ca.rechunk();
    let arr = ca.downcast_iter().next().unwrap();
    // safety: inner dtype is passed correctly
    let s = unsafe {
        Series::from_chunks_and_dtype_unchecked("", vec![arr.values().clone()], &ca.inner_dtype())
    };
    let new_inner = s.cast(child_type)?;

    let inner_dtype = new_inner.dtype().clone();
    debug_assert_eq!(&inner_dtype, child_type);

    let new_values = new_inner.array_ref(0).clone();

    let data_type =
        FixedSizeListArray::default_datatype(new_values.data_type().clone(), ca.width());
    let new_arr = FixedSizeListArray::new(data_type, new_values, arr.validity().cloned());
    Ok((Box::new(new_arr), inner_dtype))
}

#[cfg(test)]
mod test {
    use crate::prelude::*;

    #[test]
    fn test_cast_list() -> PolarsResult<()> {
        let mut builder =
            ListPrimitiveChunkedBuilder::<Int32Type>::new("a", 10, 10, DataType::Int32);
        builder.append_opt_slice(Some(&[1i32, 2, 3]));
        builder.append_opt_slice(Some(&[1i32, 2, 3]));
        let ca = builder.finish();

        let new = ca.cast(&DataType::List(DataType::Float64.into()))?;

        assert_eq!(new.dtype(), &DataType::List(DataType::Float64.into()));
        Ok(())
    }

    #[test]
    #[cfg(feature = "dtype-categorical")]
    fn test_cast_noop() {
        // check if we can cast categorical twice without panic
        let ca = Utf8Chunked::new("foo", &["bar", "ham"]);
        let out = ca.cast(&DataType::Categorical(None)).unwrap();
        let out = out.cast(&DataType::Categorical(None)).unwrap();
        assert!(matches!(out.dtype(), &DataType::Categorical(_)))
    }
}
