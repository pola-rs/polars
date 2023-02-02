//! Implementations of the ChunkCast Trait.
use std::convert::TryFrom;

use arrow::compute::cast::CastOptions;
use polars_arrow::compute::cast;

#[cfg(feature = "dtype-categorical")]
use crate::chunked_array::categorical::CategoricalChunkedBuilder;
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
    let chunks = chunks
        .iter()
        .map(|arr| arrow::compute::cast::cast(arr.as_ref(), &arrow_dtype, options))
        .collect::<arrow::error::Result<Vec<_>>>()?;
    Ok(chunks)
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
        Datetime(tu, tz) => out.into_datetime(*tu, tz.clone()),
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

impl<T> ChunkedArray<T>
where
    T: PolarsNumericType,
{
    fn cast_impl(&self, data_type: &DataType, checked: bool) -> PolarsResult<Series> {
        match data_type {
            #[cfg(feature = "dtype-categorical")]
            DataType::Categorical(_) => {
                if self.dtype() == &DataType::UInt32 {
                    // safety:
                    // we are guarded by the type system.
                    let ca = unsafe { &*(self as *const ChunkedArray<T> as *const UInt32Chunked) };
                    CategoricalChunked::from_global_indices(ca.clone()).map(|ca| ca.into_series())
                } else {
                    Err(PolarsError::ComputeError(
                        "Cannot cast numeric types to 'Categorical'".into(),
                    ))
                }
            }
            #[cfg(feature = "dtype-struct")]
            DataType::Struct(fields) => {
                // cast to first field dtype
                let fld = &fields[0];
                let dtype = &fld.dtype;
                let name = &fld.name;
                let s = cast_impl_inner(name, &self.chunks, dtype, true)?;
                Ok(StructChunked::new_unchecked(self.name(), &[s]).into_series())
            }
            _ => cast_impl_inner(self.name(), &self.chunks, data_type, checked).map(|mut s| {
                // maintain sorted if data types remain signed
                // this may still fail with overflow?
                if ((self.dtype().is_signed() && data_type.is_signed())
                    || (self.dtype().is_unsigned() && data_type.is_unsigned()))
                    && (s.null_count() == self.null_count())
                {
                    let is_sorted = self.is_sorted_flag2();
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

    fn cast_unchecked(&self, data_type: &DataType) -> PolarsResult<Series> {
        self.cast_impl(data_type, false)
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
            }
            _ => cast_impl(self.name(), &self.chunks, data_type),
        }
    }

    fn cast_unchecked(&self, data_type: &DataType) -> PolarsResult<Series> {
        self.cast(data_type)
    }
}

#[cfg(feature = "dtype-binary")]
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

#[cfg(feature = "dtype-binary")]
impl ChunkCast for BinaryChunked {
    fn cast(&self, data_type: &DataType) -> PolarsResult<Series> {
        cast_impl(self.name(), &self.chunks, data_type)
    }

    fn cast_unchecked(&self, data_type: &DataType) -> PolarsResult<Series> {
        match data_type {
            DataType::Utf8 => unsafe {
                let chunks = self
                    .downcast_iter()
                    .map(|arr| Box::new(binary_to_utf8_unchecked(arr)) as ArrayRef)
                    .collect();
                Ok(Utf8Chunked::from_chunks(self.name(), chunks).into_series())
            },
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
        if matches!(data_type, DataType::Utf8) {
            let mut ca = boolean_to_utf8(self);
            ca.rename(self.name());
            Ok(ca.into_series())
        } else {
            cast_impl(self.name(), &self.chunks, data_type)
        }
    }

    fn cast_unchecked(&self, data_type: &DataType) -> PolarsResult<Series> {
        self.cast(data_type)
    }
}

fn cast_inner_list_type(list: &ListArray<i64>, child_type: &DataType) -> PolarsResult<ArrayRef> {
    let child = list.values();
    let offsets = list.offsets();
    let child = cast::cast(child.as_ref(), &child_type.to_arrow())?;

    let data_type = ListArray::<i64>::default_datatype(child_type.to_arrow());
    let list = ListArray::new(data_type, offsets.clone(), child, list.validity().cloned());
    Ok(Box::new(list) as ArrayRef)
}

/// We cannot cast anything to or from List/LargeList
/// So this implementation casts the inner type
impl ChunkCast for ListChunked {
    fn cast(&self, data_type: &DataType) -> PolarsResult<Series> {
        use DataType::*;
        match data_type {
            List(child_type) => {
                let phys_child = child_type.to_physical();

                match (self.inner_dtype(), &**child_type) {
                    #[cfg(feature = "dtype-categorical")]
                    (Utf8, Categorical(_)) => {
                        let (arr, inner_dtype) = cast_list(self, child_type)?;
                        Ok(unsafe {
                            Series::from_chunks_and_dtype_unchecked(
                                self.name(),
                                vec![arr],
                                &List(Box::new(inner_dtype)),
                            )
                        })
                    }
                    _ if phys_child.is_primitive() => {
                        let mut ca = if child_type.to_physical() != self.inner_dtype().to_physical()
                        {
                            let chunks = self
                                .downcast_iter()
                                .map(|list| cast_inner_list_type(list, &phys_child))
                                .collect::<PolarsResult<_>>()?;
                            unsafe { ListChunked::from_chunks(self.name(), chunks) }
                        } else {
                            self.clone()
                        };
                        ca.set_inner_dtype(*child_type.clone());
                        Ok(ca.into_series())
                    }
                    _ => {
                        let arr = cast_list(self, child_type)?.0;
                        Series::try_from((self.name(), arr))
                    }
                }
            }
            _ => Err(PolarsError::ComputeError("Cannot cast list type".into())),
        }
    }

    fn cast_unchecked(&self, data_type: &DataType) -> PolarsResult<Series> {
        self.cast(data_type)
    }
}

// returns inner data type
fn cast_list(ca: &ListChunked, child_type: &DataType) -> PolarsResult<(ArrayRef, DataType)> {
    let ca = ca.rechunk();
    let arr = ca.downcast_iter().next().unwrap();
    let s = Series::try_from(("", arr.values().clone())).unwrap();
    let new_inner = s.cast(child_type)?;

    let inner_dtype = new_inner.dtype().clone();

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
