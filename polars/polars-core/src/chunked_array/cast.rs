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

    let chunks = chunks
        .iter()
        .map(|arr| arrow::compute::cast::cast(arr.as_ref(), &dtype.to_arrow(), options))
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
                Ok(CategoricalChunked::full_null(self.name(), self.len()).into_series())
            }
            _ => cast_impl_inner(self.name(), &self.chunks, data_type, checked).map(|mut s| {
                // maintain sorted if data types remain signed
                // this may still fail with overflow?
                if ((self.dtype().is_signed() && data_type.is_signed())
                    || (self.dtype().is_unsigned() && data_type.is_unsigned()))
                    && (s.null_count() == self.null_count())
                {
                    let is_sorted = self.is_sorted2();
                    s.set_sorted(is_sorted)
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
    // Safety:
    // offsets are correct as they have not changed
    let list = unsafe {
        ListArray::new_unchecked(data_type, offsets.clone(), child, list.validity().cloned())
    };
    Ok(Box::new(list) as ArrayRef)
}

/// We cannot cast anything to or from List/LargeList
/// So this implementation casts the inner type
impl ChunkCast for ListChunked {
    fn cast(&self, data_type: &DataType) -> PolarsResult<Series> {
        match data_type {
            DataType::List(child_type) => {
                let phys_child = child_type.to_physical();
                let mut ca = if child_type.to_physical() != self.inner_dtype().to_physical() {
                    let chunks = self
                        .downcast_iter()
                        .map(|list| cast_inner_list_type(list, &phys_child))
                        .collect::<PolarsResult<_>>()?;
                    ListChunked::from_chunks(self.name(), chunks)
                } else {
                    self.clone()
                };
                ca.set_inner_dtype(*child_type.clone());
                Ok(ca.into_series())
            }
            _ => Err(PolarsError::ComputeError("Cannot cast list type".into())),
        }
    }

    fn cast_unchecked(&self, data_type: &DataType) -> PolarsResult<Series> {
        self.cast(data_type)
    }
}

#[cfg(test)]
mod test {
    use crate::prelude::*;

    #[test]
    fn test_cast_list() -> PolarsResult<()> {
        let mut builder =
            ListPrimitiveChunkedBuilder::<Int32Type>::new("a", 10, 10, DataType::Int32);
        builder.append_slice(Some(&[1i32, 2, 3]));
        builder.append_slice(Some(&[1i32, 2, 3]));
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
