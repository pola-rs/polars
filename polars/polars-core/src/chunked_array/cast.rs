//! Implementations of the ChunkCast Trait.
#[cfg(feature = "dtype-categorical")]
use crate::chunked_array::categorical::CategoricalChunkedBuilder;
use crate::prelude::*;
use polars_arrow::compute::cast;
use std::convert::TryFrom;

pub(crate) fn cast_chunks(chunks: &[ArrayRef], dtype: &DataType) -> Result<Vec<ArrayRef>> {
    let chunks = chunks
        .iter()
        .map(|arr| cast::cast(arr.as_ref(), &dtype.to_arrow()))
        .map(|arr| arr.map(|x| x.into()))
        .collect::<arrow::error::Result<Vec<_>>>()?;
    Ok(chunks)
}

fn cast_impl(name: &str, chunks: &[ArrayRef], dtype: &DataType) -> Result<Series> {
    let chunks = cast_chunks(chunks, &dtype.to_physical())?;
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

impl<T> ChunkCast for ChunkedArray<T>
where
    T: PolarsNumericType,
{
    fn cast(&self, data_type: &DataType) -> Result<Series> {
        match data_type {
            #[cfg(feature = "dtype-categorical")]
            DataType::Categorical(_) => {
                Ok(CategoricalChunked::full_null(self.name(), self.len()).into_series())
            }
            _ => cast_impl(self.name(), &self.chunks, data_type),
        }
    }
}

impl ChunkCast for Utf8Chunked {
    fn cast(&self, data_type: &DataType) -> Result<Series> {
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
    fn cast(&self, data_type: &DataType) -> Result<Series> {
        if matches!(data_type, DataType::Utf8) {
            let mut ca = boolean_to_utf8(self);
            ca.rename(self.name());
            Ok(ca.into_series())
        } else {
            cast_impl(self.name(), &self.chunks, data_type)
        }
    }
}

fn cast_inner_list_type(list: &ListArray<i64>, child_type: &DataType) -> Result<ArrayRef> {
    let child = list.values();
    let offsets = list.offsets();
    let child = cast::cast(child.as_ref(), &child_type.to_arrow())?.into();

    let data_type = ListArray::<i64>::default_datatype(child_type.to_arrow());
    let list = ListArray::from_data(data_type, offsets.clone(), child, list.validity().cloned());
    Ok(Arc::new(list) as ArrayRef)
}

/// We cannot cast anything to or from List/LargeList
/// So this implementation casts the inner type
impl ChunkCast for ListChunked {
    fn cast(&self, data_type: &DataType) -> Result<Series> {
        match data_type {
            DataType::List(child_type) => {
                let mut ca = if child_type.to_physical() != self.inner_dtype() {
                    let chunks = self
                        .downcast_iter()
                        .map(|list| cast_inner_list_type(list, &**child_type))
                        .collect::<Result<_>>()?;
                    ListChunked::from_chunks(self.name(), chunks)
                } else {
                    self.clone()
                };
                ca.with_inner_type(*child_type.clone());
                Ok(ca.into_series())
            }
            _ => Err(PolarsError::ComputeError("Cannot cast list type".into())),
        }
    }
}

#[cfg(test)]
mod test {
    use crate::prelude::*;

    #[test]
    fn test_cast_list() -> Result<()> {
        let mut builder = ListPrimitiveChunkedBuilder::<i32>::new("a", 10, 10, DataType::Int32);
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
