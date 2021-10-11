//! Implementations of the ChunkCast Trait.
#[cfg(feature = "dtype-categorical")]
use crate::chunked_array::categorical::CategoricalChunkedBuilder;
use crate::prelude::*;
use arrow::compute::cast;
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
        Date | Datetime => out.into_date(),
        #[cfg(feature = "dtype-time")]
        Time => out.into_time(),
        _ => out,
    };

    Ok(out)
}

#[cfg(feature = "dtype-categorical")]
impl ChunkCast for CategoricalChunked {
    fn cast(&self, data_type: &DataType) -> Result<Series> {
        match data_type {
            DataType::Utf8 => {
                let mapping = &**self.categorical_map.as_ref().expect("should be set");

                let mut builder = Utf8ChunkedBuilder::new(self.name(), self.len(), self.len() * 5);

                let f = |idx: u32| mapping.get(idx);

                if self.null_count() == 0 {
                    self.into_no_null_iter()
                        .for_each(|idx| builder.append_value(f(idx)));
                } else {
                    self.into_iter().for_each(|opt_idx| {
                        builder.append_option(opt_idx.map(f));
                    });
                }

                let ca = builder.finish();
                Ok(ca.into_series())
            }
            DataType::UInt32 => {
                let ca = UInt32Chunked::new_from_chunks(self.name(), self.chunks.clone());
                Ok(ca.into_series())
            }
            #[cfg(feature = "dtype-categorical")]
            DataType::Categorical => Ok(self.clone().into_series()),
            _ => cast_impl(self.name(), &self.chunks, data_type),
        }
    }
}

impl<T> ChunkCast for ChunkedArray<T>
where
    T: PolarsNumericType,
{
    fn cast(&self, data_type: &DataType) -> Result<Series> {
        match (self.dtype(), data_type) {
            #[cfg(feature = "dtype-categorical")]
            (DataType::UInt32, DataType::Categorical)
            | (DataType::Categorical, DataType::Categorical) => {
                let ca = CategoricalChunked::new_from_chunks(self.name(), self.chunks.clone())
                    .set_state(self);
                Ok(ca.into_series())
            }
            _ => cast_impl(self.name(), &self.chunks, data_type),
        }
    }
}

impl ChunkCast for Utf8Chunked {
    fn cast(&self, data_type: &DataType) -> Result<Series> {
        match data_type {
            #[cfg(feature = "dtype-categorical")]
            DataType::Categorical => {
                let iter = self.into_iter();
                let mut builder = CategoricalChunkedBuilder::new(self.name(), self.len());
                builder.from_iter(iter);
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
                let chunks = self
                    .downcast_iter()
                    .map(|list| cast_inner_list_type(list, &**child_type))
                    .collect::<Result<_>>()?;
                let ca = ListChunked::new_from_chunks(self.name(), chunks);
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
        let mut builder = ListPrimitiveChunkedBuilder::<Int32Type>::new("a", 10, 10);
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
        let ca = Utf8Chunked::new_from_slice("foo", &["bar", "ham"]);
        let out = ca.cast(&DataType::Categorical).unwrap();
        let out = out.cast(&DataType::Categorical).unwrap();
        assert_eq!(out.dtype(), &DataType::Categorical)
    }
}
