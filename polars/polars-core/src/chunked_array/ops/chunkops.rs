use crate::chunked_array::builder::get_list_builder;
#[cfg(feature = "object")]
use crate::chunked_array::object::builder::ObjectChunkedBuilder;
use crate::prelude::*;
#[cfg(feature = "object")]
use arrow::array::Array;
use arrow::compute::concat;
use itertools::Itertools;
use polars_arrow::prelude::*;

pub trait ChunkOps {
    /// Aggregate to contiguous memory.
    fn rechunk(&self) -> Self
    where
        Self: std::marker::Sized;
}

impl<T> ChunkOps for ChunkedArray<T>
where
    T: PolarsNumericType,
{
    fn rechunk(&self) -> Self {
        if self.chunks().len() == 1 {
            self.clone()
        } else {
            let chunks = vec![concat::concatenate(
                self.chunks.iter().map(|a| &**a).collect_vec().as_slice(),
            )
            .unwrap()
            .into()];
            ChunkedArray::new_from_chunks(self.name(), chunks)
        }
    }
}

impl ChunkOps for BooleanChunked {
    fn rechunk(&self) -> Self {
        if self.chunks().len() == 1 {
            self.clone()
        } else {
            let chunks = vec![concat::concatenate(
                self.chunks.iter().map(|a| &**a).collect_vec().as_slice(),
            )
            .unwrap()
            .into()];
            ChunkedArray::new_from_chunks(self.name(), chunks)
        }
    }
}

impl ChunkOps for Utf8Chunked {
    fn rechunk(&self) -> Self {
        if self.chunks().len() == 1 {
            self.clone()
        } else {
            let chunks = vec![concat::concatenate(
                self.chunks.iter().map(|a| &**a).collect_vec().as_slice(),
            )
            .unwrap()
            .into()];
            ChunkedArray::new_from_chunks(self.name(), chunks)
        }
    }
}

impl ChunkOps for CategoricalChunked {
    fn rechunk(&self) -> Self
    where
        Self: std::marker::Sized,
    {
        let cat_map = self.categorical_map.clone();
        let mut ca = self.cast::<UInt32Type>().unwrap().rechunk().cast().unwrap();
        ca.categorical_map = cat_map;
        ca
    }
}

impl ChunkOps for ListChunked {
    fn rechunk(&self) -> Self {
        if self.chunks.len() == 1 {
            self.clone()
        } else {
            let values_capacity = self.get_values_size();
            if let DataType::List(dt) = self.dtype() {
                let mut builder =
                    get_list_builder(&dt.into(), values_capacity, self.len(), self.name());
                for v in self {
                    builder.append_opt_series(v.as_ref())
                }
                builder.finish()
            } else {
                unreachable!()
            }
        }
    }
}

#[cfg(feature = "object")]
impl<T> ChunkOps for ObjectChunked<T>
where
    T: PolarsObject,
{
    fn rechunk(&self) -> Self
    where
        Self: std::marker::Sized,
    {
        if self.chunks.len() == 1 {
            self.clone()
        } else {
            let mut builder = ObjectChunkedBuilder::new(self.name(), self.len());
            let chunks = self.downcast_iter();

            // todo! use iterators once implemented
            // no_null path
            if self.null_count() == 0 {
                for arr in chunks {
                    for idx in 0..arr.len() {
                        builder.append_value(arr.value(idx).clone())
                    }
                }
            } else {
                for arr in chunks {
                    for idx in 0..arr.len() {
                        if arr.is_valid(idx) {
                            builder.append_value(arr.value(idx).clone())
                        } else {
                            builder.append_null()
                        }
                    }
                }
            }
            builder.finish()
        }
    }
}

#[cfg(test)]
mod test {
    use crate::prelude::*;

    #[test]
    fn test_categorical_map_after_rechunk() {
        let s = Series::new("", &["foo", "bar", "spam"]);
        let mut a = s.cast::<CategoricalType>().unwrap();

        a.append(&a.slice(0, 2)).unwrap();
        a.rechunk();
        assert!(a.categorical().unwrap().categorical_map.is_some());
    }
}
