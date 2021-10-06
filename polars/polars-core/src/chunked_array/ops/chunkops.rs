#[cfg(feature = "object")]
use crate::chunked_array::object::builder::ObjectChunkedBuilder;
use crate::prelude::*;
#[cfg(feature = "object")]
use arrow::array::Array;
use arrow::compute::concat;
use itertools::Itertools;
#[cfg(feature = "dtype-categorical")]
use std::ops::Deref;

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

#[cfg(feature = "dtype-categorical")]
impl ChunkOps for CategoricalChunked {
    fn rechunk(&self) -> Self
    where
        Self: std::marker::Sized,
    {
        let mut out: CategoricalChunked = self.deref().rechunk().into();
        let cat_map = self.categorical_map.clone();
        out.categorical_map = cat_map;
        out
    }
}

impl ChunkOps for ListChunked {
    fn rechunk(&self) -> Self {
        if self.chunks.len() == 1 {
            self.clone()
        } else {
            let chunks = vec![concat::concatenate(
                self.chunks.iter().map(|a| &**a).collect_vec().as_slice(),
            )
            .unwrap()
            .into()];
            let mut ca = ListChunked::new_from_chunks(self.name(), chunks);
            if self.can_fast_explode() {
                ca.set_fast_explode()
            }
            ca
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
    #[cfg(feature = "dtype-categorical")]
    fn test_categorical_map_after_rechunk() {
        let s = Series::new("", &["foo", "bar", "spam"]);
        let mut a = s.cast(&DataType::Categorical).unwrap();

        a.append(&a.slice(0, 2)).unwrap();
        a.rechunk();
        assert!(a.categorical().unwrap().categorical_map.is_some());
    }
}
