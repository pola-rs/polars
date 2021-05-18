#[cfg(feature = "object")]
use crate::chunked_array::object::builder::ObjectChunkedBuilder;
use crate::prelude::*;
use crate::utils::align_chunks_binary;
#[cfg(feature = "object")]
use arrow::array::Array;
use arrow::compute::filter as filter_fn;
use std::ops::Deref;

macro_rules! check_filter_len {
    ($self:expr, $filter:expr) => {{
        if $self.len() != $filter.len() {
            return Err(PolarsError::ShapeMisMatch(
                format!(
                    "Filter's length differs from that of the ChunkedArray/ Series. \
                Length Self: {} Length mask: {}\
                Self: {:?}; mask: {:?}",
                    $self.len(),
                    $filter.len(),
                    $self,
                    $filter
                )
                .into(),
            ));
        }
    }};
}

impl<T> ChunkFilter<T> for ChunkedArray<T>
where
    T: PolarsNumericType,
{
    fn filter(&self, filter: &BooleanChunked) -> Result<ChunkedArray<T>> {
        // broadcast
        if filter.len() == 1 {
            return match filter.get(0) {
                Some(true) => Ok(self.clone()),
                _ => Ok(ChunkedArray::new_from_slice(self.name(), &[])),
            };
        }
        check_filter_len!(self, filter);
        let (left, filter) = align_chunks_binary(self, filter);

        let chunks = left
            .downcast_iter()
            .zip(filter.downcast_iter())
            .map(|(left, mask)| filter_fn(left, mask).unwrap())
            .collect::<Vec<_>>();
        Ok(ChunkedArray::new_from_chunks(self.name(), chunks))
    }
}

impl ChunkFilter<BooleanType> for BooleanChunked {
    fn filter(&self, filter: &BooleanChunked) -> Result<ChunkedArray<BooleanType>> {
        // broadcast
        if filter.len() == 1 {
            return match filter.get(0) {
                Some(true) => Ok(self.clone()),
                _ => Ok(ChunkedArray::new_from_slice(self.name(), &[])),
            };
        }
        check_filter_len!(self, filter);
        let (left, filter) = align_chunks_binary(self, filter);

        let chunks = left
            .downcast_iter()
            .zip(filter.downcast_iter())
            .map(|(left, mask)| filter_fn(left, mask).unwrap())
            .collect::<Vec<_>>();
        Ok(ChunkedArray::new_from_chunks(self.name(), chunks))
    }
}

impl ChunkFilter<Utf8Type> for Utf8Chunked {
    fn filter(&self, filter: &BooleanChunked) -> Result<ChunkedArray<Utf8Type>> {
        // broadcast
        if filter.len() == 1 {
            return match filter.get(0) {
                Some(true) => Ok(self.clone()),
                _ => Ok(self.slice(0, 0)),
            };
        }
        check_filter_len!(self, filter);
        let (left, filter) = align_chunks_binary(self, filter);

        let chunks = left
            .downcast_iter()
            .zip(filter.downcast_iter())
            .map(|(left, mask)| filter_fn(left, mask).unwrap())
            .collect::<Vec<_>>();
        Ok(ChunkedArray::new_from_chunks(self.name(), chunks))
    }
}

impl ChunkFilter<CategoricalType> for CategoricalChunked {
    fn filter(&self, filter: &BooleanChunked) -> Result<ChunkedArray<CategoricalType>>
    where
        Self: Sized,
    {
        let ca: CategoricalChunked = self.deref().filter(filter)?.cast()?;
        Ok(ca.set_state(self))
    }
}

impl ChunkFilter<ListType> for ListChunked {
    fn filter(&self, filter: &BooleanChunked) -> Result<ListChunked> {
        // broadcast
        if filter.len() == 1 {
            return match filter.get(0) {
                Some(true) => Ok(self.clone()),
                _ => Ok(self.slice(0, 0)),
            };
        }
        let (left, filter) = align_chunks_binary(self, filter);

        let chunks = left
            .downcast_iter()
            .zip(filter.downcast_iter())
            .map(|(left, mask)| filter_fn(left, mask).unwrap())
            .collect::<Vec<_>>();
        Ok(ChunkedArray::new_from_chunks(self.name(), chunks))
    }
}

#[cfg(feature = "object")]
impl<T> ChunkFilter<ObjectType<T>> for ObjectChunked<T>
where
    T: PolarsObject,
{
    fn filter(&self, filter: &BooleanChunked) -> Result<ChunkedArray<ObjectType<T>>>
    where
        Self: Sized,
    {
        // broadcast
        if filter.len() == 1 {
            return match filter.get(0) {
                Some(true) => Ok(self.clone()),
                _ => Ok(ObjectChunked::new_from_chunks(self.name(), vec![])),
            };
        }
        if self.is_empty() {
            return Err(PolarsError::NoData(
                "cannot filter empty object array".into(),
            ));
        }
        let chunks = self.downcast_iter().collect::<Vec<_>>();
        let mut builder = ObjectChunkedBuilder::<T>::new(self.name(), self.len());
        for (idx, mask) in filter.into_iter().enumerate() {
            if mask.unwrap_or(false) {
                let (chunk_idx, idx) = self.index_to_chunked_index(idx);
                unsafe {
                    let arr = chunks.get_unchecked(chunk_idx);
                    match arr.is_null(idx) {
                        true => builder.append_null(),
                        false => {
                            let v = arr.value(idx);
                            builder.append_value(v.clone())
                        }
                    }
                }
            }
        }
        Ok(builder.finish())
    }
}
