#[cfg(feature = "object")]
use arrow::array::Array;
use arrow::compute::filter::filter as filter_fn;

#[cfg(feature = "object")]
use crate::chunked_array::object::builder::ObjectChunkedBuilder;
use crate::prelude::*;
use crate::utils::align_chunks_binary;

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
    fn filter(&self, filter: &BooleanChunked) -> PolarsResult<ChunkedArray<T>> {
        // broadcast
        if filter.len() == 1 {
            return match filter.get(0) {
                Some(true) => Ok(self.clone()),
                _ => Ok(ChunkedArray::from_slice(self.name(), &[])),
            };
        }
        check_filter_len!(self, filter);
        let (left, filter) = align_chunks_binary(self, filter);

        let chunks = left
            .downcast_iter()
            .zip(filter.downcast_iter())
            .map(|(left, mask)| filter_fn(left, mask).unwrap())
            .collect::<Vec<_>>();
        Ok(self.copy_with_chunks(chunks, true))
    }
}

impl ChunkFilter<BooleanType> for BooleanChunked {
    fn filter(&self, filter: &BooleanChunked) -> PolarsResult<ChunkedArray<BooleanType>> {
        // broadcast
        if filter.len() == 1 {
            return match filter.get(0) {
                Some(true) => Ok(self.clone()),
                _ => Ok(ChunkedArray::from_slice(self.name(), &[])),
            };
        }
        check_filter_len!(self, filter);
        let (left, filter) = align_chunks_binary(self, filter);

        let chunks = left
            .downcast_iter()
            .zip(filter.downcast_iter())
            .map(|(left, mask)| filter_fn(left, mask).unwrap())
            .collect::<Vec<_>>();
        Ok(self.copy_with_chunks(chunks, true))
    }
}

impl ChunkFilter<Utf8Type> for Utf8Chunked {
    fn filter(&self, filter: &BooleanChunked) -> PolarsResult<ChunkedArray<Utf8Type>> {
        // broadcast
        if filter.len() == 1 {
            return match filter.get(0) {
                Some(true) => Ok(self.clone()),
                _ => Ok(Utf8Chunked::full_null(self.name(), 0)),
            };
        }
        check_filter_len!(self, filter);
        let (left, filter) = align_chunks_binary(self, filter);

        let chunks = left
            .downcast_iter()
            .zip(filter.downcast_iter())
            .map(|(left, mask)| filter_fn(left, mask).unwrap())
            .collect::<Vec<_>>();

        Ok(self.copy_with_chunks(chunks, true))
    }
}

#[cfg(feature = "dtype-binary")]
impl ChunkFilter<BinaryType> for BinaryChunked {
    fn filter(&self, filter: &BooleanChunked) -> PolarsResult<ChunkedArray<BinaryType>> {
        // broadcast
        if filter.len() == 1 {
            return match filter.get(0) {
                Some(true) => Ok(self.clone()),
                _ => Ok(BinaryChunked::full_null(self.name(), 0)),
            };
        }
        check_filter_len!(self, filter);
        let (left, filter) = align_chunks_binary(self, filter);

        let chunks = left
            .downcast_iter()
            .zip(filter.downcast_iter())
            .map(|(left, mask)| filter_fn(left, mask).unwrap())
            .collect::<Vec<_>>();

        Ok(self.copy_with_chunks(chunks, true))
    }
}

impl ChunkFilter<ListType> for ListChunked {
    fn filter(&self, filter: &BooleanChunked) -> PolarsResult<ListChunked> {
        // broadcast
        if filter.len() == 1 {
            return match filter.get(0) {
                Some(true) => Ok(self.clone()),
                _ => unsafe {
                    Ok(ListChunked::from_chunks(
                        self.name(),
                        vec![new_empty_array(self.dtype().to_arrow())],
                    ))
                },
            };
        }
        let (left, filter) = align_chunks_binary(self, filter);

        let chunks = left
            .downcast_iter()
            .zip(filter.downcast_iter())
            .map(|(left, mask)| filter_fn(left, mask).unwrap())
            .collect::<Vec<_>>();

        // inner type may be categorical or logical type so we clone the state.
        let mut ca = self.clone();
        ca.chunks = chunks;
        ca.compute_len();
        Ok(ca)
    }
}

#[cfg(feature = "object")]
impl<T> ChunkFilter<ObjectType<T>> for ObjectChunked<T>
where
    T: PolarsObject,
{
    fn filter(&self, filter: &BooleanChunked) -> PolarsResult<ChunkedArray<ObjectType<T>>>
    where
        Self: Sized,
    {
        // broadcast
        if filter.len() == 1 {
            return match filter.get(0) {
                Some(true) => Ok(self.clone()),
                _ => Ok(ObjectChunked::new_empty(self.name())),
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
