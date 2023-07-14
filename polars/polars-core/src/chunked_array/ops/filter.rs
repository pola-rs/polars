#[cfg(feature = "object")]
use arrow::array::Array;
use arrow::compute::filter::filter as filter_fn;

#[cfg(feature = "object")]
use crate::chunked_array::object::builder::ObjectChunkedBuilder;
use crate::prelude::*;
use crate::utils::align_chunks_binary;

macro_rules! check_filter_len {
    ($self:expr, $filter:expr) => {{
        polars_ensure!(
            $self.len() == $filter.len(),
            ShapeMismatch: "filter's length: {} differs from that of the series: {}",
            $filter.len(), $self.len()
        )
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
        unsafe { Ok(self.copy_with_chunks(chunks, true, true)) }
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
        unsafe { Ok(self.copy_with_chunks(chunks, true, true)) }
    }
}

impl ChunkFilter<Utf8Type> for Utf8Chunked {
    fn filter(&self, filter: &BooleanChunked) -> PolarsResult<ChunkedArray<Utf8Type>> {
        let out = self.as_binary().filter(filter)?;
        unsafe { Ok(out.to_utf8()) }
    }
}

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

        unsafe { Ok(self.copy_with_chunks(chunks, true, true)) }
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

#[cfg(feature = "dtype-array")]
impl ChunkFilter<FixedSizeListType> for ArrayChunked {
    fn filter(&self, filter: &BooleanChunked) -> PolarsResult<ArrayChunked> {
        // broadcast
        if filter.len() == 1 {
            return match filter.get(0) {
                Some(true) => Ok(self.clone()),
                _ => unsafe {
                    Ok(ChunkedArray::from_chunks(
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
        polars_ensure!(!self.is_empty(), NoData: "cannot filter empty object array");
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
