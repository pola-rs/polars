use polars_compute::filter::filter as filter_fn;

#[cfg(feature = "object")]
use crate::chunked_array::object::builder::ObjectChunkedBuilder;
use crate::prelude::*;

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
    T: PolarsDataType<IsObject = FalseT>,
{
    fn filter(&self, filter: &BooleanChunked) -> PolarsResult<ChunkedArray<T>> {
        // Broadcast.
        if filter.len() == 1 {
            return match filter.get(0) {
                Some(true) => Ok(self.clone()),
                _ => Ok(self.clear()),
            };
        }
        check_filter_len!(self, filter);
        Ok(unsafe {
            arity::binary_unchecked_same_type(
                self,
                filter,
                |left, mask| filter_fn(left, mask),
                true,
                true,
            )
        })
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
        // Broadcast.
        if filter.len() == 1 {
            return match filter.get(0) {
                Some(true) => Ok(self.clone()),
                _ => Ok(ObjectChunked::new_empty(self.name().clone())),
            };
        }
        check_filter_len!(self, filter);
        let chunks = self.downcast_iter().collect::<Vec<_>>();
        let mut builder = ObjectChunkedBuilder::<T>::new(self.name().clone(), self.len());
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
                        },
                    }
                }
            }
        }
        Ok(builder.finish())
    }
}
