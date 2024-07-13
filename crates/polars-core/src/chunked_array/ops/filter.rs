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
    T: PolarsDataType<HasViews = FalseT, IsObject = FalseT>,
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

// impl ChunkFilter<BooleanType> for BooleanChunked {
//     fn filter(&self, filter: &BooleanChunked) -> PolarsResult<ChunkedArray<BooleanType>> {
//         // Broadcast.
//         if filter.len() == 1 {
//             return match filter.get(0) {
//                 Some(true) => Ok(self.clone()),
//                 _ => Ok(self.clear()),
//             };
//         }
//         check_filter_len!(self, filter);
//         Ok(unsafe {
//             arity::binary_unchecked_same_type(
//                 self,
//                 filter,
//                 |left, mask| filter_fn(left, mask),
//                 true,
//                 true,
//             )
//         })
//     }
// }

impl ChunkFilter<StringType> for StringChunked {
    fn filter(&self, filter: &BooleanChunked) -> PolarsResult<ChunkedArray<StringType>> {
        let out = self.as_binary().filter(filter)?;
        unsafe { Ok(out.to_string_unchecked()) }
    }
}

impl ChunkFilter<BinaryType> for BinaryChunked {
    fn filter(&self, filter: &BooleanChunked) -> PolarsResult<ChunkedArray<BinaryType>> {
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

// impl ChunkFilter<BinaryOffsetType> for BinaryOffsetChunked {
//     fn filter(&self, filter: &BooleanChunked) -> PolarsResult<BinaryOffsetChunked> {
//         // Broadcast.
//         if filter.len() == 1 {
//             return match filter.get(0) {
//                 Some(true) => Ok(self.clone()),
//                 _ => Ok(self.clear()),
//             };
//         }
//         check_filter_len!(self, filter);
//         Ok(unsafe {
//             arity::binary_unchecked_same_type(
//                 self,
//                 filter,
//                 |left, mask| filter_fn(left, mask),
//                 true,
//                 true,
//             )
//         })
//     }
// }
//
// impl ChunkFilter<ListType> for ListChunked {
//     fn filter(&self, filter: &BooleanChunked) -> PolarsResult<ListChunked> {
//         // Broadcast.
//         if filter.len() == 1 {
//             return match filter.get(0) {
//                 Some(true) => Ok(self.clone()),
//                 _ => Ok(self.clear()),
//             };
//         }
//         check_filter_len!(self, filter);
//         Ok(unsafe {
//             arity::binary_unchecked_same_type(
//                 self,
//                 filter,
//                 |left, mask| filter_fn(left, mask),
//                 true,
//                 true,
//             )
//         })
//     }
// }
//
// #[cfg(feature = "dtype-struct")]
// impl ChunkFilter<StructType> for StructChunked {
//     fn filter(&self, filter: &BooleanChunked) -> PolarsResult<ChunkedArray<StructType>>
//     where
//         Self: Sized
//     {
//         if filter.len() == 1 {
//             return match filter.get(0) {
//                 Some(true) => Ok(self.clone()),
//                 _ => Ok(self.clear())
//             }
//         }
//     }
// }
//
// #[cfg(feature = "dtype-array")]
// impl ChunkFilter<FixedSizeListType> for ArrayChunked {
//     fn filter(&self, filter: &BooleanChunked) -> PolarsResult<ArrayChunked> {
//         // Broadcast.
//         if filter.len() == 1 {
//             return match filter.get(0) {
//                 Some(true) => Ok(self.clone()),
//                 _ => Ok(ArrayChunked::from_chunk_iter(
//                     self.name(),
//                     [FixedSizeListArray::new_empty(
//                         self.dtype().to_arrow(CompatLevel::newest()),
//                     )],
//                 )),
//             };
//         }
//         check_filter_len!(self, filter);
//         Ok(unsafe {
//             arity::binary_unchecked_same_type(
//                 self,
//                 filter,
//                 |left, mask| filter_fn(left, mask),
//                 true,
//                 true,
//             )
//         })
//     }
// }

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
                _ => Ok(ObjectChunked::new_empty(self.name())),
            };
        }
        check_filter_len!(self, filter);
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
                        },
                    }
                }
            }
        }
        Ok(builder.finish())
    }
}
