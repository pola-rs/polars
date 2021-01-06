use crate::chunked_array::builder::get_list_builder;
#[cfg(feature = "object")]
use crate::chunked_array::object::builder::ObjectChunkedBuilder;
use crate::prelude::*;
use crate::utils::Xob;
#[cfg(feature = "object")]
use arrow::array::Array;
use arrow::array::ArrayRef;
use arrow::compute::filter_primitive_array;
use std::sync::Arc;

macro_rules! impl_filter_with_nulls_in_both {
    ($self:expr, $filter:expr) => {{
        let ca = $self
            .into_iter()
            .zip($filter)
            .filter_map(|(val, valid)| match valid {
                Some(valid) => {
                    if valid {
                        Some(val)
                    } else {
                        None
                    }
                }
                None => None,
            })
            .collect();
        Ok(ca)
    }};
}

macro_rules! impl_filter_no_nulls_in_mask {
    ($self:expr, $filter:expr) => {{
        let ca = $self
            .into_iter()
            .zip($filter.into_no_null_iter())
            .filter_map(|(val, valid)| if valid { Some(val) } else { None })
            .collect();
        Ok(ca)
    }};
}

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

macro_rules! impl_filter_no_nulls {
    ($self:expr, $filter:expr) => {{
        $self
            .into_no_null_iter()
            .zip($filter.into_no_null_iter())
            .filter_map(|(val, valid)| if valid { Some(val) } else { None })
            .collect()
    }};
}

macro_rules! impl_filter_no_nulls_in_self {
    ($self:expr, $filter:expr) => {{
        $self
            .into_no_null_iter()
            .zip($filter)
            .filter_map(|(val, valid)| match valid {
                Some(valid) => {
                    if valid {
                        Some(val)
                    } else {
                        None
                    }
                }
                None => None,
            })
            .collect()
    }};
}

impl<T> ChunkFilter<T> for ChunkedArray<T>
where
    T: PolarsNumericType,
    ChunkedArray<T>: ChunkOps,
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
        if self.chunk_id == filter.chunk_id {
            let chunks = self
                .downcast_chunks()
                .iter()
                .zip(filter.downcast_chunks())
                .map(|(&left, mask)| {
                    Arc::new(filter_primitive_array(left, mask).unwrap()) as ArrayRef
                })
                .collect::<Vec<_>>();
            return Ok(ChunkedArray::new_from_chunks(self.name(), chunks));
        }
        let out = match (self.null_count(), filter.null_count()) {
            (0, 0) => {
                let ca: Xob<ChunkedArray<_>> = impl_filter_no_nulls!(self, filter);
                Ok(ca.into_inner())
            }
            (0, _) => {
                let ca: Xob<ChunkedArray<_>> = impl_filter_no_nulls_in_self!(self, filter);
                Ok(ca.into_inner())
            }
            (_, 0) => impl_filter_no_nulls_in_mask!(self, filter),
            (_, _) => impl_filter_with_nulls_in_both!(self, filter),
        };
        out.map(|mut ca| {
            ca.rename(self.name());
            ca
        })
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
        let out = match (self.null_count(), filter.null_count()) {
            (0, 0) => {
                let ca: Xob<ChunkedArray<_>> = impl_filter_no_nulls!(self, filter);
                Ok(ca.into_inner())
            }
            (0, _) => {
                let ca: Xob<ChunkedArray<_>> = impl_filter_no_nulls_in_self!(self, filter);
                Ok(ca.into_inner())
            }
            (_, 0) => impl_filter_no_nulls_in_mask!(self, filter),
            (_, _) => impl_filter_with_nulls_in_both!(self, filter),
        };
        out.map(|mut ca| {
            ca.rename(self.name());
            ca
        })
    }
}
impl ChunkFilter<Utf8Type> for Utf8Chunked {
    fn filter(&self, filter: &BooleanChunked) -> Result<ChunkedArray<Utf8Type>> {
        // broadcast
        if filter.len() == 1 {
            return match filter.get(0) {
                Some(true) => Ok(self.clone()),
                _ => Ok(Utf8Chunked::new_from_chunks(self.name(), vec![])),
            };
        }
        check_filter_len!(self, filter);
        let out: Result<Utf8Chunked> = match (self.null_count(), filter.null_count()) {
            (0, 0) => {
                let ca = impl_filter_no_nulls!(self, filter);
                Ok(ca)
            }
            (0, _) => {
                let ca = impl_filter_no_nulls_in_self!(self, filter);
                Ok(ca)
            }
            (_, 0) => impl_filter_no_nulls_in_mask!(self, filter),
            (_, _) => impl_filter_with_nulls_in_both!(self, filter),
        };

        out.map(|mut ca| {
            ca.rename(self.name());
            ca
        })
    }
}

impl ChunkFilter<ListType> for ListChunked {
    fn filter(&self, filter: &BooleanChunked) -> Result<ListChunked> {
        // broadcast
        if filter.len() == 1 {
            return match filter.get(0) {
                Some(true) => Ok(self.clone()),
                _ => Ok(ListChunked::new_from_chunks(self.name(), vec![])),
            };
        }
        let dt = self.get_inner_dtype();
        let mut builder = get_list_builder(&dt.into(), self.len(), self.name());
        filter
            .into_iter()
            .zip(self.into_iter())
            .for_each(|(opt_bool_val, opt_series)| {
                let bool_val = opt_bool_val.unwrap_or(false);
                let opt_val = match bool_val {
                    true => opt_series,
                    false => None,
                };
                builder.append_opt_series(opt_val.as_ref())
            });
        Ok(builder.finish())
    }
}

#[cfg(feature = "object")]
impl<T> ChunkFilter<ObjectType<T>> for ObjectChunked<T>
where
    T: 'static + std::fmt::Debug + Clone + Send + Sync + Default,
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
        let chunks = self.downcast_chunks();
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

#[cfg(test)]
#[cfg(feature = "object")]
mod test {
    use super::*;

    #[test]
    fn object_filter() {
        let ca = ObjectChunked::new_from_opt_slice("foo", &[Some(1), None, Some(3), None]);
        let mask = BooleanChunked::new_from_slice("", &[true, false, false, true]);
        let new = ca.filter(&mask).unwrap();
        assert_eq!(Vec::from(new.is_null()), &[Some(false), Some(true)])
    }
}
