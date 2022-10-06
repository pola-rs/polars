//! Traits to provide fast Random access to ChunkedArrays data.
//! This prevents downcasting every iteration.
//! IntoTakeRandom provides structs that implement the TakeRandom trait.
//! There are several structs that implement the fastest path for random access.
//!
use std::borrow::Cow;

use polars_arrow::compute::take::*;
pub use take_random::*;
pub use traits::*;

use crate::chunked_array::kernels::take::*;
use crate::prelude::*;
use crate::utils::NoNull;

mod take_chunked;
mod take_every;
pub(crate) mod take_random;
pub(crate) mod take_single;
mod traits;
#[cfg(feature = "chunked_ids")]
pub(crate) use take_chunked::*;

macro_rules! take_iter_n_chunks {
    ($ca:expr, $indices:expr) => {{
        let taker = $ca.take_rand();
        $indices.into_iter().map(|idx| taker.get(idx)).collect()
    }};
}

macro_rules! take_opt_iter_n_chunks {
    ($ca:expr, $indices:expr) => {{
        let taker = $ca.take_rand();
        $indices
            .into_iter()
            .map(|opt_idx| opt_idx.and_then(|idx| taker.get(idx)))
            .collect()
    }};
}

macro_rules! take_iter_n_chunks_unchecked {
    ($ca:expr, $indices:expr) => {{
        let taker = $ca.take_rand();
        $indices
            .into_iter()
            .map(|idx| taker.get_unchecked(idx))
            .collect()
    }};
}

macro_rules! take_opt_iter_n_chunks_unchecked {
    ($ca:expr, $indices:expr) => {{
        let taker = $ca.take_rand();
        $indices
            .into_iter()
            .map(|opt_idx| opt_idx.and_then(|idx| taker.get_unchecked(idx)))
            .collect()
    }};
}

impl<T> ChunkTake for ChunkedArray<T>
where
    T: PolarsNumericType,
{
    unsafe fn take_unchecked<I, INulls>(&self, indices: TakeIdx<I, INulls>) -> Self
    where
        Self: std::marker::Sized,
        I: TakeIterator,
        INulls: TakeIteratorNulls,
    {
        let mut chunks = self.downcast_iter();
        match indices {
            TakeIdx::Array(array) => {
                if array.null_count() == array.len() {
                    return Self::full_null(self.name(), array.len());
                }
                let array = match (self.null_count(), self.chunks.len()) {
                    (0, 1) => {
                        take_no_null_primitive_unchecked::<T::Native>(chunks.next().unwrap(), array)
                            as ArrayRef
                    }
                    (_, 1) => take_primitive_unchecked::<T::Native>(chunks.next().unwrap(), array)
                        as ArrayRef,
                    _ => {
                        return if !array.has_validity() {
                            let iter = array.values().iter().map(|i| *i as usize);
                            let mut ca = take_primitive_iter_n_chunks(self, iter);
                            ca.rename(self.name());
                            ca
                        } else {
                            let iter = array
                                .into_iter()
                                .map(|opt_idx| opt_idx.map(|idx| *idx as usize));
                            let mut ca = take_primitive_opt_iter_n_chunks(self, iter);
                            ca.rename(self.name());
                            ca
                        }
                    }
                };
                self.copy_with_chunks(vec![array], false)
            }
            TakeIdx::Iter(iter) => {
                if self.is_empty() {
                    return Self::full_null(self.name(), iter.size_hint().0);
                }
                let array = match (self.has_validity(), self.chunks.len()) {
                    (false, 1) => take_no_null_primitive_iter_unchecked::<T::Native, _>(
                        chunks.next().unwrap(),
                        iter,
                    ) as ArrayRef,
                    (_, 1) => {
                        take_primitive_iter_unchecked::<T::Native, _>(chunks.next().unwrap(), iter)
                            as ArrayRef
                    }
                    _ => {
                        let mut ca = take_primitive_iter_n_chunks(self, iter);
                        ca.rename(self.name());
                        return ca;
                    }
                };
                self.copy_with_chunks(vec![array], false)
            }
            TakeIdx::IterNulls(iter) => {
                if self.is_empty() {
                    return Self::full_null(self.name(), iter.size_hint().0);
                }
                let array = match (self.has_validity(), self.chunks.len()) {
                    (false, 1) => take_no_null_primitive_opt_iter_unchecked::<T::Native, _>(
                        chunks.next().unwrap(),
                        iter,
                    ) as ArrayRef,
                    (_, 1) => take_primitive_opt_iter_unchecked::<T::Native, _>(
                        chunks.next().unwrap(),
                        iter,
                    ) as ArrayRef,
                    _ => {
                        let mut ca = take_primitive_opt_iter_n_chunks(self, iter);
                        ca.rename(self.name());
                        return ca;
                    }
                };
                self.copy_with_chunks(vec![array], false)
            }
        }
    }

    fn take<I, INulls>(&self, indices: TakeIdx<I, INulls>) -> PolarsResult<Self>
    where
        Self: std::marker::Sized,
        I: TakeIterator,
        INulls: TakeIteratorNulls,
    {
        indices.check_bounds(self.len())?;
        // Safety:
        // just checked bounds
        Ok(unsafe { self.take_unchecked(indices) })
    }
}

impl ChunkTake for BooleanChunked {
    unsafe fn take_unchecked<I, INulls>(&self, indices: TakeIdx<I, INulls>) -> Self
    where
        Self: std::marker::Sized,
        I: TakeIterator,
        INulls: TakeIteratorNulls,
    {
        let mut chunks = self.downcast_iter();
        match indices {
            TakeIdx::Array(array) => {
                if array.null_count() == array.len() {
                    return Self::full_null(self.name(), array.len());
                }
                let array = match self.chunks.len() {
                    1 => take::take_unchecked(chunks.next().unwrap(), array),
                    _ => {
                        return if !array.has_validity() {
                            let iter = array.values().iter().map(|i| *i as usize);
                            let mut ca: BooleanChunked = take_iter_n_chunks!(self, iter);
                            ca.rename(self.name());
                            ca
                        } else {
                            let iter = array
                                .into_iter()
                                .map(|opt_idx| opt_idx.map(|idx| *idx as usize));
                            let mut ca: BooleanChunked = take_opt_iter_n_chunks!(self, iter);
                            ca.rename(self.name());
                            ca
                        }
                    }
                };
                self.copy_with_chunks(vec![array], false)
            }
            TakeIdx::Iter(iter) => {
                if self.is_empty() {
                    return Self::full_null(self.name(), iter.size_hint().0);
                }
                let array = match (self.has_validity(), self.chunks.len()) {
                    (false, 1) => {
                        take_no_null_bool_iter_unchecked(chunks.next().unwrap(), iter) as ArrayRef
                    }
                    (_, 1) => take_bool_iter_unchecked(chunks.next().unwrap(), iter) as ArrayRef,
                    _ => {
                        let mut ca: BooleanChunked = take_iter_n_chunks_unchecked!(self, iter);
                        ca.rename(self.name());
                        return ca;
                    }
                };
                self.copy_with_chunks(vec![array], false)
            }
            TakeIdx::IterNulls(iter) => {
                if self.is_empty() {
                    return Self::full_null(self.name(), iter.size_hint().0);
                }
                let array = match (self.has_validity(), self.chunks.len()) {
                    (false, 1) => {
                        take_no_null_bool_opt_iter_unchecked(chunks.next().unwrap(), iter)
                            as ArrayRef
                    }
                    (_, 1) => {
                        take_bool_opt_iter_unchecked(chunks.next().unwrap(), iter) as ArrayRef
                    }
                    _ => {
                        let mut ca: BooleanChunked = take_opt_iter_n_chunks_unchecked!(self, iter);
                        ca.rename(self.name());
                        return ca;
                    }
                };
                self.copy_with_chunks(vec![array], false)
            }
        }
    }

    fn take<I, INulls>(&self, indices: TakeIdx<I, INulls>) -> PolarsResult<Self>
    where
        Self: std::marker::Sized,
        I: TakeIterator,
        INulls: TakeIteratorNulls,
    {
        indices.check_bounds(self.len())?;
        // Safety:
        // just checked bounds
        Ok(unsafe { self.take_unchecked(indices) })
    }
}

impl ChunkTake for Utf8Chunked {
    unsafe fn take_unchecked<I, INulls>(&self, indices: TakeIdx<I, INulls>) -> Self
    where
        Self: std::marker::Sized,
        I: TakeIterator,
        INulls: TakeIteratorNulls,
    {
        let mut chunks = self.downcast_iter();
        match indices {
            TakeIdx::Array(array) => {
                if array.null_count() == array.len() {
                    return Self::full_null(self.name(), array.len());
                }
                let array = match self.chunks.len() {
                    1 => take_utf8_unchecked(chunks.next().unwrap(), array) as ArrayRef,
                    _ => {
                        return if !array.has_validity() {
                            let iter = array.values().iter().map(|i| *i as usize);
                            let mut ca: Utf8Chunked = take_iter_n_chunks_unchecked!(self, iter);
                            ca.rename(self.name());
                            ca
                        } else {
                            let iter = array
                                .into_iter()
                                .map(|opt_idx| opt_idx.map(|idx| *idx as usize));
                            let mut ca: Utf8Chunked = take_opt_iter_n_chunks_unchecked!(self, iter);
                            ca.rename(self.name());
                            ca
                        }
                    }
                };
                self.copy_with_chunks(vec![array], false)
            }
            TakeIdx::Iter(iter) => {
                let array = match (self.has_validity(), self.chunks.len()) {
                    (false, 1) => {
                        take_no_null_utf8_iter_unchecked(chunks.next().unwrap(), iter) as ArrayRef
                    }
                    (_, 1) => take_utf8_iter_unchecked(chunks.next().unwrap(), iter) as ArrayRef,
                    _ => {
                        let mut ca: Utf8Chunked = take_iter_n_chunks_unchecked!(self, iter);
                        ca.rename(self.name());
                        return ca;
                    }
                };
                self.copy_with_chunks(vec![array], false)
            }
            TakeIdx::IterNulls(iter) => {
                let array = match (self.has_validity(), self.chunks.len()) {
                    (false, 1) => {
                        take_no_null_utf8_opt_iter_unchecked(chunks.next().unwrap(), iter)
                            as ArrayRef
                    }
                    (_, 1) => {
                        take_utf8_opt_iter_unchecked(chunks.next().unwrap(), iter) as ArrayRef
                    }
                    _ => {
                        let mut ca: Utf8Chunked = take_opt_iter_n_chunks_unchecked!(self, iter);
                        ca.rename(self.name());
                        return ca;
                    }
                };
                self.copy_with_chunks(vec![array], false)
            }
        }
    }

    fn take<I, INulls>(&self, indices: TakeIdx<I, INulls>) -> PolarsResult<Self>
    where
        Self: std::marker::Sized,
        I: TakeIterator,
        INulls: TakeIteratorNulls,
    {
        indices.check_bounds(self.len())?;
        // Safety:
        // just checked bounds
        Ok(unsafe { self.take_unchecked(indices) })
    }
}

#[cfg(feature = "dtype-binary")]
impl ChunkTake for BinaryChunked {
    unsafe fn take_unchecked<I, INulls>(&self, indices: TakeIdx<I, INulls>) -> Self
    where
        Self: std::marker::Sized,
        I: TakeIterator,
        INulls: TakeIteratorNulls,
    {
        let mut chunks = self.downcast_iter();
        match indices {
            TakeIdx::Array(array) => {
                if array.null_count() == array.len() {
                    return Self::full_null(self.name(), array.len());
                }
                let array = match self.chunks.len() {
                    1 => take_binary_unchecked(chunks.next().unwrap(), array) as ArrayRef,
                    _ => {
                        return if !array.has_validity() {
                            let iter = array.values().iter().map(|i| *i as usize);
                            let mut ca: BinaryChunked = take_iter_n_chunks_unchecked!(self, iter);
                            ca.rename(self.name());
                            ca
                        } else {
                            let iter = array
                                .into_iter()
                                .map(|opt_idx| opt_idx.map(|idx| *idx as usize));
                            let mut ca: BinaryChunked =
                                take_opt_iter_n_chunks_unchecked!(self, iter);
                            ca.rename(self.name());
                            ca
                        }
                    }
                };
                self.copy_with_chunks(vec![array], false)
            }
            TakeIdx::Iter(iter) => {
                let array = match (self.has_validity(), self.chunks.len()) {
                    (false, 1) => {
                        take_no_null_binary_iter_unchecked(chunks.next().unwrap(), iter) as ArrayRef
                    }
                    (_, 1) => take_binary_iter_unchecked(chunks.next().unwrap(), iter) as ArrayRef,
                    _ => {
                        let mut ca: BinaryChunked = take_iter_n_chunks_unchecked!(self, iter);
                        ca.rename(self.name());
                        return ca;
                    }
                };
                self.copy_with_chunks(vec![array], false)
            }
            TakeIdx::IterNulls(iter) => {
                let array = match (self.has_validity(), self.chunks.len()) {
                    (false, 1) => {
                        take_no_null_binary_opt_iter_unchecked(chunks.next().unwrap(), iter)
                            as ArrayRef
                    }
                    (_, 1) => {
                        take_binary_opt_iter_unchecked(chunks.next().unwrap(), iter) as ArrayRef
                    }
                    _ => {
                        let mut ca: BinaryChunked = take_opt_iter_n_chunks_unchecked!(self, iter);
                        ca.rename(self.name());
                        return ca;
                    }
                };
                self.copy_with_chunks(vec![array], false)
            }
        }
    }

    fn take<I, INulls>(&self, indices: TakeIdx<I, INulls>) -> PolarsResult<Self>
    where
        Self: std::marker::Sized,
        I: TakeIterator,
        INulls: TakeIteratorNulls,
    {
        indices.check_bounds(self.len())?;
        // Safety:
        // just checked bounds
        Ok(unsafe { self.take_unchecked(indices) })
    }
}

impl ChunkTake for ListChunked {
    unsafe fn take_unchecked<I, INulls>(&self, indices: TakeIdx<I, INulls>) -> Self
    where
        Self: std::marker::Sized,
        I: TakeIterator,
        INulls: TakeIteratorNulls,
    {
        let ca_self = if self.is_nested() {
            Cow::Owned(self.rechunk())
        } else {
            Cow::Borrowed(self)
        };
        let mut chunks = ca_self.downcast_iter();
        match indices {
            TakeIdx::Array(array) => {
                if array.null_count() == array.len() {
                    return Self::full_null_with_dtype(
                        self.name(),
                        array.len(),
                        &self.inner_dtype(),
                    );
                }
                let array = match ca_self.chunks.len() {
                    1 => Box::new(take_list_unchecked(chunks.next().unwrap(), array)) as ArrayRef,
                    _ => {
                        return if !array.has_validity() {
                            let iter = array.values().iter().map(|i| *i as usize);
                            let mut ca: ListChunked =
                                take_iter_n_chunks_unchecked!(ca_self.as_ref(), iter);
                            ca.rename(ca_self.name());
                            ca
                        } else {
                            let iter = array
                                .into_iter()
                                .map(|opt_idx| opt_idx.map(|idx| *idx as usize));
                            let mut ca: ListChunked =
                                take_opt_iter_n_chunks_unchecked!(ca_self.as_ref(), iter);
                            ca.rename(ca_self.name());
                            ca
                        }
                    }
                };
                ca_self.copy_with_chunks(vec![array], false)
            }
            // todo! fast path for single chunk
            TakeIdx::Iter(iter) => {
                if ca_self.chunks.len() == 1 {
                    let idx: NoNull<IdxCa> = iter.map(|v| v as IdxSize).collect();
                    ca_self.take_unchecked((&idx.into_inner()).into())
                } else {
                    let mut ca: ListChunked = take_iter_n_chunks_unchecked!(ca_self.as_ref(), iter);
                    ca.rename(ca_self.name());
                    ca
                }
            }
            TakeIdx::IterNulls(iter) => {
                if ca_self.chunks.len() == 1 {
                    let idx: IdxCa = iter.map(|v| v.map(|v| v as IdxSize)).collect();
                    ca_self.take_unchecked((&idx).into())
                } else {
                    let mut ca: ListChunked =
                        take_opt_iter_n_chunks_unchecked!(ca_self.as_ref(), iter);
                    ca.rename(ca_self.name());
                    ca
                }
            }
        }
    }

    fn take<I, INulls>(&self, indices: TakeIdx<I, INulls>) -> PolarsResult<Self>
    where
        Self: std::marker::Sized,
        I: TakeIterator,
        INulls: TakeIteratorNulls,
    {
        indices.check_bounds(self.len())?;
        // Safety:
        // just checked bounds
        Ok(unsafe { self.take_unchecked(indices) })
    }
}

#[cfg(feature = "object")]
impl<T: PolarsObject> ChunkTake for ObjectChunked<T> {
    unsafe fn take_unchecked<I, INulls>(&self, indices: TakeIdx<I, INulls>) -> Self
    where
        Self: std::marker::Sized,
        I: TakeIterator,
        INulls: TakeIteratorNulls,
    {
        // current implementation is suboptimal, every iterator is allocated to UInt32Array
        match indices {
            TakeIdx::Array(array) => {
                if array.null_count() == array.len() {
                    return Self::full_null(self.name(), array.len());
                }

                match self.chunks.len() {
                    1 => {
                        let values = self.downcast_chunks().get(0).unwrap().values();

                        let mut ca: Self = array
                            .into_iter()
                            .map(|opt_idx| {
                                opt_idx.map(|idx| values.get_unchecked(*idx as usize).clone())
                            })
                            .collect();
                        ca.rename(self.name());
                        ca
                    }
                    _ => {
                        return if !array.has_validity() {
                            let iter = array.values().iter().map(|i| *i as usize);

                            let taker = self.take_rand();
                            let mut ca: ObjectChunked<T> =
                                iter.map(|idx| taker.get_unchecked(idx).cloned()).collect();
                            ca.rename(self.name());
                            ca
                        } else {
                            let iter = array
                                .into_iter()
                                .map(|opt_idx| opt_idx.map(|idx| *idx as usize));
                            let taker = self.take_rand();

                            let mut ca: ObjectChunked<T> = iter
                                .map(|opt_idx| {
                                    opt_idx.and_then(|idx| taker.get_unchecked(idx).cloned())
                                })
                                .collect();

                            ca.rename(self.name());
                            ca
                        }
                    }
                }
            }
            TakeIdx::Iter(iter) => {
                if self.is_empty() {
                    return Self::full_null(self.name(), iter.size_hint().0);
                }

                let taker = self.take_rand();
                let mut ca: ObjectChunked<T> =
                    iter.map(|idx| taker.get_unchecked(idx).cloned()).collect();
                ca.rename(self.name());
                ca
            }
            TakeIdx::IterNulls(iter) => {
                if self.is_empty() {
                    return Self::full_null(self.name(), iter.size_hint().0);
                }
                let taker = self.take_rand();

                let mut ca: ObjectChunked<T> = iter
                    .map(|opt_idx| opt_idx.and_then(|idx| taker.get(idx).cloned()))
                    .collect();

                ca.rename(self.name());
                ca
            }
        }
    }

    fn take<I, INulls>(&self, indices: TakeIdx<I, INulls>) -> PolarsResult<Self>
    where
        Self: std::marker::Sized,
        I: TakeIterator,
        INulls: TakeIteratorNulls,
    {
        indices.check_bounds(self.len())?;
        // Safety:
        // just checked bounds
        Ok(unsafe { self.take_unchecked(indices) })
    }
}

#[cfg(test)]
mod test {
    use crate::prelude::*;

    #[test]
    fn test_take_random() {
        let ca = Int32Chunked::from_slice("a", &[1, 2, 3]);
        assert_eq!(ca.get(0), Some(1));
        assert_eq!(ca.get(1), Some(2));
        assert_eq!(ca.get(2), Some(3));

        let ca = Utf8Chunked::from_slice("a", &["a", "b", "c"]);
        assert_eq!(ca.get(0), Some("a"));
        assert_eq!(ca.get(1), Some("b"));
        assert_eq!(ca.get(2), Some("c"));
    }
}
