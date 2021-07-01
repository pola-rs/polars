//! Traits to provide fast Random access to ChunkedArrays data.
//! This prevents downcasting every iteration.
//! IntoTakeRandom provides structs that implement the TakeRandom trait.
//! There are several structs that implement the fastest path for random access.
//!
use crate::chunked_array::kernels::take::{
    take_bool_iter, take_bool_iter_unchecked, take_bool_opt_iter_unchecked, take_no_null_bool_iter,
    take_no_null_bool_iter_unchecked, take_no_null_bool_opt_iter_unchecked,
    take_no_null_primitive_iter, take_no_null_primitive_iter_unchecked,
    take_no_null_primitive_opt_iter_unchecked, take_no_null_primitive_unchecked,
    take_no_null_utf8_iter, take_no_null_utf8_iter_unchecked, take_no_null_utf8_opt_iter_unchecked,
    take_primitive_iter, take_primitive_iter_n_chunks, take_primitive_iter_unchecked,
    take_primitive_opt_iter_n_chunks, take_primitive_opt_iter_unchecked, take_primitive_unchecked,
    take_utf8, take_utf8_iter, take_utf8_iter_unchecked, take_utf8_opt_iter_unchecked,
};
use crate::prelude::*;
use crate::utils::NoNull;
use arrow::array::{Array, ArrayRef};
use arrow::compute::kernels::take::take;
use std::ops::Deref;

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
            .map(|opt_idx| opt_idx.map(|idx| taker.get_unchecked(idx)))
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
        I: Iterator<Item = usize>,
        INulls: Iterator<Item = Option<usize>>,
    {
        let mut chunks = self.downcast_iter();
        match indices {
            TakeIdx::Array(array) => {
                if self.is_empty() || array.null_count() == array.len() {
                    return Self::full_null(self.name(), array.len());
                }
                let array = match (self.null_count(), self.chunks.len()) {
                    (0, 1) => {
                        take_no_null_primitive_unchecked(chunks.next().unwrap(), array) as ArrayRef
                    }
                    (_, 1) => take_primitive_unchecked(chunks.next().unwrap(), array) as ArrayRef,
                    _ => {
                        return if array.null_count() == 0 {
                            let iter = array.values().iter().map(|i| *i as usize);
                            let mut ca = take_primitive_iter_n_chunks(self, iter);
                            ca.rename(self.name());
                            ca
                        } else {
                            let iter = array
                                .into_iter()
                                .map(|opt_idx| opt_idx.map(|idx| idx as usize));
                            let mut ca = take_primitive_opt_iter_n_chunks(self, iter);
                            ca.rename(self.name());
                            ca
                        }
                    }
                };
                self.copy_with_chunks(vec![array])
            }
            TakeIdx::Iter(iter) => {
                if self.is_empty() {
                    return Self::full_null(self.name(), iter.size_hint().0);
                }
                let array = match (self.null_count(), self.chunks.len()) {
                    (0, 1) => take_no_null_primitive_iter_unchecked(chunks.next().unwrap(), iter)
                        as ArrayRef,
                    (_, 1) => {
                        take_primitive_iter_unchecked(chunks.next().unwrap(), iter) as ArrayRef
                    }
                    _ => {
                        let mut ca = take_primitive_iter_n_chunks(self, iter);
                        ca.rename(self.name());
                        return ca;
                    }
                };
                self.copy_with_chunks(vec![array])
            }
            TakeIdx::IterNulls(iter) => {
                if self.is_empty() {
                    return Self::full_null(self.name(), iter.size_hint().0);
                }
                let array = match (self.null_count(), self.chunks.len()) {
                    (0, 1) => {
                        take_no_null_primitive_opt_iter_unchecked(chunks.next().unwrap(), iter)
                            as ArrayRef
                    }
                    (_, 1) => {
                        take_primitive_opt_iter_unchecked(chunks.next().unwrap(), iter) as ArrayRef
                    }
                    _ => {
                        let mut ca = take_primitive_opt_iter_n_chunks(self, iter);
                        ca.rename(self.name());
                        return ca;
                    }
                };
                self.copy_with_chunks(vec![array])
            }
        }
    }

    fn take<I, INulls>(&self, indices: TakeIdx<I, INulls>) -> Self
    where
        Self: std::marker::Sized,
        I: Iterator<Item = usize>,
        INulls: Iterator<Item = Option<usize>>,
    {
        let mut chunks = self.downcast_iter();
        match indices {
            TakeIdx::Array(array) => {
                if self.is_empty() || array.null_count() == array.len() {
                    return Self::full_null(self.name(), array.len());
                }
                let array = match self.chunks.len() {
                    1 => take(chunks.next().unwrap(), array, None).unwrap(),
                    _ => {
                        let iter = array
                            .into_iter()
                            .filter_map(|opt_idx| opt_idx.map(|idx| idx as usize));

                        let mut ca = take_primitive_iter_n_chunks(self, iter);
                        ca.rename(self.name());
                        return ca;
                    }
                };
                self.copy_with_chunks(vec![array])
            }
            TakeIdx::Iter(iter) => {
                if self.is_empty() {
                    return Self::full_null(self.name(), iter.size_hint().0);
                }
                let array = match (self.null_count(), self.chunks.len()) {
                    (0, 1) => take_no_null_primitive_iter(chunks.next().unwrap(), iter) as ArrayRef,
                    (_, 1) => take_primitive_iter(chunks.next().unwrap(), iter) as ArrayRef,
                    _ => {
                        let mut ca = take_primitive_iter_n_chunks(self, iter);
                        ca.rename(self.name());
                        return ca;
                    }
                };
                self.copy_with_chunks(vec![array])
            }
            TakeIdx::IterNulls(_) => {
                panic!("not supported in take, only supported in take_unchecked for the join operation")
            }
        }
    }
}

impl ChunkTake for BooleanChunked {
    unsafe fn take_unchecked<I, INulls>(&self, indices: TakeIdx<I, INulls>) -> Self
    where
        Self: std::marker::Sized,
        I: Iterator<Item = usize>,
        INulls: Iterator<Item = Option<usize>>,
    {
        let mut chunks = self.downcast_iter();
        match indices {
            TakeIdx::Array(array) => {
                if self.is_empty() || array.null_count() == array.len() {
                    return Self::full_null(self.name(), array.len());
                }
                let array = match self.chunks.len() {
                    1 => take(chunks.next().unwrap(), array, None).unwrap(),
                    _ => {
                        return if array.null_count() == 0 {
                            let iter = array.values().iter().map(|i| *i as usize);
                            let mut ca: BooleanChunked = take_iter_n_chunks!(self, iter);
                            ca.rename(self.name());
                            ca
                        } else {
                            let iter = array
                                .into_iter()
                                .map(|opt_idx| opt_idx.map(|idx| idx as usize));
                            let mut ca: BooleanChunked = take_opt_iter_n_chunks!(self, iter);
                            ca.rename(self.name());
                            ca
                        }
                    }
                };
                self.copy_with_chunks(vec![array])
            }
            TakeIdx::Iter(iter) => {
                if self.is_empty() {
                    return Self::full_null(self.name(), iter.size_hint().0);
                }
                let array = match (self.null_count(), self.chunks.len()) {
                    (0, 1) => {
                        take_no_null_bool_iter_unchecked(chunks.next().unwrap(), iter) as ArrayRef
                    }
                    (_, 1) => take_bool_iter_unchecked(chunks.next().unwrap(), iter) as ArrayRef,
                    _ => {
                        let mut ca: BooleanChunked = take_iter_n_chunks_unchecked!(self, iter);
                        ca.rename(self.name());
                        return ca;
                    }
                };
                self.copy_with_chunks(vec![array])
            }
            TakeIdx::IterNulls(iter) => {
                if self.is_empty() {
                    return Self::full_null(self.name(), iter.size_hint().0);
                }
                let array = match (self.null_count(), self.chunks.len()) {
                    (0, 1) => take_no_null_bool_opt_iter_unchecked(chunks.next().unwrap(), iter)
                        as ArrayRef,
                    (_, 1) => {
                        take_bool_opt_iter_unchecked(chunks.next().unwrap(), iter) as ArrayRef
                    }
                    _ => {
                        let mut ca: BooleanChunked = take_opt_iter_n_chunks_unchecked!(self, iter);
                        ca.rename(self.name());
                        return ca;
                    }
                };
                self.copy_with_chunks(vec![array])
            }
        }
    }

    fn take<I, INulls>(&self, indices: TakeIdx<I, INulls>) -> Self
    where
        Self: std::marker::Sized,
        I: Iterator<Item = usize>,
        INulls: Iterator<Item = Option<usize>>,
    {
        let mut chunks = self.downcast_iter();
        match indices {
            TakeIdx::Array(array) => {
                if self.is_empty() || array.null_count() == array.len() {
                    return Self::full_null(self.name(), array.len());
                }
                let array = match self.chunks.len() {
                    1 => take(chunks.next().unwrap(), array, None).unwrap(),
                    _ => {
                        return if array.null_count() == 0 {
                            let iter = array.values().iter().map(|i| *i as usize);
                            let mut ca: BooleanChunked = take_iter_n_chunks!(self, iter);
                            ca.rename(self.name());
                            ca
                        } else {
                            let iter = array
                                .into_iter()
                                .map(|opt_idx| opt_idx.map(|idx| idx as usize));
                            let mut ca: BooleanChunked = take_opt_iter_n_chunks!(self, iter);
                            ca.rename(self.name());
                            ca
                        }
                    }
                };
                self.copy_with_chunks(vec![array])
            }
            TakeIdx::Iter(iter) => {
                if self.is_empty() {
                    return Self::full_null(self.name(), iter.size_hint().0);
                }
                let array = match (self.null_count(), self.chunks.len()) {
                    (0, 1) => take_no_null_bool_iter(chunks.next().unwrap(), iter) as ArrayRef,
                    (_, 1) => take_bool_iter(chunks.next().unwrap(), iter) as ArrayRef,
                    _ => {
                        let mut ca: BooleanChunked = take_iter_n_chunks!(self, iter);
                        ca.rename(self.name());
                        return ca;
                    }
                };
                self.copy_with_chunks(vec![array])
            }
            TakeIdx::IterNulls(_) => {
                panic!("not supported in take, only supported in take_unchecked for the join operation")
            }
        }
    }
}

impl ChunkTake for Utf8Chunked {
    unsafe fn take_unchecked<I, INulls>(&self, indices: TakeIdx<I, INulls>) -> Self
    where
        Self: std::marker::Sized,
        I: Iterator<Item = usize>,
        INulls: Iterator<Item = Option<usize>>,
    {
        let mut chunks = self.downcast_iter();
        match indices {
            TakeIdx::Array(array) => {
                if self.is_empty() || array.null_count() == array.len() {
                    return Self::full_null(self.name(), array.len());
                }
                let array = match self.chunks.len() {
                    1 => take_utf8(chunks.next().unwrap(), array) as ArrayRef,
                    _ => {
                        return if array.null_count() == 0 {
                            let iter = array.values().iter().map(|i| *i as usize);
                            let mut ca: Utf8Chunked = take_iter_n_chunks_unchecked!(self, iter);
                            ca.rename(self.name());
                            ca
                        } else {
                            let iter = array
                                .into_iter()
                                .map(|opt_idx| opt_idx.map(|idx| idx as usize));
                            let mut ca: Utf8Chunked = take_opt_iter_n_chunks_unchecked!(self, iter);
                            ca.rename(self.name());
                            ca
                        }
                    }
                };
                self.copy_with_chunks(vec![array])
            }
            TakeIdx::Iter(iter) => {
                if self.is_empty() {
                    return Self::full_null(self.name(), iter.size_hint().0);
                }
                let array = match (self.null_count(), self.chunks.len()) {
                    (0, 1) => {
                        take_no_null_utf8_iter_unchecked(chunks.next().unwrap(), iter) as ArrayRef
                    }
                    (_, 1) => take_utf8_iter_unchecked(chunks.next().unwrap(), iter) as ArrayRef,
                    _ => {
                        let mut ca: Utf8Chunked = take_iter_n_chunks_unchecked!(self, iter);
                        ca.rename(self.name());
                        return ca;
                    }
                };
                self.copy_with_chunks(vec![array])
            }
            TakeIdx::IterNulls(iter) => {
                if self.is_empty() {
                    return Self::full_null(self.name(), iter.size_hint().0);
                }
                let array = match (self.null_count(), self.chunks.len()) {
                    (0, 1) => take_no_null_utf8_opt_iter_unchecked(chunks.next().unwrap(), iter)
                        as ArrayRef,
                    (_, 1) => {
                        take_utf8_opt_iter_unchecked(chunks.next().unwrap(), iter) as ArrayRef
                    }
                    _ => {
                        let mut ca: Utf8Chunked = take_opt_iter_n_chunks_unchecked!(self, iter);
                        ca.rename(self.name());
                        return ca;
                    }
                };
                self.copy_with_chunks(vec![array])
            }
        }
    }

    fn take<I, INulls>(&self, indices: TakeIdx<I, INulls>) -> Self
    where
        Self: std::marker::Sized,
        I: Iterator<Item = usize>,
        INulls: Iterator<Item = Option<usize>>,
    {
        let mut chunks = self.downcast_iter();
        match indices {
            TakeIdx::Array(array) => {
                if self.is_empty() {
                    return Self::full_null(self.name(), array.len());
                }
                let array = match self.chunks.len() {
                    1 => take(chunks.next().unwrap(), array, None).unwrap() as ArrayRef,
                    _ => {
                        return if array.null_count() == 0 {
                            let iter = array.values().iter().map(|i| *i as usize);
                            let mut ca: Utf8Chunked = take_iter_n_chunks!(self, iter);
                            ca.rename(self.name());
                            ca
                        } else {
                            let iter = array
                                .into_iter()
                                .map(|opt_idx| opt_idx.map(|idx| idx as usize));
                            let mut ca: Utf8Chunked = take_opt_iter_n_chunks!(self, iter);
                            ca.rename(self.name());
                            ca
                        }
                    }
                };
                self.copy_with_chunks(vec![array])
            }
            TakeIdx::Iter(iter) => {
                if self.is_empty() {
                    return Self::full_null(self.name(), iter.size_hint().0);
                }
                let array = match (self.null_count(), self.chunks.len()) {
                    (0, 1) => take_no_null_utf8_iter(chunks.next().unwrap(), iter) as ArrayRef,
                    (_, 1) => take_utf8_iter(chunks.next().unwrap(), iter) as ArrayRef,
                    _ => {
                        let mut ca: Utf8Chunked = take_iter_n_chunks!(self, iter);
                        ca.rename(self.name());
                        return ca;
                    }
                };
                self.copy_with_chunks(vec![array])
            }
            TakeIdx::IterNulls(_) => {
                panic!("not supported in take, only supported in take_unchecked for the join operation")
            }
        }
    }
}

impl ChunkTake for ListChunked {
    unsafe fn take_unchecked<I, INulls>(&self, indices: TakeIdx<I, INulls>) -> Self
    where
        Self: std::marker::Sized,
        I: Iterator<Item = usize>,
        INulls: Iterator<Item = Option<usize>>,
    {
        self.take(indices)
    }

    fn take<I, INulls>(&self, indices: TakeIdx<I, INulls>) -> Self
    where
        Self: std::marker::Sized,
        I: Iterator<Item = usize>,
        INulls: Iterator<Item = Option<usize>>,
    {
        let mut chunks = self.downcast_iter();
        match indices {
            TakeIdx::Array(array) => {
                if self.is_empty() {
                    return Self::full_null(self.name(), array.len());
                }
                let array = match self.chunks.len() {
                    1 => take(chunks.next().unwrap(), array, None).unwrap() as ArrayRef,
                    _ => {
                        return if array.null_count() == 0 {
                            let iter = array.values().iter().map(|i| *i as usize);
                            let mut ca: ListChunked = take_iter_n_chunks!(self, iter);
                            ca.rename(self.name());
                            ca
                        } else {
                            let iter = array
                                .into_iter()
                                .map(|opt_idx| opt_idx.map(|idx| idx as usize));
                            let mut ca: ListChunked = take_opt_iter_n_chunks!(self, iter);
                            ca.rename(self.name());
                            ca
                        }
                    }
                };
                self.copy_with_chunks(vec![array])
            }
            TakeIdx::Iter(iter) => {
                if self.is_empty() {
                    return Self::full_null(self.name(), iter.size_hint().0);
                }
                let mut ca: ListChunked = take_iter_n_chunks!(self, iter);
                ca.rename(self.name());
                ca
            }
            TakeIdx::IterNulls(iter) => {
                if self.is_empty() {
                    return Self::full_null(self.name(), iter.size_hint().0);
                }

                let mut ca: ListChunked = take_opt_iter_n_chunks!(self, iter);
                ca.rename(self.name());
                ca
            }
        }
    }
}

impl ChunkTake for CategoricalChunked {
    unsafe fn take_unchecked<I, INulls>(&self, indices: TakeIdx<I, INulls>) -> Self
    where
        Self: std::marker::Sized,
        I: Iterator<Item = usize>,
        INulls: Iterator<Item = Option<usize>>,
    {
        let ca: CategoricalChunked = self.deref().take_unchecked(indices).into();
        ca.set_state(self)
    }

    fn take<I, INulls>(&self, indices: TakeIdx<I, INulls>) -> Self
    where
        Self: std::marker::Sized,
        I: Iterator<Item = usize>,
        INulls: Iterator<Item = Option<usize>>,
    {
        let ca: CategoricalChunked = self.deref().take(indices).into();
        ca.set_state(self)
    }
}

#[cfg(feature = "object")]
impl<T: PolarsObject> ChunkTake for ObjectChunked<T> {
    // TODO! implement unsafe unchecked
    unsafe fn take_unchecked<I, INulls>(&self, indices: TakeIdx<I, INulls>) -> Self
    where
        Self: std::marker::Sized,
        I: Iterator<Item = usize>,
        INulls: Iterator<Item = Option<usize>>,
    {
        self.take(indices)
    }

    fn take<I, INulls>(&self, indices: TakeIdx<I, INulls>) -> Self
    where
        Self: std::marker::Sized,
        I: Iterator<Item = usize>,
        INulls: Iterator<Item = Option<usize>>,
    {
        // current implementation is suboptimal, every iterator is allocated to UInt32Array
        match indices {
            TakeIdx::Array(array) => {
                if self.is_empty() {
                    return Self::full_null(self.name(), array.len());
                }
                match self.chunks.len() {
                    1 => {
                        let values = self.downcast_chunks().get(0).unwrap().values();

                        let mut ca: Self = array
                            .into_iter()
                            .map(|opt_idx| opt_idx.map(|idx| values[idx as usize].clone()))
                            .collect();
                        ca.rename(self.name());
                        ca
                    }
                    _ => {
                        return if array.null_count() == 0 {
                            let iter = array.values().iter().map(|i| *i as usize);

                            let taker = self.take_rand();
                            let mut ca: ObjectChunked<T> =
                                iter.map(|idx| taker.get(idx).cloned()).collect();
                            ca.rename(self.name());
                            ca
                        } else {
                            let iter = array
                                .into_iter()
                                .map(|opt_idx| opt_idx.map(|idx| idx as usize));
                            let taker = self.take_rand();

                            let mut ca: ObjectChunked<T> = iter
                                .map(|opt_idx| opt_idx.and_then(|idx| taker.get(idx).cloned()))
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
                let mut ca: ObjectChunked<T> = iter.map(|idx| taker.get(idx).cloned()).collect();
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
}

pub trait AsTakeIndex {
    fn as_take_iter<'a>(&'a self) -> Box<dyn Iterator<Item = usize> + 'a>;

    fn as_opt_take_iter<'a>(&'a self) -> Box<dyn Iterator<Item = Option<usize>> + 'a> {
        unimplemented!()
    }

    fn take_index_len(&self) -> usize;
}

impl<T> ChunkTakeEvery<T> for ChunkedArray<T>
where
    T: PolarsNumericType,
{
    fn take_every(&self, n: usize) -> ChunkedArray<T> {
        if self.null_count() == 0 {
            let a: NoNull<_> = self.into_no_null_iter().step_by(n).collect();
            a.into_inner()
        } else {
            self.into_iter().step_by(n).collect()
        }
    }
}

impl ChunkTakeEvery<BooleanType> for BooleanChunked {
    fn take_every(&self, n: usize) -> BooleanChunked {
        if self.null_count() == 0 {
            self.into_no_null_iter().step_by(n).collect()
        } else {
            self.into_iter().step_by(n).collect()
        }
    }
}

impl ChunkTakeEvery<Utf8Type> for Utf8Chunked {
    fn take_every(&self, n: usize) -> Utf8Chunked {
        if self.null_count() == 0 {
            self.into_no_null_iter().step_by(n).collect()
        } else {
            self.into_iter().step_by(n).collect()
        }
    }
}

impl ChunkTakeEvery<ListType> for ListChunked {
    fn take_every(&self, n: usize) -> ListChunked {
        if self.null_count() == 0 {
            self.into_no_null_iter().step_by(n).collect()
        } else {
            self.into_iter().step_by(n).collect()
        }
    }
}

impl ChunkTakeEvery<CategoricalType> for CategoricalChunked {
    fn take_every(&self, n: usize) -> CategoricalChunked {
        let mut ca = if self.null_count() == 0 {
            let ca: NoNull<UInt32Chunked> = self.into_no_null_iter().step_by(n).collect();
            ca.into_inner()
        } else {
            self.into_iter().step_by(n).collect()
        };
        ca.categorical_map = self.categorical_map.clone();
        ca.cast().unwrap()
    }
}
#[cfg(feature = "object")]
impl<T> ChunkTakeEvery<ObjectType<T>> for ObjectChunked<T> {
    fn take_every(&self, _n: usize) -> ObjectChunked<T> {
        todo!()
    }
}

#[cfg(test)]
mod test {
    use crate::prelude::*;

    #[test]
    fn test_take_random() {
        let ca = Int32Chunked::new_from_slice("a", &[1, 2, 3]);
        assert_eq!(ca.get(0), Some(1));
        assert_eq!(ca.get(1), Some(2));
        assert_eq!(ca.get(2), Some(3));

        let ca = Utf8Chunked::new_from_slice("a", &["a", "b", "c"]);
        assert_eq!(ca.get(0), Some("a"));
        assert_eq!(ca.get(1), Some("b"));
        assert_eq!(ca.get(2), Some("c"));
    }
}
