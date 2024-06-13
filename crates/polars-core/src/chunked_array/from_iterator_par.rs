//! Implementations of upstream traits for [`ChunkedArray<T>`]
use std::collections::LinkedList;
use std::sync::Mutex;

use arrow::pushable::{NoOption, Pushable};
use rayon::prelude::*;

use super::from_iterator::PolarsAsRef;
use crate::chunked_array::builder::get_list_builder;
use crate::prelude::*;
use crate::utils::flatten::flatten_par;
use crate::utils::NoNull;

/// FromParallelIterator trait
// Code taken from https://docs.rs/rayon/1.3.1/src/rayon/iter/extend.rs.html#356-366
fn vec_push<T>(mut vec: Vec<T>, elem: T) -> Vec<T> {
    vec.push(elem);
    vec
}

fn as_list<T>(item: T) -> LinkedList<T> {
    let mut list = LinkedList::new();
    list.push_back(item);
    list
}

fn list_append<T>(mut list1: LinkedList<T>, mut list2: LinkedList<T>) -> LinkedList<T> {
    list1.append(&mut list2);
    list1
}

fn collect_into_linked_list_vec<I>(par_iter: I) -> LinkedList<Vec<I::Item>>
where
    I: IntoParallelIterator,
{
    let it = par_iter.into_par_iter();
    // be careful optimizing allocations. Its hard to figure out the size
    // needed
    // https://github.com/pola-rs/polars/issues/1562
    it.fold(Vec::new, vec_push)
        .map(as_list)
        .reduce(LinkedList::new, list_append)
}

fn collect_into_linked_list<I, P, F>(par_iter: I, identity: F) -> LinkedList<P::Freeze>
where
    I: IntoParallelIterator,
    P: Pushable<I::Item> + Send + Sync,
    F: Fn() -> P + Sync + Send,
    P::Freeze: Send,
{
    let it = par_iter.into_par_iter();
    it.fold(identity, |mut v, item| {
        v.push(item);
        v
    })
    // The freeze on this line, ensures the null count is done in parallel
    .map(|p| as_list(p.freeze()))
    .reduce(LinkedList::new, list_append)
}

fn get_capacity_from_par_results<T>(ll: &LinkedList<Vec<T>>) -> usize {
    ll.iter().map(|list| list.len()).sum()
}

impl<T> FromParallelIterator<T::Native> for NoNull<ChunkedArray<T>>
where
    T: PolarsNumericType,
{
    fn from_par_iter<I: IntoParallelIterator<Item = T::Native>>(iter: I) -> Self {
        // Get linkedlist filled with different vec result from different threads
        let vectors = collect_into_linked_list_vec(iter);
        let vectors = vectors.into_iter().collect::<Vec<_>>();
        let values = flatten_par(&vectors);
        NoNull::new(ChunkedArray::new_vec("", values))
    }
}

impl<T> FromParallelIterator<Option<T::Native>> for ChunkedArray<T>
where
    T: PolarsNumericType,
{
    fn from_par_iter<I: IntoParallelIterator<Item = Option<T::Native>>>(iter: I) -> Self {
        let chunks = collect_into_linked_list(iter, MutablePrimitiveArray::new);
        Self::from_chunk_iter("", chunks).optional_rechunk()
    }
}

impl FromParallelIterator<bool> for BooleanChunked {
    fn from_par_iter<I: IntoParallelIterator<Item = bool>>(iter: I) -> Self {
        let chunks = collect_into_linked_list(iter, MutableBooleanArray::new);
        Self::from_chunk_iter("", chunks).optional_rechunk()
    }
}

impl FromParallelIterator<Option<bool>> for BooleanChunked {
    fn from_par_iter<I: IntoParallelIterator<Item = Option<bool>>>(iter: I) -> Self {
        let chunks = collect_into_linked_list(iter, MutableBooleanArray::new);
        Self::from_chunk_iter("", chunks).optional_rechunk()
    }
}

impl<Ptr> FromParallelIterator<Ptr> for StringChunked
where
    Ptr: PolarsAsRef<str> + Send + Sync + NoOption,
{
    fn from_par_iter<I: IntoParallelIterator<Item = Ptr>>(iter: I) -> Self {
        let chunks = collect_into_linked_list(iter, MutableBinaryViewArray::new);
        Self::from_chunk_iter("", chunks).optional_rechunk()
    }
}

impl<Ptr> FromParallelIterator<Ptr> for BinaryChunked
where
    Ptr: PolarsAsRef<[u8]> + Send + Sync + NoOption,
{
    fn from_par_iter<I: IntoParallelIterator<Item = Ptr>>(iter: I) -> Self {
        let chunks = collect_into_linked_list(iter, MutableBinaryViewArray::new);
        Self::from_chunk_iter("", chunks).optional_rechunk()
    }
}

impl<Ptr> FromParallelIterator<Option<Ptr>> for StringChunked
where
    Ptr: AsRef<str> + Send + Sync,
{
    fn from_par_iter<I: IntoParallelIterator<Item = Option<Ptr>>>(iter: I) -> Self {
        let chunks = collect_into_linked_list(iter, MutableBinaryViewArray::new);
        Self::from_chunk_iter("", chunks).optional_rechunk()
    }
}

impl<Ptr> FromParallelIterator<Option<Ptr>> for BinaryChunked
where
    Ptr: AsRef<[u8]> + Send + Sync,
{
    fn from_par_iter<I: IntoParallelIterator<Item = Option<Ptr>>>(iter: I) -> Self {
        let chunks = collect_into_linked_list(iter, MutableBinaryViewArray::new);
        Self::from_chunk_iter("", chunks).optional_rechunk()
    }
}

pub trait FromParIterWithDtype<K> {
    fn from_par_iter_with_dtype<I>(iter: I, name: &str, dtype: DataType) -> Self
    where
        I: IntoParallelIterator<Item = K>,
        Self: Sized;
}

fn get_value_cap(vectors: &LinkedList<Vec<Option<Series>>>) -> usize {
    vectors
        .iter()
        .map(|list| {
            list.iter()
                .map(|opt_s| opt_s.as_ref().map(|s| s.len()).unwrap_or(0))
                .sum::<usize>()
        })
        .sum::<usize>()
}

fn get_dtype(vectors: &LinkedList<Vec<Option<Series>>>) -> DataType {
    for v in vectors {
        for s in v.iter().flatten() {
            let dtype = s.dtype();
            if !matches!(dtype, DataType::Null) {
                return dtype.clone();
            }
        }
    }
    DataType::Null
}

fn materialize_list(
    name: &str,
    vectors: &LinkedList<Vec<Option<Series>>>,
    dtype: DataType,
    value_capacity: usize,
    list_capacity: usize,
) -> ListChunked {
    match &dtype {
        #[cfg(feature = "object")]
        DataType::Object(_, _) => {
            let s = vectors
                .iter()
                .flatten()
                .find_map(|opt_s| opt_s.as_ref())
                .unwrap();
            let mut builder = s.get_list_builder(name, value_capacity, list_capacity);

            for v in vectors {
                for val in v {
                    builder.append_opt_series(val.as_ref()).unwrap();
                }
            }
            builder.finish()
        },
        dtype => {
            let mut builder = get_list_builder(dtype, value_capacity, list_capacity, name).unwrap();
            for v in vectors {
                for val in v {
                    builder.append_opt_series(val.as_ref()).unwrap();
                }
            }
            builder.finish()
        },
    }
}

impl FromParallelIterator<Option<Series>> for ListChunked {
    fn from_par_iter<I>(par_iter: I) -> Self
    where
        I: IntoParallelIterator<Item = Option<Series>>,
    {
        let vectors = collect_into_linked_list_vec(par_iter);

        let list_capacity: usize = get_capacity_from_par_results(&vectors);
        let value_capacity = get_value_cap(&vectors);
        let dtype = get_dtype(&vectors);
        if let DataType::Null = dtype {
            ListChunked::full_null_with_dtype("", list_capacity, &DataType::Null)
        } else {
            materialize_list("", &vectors, dtype, value_capacity, list_capacity)
        }
    }
}

impl FromParIterWithDtype<Option<Series>> for ListChunked {
    fn from_par_iter_with_dtype<I>(iter: I, name: &str, dtype: DataType) -> Self
    where
        I: IntoParallelIterator<Item = Option<Series>>,
        Self: Sized,
    {
        let vectors = collect_into_linked_list_vec(iter);

        let list_capacity: usize = get_capacity_from_par_results(&vectors);
        let value_capacity = get_value_cap(&vectors);
        if let DataType::List(dtype) = dtype {
            materialize_list(name, &vectors, *dtype, value_capacity, list_capacity)
        } else {
            panic!("expected list dtype")
        }
    }
}

pub trait ChunkedCollectParIterExt: ParallelIterator {
    fn collect_ca_with_dtype<B: FromParIterWithDtype<Self::Item>>(
        self,
        name: &str,
        dtype: DataType,
    ) -> B
    where
        Self: Sized,
    {
        B::from_par_iter_with_dtype(self, name, dtype)
    }
}

impl<I: ParallelIterator> ChunkedCollectParIterExt for I {}

// Adapted from rayon
impl<C, T, E> FromParIterWithDtype<Result<T, E>> for Result<C, E>
where
    C: FromParIterWithDtype<T>,
    T: Send,
    E: Send,
{
    fn from_par_iter_with_dtype<I>(par_iter: I, name: &str, dtype: DataType) -> Self
    where
        I: IntoParallelIterator<Item = Result<T, E>>,
    {
        fn ok<T, E>(saved: &Mutex<Option<E>>) -> impl Fn(Result<T, E>) -> Option<T> + '_ {
            move |item| match item {
                Ok(item) => Some(item),
                Err(error) => {
                    // We don't need a blocking `lock()`, as anybody
                    // else holding the lock will also be writing
                    // `Some(error)`, and then ours is irrelevant.
                    if let Ok(mut guard) = saved.try_lock() {
                        if guard.is_none() {
                            *guard = Some(error);
                        }
                    }
                    None
                },
            }
        }

        let saved_error = Mutex::new(None);
        let iter = par_iter.into_par_iter().map(ok(&saved_error)).while_some();

        let collection = C::from_par_iter_with_dtype(iter, name, dtype);

        match saved_error.into_inner().unwrap() {
            Some(error) => Err(error),
            None => Ok(collection),
        }
    }
}
