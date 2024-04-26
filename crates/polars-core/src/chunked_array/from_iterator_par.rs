//! Implementations of upstream traits for [`ChunkedArray<T>`]
use std::collections::LinkedList;

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
        Self::from_chunk_iter("", chunks)
    }
}

impl FromParallelIterator<bool> for BooleanChunked {
    fn from_par_iter<I: IntoParallelIterator<Item = bool>>(iter: I) -> Self {
        let chunks = collect_into_linked_list(iter, MutableBooleanArray::new);
        Self::from_chunk_iter("", chunks)
    }
}

impl FromParallelIterator<Option<bool>> for BooleanChunked {
    fn from_par_iter<I: IntoParallelIterator<Item = Option<bool>>>(iter: I) -> Self {
        let chunks = collect_into_linked_list(iter, MutableBooleanArray::new);
        Self::from_chunk_iter("", chunks)
    }
}

impl<Ptr> FromParallelIterator<Ptr> for StringChunked
where
    Ptr: PolarsAsRef<str> + Send + Sync + NoOption,
{
    fn from_par_iter<I: IntoParallelIterator<Item = Ptr>>(iter: I) -> Self {
        let chunks = collect_into_linked_list(iter, MutableBinaryViewArray::new);
        Self::from_chunk_iter("", chunks)
    }
}

impl<Ptr> FromParallelIterator<Ptr> for BinaryChunked
where
    Ptr: PolarsAsRef<[u8]> + Send + Sync + NoOption,
{
    fn from_par_iter<I: IntoParallelIterator<Item = Ptr>>(iter: I) -> Self {
        let chunks = collect_into_linked_list(iter, MutableBinaryViewArray::new);
        Self::from_chunk_iter("", chunks)
    }
}

impl<Ptr> FromParallelIterator<Option<Ptr>> for StringChunked
where
    Ptr: AsRef<str> + Send + Sync,
{
    fn from_par_iter<I: IntoParallelIterator<Item = Option<Ptr>>>(iter: I) -> Self {
        let chunks = collect_into_linked_list(iter, MutableBinaryViewArray::new);
        Self::from_chunk_iter("", chunks)
    }
}

impl<Ptr> FromParallelIterator<Option<Ptr>> for BinaryChunked
where
    Ptr: AsRef<[u8]> + Send + Sync,
{
    fn from_par_iter<I: IntoParallelIterator<Item = Option<Ptr>>>(iter: I) -> Self {
        let chunks = collect_into_linked_list(iter, MutableBinaryViewArray::new);
        Self::from_chunk_iter("", chunks)
    }
}

/// From trait
impl FromParallelIterator<Option<Series>> for ListChunked {
    fn from_par_iter<I>(iter: I) -> Self
    where
        I: IntoParallelIterator<Item = Option<Series>>,
    {
        let mut dtype = None;
        let vectors = collect_into_linked_list_vec(iter);

        let list_capacity: usize = get_capacity_from_par_results(&vectors);
        let value_capacity = vectors
            .iter()
            .map(|list| {
                list.iter()
                    .map(|opt_s| {
                        opt_s
                            .as_ref()
                            .map(|s| {
                                if dtype.is_none() && !matches!(s.dtype(), DataType::Null) {
                                    dtype = Some(s.dtype().clone())
                                }
                                s.len()
                            })
                            .unwrap_or(0)
                    })
                    .sum::<usize>()
            })
            .sum::<usize>();

        match &dtype {
            #[cfg(feature = "object")]
            Some(DataType::Object(_, _)) => {
                let s = vectors
                    .iter()
                    .flatten()
                    .find_map(|opt_s| opt_s.as_ref())
                    .unwrap();
                let mut builder = s.get_list_builder("collected", value_capacity, list_capacity);

                for v in vectors {
                    for val in v {
                        builder.append_opt_series(val.as_ref()).unwrap();
                    }
                }
                builder.finish()
            },
            Some(dtype) => {
                let mut builder =
                    get_list_builder(dtype, value_capacity, list_capacity, "collected").unwrap();
                for v in &vectors {
                    for val in v {
                        builder.append_opt_series(val.as_ref()).unwrap();
                    }
                }
                builder.finish()
            },
            None => ListChunked::full_null_with_dtype("collected", list_capacity, &DataType::Null),
        }
    }
}

#[cfg(test)]
mod test {
    use crate::prelude::*;

    #[test]
    fn test_collect_into_list() {
        let s1 = Series::new("", &[true, false, true]);
        let s2 = Series::new("", &[true, false, true]);

        let ll: ListChunked = [&s1, &s2].iter().copied().collect();
        assert_eq!(ll.len(), 2);
        assert_eq!(ll.null_count(), 0);
        let ll: ListChunked = [None, Some(s2)].into_iter().collect();
        assert_eq!(ll.len(), 2);
        assert_eq!(ll.null_count(), 1);
    }
}
