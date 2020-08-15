//! Implementations of upstream traits for ChunkedArray<T>
use crate::prelude::*;
use crate::utils::get_iter_capacity;
use crate::utils::Xob;
use rayon::iter::{FromParallelIterator, IntoParallelIterator};
use rayon::prelude::*;
use std::collections::LinkedList;
use std::iter::FromIterator;

/// FromIterator trait

impl<T> FromIterator<Option<T::Native>> for ChunkedArray<T>
where
    T: ArrowPrimitiveType,
{
    fn from_iter<I: IntoIterator<Item = Option<T::Native>>>(iter: I) -> Self {
        let iter = iter.into_iter();
        let mut builder = PrimitiveChunkedBuilder::new("", get_iter_capacity(&iter));

        for opt_val in iter {
            builder.append_option(opt_val);
        }
        builder.finish()
    }
}

// Xob is only a wrapper needed for specialization
impl<T> FromIterator<T::Native> for Xob<ChunkedArray<T>>
where
    T: ArrowPrimitiveType,
{
    fn from_iter<I: IntoIterator<Item = T::Native>>(iter: I) -> Self {
        let iter = iter.into_iter();
        let mut builder = PrimitiveChunkedBuilder::new("", get_iter_capacity(&iter));

        for val in iter {
            builder.append_value(val);
        }
        Xob::new(builder.finish())
    }
}

impl FromIterator<bool> for BooleanChunked {
    fn from_iter<I: IntoIterator<Item = bool>>(iter: I) -> Self {
        let iter = iter.into_iter();
        let mut builder = PrimitiveChunkedBuilder::new("", get_iter_capacity(&iter));

        for val in iter {
            builder.append_value(val);
        }
        builder.finish()
    }
}

// FromIterator for Utf8Chunked variants.

impl<'a> FromIterator<&'a str> for Utf8Chunked {
    fn from_iter<I: IntoIterator<Item = &'a str>>(iter: I) -> Self {
        let iter = iter.into_iter();
        let mut builder = Utf8ChunkedBuilder::new("", get_iter_capacity(&iter));

        for val in iter {
            builder.append_value(val);
        }
        builder.finish()
    }
}

impl<'a> FromIterator<&'a &'a str> for Utf8Chunked {
    fn from_iter<I: IntoIterator<Item = &'a &'a str>>(iter: I) -> Self {
        let iter = iter.into_iter();
        let mut builder = Utf8ChunkedBuilder::new("", get_iter_capacity(&iter));

        for val in iter {
            builder.append_value(val);
        }
        builder.finish()
    }
}

impl<'a> FromIterator<Option<&'a str>> for Utf8Chunked {
    fn from_iter<I: IntoIterator<Item = Option<&'a str>>>(iter: I) -> Self {
        let iter = iter.into_iter();
        let mut builder = Utf8ChunkedBuilder::new("", get_iter_capacity(&iter));

        for opt_val in iter {
            match opt_val {
                None => builder.append_null(),
                Some(val) => builder.append_value(val),
            }
        }
        builder.finish()
    }
}

impl FromIterator<String> for Utf8Chunked {
    fn from_iter<I: IntoIterator<Item = String>>(iter: I) -> Self {
        let iter = iter.into_iter();
        let mut builder = Utf8ChunkedBuilder::new("", get_iter_capacity(&iter));

        for val in iter {
            builder.append_value(val.as_str());
        }
        builder.finish()
    }
}

impl FromIterator<Option<String>> for Utf8Chunked {
    fn from_iter<I: IntoIterator<Item = Option<String>>>(iter: I) -> Self {
        let iter = iter.into_iter();
        let mut builder = Utf8ChunkedBuilder::new("", get_iter_capacity(&iter));

        for opt_val in iter {
            match opt_val {
                None => builder.append_null(),
                Some(val) => builder.append_value(val.as_str()),
            }
        }
        builder.finish()
    }
}

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

fn collect_into_linked_list<I>(par_iter: I) -> LinkedList<Vec<I::Item>>
where
    I: IntoParallelIterator,
{
    par_iter
        .into_par_iter()
        .fold(Vec::new, vec_push)
        .map(as_list)
        .reduce(LinkedList::new, list_append)
}

fn get_capacity_from_par_results<T>(ll: &LinkedList<Vec<T>>) -> usize {
    ll.iter().map(|list| list.len()).sum()
}

impl<T> FromParallelIterator<Option<T::Native>> for ChunkedArray<T>
where
    T: ArrowPrimitiveType,
{
    fn from_par_iter<I: IntoParallelIterator<Item = Option<T::Native>>>(iter: I) -> Self {
        // Get linkedlist filled with different vec result from different threads
        let vectors = collect_into_linked_list(iter);
        let capacity: usize = get_capacity_from_par_results(&vectors);

        let mut builder = PrimitiveChunkedBuilder::new("", capacity);
        // Unpack all these results and append them single threaded
        vectors.iter().for_each(|vec| {
            for opt_val in vec {
                builder.append_option(*opt_val);
            }
        });

        builder.finish()
    }
}

impl FromParallelIterator<bool> for BooleanChunked {
    fn from_par_iter<I: IntoParallelIterator<Item = bool>>(iter: I) -> Self {
        let vectors = collect_into_linked_list(iter);
        let capacity: usize = get_capacity_from_par_results(&vectors);

        let mut builder = PrimitiveChunkedBuilder::new("", capacity);
        // Unpack all these results and append them single threaded
        vectors.iter().for_each(|vec| {
            for val in vec {
                builder.append_value(*val);
            }
        });

        builder.finish()
    }
}

impl FromParallelIterator<String> for Utf8Chunked {
    fn from_par_iter<I: IntoParallelIterator<Item = String>>(iter: I) -> Self {
        let vectors = collect_into_linked_list(iter);
        let capacity: usize = get_capacity_from_par_results(&vectors);

        let mut builder = Utf8ChunkedBuilder::new("", capacity);
        // Unpack all these results and append them single threaded
        vectors.iter().for_each(|vec| {
            for val in vec {
                builder.append_value(val.as_str());
            }
        });

        builder.finish()
    }
}

impl FromParallelIterator<Option<String>> for Utf8Chunked {
    fn from_par_iter<I: IntoParallelIterator<Item = Option<String>>>(iter: I) -> Self {
        let vectors = collect_into_linked_list(iter);
        let capacity: usize = get_capacity_from_par_results(&vectors);

        let mut builder = Utf8ChunkedBuilder::new("", capacity);
        // Unpack all these results and append them single threaded
        vectors.iter().for_each(|vec| {
            for val in vec {
                builder.append_option(val.as_ref());
            }
        });
        builder.finish()
    }
}

impl<'a> FromParallelIterator<Option<&'a str>> for Utf8Chunked {
    fn from_par_iter<I: IntoParallelIterator<Item = Option<&'a str>>>(iter: I) -> Self {
        let vectors = collect_into_linked_list(iter);
        let capacity: usize = get_capacity_from_par_results(&vectors);

        let mut builder = Utf8ChunkedBuilder::new("", capacity);
        // Unpack all these results and append them single threaded
        vectors.iter().for_each(|vec| {
            for val in vec {
                builder.append_option(val.as_ref());
            }
        });
        builder.finish()
    }
}

impl<'a> FromParallelIterator<&'a str> for Utf8Chunked {
    fn from_par_iter<I: IntoParallelIterator<Item = &'a str>>(iter: I) -> Self {
        let vectors = collect_into_linked_list(iter);
        let capacity: usize = get_capacity_from_par_results(&vectors);

        let mut builder = Utf8ChunkedBuilder::new("", capacity);
        // Unpack all these results and append them single threaded
        vectors.iter().for_each(|vec| {
            for val in vec {
                builder.append_value(val);
            }
        });
        builder.finish()
    }
}

/// From trait

// TODO: use macro
// Only the one which takes Utf8Chunked by reference is implemented.
// We cannot return a & str owned by this function.
impl<'a> From<&'a Utf8Chunked> for Vec<Option<&'a str>> {
    fn from(ca: &'a Utf8Chunked) -> Self {
        let mut vec = Vec::with_capacity_aligned(ca.len());
        ca.into_iter().for_each(|opt| vec.push(opt));
        vec
    }
}

impl From<Utf8Chunked> for Vec<Option<String>> {
    fn from(ca: Utf8Chunked) -> Self {
        let mut vec = Vec::with_capacity_aligned(ca.len());
        ca.into_iter()
            .for_each(|opt| vec.push(opt.map(|s| s.to_string())));
        vec
    }
}

impl<'a> From<&'a BooleanChunked> for Vec<Option<bool>> {
    fn from(ca: &'a BooleanChunked) -> Self {
        let mut vec = Vec::with_capacity_aligned(ca.len());
        ca.into_iter().for_each(|opt| vec.push(opt));
        vec
    }
}

impl From<BooleanChunked> for Vec<Option<bool>> {
    fn from(ca: BooleanChunked) -> Self {
        let mut vec = Vec::with_capacity_aligned(ca.len());
        ca.into_iter().for_each(|opt| vec.push(opt));
        vec
    }
}

impl<'a, T> From<&'a ChunkedArray<T>> for Vec<Option<T::Native>>
where
    T: PolarsNumericType,
    &'a ChunkedArray<T>: IntoIterator<Item = Option<T::Native>>,
    ChunkedArray<T>: ChunkOps,
{
    fn from(ca: &'a ChunkedArray<T>) -> Self {
        let mut vec = Vec::with_capacity_aligned(ca.len());
        ca.into_iter().for_each(|opt| vec.push(opt));
        vec
    }
}
// TODO: macro implementation of Vec From for all types. ChunkedArray<T> (no reference) doesn't implement
//    &'a ChunkedArray<T>: IntoIterator<Item = Option<T::Native>>,
