//! Implementations of upstream traits for ChunkedArray<T>
use crate::chunked_array::builder::get_large_list_builder;
use crate::prelude::*;
use crate::utils::get_iter_capacity;
use crate::utils::Xob;
use rayon::iter::{FromParallelIterator, IntoParallelIterator};
use rayon::prelude::*;
use std::collections::LinkedList;
use std::iter::FromIterator;
use std::marker::PhantomData;
use std::sync::Arc;

impl<T> Default for ChunkedArray<T> {
    fn default() -> Self {
        ChunkedArray {
            field: Arc::new(Field::new("default", ArrowDataType::Null, false)),
            chunks: Vec::with_capacity(0),
            chunk_id: Vec::with_capacity(0),
            phantom: PhantomData,
        }
    }
}

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
    // We use AlignedVec because it is way faster than Arrows builder. We can do this because we
    // know we don't have null values.
    fn from_iter<I: IntoIterator<Item = T::Native>>(iter: I) -> Self {
        // bools are bit packed
        if let ArrowDataType::Boolean = T::get_data_type() {
            let iter = iter.into_iter();
            let mut builder = PrimitiveChunkedBuilder::new("", get_iter_capacity(&iter));

            for val in iter {
                builder.append_value(val);
            }
            Xob::new(builder.finish())
        } else {
            let iter = iter.into_iter();
            let mut v = AlignedVec::with_capacity_aligned(get_iter_capacity(&iter));

            for val in iter {
                unsafe { v.push(val) }
            }
            // TODO: shrink capacity
            Xob::new(ChunkedArray::new_from_aligned_vec("", v))
        }
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

macro_rules! impl_from_iter_utf8 {
    ($iter:ident) => {{
        let iter = $iter.into_iter();
        let mut builder = Utf8ChunkedBuilder::new("", get_iter_capacity(&iter));

        for opt_val in iter {
            builder.append_option(opt_val.as_ref())
        }
        builder.finish()
    }};
}

impl<'a> FromIterator<Option<&'a str>> for Utf8Chunked {
    fn from_iter<I: IntoIterator<Item = Option<&'a str>>>(iter: I) -> Self {
        impl_from_iter_utf8!(iter)
    }
}

impl<'a> FromIterator<&'a Option<&'a str>> for Utf8Chunked {
    fn from_iter<I: IntoIterator<Item = &'a Option<&'a str>>>(iter: I) -> Self {
        impl_from_iter_utf8!(iter)
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

impl FromIterator<Series> for LargeListChunked {
    fn from_iter<I: IntoIterator<Item = Series>>(iter: I) -> Self {
        let mut it = iter.into_iter();
        let capacity = get_iter_capacity(&it);

        // first take one to get the dtype. We panic if we have an empty iterator
        let v = it.next().unwrap();
        let mut builder = get_large_list_builder(v.dtype(), capacity, "collected");

        builder.append_opt_series(&Some(v));
        while let Some(s) = it.next() {
            builder.append_opt_series(&Some(s));
        }
        builder.finish()
    }
}

impl<'a> FromIterator<&'a Series> for LargeListChunked {
    fn from_iter<I: IntoIterator<Item = &'a Series>>(iter: I) -> Self {
        let mut it = iter.into_iter();
        let capacity = get_iter_capacity(&it);

        // first take one to get the dtype. We panic if we have an empty iterator
        let v = it.next().unwrap();
        let mut builder = get_large_list_builder(v.dtype(), capacity, "collected");

        builder.append_series(v);
        while let Some(s) = it.next() {
            builder.append_series(s);
        }

        builder.finish()
    }
}

macro_rules! impl_from_iter_opt_series {
    ($iter:ident) => {{
        // we don't know the type of the series until we get Some(Series) from the iterator.
        // until that happens we count the number of None's so that we can first fill the None's until
        // we know the type

        let mut it = $iter.into_iter();

        let v;
        let mut cnt = 0;

        loop {
            let opt_v = it.next();

            match opt_v {
                Some(opt_v) => match opt_v {
                    Some(val) => {
                        v = val;
                        break;
                    }
                    None => cnt += 1,
                },
                // end of iterator
                None => {
                    // type is not known
                    panic!("Type of Series cannot be determined as they are all null")
                }
            }
        }
        let capacity = get_iter_capacity(&it);
        let mut builder = get_large_list_builder(v.dtype(), capacity, "collected");

        // first fill all None's we encountered
        while cnt > 0 {
            builder.append_opt_series(&None);
            cnt -= 1;
        }

        // now the first non None
        builder.append_series(&v);

        // now we have added all Nones, we can consume the rest of the iterator.
        while let Some(opt_s) = it.next() {
            builder.append_opt_series(&opt_s);
        }

        builder.finish()

    }}
}

impl FromIterator<Option<Series>> for LargeListChunked {
    fn from_iter<I: IntoIterator<Item = Option<Series>>>(iter: I) -> Self {
        impl_from_iter_opt_series!(iter)
    }
}

impl<'a> FromIterator<&'a Option<Series>> for LargeListChunked {
    fn from_iter<I: IntoIterator<Item = &'a Option<Series>>>(iter: I) -> Self {
        impl_from_iter_opt_series!(iter)
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

impl<T> FromParallelIterator<T::Native> for Xob<ChunkedArray<T>>
where
    T: ArrowPrimitiveType,
{
    fn from_par_iter<I: IntoParallelIterator<Item = T::Native>>(iter: I) -> Self {
        // Get linkedlist filled with different vec result from different threads
        let vectors = collect_into_linked_list(iter);
        let capacity: usize = get_capacity_from_par_results(&vectors);

        let mut builder = PrimitiveChunkedBuilder::new("", capacity);
        // Unpack all these results and append them single threaded
        vectors.iter().for_each(|vec| {
            for val in vec {
                builder.append_value(*val);
            }
        });

        Xob::new(builder.finish())
    }
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
        let mut vec = Vec::with_capacity(ca.len());
        ca.into_iter().for_each(|opt| vec.push(opt));
        vec
    }
}

impl From<Utf8Chunked> for Vec<Option<String>> {
    fn from(ca: Utf8Chunked) -> Self {
        let mut vec = Vec::with_capacity(ca.len());
        ca.into_iter()
            .for_each(|opt| vec.push(opt.map(|s| s.to_string())));
        vec
    }
}

impl<'a> From<&'a BooleanChunked> for Vec<Option<bool>> {
    fn from(ca: &'a BooleanChunked) -> Self {
        let mut vec = Vec::with_capacity(ca.len());
        ca.into_iter().for_each(|opt| vec.push(opt));
        vec
    }
}

impl From<BooleanChunked> for Vec<Option<bool>> {
    fn from(ca: BooleanChunked) -> Self {
        let mut vec = Vec::with_capacity(ca.len());
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
        let mut vec = Vec::with_capacity(ca.len());
        ca.into_iter().for_each(|opt| vec.push(opt));
        vec
    }
}
// TODO: macro implementation of Vec From for all types. ChunkedArray<T> (no reference) doesn't implement
//    &'a ChunkedArray<T>: IntoIterator<Item = Option<T::Native>>,

#[cfg(test)]
mod test {
    use crate::prelude::*;

    #[test]
    fn test_collect_into_large_listt() {
        let s1 = Series::new("", &[true, false, true]);
        let s2 = Series::new("", &[true, false, true]);

        let ll: LargeListChunked = [&s1, &s2].iter().map(|&s| s).collect();
        assert_eq!(ll.len(), 2);
        assert_eq!(ll.null_count(), 0);
        let ll: LargeListChunked = [None, Some(s2)].iter().collect();
        assert_eq!(ll.len(), 2);
        assert_eq!(ll.null_count(), 1);
    }
}
