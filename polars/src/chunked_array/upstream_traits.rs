//! Implementations of upstream traits for ChunkedArray<T>
use crate::prelude::*;
use crate::utils::Xob;
use std::iter::FromIterator;

/// FromIterator trait

fn get_iter_capacity<T, I: Iterator<Item = T>>(iter: &I) -> usize {
    match iter.size_hint() {
        (_lower, Some(upper)) => upper,
        (0, None) => 1024,
        (lower, None) => lower,
    }
}

impl<T> FromIterator<Option<T::Native>> for ChunkedArray<T>
where
    T: ArrowPrimitiveType,
{
    fn from_iter<I: IntoIterator<Item = Option<T::Native>>>(iter: I) -> Self {
        let iter = iter.into_iter();
        let mut builder = PrimitiveChunkedBuilder::new("", get_iter_capacity(&iter));

        for opt_val in iter {
            builder.append_option(opt_val).expect("could not append");
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
            builder.append_value(val).expect("could not append");
        }
        Xob::new(builder.finish())
    }
}

impl FromIterator<bool> for BooleanChunked {
    fn from_iter<I: IntoIterator<Item = bool>>(iter: I) -> Self {
        let iter = iter.into_iter();
        let mut builder = PrimitiveChunkedBuilder::new("", get_iter_capacity(&iter));

        for val in iter {
            builder.append_value(val).expect("could not append");
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
            builder.append_value(val).expect("could not append");
        }
        builder.finish()
    }
}

impl<'a> FromIterator<&'a &'a str> for Utf8Chunked {
    fn from_iter<I: IntoIterator<Item = &'a &'a str>>(iter: I) -> Self {
        let iter = iter.into_iter();
        let mut builder = Utf8ChunkedBuilder::new("", get_iter_capacity(&iter));

        for val in iter {
            builder.append_value(val).expect("could not append");
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
                None => builder.append_null().expect("should not fail"),
                Some(val) => builder.append_value(val).expect("should not fail"),
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
            builder
                .append_value(val.as_str())
                .expect("could not append");
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
                None => builder.append_null().expect("should not fail"),
                Some(val) => builder.append_value(val.as_str()).expect("should not fail"),
            }
        }
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
