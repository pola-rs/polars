mod boolean;
#[cfg(feature = "dtype-categorical")]
pub mod categorical;
#[cfg(feature = "dtype-array")]
pub mod fixed_size_list;
pub mod list;
mod null;
mod primitive;
mod string;

use std::sync::Arc;

use arrow::bitmap::Bitmap;
pub use boolean::*;
#[cfg(feature = "dtype-categorical")]
pub use categorical::*;
#[cfg(feature = "dtype-array")]
pub(crate) use fixed_size_list::*;
pub use list::*;
pub use null::*;
use polars_array::PlBinaryViewArrayBuilder;
pub use primitive::*;
pub use string::*;

use crate::chunked_array::to_primitive;
use crate::prelude::*;
use crate::utils::{NoNull, get_iter_capacity};

// N: the value type; T: the sentinel type
pub trait ChunkedBuilder<N, T: PolarsDataType> {
    fn append_value(&mut self, val: N);
    fn append_null(&mut self);

    #[inline]
    fn append_option(&mut self, opt_val: Option<N>) {
        match opt_val {
            Some(v) => self.append_value(v),
            None => self.append_null(),
        }
    }

    fn finish(self) -> ChunkedArray<T>;
}

// Used in polars/src/chunked_array/apply.rs:24 to collect from aligned vecs and null bitmaps
impl<T> FromIterator<(Vec<T::Native>, Option<Bitmap>)> for ChunkedArray<T>
where
    T: PolarsNumericType,
{
    fn from_iter<I: IntoIterator<Item = (Vec<T::Native>, Option<Bitmap>)>>(iter: I) -> Self {
        let chunks = iter
            .into_iter()
            .map(|(values, opt_buffer)| to_primitive::<T>(values, opt_buffer));
        ChunkedArray::from_chunk_iter(PlSmallStr::EMPTY, chunks)
    }
}

pub trait NewChunkedArray<T, N> {
    fn from_slice(name: PlSmallStr, v: &[N]) -> Self;
    fn from_slice_options(name: PlSmallStr, opt_v: &[Option<N>]) -> Self;

    /// Create a new ChunkedArray from an iterator.
    fn from_iter_options(name: PlSmallStr, it: impl Iterator<Item = Option<N>>) -> Self;

    /// Create a new ChunkedArray from an iterator.
    fn from_iter_values(name: PlSmallStr, it: impl Iterator<Item = N>) -> Self;
}

impl<T> NewChunkedArray<T, T::Native> for ChunkedArray<T>
where
    T: PolarsNumericType,
{
    fn from_slice(name: PlSmallStr, v: &[T::Native]) -> Self {
        ChunkedArray::with_chunk(name, PlPrimitiveArray::from_slice(v))
    }

    fn from_slice_options(name: PlSmallStr, opt_v: &[Option<T::Native>]) -> Self {
        Self::from_iter_options(name, opt_v.iter().copied())
    }

    fn from_iter_options(
        name: PlSmallStr,
        it: impl Iterator<Item = Option<T::Native>>,
    ) -> ChunkedArray<T> {
        let mut builder = PrimitiveChunkedBuilder::new(name, get_iter_capacity(&it));
        it.for_each(|opt| builder.append_option(opt));
        builder.finish()
    }

    /// Create a new ChunkedArray from an iterator.
    fn from_iter_values(name: PlSmallStr, it: impl Iterator<Item = T::Native>) -> ChunkedArray<T> {
        let ca: NoNull<ChunkedArray<_>> = it.collect();
        let mut ca = ca.into_inner();
        ca.rename(name);
        ca
    }
}

impl NewChunkedArray<BooleanType, bool> for BooleanChunked {
    fn from_slice(name: PlSmallStr, v: &[bool]) -> Self {
        Self::from_iter_values(name, v.iter().copied())
    }

    fn from_slice_options(name: PlSmallStr, opt_v: &[Option<bool>]) -> Self {
        Self::from_iter_options(name, opt_v.iter().copied())
    }

    fn from_iter_options(
        name: PlSmallStr,
        it: impl Iterator<Item = Option<bool>>,
    ) -> ChunkedArray<BooleanType> {
        let mut builder = BooleanChunkedBuilder::new(name, get_iter_capacity(&it));
        it.for_each(|opt| builder.append_option(opt));
        builder.finish()
    }

    /// Create a new ChunkedArray from an iterator.
    fn from_iter_values(
        name: PlSmallStr,
        it: impl Iterator<Item = bool>,
    ) -> ChunkedArray<BooleanType> {
        let mut ca: ChunkedArray<_> = it.collect();
        ca.rename(name);
        ca
    }
}

impl<S> NewChunkedArray<StringType, S> for StringChunked
where
    S: AsRef<str>,
{
    fn from_slice(name: PlSmallStr, v: &[S]) -> Self {
        Self::from_iter_values(name, v.iter())
    }

    fn from_slice_options(name: PlSmallStr, opt_v: &[Option<S>]) -> Self {
        let arr: PlUtf8ViewArray = opt_v
            .iter()
            .map(|v| v.as_ref().map(S::as_ref))
            .collect_arr();
        ChunkedArray::with_chunk(name, arr)
    }

    fn from_iter_options(name: PlSmallStr, it: impl Iterator<Item = Option<S>>) -> Self {
        // The values are owned, so each is appended while it is still alive rather than collected
        // as a borrow of it.
        let mut builder = PlUtf8ViewArrayBuilder::with_capacity(get_iter_capacity(&it));
        it.for_each(|v| builder.push(v.as_ref().map(S::as_ref)));
        ChunkedArray::with_chunk(name, builder.freeze())
    }

    /// Create a new ChunkedArray from an iterator.
    fn from_iter_values(name: PlSmallStr, it: impl Iterator<Item = S>) -> Self {
        let mut builder = PlUtf8ViewArrayBuilder::with_capacity(get_iter_capacity(&it));
        it.for_each(|v| builder.push_value(v.as_ref()));
        ChunkedArray::with_chunk(name, builder.freeze())
    }
}

impl<B> NewChunkedArray<BinaryType, B> for BinaryChunked
where
    B: AsRef<[u8]>,
{
    fn from_slice(name: PlSmallStr, v: &[B]) -> Self {
        Self::from_iter_values(name, v.iter())
    }

    fn from_slice_options(name: PlSmallStr, opt_v: &[Option<B>]) -> Self {
        let arr: PlBinaryViewArray = opt_v
            .iter()
            .map(|v| v.as_ref().map(B::as_ref))
            .collect_arr();
        ChunkedArray::with_chunk(name, arr)
    }

    fn from_iter_options(name: PlSmallStr, it: impl Iterator<Item = Option<B>>) -> Self {
        // The values are owned, so each is appended while it is still alive rather than collected
        // as a borrow of it.
        let mut builder = PlBinaryViewArrayBuilder::with_capacity(get_iter_capacity(&it));
        it.for_each(|v| builder.push(v.as_ref().map(B::as_ref)));
        ChunkedArray::with_chunk(name, builder.freeze())
    }

    /// Create a new ChunkedArray from an iterator.
    fn from_iter_values(name: PlSmallStr, it: impl Iterator<Item = B>) -> Self {
        let mut builder = PlBinaryViewArrayBuilder::with_capacity(get_iter_capacity(&it));
        it.for_each(|v| builder.push_value(v.as_ref()));
        ChunkedArray::with_chunk(name, builder.freeze())
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_primitive_builder() {
        let mut builder =
            PrimitiveChunkedBuilder::<UInt32Type>::new(PlSmallStr::from_static("foo"), 6);
        let values = &[Some(1), None, Some(2), Some(3), None, Some(4)];
        for val in values {
            builder.append_option(*val);
        }
        let ca = builder.finish();
        assert_eq!(Vec::from(&ca), values);
    }

    #[test]
    fn test_list_builder() {
        let mut builder = ListPrimitiveChunkedBuilder::<Int32Type>::new(
            PlSmallStr::from_static("a"),
            10,
            5,
            DataType::Int32,
        );

        // Create a series containing two chunks.
        let mut s1 =
            Int32Chunked::from_slice(PlSmallStr::from_static("a"), &[1, 2, 3]).into_series();
        let s2 = Int32Chunked::from_slice(PlSmallStr::from_static("b"), &[4, 5, 6]).into_series();
        s1.append(&s2).unwrap();

        builder.append_series(&s1).unwrap();
        builder.append_series(&s2).unwrap();
        let ls = builder.finish();
        if let AnyValue::List(s) = ls.get_any_value(0).unwrap() {
            // many chunks are aggregated to one in the ListArray
            assert_eq!(s.len(), 6)
        } else {
            panic!()
        }
        if let AnyValue::List(s) = ls.get_any_value(1).unwrap() {
            assert_eq!(s.len(), 3)
        } else {
            panic!()
        }

        // Test list collect.
        let out = [&s1, &s2].iter().copied().collect::<ListChunked>();
        assert_eq!(out.get_as_series(0).unwrap().len(), 6);
        assert_eq!(out.get_as_series(1).unwrap().len(), 3);

        let mut builder = ListPrimitiveChunkedBuilder::<Int32Type>::new(
            PlSmallStr::from_static("a"),
            10,
            5,
            DataType::Int32,
        );
        builder.append_series(&s1).unwrap();
        builder.append_null();

        let out = builder.finish();
        let out = out
            .explode(ExplodeOptions {
                empty_as_null: true,
                keep_nulls: true,
            })
            .unwrap();
        assert_eq!(out.len(), 7);
        assert_eq!(out.get(6).unwrap(), AnyValue::Null);
    }
}
