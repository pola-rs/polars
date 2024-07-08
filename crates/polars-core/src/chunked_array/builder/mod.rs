mod boolean;
#[cfg(feature = "dtype-array")]
pub mod fixed_size_list;
pub mod list;
mod null;
mod primitive;
mod string;

use std::sync::Arc;

use arrow::array::*;
use arrow::bitmap::Bitmap;
pub use boolean::*;
#[cfg(feature = "dtype-array")]
pub(crate) use fixed_size_list::*;
pub use list::*;
pub use null::*;
pub use primitive::*;
pub use string::*;

use crate::chunked_array::to_primitive;
use crate::prelude::*;
use crate::utils::{get_iter_capacity, NoNull};

// N: the value type; T: the sentinel type
pub trait ChunkedBuilder<N, T: PolarsDataType> {
    fn append_value(&mut self, val: N);
    fn append_null(&mut self);
    fn append_option(&mut self, opt_val: Option<N>) {
        match opt_val {
            Some(v) => self.append_value(v),
            None => self.append_null(),
        }
    }
    fn finish(self) -> ChunkedArray<T>;

    fn shrink_to_fit(&mut self);
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
        ChunkedArray::from_chunk_iter("from_iter", chunks)
    }
}

pub trait NewChunkedArray<T, N> {
    fn from_slice(name: &str, v: &[N]) -> Self;
    fn from_slice_options(name: &str, opt_v: &[Option<N>]) -> Self;

    /// Create a new ChunkedArray from an iterator.
    fn from_iter_options(name: &str, it: impl Iterator<Item = Option<N>>) -> Self;

    /// Create a new ChunkedArray from an iterator.
    fn from_iter_values(name: &str, it: impl Iterator<Item = N>) -> Self;
}

impl<T> NewChunkedArray<T, T::Native> for ChunkedArray<T>
where
    T: PolarsNumericType,
{
    fn from_slice(name: &str, v: &[T::Native]) -> Self {
        let arr = PrimitiveArray::from_slice(v).to(T::get_dtype().to_arrow(CompatLevel::newest()));
        ChunkedArray::with_chunk(name, arr)
    }

    fn from_slice_options(name: &str, opt_v: &[Option<T::Native>]) -> Self {
        Self::from_iter_options(name, opt_v.iter().copied())
    }

    fn from_iter_options(
        name: &str,
        it: impl Iterator<Item = Option<T::Native>>,
    ) -> ChunkedArray<T> {
        let mut builder = PrimitiveChunkedBuilder::new(name, get_iter_capacity(&it));
        it.for_each(|opt| builder.append_option(opt));
        builder.finish()
    }

    /// Create a new ChunkedArray from an iterator.
    fn from_iter_values(name: &str, it: impl Iterator<Item = T::Native>) -> ChunkedArray<T> {
        let ca: NoNull<ChunkedArray<_>> = it.collect();
        let mut ca = ca.into_inner();
        ca.rename(name);
        ca
    }
}

impl NewChunkedArray<BooleanType, bool> for BooleanChunked {
    fn from_slice(name: &str, v: &[bool]) -> Self {
        Self::from_iter_values(name, v.iter().copied())
    }

    fn from_slice_options(name: &str, opt_v: &[Option<bool>]) -> Self {
        Self::from_iter_options(name, opt_v.iter().copied())
    }

    fn from_iter_options(
        name: &str,
        it: impl Iterator<Item = Option<bool>>,
    ) -> ChunkedArray<BooleanType> {
        let mut builder = BooleanChunkedBuilder::new(name, get_iter_capacity(&it));
        it.for_each(|opt| builder.append_option(opt));
        builder.finish()
    }

    /// Create a new ChunkedArray from an iterator.
    fn from_iter_values(name: &str, it: impl Iterator<Item = bool>) -> ChunkedArray<BooleanType> {
        let mut ca: ChunkedArray<_> = it.collect();
        ca.rename(name);
        ca
    }
}

impl<S> NewChunkedArray<StringType, S> for StringChunked
where
    S: AsRef<str>,
{
    fn from_slice(name: &str, v: &[S]) -> Self {
        let arr = Utf8ViewArray::from_slice_values(v);
        ChunkedArray::with_chunk(name, arr)
    }

    fn from_slice_options(name: &str, opt_v: &[Option<S>]) -> Self {
        let arr = Utf8ViewArray::from_slice(opt_v);
        ChunkedArray::with_chunk(name, arr)
    }

    fn from_iter_options(name: &str, it: impl Iterator<Item = Option<S>>) -> Self {
        let arr = MutableBinaryViewArray::from_iterator(it).freeze();
        ChunkedArray::with_chunk(name, arr)
    }

    /// Create a new ChunkedArray from an iterator.
    fn from_iter_values(name: &str, it: impl Iterator<Item = S>) -> Self {
        let arr = MutableBinaryViewArray::from_values_iter(it).freeze();
        ChunkedArray::with_chunk(name, arr)
    }
}

impl<B> NewChunkedArray<BinaryType, B> for BinaryChunked
where
    B: AsRef<[u8]>,
{
    fn from_slice(name: &str, v: &[B]) -> Self {
        let arr = BinaryViewArray::from_slice_values(v);
        ChunkedArray::with_chunk(name, arr)
    }

    fn from_slice_options(name: &str, opt_v: &[Option<B>]) -> Self {
        let arr = BinaryViewArray::from_slice(opt_v);
        ChunkedArray::with_chunk(name, arr)
    }

    fn from_iter_options(name: &str, it: impl Iterator<Item = Option<B>>) -> Self {
        let arr = MutableBinaryViewArray::from_iterator(it).freeze();
        ChunkedArray::with_chunk(name, arr)
    }

    /// Create a new ChunkedArray from an iterator.
    fn from_iter_values(name: &str, it: impl Iterator<Item = B>) -> Self {
        let arr = MutableBinaryViewArray::from_values_iter(it).freeze();
        ChunkedArray::with_chunk(name, arr)
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_primitive_builder() {
        let mut builder = PrimitiveChunkedBuilder::<UInt32Type>::new("foo", 6);
        let values = &[Some(1), None, Some(2), Some(3), None, Some(4)];
        for val in values {
            builder.append_option(*val);
        }
        let ca = builder.finish();
        assert_eq!(Vec::from(&ca), values);
    }

    #[test]
    fn test_list_builder() {
        let mut builder =
            ListPrimitiveChunkedBuilder::<Int32Type>::new("a", 10, 5, DataType::Int32);

        // Create a series containing two chunks.
        let mut s1 = Int32Chunked::from_slice("a", &[1, 2, 3]).into_series();
        let s2 = Int32Chunked::from_slice("b", &[4, 5, 6]).into_series();
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

        let mut builder =
            ListPrimitiveChunkedBuilder::<Int32Type>::new("a", 10, 5, DataType::Int32);
        builder.append_series(&s1).unwrap();
        builder.append_null();

        let out = builder.finish();
        let out = out.explode().unwrap();
        assert_eq!(out.len(), 7);
        assert_eq!(out.get(6).unwrap(), AnyValue::Null);
    }
}
