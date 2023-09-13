//! Methods for collecting into a ChunkedArray.
//! To enable:
//!     where ChunkedArray<T>: ChunkedCollect<T>
//!
//! For types that don't have dtype parameters:
//! iter.(try_)to_ca(_trusted) (name)
//!
//! For all types:
//! iter.(try_)to_ca(_trusted)_like (other_df)  Copies name/dtype from other_df
//! iter.(try_)to_ca(_trusted)_with_dtype (name, df)
//!
//! The try variants work on iterators of Results, the trusted variants do not
//! check the length of the iterator.

use std::sync::Arc;

use arrow::array::{BinaryArray, MutableBinaryArray, MutableBinaryValuesArray, PrimitiveArray};
use arrow::bitmap::Bitmap;
use polars_arrow::trusted_len::{TrustedLen, TrustedLenPush};

use crate::chunked_array::ChunkedArray;
use crate::datatypes::{
    BinaryChunked, DataType, Field, NumericNative, PolarsDataType,
    PolarsNumericType, PolarsParameterFreeDataType, Utf8Chunked,
};

// Convenience trait for specifying bounds.
pub trait ChunkedCollect<T: PolarsDataType>:
    for<'a> ChunkedFromIter<T::Physical<'a>> + for<'a> ChunkedFromIter<Option<T::Physical<'a>>>
{
}
impl<T> ChunkedCollect<T> for ChunkedArray<T>
where
    T: PolarsDataType,
    for<'a> ChunkedArray<T>:
        ChunkedFromIter<T::Physical<'a>> + ChunkedFromIter<Option<T::Physical<'a>>>,
{
}

// We explicitly use `try_from_iter` instead of a blanket implementation on
// Result<T, E>: ChunkedFromIter<Result<T, E>> for two reasons:
// 1. If we did that ChunkedCollect could not encapsulate that this is required/exists.
// 2. We want to specialize some implementations to be faster around Results.
pub trait ChunkedFromIter<T>: Sized {
    fn ca_from_iter<I>(iter: I, field: Arc<Field>) -> Self
    where
        I: IntoIterator<Item = T>;

    fn ca_from_iter_trusted<I>(iter: I, field: Arc<Field>) -> Self
    where
        I: IntoIterator<Item = T>,
        I::IntoIter: TrustedLen;

    fn try_ca_from_iter<I, E>(iter: I, field: Arc<Field>) -> Result<Self, E>
    where
        I: IntoIterator<Item = Result<T, E>>;

    fn try_ca_from_iter_trusted<I, E>(iter: I, field: Arc<Field>) -> Result<Self, E>
    where
        I: IntoIterator<Item = Result<T, E>>,
        I::IntoIter: TrustedLen;
}

pub trait ChunkedCollectIterExt<T: PolarsDataType>: Iterator + Sized {
    #[inline]
    fn to_ca_with_dtype(self, name: &str, dtype: DataType) -> ChunkedArray<T>
    where
        ChunkedArray<T>: ChunkedFromIter<Self::Item>,
    {
        ChunkedArray::ca_from_iter(self, Arc::new(Field::new(name, dtype)))
    }

    #[inline]
    fn to_ca_like(self, name_dtype_src: &ChunkedArray<T>) -> ChunkedArray<T>
    where
        ChunkedArray<T>: ChunkedFromIter<Self::Item>,
    {
        ChunkedArray::ca_from_iter(self, Arc::clone(&name_dtype_src.field))
    }

    #[inline]
    fn to_ca_trusted_with_dtype(self, name: &str, dtype: DataType) -> ChunkedArray<T>
    where
        ChunkedArray<T>: ChunkedFromIter<Self::Item>,
        Self: TrustedLen,
    {
        ChunkedArray::ca_from_iter_trusted(self, Arc::new(Field::new(name, dtype)))
    }

    #[inline]
    fn to_ca_trusted_like(self, name_dtype_src: &ChunkedArray<T>) -> ChunkedArray<T>
    where
        ChunkedArray<T>: ChunkedFromIter<Self::Item>,
        Self: TrustedLen,
    {
        ChunkedArray::ca_from_iter_trusted(self, Arc::clone(&name_dtype_src.field))
    }

    #[inline]
    fn try_to_ca_with_dtype<U, E>(self, name: &str, dtype: DataType) -> Result<ChunkedArray<T>, E>
    where
        ChunkedArray<T>: ChunkedFromIter<U>,
        Self: Iterator<Item = Result<U, E>>,
    {
        ChunkedArray::try_ca_from_iter(self, Arc::new(Field::new(name, dtype)))
    }

    #[inline]
    fn try_to_ca_like<U, E>(self, name_dtype_src: &ChunkedArray<T>) -> Result<ChunkedArray<T>, E>
    where
        ChunkedArray<T>: ChunkedFromIter<U>,
        Self: Iterator<Item = Result<U, E>>,
    {
        ChunkedArray::try_ca_from_iter(self, Arc::clone(&name_dtype_src.field))
    }

    #[inline]
    fn try_to_ca_trusted_with_dtype<U, E>(
        self,
        name: &str,
        dtype: DataType,
    ) -> Result<ChunkedArray<T>, E>
    where
        ChunkedArray<T>: ChunkedFromIter<U>,
        Self: Iterator<Item = Result<U, E>> + TrustedLen,
    {
        ChunkedArray::try_ca_from_iter_trusted(self, Arc::new(Field::new(name, dtype)))
    }

    #[inline]
    fn try_to_ca_trusted_like<U, E>(
        self,
        name_dtype_src: &ChunkedArray<T>,
    ) -> Result<ChunkedArray<T>, E>
    where
        ChunkedArray<T>: ChunkedFromIter<U>,
        Self: Iterator<Item = Result<U, E>> + TrustedLen,
    {
        ChunkedArray::try_ca_from_iter_trusted(self, Arc::clone(&name_dtype_src.field))
    }
}

impl<T: PolarsDataType, I: Iterator> ChunkedCollectIterExt<T> for I {}

pub trait ChunkedCollectIterInferExt<T: PolarsParameterFreeDataType>: Iterator + Sized {
    #[inline]
    fn to_ca(self, name: &str) -> ChunkedArray<T>
    where
        ChunkedArray<T>: ChunkedFromIter<Self::Item>,
    {
        ChunkedArray::ca_from_iter(self, Arc::new(Field::new(name, T::get_dtype())))
    }

    #[inline]
    fn to_ca_trusted(self, name: &str) -> ChunkedArray<T>
    where
        ChunkedArray<T>: ChunkedFromIter<Self::Item>,
        Self: TrustedLen,
    {
        ChunkedArray::ca_from_iter_trusted(self, Arc::new(Field::new(name, T::get_dtype())))
    }

    #[inline]
    fn try_to_ca<U, E>(self, name: &str) -> Result<ChunkedArray<T>, E>
    where
        ChunkedArray<T>: ChunkedFromIter<U>,
        Self: Iterator<Item = Result<U, E>>,
    {
        ChunkedArray::try_ca_from_iter(self, Arc::new(Field::new(name, T::get_dtype())))
    }

    #[inline]
    fn try_to_ca_trusted<U, E>(self, name: &str) -> Result<ChunkedArray<T>, E>
    where
        ChunkedArray<T>: ChunkedFromIter<U>,
        Self: Iterator<Item = Result<U, E>> + TrustedLen,
    {
        ChunkedArray::try_ca_from_iter_trusted(self, Arc::new(Field::new(name, T::get_dtype())))
    }
}

impl<T: PolarsParameterFreeDataType, I: Iterator> ChunkedCollectIterInferExt<T> for I {}

// Using this trick we can implement both ChunkedFromIter<T::Physical> and
// ChunkedFromIter<Option<T::Physical>> for ChunkedArray<T>.
trait ChunkedFromIterOptionTrick<T, U>: Sized {
    fn ca_from_iter_impl<I>(iter: I, field: Arc<Field>) -> Self
    where
        I: IntoIterator<Item = U>;

    #[inline]
    fn ca_from_iter_trusted_impl<I>(iter: I, field: Arc<Field>) -> Self
    where
        I: IntoIterator<Item = U>,
        I::IntoIter: TrustedLen,
    {
        // Default implementation just ignores trusted length.
        Self::ca_from_iter_impl(iter, field)
    }

    fn try_ca_from_iter_impl<I, E>(iter: I, field: Arc<Field>) -> Result<Self, E>
    where
        I: IntoIterator<Item = Result<U, E>>,
    {
        let mut err = None;
        let ok_iter = iter.into_iter().scan((), |_, x| match x {
            Ok(x) => Some(x),
            Err(x) => {
                err = Some(x);
                None
            },
        });
        let ret = Self::ca_from_iter_impl(ok_iter, field);

        match err {
            None => Ok(ret),
            Some(err) => Err(err),
        }
    }

    fn try_ca_from_iter_trusted_impl<I, E>(iter: I, field: Arc<Field>) -> Result<Self, E>
    where
        I: IntoIterator<Item = Result<U, E>>,
        I::IntoIter: TrustedLen,
    {
        // Naive implementation becomes shorter on Err, can't trust anymore.
        Self::try_ca_from_iter_impl(iter, field)
    }
}

// Blanket implementation, the real implementation is on ChunkedFromIterOptionTrick.
impl<'a, T: PolarsDataType, U> ChunkedFromIter<U> for ChunkedArray<T>
where
    ChunkedArray<T>: ChunkedFromIterOptionTrick<T::Physical<'a>, U>,
{
    #[inline]
    fn ca_from_iter<I>(iter: I, field: Arc<Field>) -> Self
    where
        I: IntoIterator<Item = U>,
    {
        Self::ca_from_iter_impl(iter, field)
    }

    #[inline]
    fn ca_from_iter_trusted<I>(iter: I, field: Arc<Field>) -> Self
    where
        I: IntoIterator<Item = U>,
        I::IntoIter: TrustedLen,
    {
        Self::ca_from_iter_trusted_impl(iter, field)
    }

    #[inline]
    fn try_ca_from_iter<I, E>(iter: I, field: Arc<Field>) -> Result<Self, E>
    where
        I: IntoIterator<Item = Result<U, E>>,
    {
        Self::try_ca_from_iter_impl(iter, field)
    }

    #[inline]
    fn try_ca_from_iter_trusted<I, E>(iter: I, field: Arc<Field>) -> Result<Self, E>
    where
        I: IntoIterator<Item = Result<U, E>>,
        I::IntoIter: TrustedLen,
    {
        Self::try_ca_from_iter_trusted_impl(iter, field)
    }
}

// ---------------
// Implementations
// ---------------
macro_rules! impl_collect_vec_validity {
    ($iter: ident, $x:ident, $unpack:expr) => {{
        let mut iter = $iter.into_iter();
        let mut buf: Vec<T> = Vec::new();
        let mut bitmap: Vec<u8> = Vec::new();
        let lo = iter.size_hint().0;
        buf.reserve(8 + lo);
        bitmap.reserve(8 + 8 * (lo / 64));

        let mut nonnull_count = 0;
        let mut mask = 0u8;
        'exhausted: loop {
            unsafe {
                // SAFETY: when we enter this loop we always have at least one
                // capacity in bitmap, and at least 8 in buf.
                for i in 0..8 {
                    let Some($x) = iter.next() else {
                        break 'exhausted;
                    };
                    let x = $unpack;
                    let nonnull = x.is_some();
                    mask |= (nonnull as u8) << i;
                    nonnull_count += nonnull as usize;
                    buf.push_unchecked(x.unwrap_or_default());
                }

                bitmap.push_unchecked(mask);
                mask = 0;
            }

            buf.reserve(8);
            if bitmap.len() == bitmap.capacity() {
                bitmap.reserve(8); // Waste some space to make branch more predictable.
            }
        }

        unsafe {
            // SAFETY: when we broke to 'exhausted we had capacity by the loop invariant.
            // It's also no problem if we make the mask bigger than strictly necessary.
            bitmap.push_unchecked(mask);
        }

        let null_count = buf.len() - nonnull_count;
        let arrow_bitmap = if null_count > 0 {
            unsafe {
                // SAFETY: we made sure the null_count is correct.
                Some(Bitmap::from_inner(Arc::new(bitmap.into()), 0, buf.len(), null_count).unwrap())
            }
        } else {
            None
        };

        (buf, arrow_bitmap)
    }};
}

macro_rules! impl_trusted_collect_vec_validity {
    ($iter: ident, $x:ident, $unpack:expr) => {{
        let mut iter = $iter.into_iter();
        let mut buf: Vec<T> = Vec::new();
        let mut bitmap: Vec<u8> = Vec::new();
        let n = iter.size_hint().1.expect("must have an upper bound");
        buf.reserve(n);
        bitmap.reserve(8 + 8 * (n / 64));

        let mut nonnull_count = 0;
        while buf.len() + 8 <= n {
            unsafe {
                let mut mask = 0u8;
                for i in 0..8 {
                    let $x = iter.next().unwrap_unchecked();
                    let x = $unpack;
                    let nonnull = x.is_some();
                    mask |= (nonnull as u8) << i;
                    nonnull_count += nonnull as usize;
                    buf.push_unchecked(x.unwrap_or_default());
                }
                bitmap.push_unchecked(mask);
            }
        }

        if buf.len() < n {
            unsafe {
                let mut mask = 0u8;
                for i in 0..n - buf.len() {
                    let $x = iter.next().unwrap_unchecked();
                    let x = $unpack;
                    let nonnull = x.is_some();
                    mask |= (nonnull as u8) << i;
                    nonnull_count += nonnull as usize;
                    buf.push_unchecked(x.unwrap_or_default());
                }
                bitmap.push_unchecked(mask);
            }
        }

        let null_count = buf.len() - nonnull_count;
        let arrow_bitmap = if null_count > 0 {
            unsafe {
                // SAFETY: we made sure the null_count is correct.
                Some(Bitmap::from_inner(Arc::new(bitmap.into()), 0, buf.len(), null_count).unwrap())
            }
        } else {
            None
        };

        (buf, arrow_bitmap)
    }};
}

fn collect_vec_validity<T: Default, I: IntoIterator<Item = Option<T>>>(
    iter: I,
) -> (Vec<T>, Option<Bitmap>) {
    impl_collect_vec_validity!(iter, x, x)
}

fn try_collect_vec_validity<T: Default, E, I: IntoIterator<Item = Result<Option<T>, E>>>(
    iter: I,
) -> Result<(Vec<T>, Option<Bitmap>), E> {
    Ok(impl_collect_vec_validity!(iter, x, x?))
}

fn trusted_collect_vec_validity<T, I>(iter: I) -> (Vec<T>, Option<Bitmap>)
where
    T: Default,
    I: IntoIterator<Item = Option<T>>,
    I::IntoIter: TrustedLen,
{
    impl_trusted_collect_vec_validity!(iter, x, x)
}

fn try_trusted_collect_vec_validity<T, E, I>(iter: I) -> Result<(Vec<T>, Option<Bitmap>), E>
where
    T: Default,
    I: IntoIterator<Item = Result<Option<T>, E>>,
    I::IntoIter: TrustedLen,
{
    Ok(impl_trusted_collect_vec_validity!(iter, x, x?))
}

impl<'a, P: NumericNative, T: PolarsNumericType<Native = P>> ChunkedFromIterOptionTrick<P, P>
    for ChunkedArray<T>
{
    fn ca_from_iter_impl<I>(iter: I, field: Arc<Field>) -> Self
    where
        I: IntoIterator<Item = P>,
    {
        let v: Vec<T::Native> = iter.into_iter().collect();
        ChunkedArray::from_chunk_iter_and_field(field, [PrimitiveArray::from_vec(v)])
    }

    fn ca_from_iter_trusted_impl<I>(iter: I, field: Arc<Field>) -> Self
    where
        I: IntoIterator<Item = P>,
        I::IntoIter: TrustedLen,
    {
        let v: Vec<T::Native> = Vec::from_trusted_len_iter(iter);
        ChunkedArray::from_chunk_iter_and_field(field, [PrimitiveArray::from_vec(v)])
    }

    fn try_ca_from_iter_impl<I, E>(iter: I, field: Arc<Field>) -> Result<Self, E>
    where
        I: IntoIterator<Item = Result<P, E>>,
    {
        let v: Result<Vec<T::Native>, E> = iter.into_iter().collect();
        let arr = PrimitiveArray::from_vec(v?);
        Ok(ChunkedArray::from_chunk_iter_and_field(field, [arr]))
    }

    #[inline]
    fn try_ca_from_iter_trusted_impl<I, E>(iter: I, field: Arc<Field>) -> Result<Self, E>
    where
        I: IntoIterator<Item = Result<P, E>>,
        I::IntoIter: TrustedLen,
    {
        let v: Vec<T::Native> = Vec::try_from_trusted_len_iter(iter)?;
        let arr = PrimitiveArray::from_vec(v);
        Ok(ChunkedArray::from_chunk_iter_and_field(field, [arr]))
    }
}

impl<'a, P: NumericNative, T: PolarsNumericType<Native = P>>
    ChunkedFromIterOptionTrick<P, Option<P>> for ChunkedArray<T>
{
    fn ca_from_iter_impl<I>(iter: I, field: Arc<Field>) -> Self
    where
        I: IntoIterator<Item = Option<P>>,
    {
        let (buf, validity) = collect_vec_validity(iter);
        let arr = PrimitiveArray::new(field.dtype.to_arrow(), buf.into(), validity);
        ChunkedArray::from_chunk_iter_and_field(field, [arr])
    }

    fn ca_from_iter_trusted_impl<I>(iter: I, field: Arc<Field>) -> Self
    where
        I: IntoIterator<Item = Option<P>>,
        I::IntoIter: TrustedLen,
    {
        let (buf, validity) = trusted_collect_vec_validity(iter);
        let arr = PrimitiveArray::new(field.dtype.to_arrow(), buf.into(), validity);
        ChunkedArray::from_chunk_iter_and_field(field, [arr])
    }

    fn try_ca_from_iter_impl<I, E>(iter: I, field: Arc<Field>) -> Result<Self, E>
    where
        I: IntoIterator<Item = Result<Option<P>, E>>,
    {
        let (buf, validity) = try_collect_vec_validity(iter)?;
        let arr = PrimitiveArray::new(field.dtype.to_arrow(), buf.into(), validity);
        Ok(ChunkedArray::from_chunk_iter_and_field(field, [arr]))
    }

    #[inline]
    fn try_ca_from_iter_trusted_impl<I, E>(iter: I, field: Arc<Field>) -> Result<Self, E>
    where
        I: IntoIterator<Item = Result<Option<P>, E>>,
        I::IntoIter: TrustedLen,
    {
        let (buf, validity) = try_trusted_collect_vec_validity(iter)?;
        let arr = PrimitiveArray::new(field.dtype.to_arrow(), buf.into(), validity);
        Ok(ChunkedArray::from_chunk_iter_and_field(field, [arr]))
    }
}

impl<T: AsRef<[u8]>> ChunkedFromIterOptionTrick<T, T> for BinaryChunked {
    fn ca_from_iter_impl<I>(iter: I, field: Arc<Field>) -> Self
    where
        I: IntoIterator<Item = T>,
    {
        let arr = BinaryArray::<i64>::from_iter_values(iter.into_iter());
        ChunkedArray::from_chunk_iter_and_field(field, [arr])
    }

    fn ca_from_iter_trusted_impl<I>(iter: I, field: Arc<Field>) -> Self
    where
        I: IntoIterator<Item = T>,
        I::IntoIter: TrustedLen,
    {
        let arr = unsafe {
            MutableBinaryArray::<i64>::from_trusted_len_values_iter_unchecked(iter.into_iter())
        };
        ChunkedArray::from_chunk_iter_and_field(field, [arr.into()])
    }

    fn try_ca_from_iter_impl<I, E>(iter: I, field: Arc<Field>) -> Result<Self, E>
    where
        I: IntoIterator<Item = Result<T, E>>,
    {
        // No built-in for this?
        let mut arr = MutableBinaryValuesArray::<i64>::new();
        let mut iter = iter.into_iter();
        arr.reserve(iter.size_hint().0, 0);
        iter.try_for_each(|x| -> Result<(), E> {
            arr.push(x?);
            Ok(())
        })?;
        Ok(ChunkedArray::from_chunk_iter_and_field(field, [arr.into()]))
    }

    // No built-in for this, not really faster than default impl anyway.
    // fn try_ca_from_iter_trusted_impl<I, E>(iter: I, field: Arc<Field>) -> Result<Self, E>
}

impl<T: AsRef<[u8]>> ChunkedFromIterOptionTrick<T, Option<T>> for BinaryChunked {
    fn ca_from_iter_impl<I>(iter: I, field: Arc<Field>) -> Self
    where
        I: IntoIterator<Item = Option<T>>,
    {
        let arr = BinaryArray::<i64>::from_iter(iter.into_iter());
        ChunkedArray::from_chunk_iter_and_field(field, [arr])
    }

    fn ca_from_iter_trusted_impl<I>(iter: I, field: Arc<Field>) -> Self
    where
        I: IntoIterator<Item = Option<T>>,
        I::IntoIter: TrustedLen,
    {
        let arr = unsafe { BinaryArray::<i64>::from_trusted_len_iter_unchecked(iter.into_iter()) };
        ChunkedArray::from_chunk_iter_and_field(field, [arr])
    }

    fn try_ca_from_iter_impl<I, E>(iter: I, field: Arc<Field>) -> Result<Self, E>
    where
        I: IntoIterator<Item = Result<Option<T>, E>>,
    {
        // No built-in for this?
        let mut arr = MutableBinaryArray::<i64>::new();
        let mut iter = iter.into_iter();
        arr.reserve(iter.size_hint().0, 0);
        iter.try_for_each(|x| -> Result<(), E> {
            arr.push(x?);
            Ok(())
        })?;
        Ok(ChunkedArray::from_chunk_iter_and_field(field, [arr.into()]))
    }

    #[inline]
    fn try_ca_from_iter_trusted_impl<I, E>(iter: I, field: Arc<Field>) -> Result<Self, E>
    where
        I: IntoIterator<Item = Result<Option<T>, E>>,
        I::IntoIter: TrustedLen,
    {
        let arr = unsafe { BinaryArray::<i64>::try_from_trusted_len_iter_unchecked(iter.into_iter()) }?;
        Ok(ChunkedArray::from_chunk_iter_and_field(field, [arr]))
    }
}

/*
impl<T: AsRef<str>> ChunkedFromIterOptionTrick<T, T> for Utf8Chunked {
    fn ca_from_iter_impl<I>(iter: I, field: Arc<Field>) -> Self
    where
        I: IntoIterator<Item = T>,
    {
        let arr = BinaryArray::<i64>::from_iter_values(iter.into_iter());
        ChunkedArray::from_chunk_iter_and_field(field, [arr])
    }

    fn ca_from_iter_trusted_impl<I>(iter: I, field: Arc<Field>) -> Self
    where
        I: IntoIterator<Item = T>,
        I::IntoIter: TrustedLen,
    {
        let arr = unsafe {
            MutableBinaryArray::<i64>::from_trusted_len_values_iter_unchecked(iter.into_iter())
        };
        ChunkedArray::from_chunk_iter_and_field(field, [arr.into()])
    }

    fn try_ca_from_iter_impl<I, E>(iter: I, field: Arc<Field>) -> Result<Self, E>
    where
        I: IntoIterator<Item = Result<T, E>>,
    {
        // No built-in for this?
        let mut arr = MutableBinaryValuesArray::<i64>::new();
        let mut iter = iter.into_iter();
        arr.reserve(iter.size_hint().0, 0);
        iter.try_for_each(|x| -> Result<(), E> {
            arr.push(x?);
            Ok(())
        })?;
        Ok(ChunkedArray::from_chunk_iter_and_field(field, [arr.into()]))
    }

    // No built-in for this, not really faster than default impl anyway.
    // fn try_ca_from_iter_trusted_impl<I, E>(iter: I, field: Arc<Field>) -> Result<Self, E>
}
*/




/*
impl_polars_datatype!(BinaryType, Binary, BinaryArray<i64>, 'a, &'a [u8]);
impl_polars_datatype!(BooleanType, Boolean, BooleanArray, 'a, bool);

pub struct ListType {}

#[cfg(feature = "dtype-array")]
pub struct FixedSizeListType {}
#[cfg(feature = "object")]
pub struct ObjectType<T>(T);


*/
