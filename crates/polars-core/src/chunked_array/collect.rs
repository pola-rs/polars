use std::sync::Arc;

use arrow::array::PrimitiveArray;
use arrow::bitmap::MutableBitmap;
use polars_arrow::trusted_len::{TrustedLen, TrustedLenPush};

use crate::chunked_array::ChunkedArray;
use crate::datatypes::{
    DataType, DecimalType, Field, NumericNative, PolarsDataType, PolarsNumericType,
    PolarsParameterFreeDataType, UInt8Chunked,
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

// Exhausts the iterator, passing smaller subiterators to f that are guaranteed
// to have a correct upper bound.
fn try_exhaust_iterator_chunked<T, E, I, F>(iter: I, mut f: F) -> Result<(), E>
where
    I: IntoIterator<Item = T>,
    F: FnMut(std::iter::Take<&mut I::IntoIter>) -> Result<bool, E>,
{
    let mut iter = iter.into_iter();
    let mut min_count = 8;
    loop {
        let (lo, hi) = iter.size_hint();
        if hi == Some(0) {
            return Ok(());
        }
        let count = std::cmp::max(lo, min_count);
        if f(iter.by_ref().take(count))? {
            return Ok(());
        }

        // We don't want to immediately reserve space for 1024 elements for
        // small inputs, but eventually we do want to process relatively
        // large blocks of data, even if the iterator hint keeps returning
        // small or zero size hints.
        if min_count < 1024 {
            min_count *= 2;
        }
    }
}

/// SAFETY: the iterator must have a correct upper size hint.
unsafe fn extend_vec_validity<T: Default, I: Iterator<Item = Option<T>>>(
    iter: I,
    buf: &mut Vec<T>,
    validity: &mut MutableBitmap,
) -> bool {
    let upper = iter.size_hint().1.expect("must have an upper bound");
    buf.reserve(upper);
    validity.reserve(upper);

    let mut iter_exhausted = true;
    for x in iter {
        iter_exhausted = false;
        // SAFETY: we reserved for count, and the iterator is no longer than that.
        unsafe {
            validity.push_unchecked(x.is_some());
            buf.push_unchecked(x.unwrap_or_default());
        }
    }

    iter_exhausted
}

/// SAFETY: the iterator must have a correct upper size hint.
unsafe fn try_extend_vec_validity<T: Default, E, I: Iterator<Item = Result<Option<T>, E>>>(
    iter: I,
    buf: &mut Vec<T>,
    validity: &mut MutableBitmap,
) -> Result<bool, E> {
    let upper = iter.size_hint().1.expect("must have an upper bound");
    buf.reserve(upper);
    validity.reserve(upper);

    let mut iter_exhausted = true;
    for x in iter {
        iter_exhausted = false;
        // SAFETY: we reserved for count, and the iterator is no longer than that.
        unsafe {
            let x = x?;
            validity.push_unchecked(x.is_some());
            buf.push_unchecked(x.unwrap_or_default());
        }
    }

    Ok(iter_exhausted)
}

impl<'a, P: NumericNative, T: PolarsNumericType<Native = P>>
    ChunkedFromIterOptionTrick<P, Option<P>> for ChunkedArray<T>
{
    fn ca_from_iter_impl<I>(iter: I, field: Arc<Field>) -> Self
    where
        I: IntoIterator<Item = Option<P>>,
    {
        let mut buf = Vec::new();
        let mut validity = MutableBitmap::new();
        let _ignore: Result<(), ()> = try_exhaust_iterator_chunked(iter, |chunk_iter| unsafe {
            // SAFETY: try_exhaust_iterator_chunked guarantees the upper bound of
            // chunk_iter is correct.
            Ok(extend_vec_validity(chunk_iter, &mut buf, &mut validity))
        });

        let arr = PrimitiveArray::new(field.dtype.to_arrow(), buf.into(), validity.into());
        ChunkedArray::from_chunk_iter_and_field(field, [arr])
    }

    fn ca_from_iter_trusted_impl<I>(iter: I, field: Arc<Field>) -> Self
    where
        I: IntoIterator<Item = Option<P>>,
        I::IntoIter: TrustedLen,
    {
        let mut buf = Vec::new();
        let mut validity = MutableBitmap::new();
        let iter = iter.into_iter();
        unsafe {
            // SAFETY: TrustedLen guarantees this is safe, and will exactly
            // and completely exhaust the iterator.
            extend_vec_validity(iter, &mut buf, &mut validity);
        }

        let arr = PrimitiveArray::new(field.dtype.to_arrow(), buf.into(), validity.into());
        ChunkedArray::from_chunk_iter_and_field(field, [arr])
    }

    fn try_ca_from_iter_impl<I, E>(iter: I, field: Arc<Field>) -> Result<Self, E>
    where
        I: IntoIterator<Item = Result<Option<P>, E>>,
    {
        let mut buf = Vec::new();
        let mut validity = MutableBitmap::new();
        try_exhaust_iterator_chunked(iter, |chunk_iter| unsafe {
            // SAFETY: try_exhaust_iterator_chunked guarantees the upper bound of
            // chunk_iter is correct.
            try_extend_vec_validity(chunk_iter, &mut buf, &mut validity)
        })?;

        let arr = PrimitiveArray::new(field.dtype.to_arrow(), buf.into(), validity.into());
        Ok(ChunkedArray::from_chunk_iter_and_field(field, [arr]))
    }

    #[inline]
    fn try_ca_from_iter_trusted_impl<I, E>(iter: I, field: Arc<Field>) -> Result<Self, E>
    where
        I: IntoIterator<Item = Result<Option<P>, E>>,
        I::IntoIter: TrustedLen,
    {
        let mut buf = Vec::new();
        let mut validity = MutableBitmap::new();
        let iter = iter.into_iter();
        unsafe {
            // SAFETY: TrustedLen guarantees this is safe, and will exactly
            // and completely exhaust the iterator.
            try_extend_vec_validity(iter, &mut buf, &mut validity)?;
        }

        let arr = PrimitiveArray::new(field.dtype.to_arrow(), buf.into(), validity.into());
        Ok(ChunkedArray::from_chunk_iter_and_field(field, [arr]))
    }
}

impl<'a> ChunkedFromIterOptionTrick<DecimalType, DecimalType> for ChunkedArray<DecimalType> {
    fn ca_from_iter_impl<I>(iter: I, field: Arc<Field>) -> Self
    where
        I: IntoIterator<Item = DecimalType>,
    {
        todo!()
    }

    fn ca_from_iter_trusted_impl<I>(iter: I, field: Arc<Field>) -> Self
    where
        I: IntoIterator<Item = DecimalType>,
    {
        todo!()
    }
}

impl<'a> ChunkedFromIterOptionTrick<DecimalType, Option<DecimalType>>
    for ChunkedArray<DecimalType>
{
    fn ca_from_iter_impl<I>(iter: I, field: Arc<Field>) -> Self
    where
        I: IntoIterator<Item = Option<DecimalType>>,
    {
        todo!()
    }

    fn ca_from_iter_trusted_impl<I>(iter: I, field: Arc<Field>) -> Self
    where
        I: IntoIterator<Item = Option<DecimalType>>,
    {
        todo!()
    }
}

/*

trait MaybeOptionChunkedCollect {


}

impl<'a, T: PolarsDataType, U> ChunkedFromIterOptionTrick<U, U> for ChunkedArray<T> {
    fn ca_from_iter<I>(iter: I, name: &str, dtype: DataType) -> Self
       where I: IntoIterator<Item = T::Physical<'a>> {
        todo!()
    }
}

impl<'a, T: PolarsDataType, U> ChunkedFromIter<U, Option<U>> for ChunkedArray<T> {
    fn ca_from_iter<I>(iter: I, name: &str, dtype: DataType) -> Self
       where I: IntoIterator<Item = Option<T::Physical<'a>>> {
        todo!()
    }
}
*/

// impl<T> MaybeOptionChunkedCollect for T { }
// impl<T> MaybeOptionChunkedCollect for Option<T> { }

/*
impl<T, I: Iterator<Item=T>, CA: ChunkedFromIter<T>> ChunkedCollect<CA> for I {
    fn collect_ca(self) -> CA {
        todo!()
    }
}

impl<T, I: Iterator<Item=Option<T>>, CA: ChunkedFromIter<T>> ChunkedCollect<CA> for I {
    fn collect_ca(self) -> CA {
        todo!()
    }
}
*/

/*
impl<T: PolarsDataType> ChunkedArray<T> {
    fn from_iter<'a>(iter: T) -> Self
       where T: IntoIterator<Item = T::Physical<'a>> {
        todo!()

    }
}
*/

/*
impl<T: PolarsNumericType> ChunkedFromIter<T::Native> for ChunkedArray<T> {
    fn from_iter<I>(iter: I) -> Self
       where I: IntoIterator<Item = T::Native> {
        todo!()
    }

    fn from_nonnull_iter<I>(iter: I) -> Self
       where I: IntoIterator<Item = T::Native> {
        todo!()
    }
}
*/

// impl<T: PolarsNumericType> ChunkedFromIter<Option<T::Native>> for ChunkedArray<T> {

// }
