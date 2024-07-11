use std::borrow::Cow;
use std::sync::Arc;

use polars_utils::no_call_const;

use crate::array::static_array::{ParameterFreeDtypeStaticArray, StaticArray};
use crate::array::{
    Array, BinaryArray, BinaryViewArray, BooleanArray, FixedSizeListArray, ListArray,
    MutableBinaryArray, MutableBinaryValuesArray, MutableBinaryViewArray, PrimitiveArray,
    StructArray, Utf8Array, Utf8ViewArray,
};
use crate::bitmap::Bitmap;
use crate::datatypes::ArrowDataType;
#[cfg(feature = "dtype-array")]
use crate::legacy::prelude::fixed_size_list::AnonymousBuilder as AnonymousFixedSizeListArrayBuilder;
use crate::legacy::prelude::list::AnonymousBuilder as AnonymousListArrayBuilder;
use crate::legacy::trusted_len::TrustedLenPush;
use crate::trusted_len::TrustedLen;
use crate::types::NativeType;

pub trait ArrayFromIterDtype<T>: Sized {
    fn arr_from_iter_with_dtype<I: IntoIterator<Item = T>>(dtype: ArrowDataType, iter: I) -> Self;

    #[inline(always)]
    fn arr_from_iter_trusted_with_dtype<I>(dtype: ArrowDataType, iter: I) -> Self
    where
        I: IntoIterator<Item = T>,
        I::IntoIter: TrustedLen,
    {
        Self::arr_from_iter_with_dtype(dtype, iter)
    }

    fn try_arr_from_iter_with_dtype<E, I: IntoIterator<Item = Result<T, E>>>(
        dtype: ArrowDataType,
        iter: I,
    ) -> Result<Self, E>;

    #[inline(always)]
    fn try_arr_from_iter_trusted_with_dtype<E, I>(dtype: ArrowDataType, iter: I) -> Result<Self, E>
    where
        I: IntoIterator<Item = Result<T, E>>,
        I::IntoIter: TrustedLen,
    {
        Self::try_arr_from_iter_with_dtype(dtype, iter)
    }
}

pub trait ArrayFromIter<T>: Sized {
    fn arr_from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self;

    #[inline(always)]
    fn arr_from_iter_trusted<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = T>,
        I::IntoIter: TrustedLen,
    {
        Self::arr_from_iter(iter)
    }

    fn try_arr_from_iter<E, I: IntoIterator<Item = Result<T, E>>>(iter: I) -> Result<Self, E>;

    #[inline(always)]
    fn try_arr_from_iter_trusted<E, I>(iter: I) -> Result<Self, E>
    where
        I: IntoIterator<Item = Result<T, E>>,
        I::IntoIter: TrustedLen,
    {
        Self::try_arr_from_iter(iter)
    }
}

impl<T, A: ParameterFreeDtypeStaticArray + ArrayFromIter<T>> ArrayFromIterDtype<T> for A {
    #[inline(always)]
    fn arr_from_iter_with_dtype<I: IntoIterator<Item = T>>(dtype: ArrowDataType, iter: I) -> Self {
        // FIXME: currently some Object arrays have Unknown dtype, when this is fixed remove this bypass.
        if dtype != ArrowDataType::Unknown {
            debug_assert_eq!(
                std::mem::discriminant(&dtype),
                std::mem::discriminant(&A::get_dtype())
            );
        }
        Self::arr_from_iter(iter)
    }

    #[inline(always)]
    fn arr_from_iter_trusted_with_dtype<I>(dtype: ArrowDataType, iter: I) -> Self
    where
        I: IntoIterator<Item = T>,
        I::IntoIter: TrustedLen,
    {
        // FIXME: currently some Object arrays have Unknown dtype, when this is fixed remove this bypass.
        if dtype != ArrowDataType::Unknown {
            debug_assert_eq!(
                std::mem::discriminant(&dtype),
                std::mem::discriminant(&A::get_dtype())
            );
        }
        Self::arr_from_iter_trusted(iter)
    }

    #[inline(always)]
    fn try_arr_from_iter_with_dtype<E, I: IntoIterator<Item = Result<T, E>>>(
        dtype: ArrowDataType,
        iter: I,
    ) -> Result<Self, E> {
        // FIXME: currently some Object arrays have Unknown dtype, when this is fixed remove this bypass.
        if dtype != ArrowDataType::Unknown {
            debug_assert_eq!(
                std::mem::discriminant(&dtype),
                std::mem::discriminant(&A::get_dtype())
            );
        }
        Self::try_arr_from_iter(iter)
    }

    #[inline(always)]
    fn try_arr_from_iter_trusted_with_dtype<E, I>(dtype: ArrowDataType, iter: I) -> Result<Self, E>
    where
        I: IntoIterator<Item = Result<T, E>>,
        I::IntoIter: TrustedLen,
    {
        // FIXME: currently some Object arrays have Unknown dtype, when this is fixed remove this bypass.
        if dtype != ArrowDataType::Unknown {
            debug_assert_eq!(
                std::mem::discriminant(&dtype),
                std::mem::discriminant(&A::get_dtype())
            );
        }
        Self::try_arr_from_iter_trusted(iter)
    }
}

pub trait ArrayCollectIterExt<A: StaticArray>: Iterator + Sized {
    #[inline(always)]
    fn collect_arr(self) -> A
    where
        A: ArrayFromIter<Self::Item>,
    {
        A::arr_from_iter(self)
    }

    #[inline(always)]
    fn collect_arr_trusted(self) -> A
    where
        A: ArrayFromIter<Self::Item>,
        Self: TrustedLen,
    {
        A::arr_from_iter_trusted(self)
    }

    #[inline(always)]
    fn try_collect_arr<U, E>(self) -> Result<A, E>
    where
        A: ArrayFromIter<U>,
        Self: Iterator<Item = Result<U, E>>,
    {
        A::try_arr_from_iter(self)
    }

    #[inline(always)]
    fn try_collect_arr_trusted<U, E>(self) -> Result<A, E>
    where
        A: ArrayFromIter<U>,
        Self: Iterator<Item = Result<U, E>> + TrustedLen,
    {
        A::try_arr_from_iter_trusted(self)
    }

    #[inline(always)]
    fn collect_arr_with_dtype(self, dtype: ArrowDataType) -> A
    where
        A: ArrayFromIterDtype<Self::Item>,
    {
        A::arr_from_iter_with_dtype(dtype, self)
    }

    #[inline(always)]
    fn collect_arr_trusted_with_dtype(self, dtype: ArrowDataType) -> A
    where
        A: ArrayFromIterDtype<Self::Item>,
        Self: TrustedLen,
    {
        A::arr_from_iter_trusted_with_dtype(dtype, self)
    }

    #[inline(always)]
    fn try_collect_arr_with_dtype<U, E>(self, dtype: ArrowDataType) -> Result<A, E>
    where
        A: ArrayFromIterDtype<U>,
        Self: Iterator<Item = Result<U, E>>,
    {
        A::try_arr_from_iter_with_dtype(dtype, self)
    }

    #[inline(always)]
    fn try_collect_arr_trusted_with_dtype<U, E>(self, dtype: ArrowDataType) -> Result<A, E>
    where
        A: ArrayFromIterDtype<U>,
        Self: Iterator<Item = Result<U, E>> + TrustedLen,
    {
        A::try_arr_from_iter_trusted_with_dtype(dtype, self)
    }
}

impl<A: StaticArray, I: Iterator> ArrayCollectIterExt<A> for I {}

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
                    #[allow(clippy::all)]
                    // #[allow(clippy::redundant_locals)]  Clippy lint too new
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
                Some(Bitmap::from_inner_unchecked(
                    Arc::new(bitmap.into()),
                    0,
                    buf.len(),
                    Some(null_count),
                ))
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
                    #[allow(clippy::all)]
                    // #[allow(clippy::redundant_locals)]  Clippy lint too new
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
                Some(Bitmap::from_inner_unchecked(
                    Arc::new(bitmap.into()),
                    0,
                    buf.len(),
                    Some(null_count),
                ))
            }
        } else {
            None
        };

        (buf, arrow_bitmap)
    }};
}

impl<T: NativeType> ArrayFromIter<T> for PrimitiveArray<T> {
    #[inline]
    fn arr_from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        PrimitiveArray::from_vec(iter.into_iter().collect())
    }

    #[inline]
    fn arr_from_iter_trusted<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = T>,
        I::IntoIter: TrustedLen,
    {
        PrimitiveArray::from_vec(Vec::from_trusted_len_iter(iter))
    }

    #[inline]
    fn try_arr_from_iter<E, I: IntoIterator<Item = Result<T, E>>>(iter: I) -> Result<Self, E> {
        let v: Result<Vec<T>, E> = iter.into_iter().collect();
        Ok(PrimitiveArray::from_vec(v?))
    }

    #[inline]
    fn try_arr_from_iter_trusted<E, I>(iter: I) -> Result<Self, E>
    where
        I: IntoIterator<Item = Result<T, E>>,
        I::IntoIter: TrustedLen,
    {
        let v = Vec::try_from_trusted_len_iter(iter);
        Ok(PrimitiveArray::from_vec(v?))
    }
}

impl<T: NativeType> ArrayFromIter<Option<T>> for PrimitiveArray<T> {
    fn arr_from_iter<I: IntoIterator<Item = Option<T>>>(iter: I) -> Self {
        let (buf, validity) = impl_collect_vec_validity!(iter, x, x);
        PrimitiveArray::new(T::PRIMITIVE.into(), buf.into(), validity)
    }

    fn arr_from_iter_trusted<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = Option<T>>,
        I::IntoIter: TrustedLen,
    {
        let (buf, validity) = impl_trusted_collect_vec_validity!(iter, x, x);
        PrimitiveArray::new(T::PRIMITIVE.into(), buf.into(), validity)
    }

    fn try_arr_from_iter<E, I: IntoIterator<Item = Result<Option<T>, E>>>(
        iter: I,
    ) -> Result<Self, E> {
        let (buf, validity) = impl_collect_vec_validity!(iter, x, x?);
        Ok(PrimitiveArray::new(
            T::PRIMITIVE.into(),
            buf.into(),
            validity,
        ))
    }

    fn try_arr_from_iter_trusted<E, I>(iter: I) -> Result<Self, E>
    where
        I: IntoIterator<Item = Result<Option<T>, E>>,
        I::IntoIter: TrustedLen,
    {
        let (buf, validity) = impl_trusted_collect_vec_validity!(iter, x, x?);
        Ok(PrimitiveArray::new(
            T::PRIMITIVE.into(),
            buf.into(),
            validity,
        ))
    }
}

// We don't use AsRef here because it leads to problems with conflicting implementations,
// as Rust considers that AsRef<[u8]> for Option<&[u8]> could be implemented.
trait IntoBytes {
    type AsRefT: AsRef<[u8]>;
    fn into_bytes(self) -> Self::AsRefT;
}
trait TrivialIntoBytes: AsRef<[u8]> {}
impl<T: TrivialIntoBytes> IntoBytes for T {
    type AsRefT = Self;
    fn into_bytes(self) -> Self {
        self
    }
}
impl TrivialIntoBytes for Vec<u8> {}
impl<'a> TrivialIntoBytes for Cow<'a, [u8]> {}
impl<'a> TrivialIntoBytes for &'a [u8] {}
impl TrivialIntoBytes for String {}
impl<'a> TrivialIntoBytes for &'a str {}
impl<'a> IntoBytes for Cow<'a, str> {
    type AsRefT = Cow<'a, [u8]>;
    fn into_bytes(self) -> Cow<'a, [u8]> {
        match self {
            Cow::Borrowed(a) => Cow::Borrowed(a.as_bytes()),
            Cow::Owned(s) => Cow::Owned(s.into_bytes()),
        }
    }
}

impl<T: IntoBytes> ArrayFromIter<T> for BinaryArray<i64> {
    fn arr_from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        BinaryArray::from_iter_values(iter.into_iter().map(|s| s.into_bytes()))
    }

    fn arr_from_iter_trusted<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = T>,
        I::IntoIter: TrustedLen,
    {
        unsafe {
            // SAFETY: our iterator is TrustedLen.
            MutableBinaryArray::from_trusted_len_values_iter_unchecked(
                iter.into_iter().map(|s| s.into_bytes()),
            )
            .into()
        }
    }

    fn try_arr_from_iter<E, I: IntoIterator<Item = Result<T, E>>>(iter: I) -> Result<Self, E> {
        // No built-in for this?
        let mut arr = MutableBinaryValuesArray::new();
        let mut iter = iter.into_iter();
        arr.reserve(iter.size_hint().0, 0);
        iter.try_for_each(|x| -> Result<(), E> {
            arr.push(x?.into_bytes());
            Ok(())
        })?;
        Ok(arr.into())
    }

    // No faster implementation than this available, fall back to default.
    // fn try_arr_from_iter_trusted<E, I>(iter: I) -> Result<Self, E>
}

impl<T: IntoBytes> ArrayFromIter<Option<T>> for BinaryArray<i64> {
    #[inline]
    fn arr_from_iter<I: IntoIterator<Item = Option<T>>>(iter: I) -> Self {
        BinaryArray::from_iter(iter.into_iter().map(|s| Some(s?.into_bytes())))
    }

    #[inline]
    fn arr_from_iter_trusted<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = Option<T>>,
        I::IntoIter: TrustedLen,
    {
        unsafe {
            // SAFETY: the iterator is TrustedLen.
            BinaryArray::from_trusted_len_iter_unchecked(
                iter.into_iter().map(|s| Some(s?.into_bytes())),
            )
        }
    }

    fn try_arr_from_iter<E, I: IntoIterator<Item = Result<Option<T>, E>>>(
        iter: I,
    ) -> Result<Self, E> {
        // No built-in for this?
        let mut arr = MutableBinaryArray::new();
        let mut iter = iter.into_iter();
        arr.reserve(iter.size_hint().0, 0);
        iter.try_for_each(|x| -> Result<(), E> {
            arr.push(x?.map(|s| s.into_bytes()));
            Ok(())
        })?;
        Ok(arr.into())
    }

    fn try_arr_from_iter_trusted<E, I>(iter: I) -> Result<Self, E>
    where
        I: IntoIterator<Item = Result<Option<T>, E>>,
        I::IntoIter: TrustedLen,
    {
        unsafe {
            // SAFETY: the iterator is TrustedLen.
            BinaryArray::try_from_trusted_len_iter_unchecked(
                iter.into_iter().map(|s| s.map(|s| Some(s?.into_bytes()))),
            )
        }
    }
}

impl<T: IntoBytes> ArrayFromIter<T> for BinaryViewArray {
    #[inline]
    fn arr_from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        MutableBinaryViewArray::from_values_iter(iter.into_iter().map(|a| a.into_bytes())).into()
    }

    #[inline]
    fn arr_from_iter_trusted<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = T>,
        I::IntoIter: TrustedLen,
    {
        Self::arr_from_iter(iter)
    }

    fn try_arr_from_iter<E, I: IntoIterator<Item = Result<T, E>>>(iter: I) -> Result<Self, E> {
        let mut iter = iter.into_iter();
        let mut arr = MutableBinaryViewArray::with_capacity(iter.size_hint().0);
        iter.try_for_each(|x| -> Result<(), E> {
            arr.push_value_ignore_validity(x?.into_bytes());
            Ok(())
        })?;
        Ok(arr.into())
    }

    // No faster implementation than this available, fall back to default.
    // fn try_arr_from_iter_trusted<E, I>(iter: I) -> Result<Self, E>
}

impl<T: IntoBytes> ArrayFromIter<Option<T>> for BinaryViewArray {
    #[inline]
    fn arr_from_iter<I: IntoIterator<Item = Option<T>>>(iter: I) -> Self {
        MutableBinaryViewArray::from_iter(
            iter.into_iter().map(|opt_a| opt_a.map(|a| a.into_bytes())),
        )
        .into()
    }

    #[inline]
    fn arr_from_iter_trusted<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = Option<T>>,
        I::IntoIter: TrustedLen,
    {
        Self::arr_from_iter(iter)
    }

    fn try_arr_from_iter<E, I: IntoIterator<Item = Result<Option<T>, E>>>(
        iter: I,
    ) -> Result<Self, E> {
        let mut iter = iter.into_iter();
        let mut arr = MutableBinaryViewArray::with_capacity(iter.size_hint().0);
        iter.try_for_each(|x| -> Result<(), E> {
            let x = x?;
            arr.push(x.map(|x| x.into_bytes()));
            Ok(())
        })?;
        Ok(arr.into())
    }

    // No faster implementation than this available, fall back to default.
    // fn try_arr_from_iter_trusted<E, I>(iter: I) -> Result<Self, E>
}

/// We use this to reuse the binary collect implementation for strings.
/// # Safety
/// The array must be valid UTF-8.
unsafe fn into_utf8array(arr: BinaryArray<i64>) -> Utf8Array<i64> {
    unsafe {
        let (_dt, offsets, values, validity) = arr.into_inner();
        Utf8Array::new_unchecked(ArrowDataType::LargeUtf8, offsets, values, validity)
    }
}

trait StrIntoBytes: IntoBytes {}
impl StrIntoBytes for String {}
impl<'a> StrIntoBytes for &'a str {}
impl<'a> StrIntoBytes for Cow<'a, str> {}

impl<T: StrIntoBytes> ArrayFromIter<T> for Utf8ViewArray {
    #[inline]
    fn arr_from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        unsafe { BinaryViewArray::arr_from_iter(iter).to_utf8view_unchecked() }
    }

    #[inline]
    fn arr_from_iter_trusted<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = T>,
        I::IntoIter: TrustedLen,
    {
        Self::arr_from_iter(iter)
    }

    fn try_arr_from_iter<E, I: IntoIterator<Item = Result<T, E>>>(iter: I) -> Result<Self, E> {
        unsafe { BinaryViewArray::try_arr_from_iter(iter).map(|arr| arr.to_utf8view_unchecked()) }
    }

    // No faster implementation than this available, fall back to default.
    // fn try_arr_from_iter_trusted<E, I>(iter: I) -> Result<Self, E>
}

impl<T: StrIntoBytes> ArrayFromIter<Option<T>> for Utf8ViewArray {
    #[inline]
    fn arr_from_iter<I: IntoIterator<Item = Option<T>>>(iter: I) -> Self {
        unsafe { BinaryViewArray::arr_from_iter(iter).to_utf8view_unchecked() }
    }

    #[inline]
    fn arr_from_iter_trusted<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = Option<T>>,
        I::IntoIter: TrustedLen,
    {
        Self::arr_from_iter(iter)
    }

    fn try_arr_from_iter<E, I: IntoIterator<Item = Result<Option<T>, E>>>(
        iter: I,
    ) -> Result<Self, E> {
        unsafe { BinaryViewArray::try_arr_from_iter(iter).map(|arr| arr.to_utf8view_unchecked()) }
    }

    // No faster implementation than this available, fall back to default.
    // fn try_arr_from_iter_trusted<E, I>(iter: I) -> Result<Self, E>
}

impl<T: StrIntoBytes> ArrayFromIter<T> for Utf8Array<i64> {
    #[inline(always)]
    fn arr_from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        unsafe { into_utf8array(iter.into_iter().collect_arr()) }
    }

    #[inline(always)]
    fn arr_from_iter_trusted<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = T>,
        I::IntoIter: TrustedLen,
    {
        unsafe { into_utf8array(iter.into_iter().collect_arr()) }
    }

    #[inline(always)]
    fn try_arr_from_iter<E, I: IntoIterator<Item = Result<T, E>>>(iter: I) -> Result<Self, E> {
        let arr = iter.into_iter().try_collect_arr()?;
        unsafe { Ok(into_utf8array(arr)) }
    }

    #[inline(always)]
    fn try_arr_from_iter_trusted<E, I: IntoIterator<Item = Result<T, E>>>(
        iter: I,
    ) -> Result<Self, E> {
        let arr = iter.into_iter().try_collect_arr()?;
        unsafe { Ok(into_utf8array(arr)) }
    }
}

impl<T: StrIntoBytes> ArrayFromIter<Option<T>> for Utf8Array<i64> {
    #[inline(always)]
    fn arr_from_iter<I: IntoIterator<Item = Option<T>>>(iter: I) -> Self {
        unsafe { into_utf8array(iter.into_iter().collect_arr()) }
    }

    #[inline(always)]
    fn arr_from_iter_trusted<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = Option<T>>,
        I::IntoIter: TrustedLen,
    {
        unsafe { into_utf8array(iter.into_iter().collect_arr()) }
    }

    #[inline(always)]
    fn try_arr_from_iter<E, I: IntoIterator<Item = Result<Option<T>, E>>>(
        iter: I,
    ) -> Result<Self, E> {
        let arr = iter.into_iter().try_collect_arr()?;
        unsafe { Ok(into_utf8array(arr)) }
    }

    #[inline(always)]
    fn try_arr_from_iter_trusted<E, I: IntoIterator<Item = Result<Option<T>, E>>>(
        iter: I,
    ) -> Result<Self, E> {
        let arr = iter.into_iter().try_collect_arr()?;
        unsafe { Ok(into_utf8array(arr)) }
    }
}

macro_rules! impl_collect_bool_validity {
    ($iter: ident, $x:ident, $unpack:expr, $truth:expr, $nullity:expr, $with_valid:literal) => {{
        let mut iter = $iter.into_iter();
        let mut buf: Vec<u8> = Vec::new();
        let mut validity: Vec<u8> = Vec::new();
        let lo = iter.size_hint().0;
        buf.reserve(8 + 8 * (lo / 64));
        if $with_valid {
            validity.reserve(8 + 8 * (lo / 64));
        }

        let mut len = 0;
        let mut buf_mask = 0u8;
        let mut true_count = 0;
        let mut valid_mask = 0u8;
        let mut nonnull_count = 0;
        'exhausted: loop {
            unsafe {
                for i in 0..8 {
                    let Some($x) = iter.next() else {
                        break 'exhausted;
                    };
                    #[allow(clippy::all)]
                    // #[allow(clippy::redundant_locals)]  Clippy lint too new
                    let $x = $unpack;
                    let is_true: bool = $truth;
                    buf_mask |= (is_true as u8) << i;
                    true_count += is_true as usize;
                    if $with_valid {
                        let nonnull: bool = $nullity;
                        valid_mask |= (nonnull as u8) << i;
                        nonnull_count += nonnull as usize;
                    }
                    len += 1;
                }

                buf.push_unchecked(buf_mask);
                buf_mask = 0;
                if $with_valid {
                    validity.push_unchecked(valid_mask);
                    valid_mask = 0;
                }
            }

            if buf.len() == buf.capacity() {
                buf.reserve(8); // Waste some space to make branch more predictable.
                if $with_valid {
                    validity.reserve(8);
                }
            }
        }

        unsafe {
            // SAFETY: when we broke to 'exhausted we had capacity by the loop invariant.
            // It's also no problem if we make the mask bigger than strictly necessary.
            buf.push_unchecked(buf_mask);
            if $with_valid {
                validity.push_unchecked(valid_mask);
            }
        }

        let false_count = len - true_count;
        let values = unsafe {
            Bitmap::from_inner_unchecked(Arc::new(buf.into()), 0, len, Some(false_count))
        };

        let null_count = len - nonnull_count;
        let validity_bitmap = if $with_valid && null_count > 0 {
            unsafe {
                // SAFETY: we made sure the null_count is correct.
                Some(Bitmap::from_inner_unchecked(
                    Arc::new(validity.into()),
                    0,
                    len,
                    Some(null_count),
                ))
            }
        } else {
            None
        };

        (values, validity_bitmap)
    }};
}

impl ArrayFromIter<bool> for BooleanArray {
    fn arr_from_iter<I: IntoIterator<Item = bool>>(iter: I) -> Self {
        let dt = ArrowDataType::Boolean;
        let (values, _valid) = impl_collect_bool_validity!(iter, x, x, x, false, false);
        BooleanArray::new(dt, values, None)
    }

    // TODO: are efficient trusted collects for booleans worth it?
    // fn arr_from_iter_trusted<I>(iter: I) -> Self

    fn try_arr_from_iter<E, I: IntoIterator<Item = Result<bool, E>>>(iter: I) -> Result<Self, E> {
        let dt = ArrowDataType::Boolean;
        let (values, _valid) = impl_collect_bool_validity!(iter, x, x?, x, false, false);
        Ok(BooleanArray::new(dt, values, None))
    }

    // fn try_arr_from_iter_trusted<E, I: IntoIterator<Item = Result<bool, E>>>(
}

impl ArrayFromIter<Option<bool>> for BooleanArray {
    fn arr_from_iter<I: IntoIterator<Item = Option<bool>>>(iter: I) -> Self {
        let dt = ArrowDataType::Boolean;
        let (values, valid) =
            impl_collect_bool_validity!(iter, x, x, x.unwrap_or(false), x.is_some(), true);
        BooleanArray::new(dt, values, valid)
    }

    // fn arr_from_iter_trusted<I>(iter: I) -> Self

    fn try_arr_from_iter<E, I: IntoIterator<Item = Result<Option<bool>, E>>>(
        iter: I,
    ) -> Result<Self, E> {
        let dt = ArrowDataType::Boolean;
        let (values, valid) =
            impl_collect_bool_validity!(iter, x, x?, x.unwrap_or(false), x.is_some(), true);
        Ok(BooleanArray::new(dt, values, valid))
    }

    // fn try_arr_from_iter_trusted<E, I: IntoIterator<Item = Result<Option<bool>, E>>>(
}

// We don't use AsRef here because it leads to problems with conflicting implementations,
// as Rust considers that AsRef<dyn Array> for Option<&dyn Array> could be implemented.
trait AsArray {
    fn as_array(&self) -> &dyn Array;
    #[cfg(feature = "dtype-array")]
    fn into_boxed_array(self) -> Box<dyn Array>; // Prevents unnecessary re-boxing.
}
impl AsArray for Box<dyn Array> {
    fn as_array(&self) -> &dyn Array {
        self.as_ref()
    }
    #[cfg(feature = "dtype-array")]
    fn into_boxed_array(self) -> Box<dyn Array> {
        self
    }
}
impl<'a> AsArray for &'a dyn Array {
    fn as_array(&self) -> &'a dyn Array {
        *self
    }
    #[cfg(feature = "dtype-array")]
    fn into_boxed_array(self) -> Box<dyn Array> {
        self.to_boxed()
    }
}

// TODO: more efficient (fixed size) list collect routines.
impl<T: AsArray> ArrayFromIterDtype<T> for ListArray<i64> {
    fn arr_from_iter_with_dtype<I: IntoIterator<Item = T>>(dtype: ArrowDataType, iter: I) -> Self {
        let iter_values: Vec<T> = iter.into_iter().collect();
        let mut builder = AnonymousListArrayBuilder::new(iter_values.len());
        for arr in &iter_values {
            builder.push(arr.as_array());
        }
        let inner = dtype
            .inner_dtype()
            .expect("expected nested type in ListArray collect");
        builder
            .finish(Some(&inner.underlying_physical_type()))
            .unwrap()
    }

    fn try_arr_from_iter_with_dtype<E, I: IntoIterator<Item = Result<T, E>>>(
        dtype: ArrowDataType,
        iter: I,
    ) -> Result<Self, E> {
        let iter_values = iter.into_iter().collect::<Result<Vec<_>, E>>()?;
        Ok(Self::arr_from_iter_with_dtype(dtype, iter_values))
    }
}

impl<T: AsArray> ArrayFromIterDtype<Option<T>> for ListArray<i64> {
    fn arr_from_iter_with_dtype<I: IntoIterator<Item = Option<T>>>(
        dtype: ArrowDataType,
        iter: I,
    ) -> Self {
        let iter_values: Vec<Option<T>> = iter.into_iter().collect();
        let mut builder = AnonymousListArrayBuilder::new(iter_values.len());
        for arr in &iter_values {
            builder.push_opt(arr.as_ref().map(|a| a.as_array()));
        }
        let inner = dtype
            .inner_dtype()
            .expect("expected nested type in ListArray collect");
        builder
            .finish(Some(&inner.underlying_physical_type()))
            .unwrap()
    }

    fn try_arr_from_iter_with_dtype<E, I: IntoIterator<Item = Result<Option<T>, E>>>(
        dtype: ArrowDataType,
        iter: I,
    ) -> Result<Self, E> {
        let iter_values = iter.into_iter().collect::<Result<Vec<_>, E>>()?;
        let mut builder = AnonymousListArrayBuilder::new(iter_values.len());
        for arr in &iter_values {
            builder.push_opt(arr.as_ref().map(|a| a.as_array()));
        }
        let inner = dtype
            .inner_dtype()
            .expect("expected nested type in ListArray collect");
        Ok(builder
            .finish(Some(&inner.underlying_physical_type()))
            .unwrap())
    }
}

impl<T: AsArray> ArrayFromIter<Option<T>> for ListArray<i64> {
    fn arr_from_iter<I: IntoIterator<Item = Option<T>>>(iter: I) -> Self {
        let iter = iter.into_iter();
        let iter_values: Vec<Option<T>> = iter.into_iter().collect();
        let mut builder = AnonymousListArrayBuilder::new(iter_values.len());
        for arr in &iter_values {
            builder.push_opt(arr.as_ref().map(|a| a.as_array()));
        }
        builder.finish(None).unwrap()
    }

    fn try_arr_from_iter<E, I: IntoIterator<Item = Result<Option<T>, E>>>(
        iter: I,
    ) -> Result<Self, E> {
        let iter_values = iter.into_iter().collect::<Result<Vec<_>, E>>()?;
        let mut builder = AnonymousListArrayBuilder::new(iter_values.len());
        for arr in &iter_values {
            builder.push_opt(arr.as_ref().map(|a| a.as_array()));
        }
        Ok(builder.finish(None).unwrap())
    }
}

impl ArrayFromIterDtype<Box<dyn Array>> for FixedSizeListArray {
    #[allow(unused_variables)]
    fn arr_from_iter_with_dtype<I: IntoIterator<Item = Box<dyn Array>>>(
        dtype: ArrowDataType,
        iter: I,
    ) -> Self {
        #[cfg(feature = "dtype-array")]
        {
            let ArrowDataType::FixedSizeList(_, width) = &dtype else {
                panic!("FixedSizeListArray::arr_from_iter_with_dtype called with non-Array dtype");
            };
            let iter_values: Vec<_> = iter.into_iter().collect();
            let mut builder = AnonymousFixedSizeListArrayBuilder::new(iter_values.len(), *width);
            for arr in iter_values {
                builder.push(arr.into_boxed_array());
            }
            let inner = dtype
                .inner_dtype()
                .expect("expected nested type in ListArray collect");
            builder
                .finish(Some(&inner.underlying_physical_type()))
                .unwrap()
        }
        #[cfg(not(feature = "dtype-array"))]
        panic!("activate 'dtype-array'")
    }

    fn try_arr_from_iter_with_dtype<E, I: IntoIterator<Item = Result<Box<dyn Array>, E>>>(
        dtype: ArrowDataType,
        iter: I,
    ) -> Result<Self, E> {
        let iter_values = iter.into_iter().collect::<Result<Vec<_>, E>>()?;
        Ok(Self::arr_from_iter_with_dtype(dtype, iter_values))
    }
}

impl ArrayFromIterDtype<Option<Box<dyn Array>>> for FixedSizeListArray {
    #[allow(unused_variables)]
    fn arr_from_iter_with_dtype<I: IntoIterator<Item = Option<Box<dyn Array>>>>(
        dtype: ArrowDataType,
        iter: I,
    ) -> Self {
        #[cfg(feature = "dtype-array")]
        {
            let ArrowDataType::FixedSizeList(_, width) = &dtype else {
                panic!(
                    "FixedSizeListArray::arr_from_iter_with_dtype called with non-FixedSizeList dtype"
                );
            };
            let iter_values: Vec<_> = iter.into_iter().collect();
            let mut builder = AnonymousFixedSizeListArrayBuilder::new(iter_values.len(), *width);
            for arr in iter_values {
                match arr {
                    Some(a) => builder.push(a.into_boxed_array()),
                    None => builder.push_null(),
                }
            }
            let inner = dtype
                .inner_dtype()
                .expect("expected nested type in ListArray collect");
            builder
                .finish(Some(&inner.underlying_physical_type()))
                .unwrap()
        }
        #[cfg(not(feature = "dtype-array"))]
        panic!("activate 'dtype-array'")
    }

    fn try_arr_from_iter_with_dtype<
        E,
        I: IntoIterator<Item = Result<Option<Box<dyn Array>>, E>>,
    >(
        dtype: ArrowDataType,
        iter: I,
    ) -> Result<Self, E> {
        let iter_values = iter.into_iter().collect::<Result<Vec<_>, E>>()?;
        Ok(Self::arr_from_iter_with_dtype(dtype, iter_values))
    }
}

impl ArrayFromIter<Option<()>> for StructArray {
    fn arr_from_iter<I: IntoIterator<Item = Option<()>>>(_iter: I) -> Self {
        no_call_const!()
    }

    fn try_arr_from_iter<E, I: IntoIterator<Item = Result<Option<()>, E>>>(
        _iter: I,
    ) -> Result<Self, E> {
        no_call_const!()
    }
}

impl ArrayFromIter<()> for StructArray {
    fn arr_from_iter<I: IntoIterator<Item = ()>>(_iter: I) -> Self {
        no_call_const!()
    }

    fn try_arr_from_iter<E, I: IntoIterator<Item = Result<(), E>>>(_iter: I) -> Result<Self, E> {
        no_call_const!()
    }
}

impl ArrayFromIterDtype<()> for StructArray {
    fn arr_from_iter_with_dtype<I: IntoIterator<Item = ()>>(
        _dtype: ArrowDataType,
        _iter: I,
    ) -> Self {
        no_call_const!()
    }

    fn try_arr_from_iter_with_dtype<E, I: IntoIterator<Item = Result<(), E>>>(
        _dtype: ArrowDataType,
        _iter: I,
    ) -> Result<Self, E> {
        no_call_const!()
    }
}

impl ArrayFromIterDtype<Option<()>> for StructArray {
    fn arr_from_iter_with_dtype<I: IntoIterator<Item = Option<()>>>(
        _dtype: ArrowDataType,
        _iter: I,
    ) -> Self {
        no_call_const!()
    }

    fn try_arr_from_iter_with_dtype<E, I: IntoIterator<Item = Result<Option<()>, E>>>(
        _dtype: ArrowDataType,
        _iter: I,
    ) -> Result<Self, E> {
        no_call_const!()
    }
}
