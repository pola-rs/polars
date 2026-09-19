//! Implementations of the ChunkApply Trait.
#![allow(unsafe_op_in_unsafe_fn)]
use std::borrow::Cow;

use polars_buffer::Buffer;

use crate::chunked_array::arity::{unary_elementwise, unary_elementwise_values};
use crate::chunked_array::cast::CastOptions;
use crate::prelude::*;
use crate::series::IsSorted;

/// The answer of `op` over a chunk whose values repeat one value under a mask that does not.
#[inline(never)]
fn scalar_values_under_mask<'a, A, Arr, K, F>(arr: &'a A, op: &mut F) -> Option<Arr>
where
    A: StaticArray,
    Arr: StaticArray + ArrayFromIter<K>,
    F: FnMut(A::ValueT<'a>) -> K,
{
    let length = arr.len();
    if arr.null_count() == length {
        return None;
    }

    let value = arr.scalar_value_ignore_validity()?;
    let single: Arr = std::iter::once(op(value)).collect_arr();

    Some(
        single
            .new_from_index_typed(0, length)
            .with_validity_typed(arr.validity().map(PlBitmap::from)),
    )
}

/// [`scalar_values_under_mask`] for an `op` that may fail.
#[inline(never)]
fn try_scalar_values_under_mask<'a, A, Arr, K, E, F>(
    arr: &'a A,
    op: &mut F,
) -> Result<Option<Arr>, E>
where
    A: StaticArray,
    Arr: StaticArray + ArrayFromIter<K>,
    F: FnMut(A::ValueT<'a>) -> Result<K, E>,
{
    let length = arr.len();
    if arr.null_count() == length {
        return Ok(None);
    }

    let Some(value) = arr.scalar_value_ignore_validity() else {
        return Ok(None);
    };
    let single: Arr = std::iter::once(op(value)?).collect_arr();

    Ok(Some(
        single
            .new_from_index_typed(0, length)
            .with_validity_typed(arr.validity().map(PlBitmap::from)),
    ))
}

/// [`scalar_values_under_mask`] for an `f` that writes its answer into a buffer.
#[inline(never)]
fn scalar_string_under_mask<'a, A, F>(
    arr: &'a A,
    buf: &mut String,
    f: &mut F,
) -> Option<PlUtf8ViewArray>
where
    A: StaticArray,
    F: FnMut(A::ValueT<'a>, &mut String),
{
    let length = arr.len();
    if arr.null_count() == length {
        return None;
    }

    let value = arr.scalar_value_ignore_validity()?;
    buf.clear();
    f(value, buf);

    Some(PlUtf8ViewArray::new_scalar(buf, length).with_validity(arr.validity().map(PlBitmap::from)))
}

/// [`scalar_string_under_mask`] for an `f` that may fail.
#[inline(never)]
fn try_scalar_string_under_mask<'a, A, E, F>(
    arr: &'a A,
    buf: &mut String,
    f: &mut F,
) -> Result<Option<PlUtf8ViewArray>, E>
where
    A: StaticArray,
    F: FnMut(A::ValueT<'a>, &mut String) -> Result<(), E>,
{
    let length = arr.len();
    if arr.null_count() == length {
        return Ok(None);
    }

    let Some(value) = arr.scalar_value_ignore_validity() else {
        return Ok(None);
    };
    buf.clear();
    f(value, buf)?;

    Ok(Some(
        PlUtf8ViewArray::new_scalar(buf, length).with_validity(arr.validity().map(PlBitmap::from)),
    ))
}

impl<T> ChunkedArray<T>
where
    T: PolarsDataType,
{
    /// Applies a function only to the non-null elements, propagating nulls.
    pub fn apply_nonnull_values_generic<'a, U, K, F>(
        &'a self,
        _dtype: DataType,
        op: F,
    ) -> ChunkedArray<U>
    where
        U: PolarsDataType,
        F: Fn(T::Physical<'a>) -> K,
        U::Array: ArrayFromIter<K> + ArrayFromIter<Option<K>>,
    {
        let iter = self.downcast_iter().map(|arr| {
            let length = arr.len();
            if length > 1 {
                if let Some(Some(value)) = arr.scalar_value() {
                    let single: U::Array = std::iter::once(op(value)).collect_arr();
                    return single.new_from_index_typed(0, length);
                }

                if !arr.is_flat()
                    && let Some(single) = scalar_values_under_mask(arr, &mut &op)
                {
                    return single;
                }
            }

            if arr.null_count() == 0 {
                let out: U::Array = arr.values_iter().map(&op).collect_arr_trusted();
                out.with_validity_typed(arr.validity().map(PlBitmap::from))
            } else {
                let out: U::Array = arr.iter().map(|opt| opt.map(&op)).collect_arr_trusted();
                out.with_validity_typed(arr.validity().map(PlBitmap::from))
            }
        });

        ChunkedArray::from_chunk_iter(self.name().clone(), iter)
    }

    /// [`Self::apply_nonnull_values_generic`] for an `op` that carries state between elements.
    pub fn apply_nonnull_values_generic_mut<'a, U, K, F>(
        &'a self,
        _dtype: DataType,
        mut op: F,
    ) -> ChunkedArray<U>
    where
        U: PolarsDataType,
        F: FnMut(T::Physical<'a>) -> K,
        U::Array: ArrayFromIter<K> + ArrayFromIter<Option<K>>,
    {
        let iter = self.downcast_iter().map(|arr| {
            if arr.null_count() == 0 {
                let out: U::Array = arr.values_iter().map(&mut op).collect_arr_trusted();
                out.with_validity_typed(arr.validity().map(PlBitmap::from))
            } else {
                let out: U::Array = arr.iter().map(|opt| opt.map(&mut op)).collect_arr_trusted();
                out.with_validity_typed(arr.validity().map(PlBitmap::from))
            }
        });

        ChunkedArray::from_chunk_iter(self.name().clone(), iter)
    }

    /// Applies a function only to the non-null elements, propagating nulls.
    pub fn try_apply_nonnull_values_generic<'a, U, K, F, E>(
        &'a self,
        op: F,
    ) -> Result<ChunkedArray<U>, E>
    where
        U: PolarsDataType,
        F: Fn(T::Physical<'a>) -> Result<K, E>,
        U::Array: ArrayFromIter<K> + ArrayFromIter<Option<K>>,
    {
        let iter = self.downcast_iter().map(|arr| {
            let length = arr.len();
            if length > 1 {
                if let Some(Some(value)) = arr.scalar_value() {
                    let single: U::Array = std::iter::once(op(value)?).collect_arr();
                    return Ok(single.new_from_index_typed(0, length));
                }

                if !arr.is_flat()
                    && let Some(single) = try_scalar_values_under_mask(arr, &mut &op)?
                {
                    return Ok(single);
                }
            }

            let arr = if arr.null_count() == 0 {
                let out: U::Array = arr.values_iter().map(&op).try_collect_arr_trusted()?;
                out.with_validity_typed(arr.validity().map(PlBitmap::from))
            } else {
                let out: U::Array = arr
                    .iter()
                    .map(|opt| opt.map(&op).transpose())
                    .try_collect_arr_trusted()?;
                out.with_validity_typed(arr.validity().map(PlBitmap::from))
            };
            Ok(arr)
        });

        ChunkedArray::try_from_chunk_iter(self.name().clone(), iter)
    }

    pub fn apply_into_string_amortized<'a, F>(&'a self, mut f: F) -> StringChunked
    where
        F: FnMut(T::Physical<'a>, &mut String),
    {
        let mut buf = String::new();
        let chunks = self
            .downcast_iter()
            .map(|arr| {
                let length = arr.len();
                if length > 1 {
                    if let Some(element) = arr.scalar_value() {
                        return match element {
                            None => PlUtf8ViewArray::new_full_null(length),
                            Some(v) => {
                                buf.clear();
                                f(v, &mut buf);
                                PlUtf8ViewArray::new_scalar(&buf, length)
                            },
                        };
                    }

                    if !arr.is_flat()
                        && let Some(single) = scalar_string_under_mask(arr, &mut buf, &mut f)
                    {
                        return single;
                    }
                }

                let mut mutarr = PlUtf8ViewArrayBuilder::with_capacity(length);
                arr.iter().for_each(|opt| match opt {
                    None => mutarr.push_null(),
                    Some(v) => {
                        buf.clear();
                        f(v, &mut buf);
                        mutarr.push_value(&buf)
                    },
                });
                mutarr.freeze()
            })
            .collect::<Vec<_>>();
        ChunkedArray::from_chunk_iter(self.name().clone(), chunks)
    }

    pub fn try_apply_into_string_amortized<'a, F, E>(&'a self, mut f: F) -> Result<StringChunked, E>
    where
        F: FnMut(T::Physical<'a>, &mut String) -> Result<(), E>,
    {
        let mut buf = String::new();
        let chunks = self
            .downcast_iter()
            .map(|arr| {
                let length = arr.len();
                if length > 1 {
                    if let Some(element) = arr.scalar_value() {
                        return match element {
                            None => Ok(PlUtf8ViewArray::new_full_null(length)),
                            Some(v) => {
                                buf.clear();
                                f(v, &mut buf)?;
                                Ok(PlUtf8ViewArray::new_scalar(&buf, length))
                            },
                        };
                    }

                    if !arr.is_flat()
                        && let Some(single) = try_scalar_string_under_mask(arr, &mut buf, &mut f)?
                    {
                        return Ok(single);
                    }
                }

                let mut mutarr = PlUtf8ViewArrayBuilder::with_capacity(length);
                for opt in arr.iter() {
                    match opt {
                        None => mutarr.push_null(),
                        Some(v) => {
                            buf.clear();
                            f(v, &mut buf)?;
                            mutarr.push_value(&buf)
                        },
                    };
                }
                Ok(mutarr.freeze())
            })
            .collect::<Vec<_>>();
        ChunkedArray::try_from_chunk_iter(self.name().clone(), chunks)
    }
}

fn apply_in_place_impl<S, F>(name: PlSmallStr, chunks: Vec<PlArrayRef>, f: F) -> ChunkedArray<S>
where
    F: Fn(S::Native) -> S::Native + Copy,
    S: PolarsNumericType,
{
    let chunks = chunks.into_iter().map(|arr| {
        let typed = arr
            .as_any()
            .downcast_ref::<PlPrimitiveArray<S::Native>>()
            .unwrap();

        if let Some(value) = typed.scalar_value_ignore_validity() {
            let validity = typed.validity().map(PlBitmap::from);
            return PlPrimitiveArray::new_scalar(f(value), typed.len()).with_validity(validity);
        }

        let mut owned = typed.clone();
        drop(arr);

        let values = owned
            .flat_values_mut()
            .expect("the chunk is flat: a scalar one returned above");
        match values.get_mut_slice() {
            Some(slice) => {
                for value in slice {
                    *value = f(*value);
                }
            },
            None => {
                let mapped: Vec<_> = values.as_slice().iter().map(|value| f(*value)).collect();
                *values = Buffer::from(mapped);
            },
        }
        owned
    });

    ChunkedArray::from_chunk_iter(name, chunks)
}

impl<T: PolarsNumericType> ChunkedArray<T> {
    /// Cast a numeric array to another numeric data type and apply a function in place.
    /// This saves an allocation.
    pub fn cast_and_apply_in_place<F, S>(&self, f: F) -> ChunkedArray<S>
    where
        F: Fn(S::Native) -> S::Native + Copy,
        S: PolarsNumericType,
    {
        // if we cast, we create a new arrow buffer
        // then we clone the arrays and drop the cast arrays
        // this will ensure we have a single ref count
        // and we can mutate in place
        let chunks = {
            let s = self
                .cast_with_options(&S::get_static_dtype(), CastOptions::Overflowing)
                .unwrap();
            s.chunks().clone()
        };
        apply_in_place_impl(self.name().clone(), chunks, f)
    }

    /// Cast a numeric array to another numeric data type and apply a function in place.
    /// This saves an allocation.
    pub fn apply_in_place<F>(mut self, f: F) -> Self
    where
        F: Fn(T::Native) -> T::Native + Copy,
    {
        let chunks = std::mem::take(&mut self.chunks);
        apply_in_place_impl(self.name().clone(), chunks, f)
    }
}

impl<T: PolarsNumericType> ChunkedArray<T> {
    pub fn apply_mut<F>(&mut self, f: F)
    where
        F: Fn(T::Native) -> T::Native + Copy,
    {
        // SAFETY, we do no t change the lengths
        unsafe {
            self.downcast_iter_mut().for_each(|arr| {
                if let Some(slots) = arr.flat_or_scalar_values_mut() {
                    slots.iter_mut().for_each(|v| *v = f(*v));
                    return;
                }

                let length = arr.len();
                let validity = arr.validity().map(PlBitmap::from);
                let mapped = match arr.scalar_value_ignore_validity() {
                    Some(value) => PlPrimitiveArray::new_scalar(f(value), length),
                    None => PlPrimitiveArray::from_vec(arr.values_iter().map(f).collect()),
                };
                *arr = mapped.with_validity(validity);
            })
        };
        // can be in any order now
        self.compute_len();
        self.set_sorted_flag(IsSorted::Not);
    }
}

impl<'a, T> ChunkApply<'a, T::Native> for ChunkedArray<T>
where
    T: PolarsNumericType,
{
    type FuncRet = T::Native;

    fn apply_values<F>(&'a self, f: F) -> Self
    where
        F: Fn(T::Native) -> T::Native + Copy,
    {
        let chunks = self.downcast_iter().map(|arr| {
            let validity = arr.validity().map(PlBitmap::from);
            if let Some(value) = arr.scalar_value_ignore_validity() {
                return PlPrimitiveArray::new_scalar(f(value), arr.len())
                    .with_validity_typed(validity);
            }

            let flat = arr.to_flat();
            let out: T::Array = flat.as_slice().iter().copied().map(f).collect_arr();
            out.with_validity_typed(validity)
        });
        ChunkedArray::from_chunk_iter(self.name().clone(), chunks)
    }

    fn apply<F>(&'a self, f: F) -> Self
    where
        F: Fn(Option<T::Native>) -> Option<T::Native> + Copy,
    {
        unary_elementwise(self, f)
    }

    fn apply_to_slice<F, V>(&'a self, f: F, slice: &mut [V])
    where
        F: Fn(Option<T::Native>, &V) -> V,
    {
        assert!(slice.len() >= self.len());

        let mut idx = 0;
        self.downcast_iter().for_each(|arr| {
            arr.into_iter().for_each(|opt_val| {
                // SAFETY:
                // length asserted above
                let item = unsafe { slice.get_unchecked_mut(idx) };
                *item = f(opt_val, item);
                idx += 1;
            })
        });
    }
}

impl<'a> ChunkApply<'a, bool> for BooleanChunked {
    type FuncRet = bool;

    fn apply_values<F>(&self, f: F) -> Self
    where
        F: Fn(bool) -> bool + Copy,
    {
        let constant = |value: bool| {
            let chunks = self
                .downcast_iter()
                .map(|arr| {
                    PlBooleanArray::new_scalar(value, arr.len())
                        .with_validity(arr.validity().map(PlBitmap::from))
                })
                .collect::<Vec<_>>();
            Self::from_chunk_iter(self.name().clone(), chunks)
        };
        match (f(false), f(true)) {
            (false, false) => constant(false),
            (false, true) => self.clone(),
            (true, false) => !self,
            (true, true) => constant(true),
        }
    }

    fn apply<F>(&'a self, f: F) -> Self
    where
        F: Fn(Option<bool>) -> Option<bool> + Copy,
    {
        unary_elementwise(self, f)
    }

    fn apply_to_slice<F, T>(&'a self, f: F, slice: &mut [T])
    where
        F: Fn(Option<bool>, &T) -> T,
    {
        assert!(slice.len() >= self.len());

        let mut idx = 0;
        self.downcast_iter().for_each(|arr| {
            arr.into_iter().for_each(|opt_val| {
                // SAFETY:
                // length asserted above
                let item = unsafe { slice.get_unchecked_mut(idx) };
                *item = f(opt_val, item);
                idx += 1;
            })
        });
    }
}

impl StringChunked {
    pub fn apply_mut<'a, F>(&'a self, mut f: F) -> Self
    where
        F: FnMut(&'a str) -> &'a str,
    {
        let chunks = self.downcast_iter().map(|arr| {
            let length = arr.len();
            if length > 1 {
                if let Some(value) = arr.scalar_value_ignore_validity() {
                    return PlUtf8ViewArray::new_scalar(f(value), length)
                        .with_validity(arr.validity().map(PlBitmap::from));
                }
            }

            let iter = arr.values_iter().map(&mut f);
            let new = PlUtf8ViewArray::arr_from_iter(iter);
            new.with_validity(arr.validity().map(PlBitmap::from))
        });
        StringChunked::from_chunk_iter(self.name().clone(), chunks)
    }
}

impl BinaryChunked {
    pub fn apply_mut<'a, F>(&'a self, mut f: F) -> Self
    where
        F: FnMut(&'a [u8]) -> &'a [u8],
    {
        let chunks = self.downcast_iter().map(|arr| {
            let length = arr.len();
            if length > 1 {
                if let Some(value) = arr.scalar_value_ignore_validity() {
                    return PlBinaryViewArray::new_scalar(f(value), length)
                        .with_validity(arr.validity().map(PlBitmap::from));
                }
            }

            let iter = arr.values_iter().map(&mut f);
            let new = PlBinaryViewArray::arr_from_iter(iter);
            new.with_validity(arr.validity().map(PlBitmap::from))
        });
        BinaryChunked::from_chunk_iter(self.name().clone(), chunks)
    }
}

impl<'a> ChunkApply<'a, &'a str> for StringChunked {
    type FuncRet = Cow<'a, str>;

    fn apply_values<F>(&'a self, f: F) -> Self
    where
        F: Fn(&'a str) -> Cow<'a, str> + Copy,
    {
        unary_elementwise_values(self, f)
    }

    fn apply<F>(&'a self, f: F) -> Self
    where
        F: Fn(Option<&'a str>) -> Option<Cow<'a, str>> + Copy,
    {
        unary_elementwise(self, f)
    }

    fn apply_to_slice<F, T>(&'a self, f: F, slice: &mut [T])
    where
        F: Fn(Option<&'a str>, &T) -> T,
    {
        assert!(slice.len() >= self.len());

        let mut idx = 0;
        self.downcast_iter().for_each(|arr| {
            arr.into_iter().for_each(|opt_val| {
                // SAFETY:
                // length asserted above
                let item = unsafe { slice.get_unchecked_mut(idx) };
                *item = f(opt_val, item);
                idx += 1;
            })
        });
    }
}

impl<'a> ChunkApply<'a, &'a [u8]> for BinaryChunked {
    type FuncRet = Cow<'a, [u8]>;

    fn apply_values<F>(&'a self, f: F) -> Self
    where
        F: Fn(&'a [u8]) -> Cow<'a, [u8]> + Copy,
    {
        unary_elementwise_values(self, f)
    }

    fn apply<F>(&'a self, f: F) -> Self
    where
        F: Fn(Option<&'a [u8]>) -> Option<Cow<'a, [u8]>> + Copy,
    {
        unary_elementwise(self, f)
    }

    fn apply_to_slice<F, T>(&'a self, f: F, slice: &mut [T])
    where
        F: Fn(Option<&'a [u8]>, &T) -> T,
    {
        assert!(slice.len() >= self.len());

        let mut idx = 0;
        self.downcast_iter().for_each(|arr| {
            arr.into_iter().for_each(|opt_val| {
                // SAFETY:
                // length asserted above
                let item = unsafe { slice.get_unchecked_mut(idx) };
                *item = f(opt_val, item);
                idx += 1;
            })
        });
    }
}

impl<'a> ChunkApply<'a, Series> for ListChunked {
    type FuncRet = Series;

    /// Apply a closure `F` elementwise.
    fn apply_values<F>(&'a self, f: F) -> Self
    where
        F: Fn(Series) -> Series + Copy,
    {
        if self.is_empty() {
            return self.clone();
        }
        let mut fast_explode = true;
        let mut function = |s: Series| {
            let out = f(s);
            if out.is_empty() {
                fast_explode = false;
            }
            out
        };
        let mut ca: ListChunked = {
            self.series_iter()
                .map(|opt_v| opt_v.map(&mut function))
                .collect_trusted()
        };
        if fast_explode {
            ca.set_fast_explode()
        }
        ca
    }

    fn apply<F>(&'a self, f: F) -> Self
    where
        F: Fn(Option<Series>) -> Option<Series> + Copy,
    {
        if self.is_empty() {
            return self.clone();
        }
        self.series_iter().map(f).collect_trusted()
    }

    fn apply_to_slice<F, T>(&'a self, f: F, slice: &mut [T])
    where
        F: Fn(Option<Series>, &T) -> T,
    {
        assert!(slice.len() >= self.len());

        let inner_dtype = self.inner_dtype().to_physical();
        let mut idx = 0;
        self.downcast_iter().for_each(|arr| {
            arr.iter().for_each(|opt_val| {
                let opt_val = opt_val.map(|values| unsafe {
                    Series::from_chunks_and_dtype_unchecked(
                        PlSmallStr::EMPTY,
                        vec![values],
                        &inner_dtype,
                    )
                });

                // SAFETY:
                // length asserted above
                let item = unsafe { slice.get_unchecked_mut(idx) };
                *item = f(opt_val, item);
                idx += 1;
            })
        });
    }
}

#[cfg(feature = "object")]
impl<'a, T> ChunkApply<'a, &'a T> for ObjectChunked<T>
where
    T: PolarsObject,
{
    type FuncRet = T;

    fn apply_values<F>(&'a self, f: F) -> Self
    where
        F: Fn(&'a T) -> T + Copy,
    {
        let mut ca: ObjectChunked<T> = self.iter().map(|opt_v| opt_v.map(f)).collect();
        ca.rename(self.name().clone());
        ca
    }

    fn apply<F>(&'a self, f: F) -> Self
    where
        F: Fn(Option<&'a T>) -> Option<T> + Copy,
    {
        let mut ca: ObjectChunked<T> = self.iter().map(f).collect();
        ca.rename(self.name().clone());
        ca
    }

    fn apply_to_slice<F, V>(&'a self, f: F, slice: &mut [V])
    where
        F: Fn(Option<&'a T>, &V) -> V,
    {
        assert!(slice.len() >= self.len());
        let mut idx = 0;
        self.downcast_iter().for_each(|arr| {
            arr.into_iter().for_each(|opt_val| {
                // SAFETY:
                // length asserted above
                let item = unsafe { slice.get_unchecked_mut(idx) };
                *item = f(opt_val, item);
                idx += 1;
            })
        });
    }
}

impl StringChunked {
    /// # Safety
    /// Update the views. All invariants of the views apply.
    pub unsafe fn apply_views<F: FnMut(View, &str) -> View + Copy>(&self, update_view: F) -> Self {
        let mut out = self.clone();
        for arr in out.downcast_iter_mut() {
            *arr = arr.apply_views(update_view);
        }
        out
    }
}
