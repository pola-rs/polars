//! Implementations of the ChunkApply Trait.
use std::borrow::Cow;
use std::convert::TryFrom;

use arrow::array::{BooleanArray, PrimitiveArray};
use arrow::bitmap::utils::{get_bit_unchecked, set_bit_unchecked};
use arrow::legacy::bitmap::unary_mut;

use crate::prelude::*;
use crate::series::IsSorted;
use crate::utils::CustomIterTools;

impl<T> ChunkedArray<T>
where
    T: PolarsDataType,
{
    // Applies a function to all elements, regardless of whether they
    // are null or not, after which the null mask is copied from the
    // original array.
    pub fn apply_values_generic<'a, U, K, F>(&'a self, mut op: F) -> ChunkedArray<U>
    where
        U: PolarsDataType,
        F: FnMut(T::Physical<'a>) -> K,
        U::Array: ArrayFromIter<K>,
    {
        let iter = self.downcast_iter().map(|arr| {
            let out: U::Array = arr.values_iter().map(&mut op).collect_arr();
            out.with_validity_typed(arr.validity().cloned())
        });

        ChunkedArray::from_chunk_iter(self.name(), iter)
    }

    /// Applies a function to all elements, regardless of whether they
    /// are null or not, after which the null mask is copied from the
    /// original array.
    pub fn try_apply_values_generic<'a, U, K, F, E>(
        &'a self,
        mut op: F,
    ) -> Result<ChunkedArray<U>, E>
    where
        U: PolarsDataType,
        F: FnMut(T::Physical<'a>) -> Result<K, E>,
        U::Array: ArrayFromIter<K>,
    {
        let iter = self.downcast_iter().map(|arr| {
            let element_iter = arr.values_iter().map(&mut op);
            let array: U::Array = element_iter.try_collect_arr()?;
            Ok(array.with_validity_typed(arr.validity().cloned()))
        });

        ChunkedArray::try_from_chunk_iter(self.name(), iter)
    }

    /// Applies a function only to the non-null elements, propagating nulls.
    pub fn apply_nonnull_values_generic<'a, U, K, F>(
        &'a self,
        dtype: DataType,
        mut op: F,
    ) -> ChunkedArray<U>
    where
        U: PolarsDataType,
        F: FnMut(T::Physical<'a>) -> K,
        U::Array: ArrayFromIterDtype<K> + ArrayFromIterDtype<Option<K>>,
    {
        let iter = self.downcast_iter().map(|arr| {
            if arr.null_count() == 0 {
                let out: U::Array = arr
                    .values_iter()
                    .map(&mut op)
                    .collect_arr_with_dtype(dtype.to_arrow(true));
                out.with_validity_typed(arr.validity().cloned())
            } else {
                let out: U::Array = arr
                    .iter()
                    .map(|opt| opt.map(&mut op))
                    .collect_arr_with_dtype(dtype.to_arrow(true));
                out.with_validity_typed(arr.validity().cloned())
            }
        });

        ChunkedArray::from_chunk_iter(self.name(), iter)
    }

    /// Applies a function only to the non-null elements, propagating nulls.
    pub fn try_apply_nonnull_values_generic<'a, U, K, F, E>(
        &'a self,
        mut op: F,
    ) -> Result<ChunkedArray<U>, E>
    where
        U: PolarsDataType,
        F: FnMut(T::Physical<'a>) -> Result<K, E>,
        U::Array: ArrayFromIter<K> + ArrayFromIter<Option<K>>,
    {
        let iter = self.downcast_iter().map(|arr| {
            let arr = if arr.null_count() == 0 {
                let out: U::Array = arr.values_iter().map(&mut op).try_collect_arr()?;
                out.with_validity_typed(arr.validity().cloned())
            } else {
                let out: U::Array = arr
                    .iter()
                    .map(|opt| opt.map(&mut op).transpose())
                    .try_collect_arr()?;
                out.with_validity_typed(arr.validity().cloned())
            };
            Ok(arr)
        });

        ChunkedArray::try_from_chunk_iter(self.name(), iter)
    }

    pub fn apply_generic<'a, U, K, F>(&'a self, mut op: F) -> ChunkedArray<U>
    where
        U: PolarsDataType,
        F: FnMut(Option<T::Physical<'a>>) -> Option<K>,
        U::Array: ArrayFromIter<Option<K>>,
    {
        if self.null_count() == 0 {
            let iter = self
                .downcast_iter()
                .map(|arr| arr.values_iter().map(|x| op(Some(x))).collect_arr());
            ChunkedArray::from_chunk_iter(self.name(), iter)
        } else {
            let iter = self
                .downcast_iter()
                .map(|arr| arr.iter().map(&mut op).collect_arr());
            ChunkedArray::from_chunk_iter(self.name(), iter)
        }
    }

    pub fn try_apply_generic<'a, U, K, F, E>(&'a self, op: F) -> Result<ChunkedArray<U>, E>
    where
        U: PolarsDataType,
        F: FnMut(Option<T::Physical<'a>>) -> Result<Option<K>, E> + Copy,
        U::Array: ArrayFromIter<Option<K>>,
    {
        let iter = self.downcast_iter().map(|arr| {
            let array: U::Array = arr.iter().map(op).try_collect_arr()?;
            Ok(array.with_validity_typed(arr.validity().cloned()))
        });

        ChunkedArray::try_from_chunk_iter(self.name(), iter)
    }
}

fn apply_in_place_impl<S, F>(name: &str, chunks: Vec<ArrayRef>, f: F) -> ChunkedArray<S>
where
    F: Fn(S::Native) -> S::Native + Copy,
    S: PolarsNumericType,
{
    use arrow::Either::*;
    let chunks = chunks.into_iter().map(|arr| {
        let owned_arr = arr
            .as_any()
            .downcast_ref::<PrimitiveArray<S::Native>>()
            .unwrap()
            .clone();
        // Make sure we have a single ref count coming in.
        drop(arr);

        let compute_immutable = |arr: &PrimitiveArray<S::Native>| {
            arrow::compute::arity::unary(arr, f, S::get_dtype().to_arrow(true))
        };

        if owned_arr.values().is_sliced() {
            compute_immutable(&owned_arr)
        } else {
            match owned_arr.into_mut() {
                Left(immutable) => compute_immutable(&immutable),
                Right(mut mutable) => {
                    let vals = mutable.values_mut_slice();
                    vals.iter_mut().for_each(|v| *v = f(*v));
                    mutable.into()
                },
            }
        }
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
        // then we clone the arrays and drop the casted arrays
        // this will ensure we have a single ref count
        // and we can mutate in place
        let chunks = {
            let s = self.cast(&S::get_dtype()).unwrap();
            s.chunks().clone()
        };
        apply_in_place_impl(self.name(), chunks, f)
    }

    /// Cast a numeric array to another numeric data type and apply a function in place.
    /// This saves an allocation.
    pub fn apply_in_place<F>(mut self, f: F) -> Self
    where
        F: Fn(T::Native) -> T::Native + Copy,
    {
        let chunks = std::mem::take(&mut self.chunks);
        apply_in_place_impl(self.name(), chunks, f)
    }
}

impl<T: PolarsNumericType> ChunkedArray<T> {
    pub fn apply_mut<F>(&mut self, f: F)
    where
        F: Fn(T::Native) -> T::Native + Copy,
    {
        // safety, we do no t change the lengths
        unsafe {
            self.downcast_iter_mut()
                .for_each(|arr| arrow::compute::arity_assign::unary(arr, f))
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
        let chunks = self
            .data_views()
            .zip(self.iter_validities())
            .map(|(slice, validity)| {
                let arr: T::Array = slice.iter().copied().map(f).collect_arr();
                arr.with_validity(validity.cloned())
            });
        ChunkedArray::from_chunk_iter(self.name(), chunks)
    }

    fn try_apply<F>(&'a self, f: F) -> PolarsResult<Self>
    where
        F: Fn(T::Native) -> PolarsResult<T::Native> + Copy,
    {
        let mut ca: ChunkedArray<T> = self
            .data_views()
            .zip(self.iter_validities())
            .map(|(slice, validity)| {
                let vec: PolarsResult<Vec<_>> = slice.iter().copied().map(f).collect();
                Ok((vec?, validity.cloned()))
            })
            .collect::<PolarsResult<_>>()?;
        ca.rename(self.name());
        Ok(ca)
    }

    fn apply<F>(&'a self, f: F) -> Self
    where
        F: Fn(Option<T::Native>) -> Option<T::Native> + Copy,
    {
        let chunks = self.downcast_iter().map(|arr| {
            let iter = arr.into_iter().map(|opt_v| f(opt_v.copied()));
            PrimitiveArray::<T::Native>::from_trusted_len_iter(iter)
        });
        Self::from_chunk_iter(self.name(), chunks)
    }

    fn apply_to_slice<F, V>(&'a self, f: F, slice: &mut [V])
    where
        F: Fn(Option<T::Native>, &V) -> V,
    {
        assert!(slice.len() >= self.len());

        let mut idx = 0;
        self.downcast_iter().for_each(|arr| {
            arr.into_iter().for_each(|opt_val| {
                // Safety:
                // length asserted above
                let item = unsafe { slice.get_unchecked_mut(idx) };
                *item = f(opt_val.copied(), item);
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
        self.apply_kernel(&|arr| {
            let values = arrow::bitmap::unary(arr.values(), |chunk| {
                let bytes = chunk.to_ne_bytes();

                // different output as that might lead
                // to better internal parallelism
                let mut out = 0u64.to_ne_bytes();
                for i in 0..64 {
                    unsafe {
                        let val = get_bit_unchecked(&bytes, i);
                        let res = f(val);
                        set_bit_unchecked(&mut out, i, res)
                    };
                }
                u64::from_ne_bytes(out)
            });
            BooleanArray::from_data_default(values, arr.validity().cloned()).boxed()
        })
    }

    fn try_apply<F>(&self, f: F) -> PolarsResult<Self>
    where
        F: Fn(bool) -> PolarsResult<bool> + Copy,
    {
        let mut failed: Option<PolarsError> = None;
        let chunks = self.downcast_iter().map(|arr| {
            let values = unary_mut(arr.values(), |chunk| {
                let bytes = chunk.to_ne_bytes();

                if failed.is_some() {
                    0
                } else {
                    let mut out = 0u64.to_ne_bytes();
                    // We reverse the order of the loop so we keep the first error, if any.
                    for i in (0..64).rev() {
                        unsafe {
                            let val = get_bit_unchecked(&bytes, i);
                            match f(val) {
                                Ok(res) => set_bit_unchecked(&mut out, i, res),
                                Err(e) => failed = Some(e),
                            }
                        };
                    }
                    u64::from_ne_bytes(out)
                }
            });

            BooleanArray::from_data_default(values, arr.validity().cloned())
        });

        let ret = BooleanChunked::from_chunk_iter(self.name(), chunks);
        if let Some(e) = failed {
            return Err(e);
        }
        Ok(ret)
    }

    fn apply<F>(&'a self, f: F) -> Self
    where
        F: Fn(Option<bool>) -> Option<bool> + Copy,
    {
        self.apply_generic(f)
    }

    fn apply_to_slice<F, T>(&'a self, f: F, slice: &mut [T])
    where
        F: Fn(Option<bool>, &T) -> T,
    {
        assert!(slice.len() >= self.len());

        let mut idx = 0;
        self.downcast_iter().for_each(|arr| {
            arr.into_iter().for_each(|opt_val| {
                // Safety:
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
            let iter = arr.values_iter().map(&mut f);
            let new = Utf8ViewArray::arr_from_iter(iter);
            new.with_validity(arr.validity().cloned())
        });
        StringChunked::from_chunk_iter(self.name(), chunks)
    }

    /// Utility that reuses an string buffer to amortize allocations.
    /// Prefer this over an `apply` that returns an owned `String`.
    pub fn apply_to_buffer<'a, F>(&'a self, mut f: F) -> Self
    where
        F: FnMut(&'a str, &mut String),
    {
        let mut buf = String::new();
        let outer = |s: &'a str| {
            buf.clear();
            f(s, &mut buf);
            unsafe { std::mem::transmute::<&str, &'a str>(buf.as_str()) }
        };
        self.apply_mut(outer)
    }
}

impl BinaryChunked {
    pub fn apply_mut<'a, F>(&'a self, mut f: F) -> Self
    where
        F: FnMut(&'a [u8]) -> &'a [u8],
    {
        let chunks = self.downcast_iter().map(|arr| {
            let iter = arr.values_iter().map(&mut f);
            let new = BinaryViewArray::arr_from_iter(iter);
            new.with_validity(arr.validity().cloned())
        });
        BinaryChunked::from_chunk_iter(self.name(), chunks)
    }
}

impl<'a> ChunkApply<'a, &'a str> for StringChunked {
    type FuncRet = Cow<'a, str>;

    fn apply_values<F>(&'a self, f: F) -> Self
    where
        F: Fn(&'a str) -> Cow<'a, str> + Copy,
    {
        ChunkedArray::apply_values_generic(self, f)
    }

    fn try_apply<F>(&'a self, f: F) -> PolarsResult<Self>
    where
        F: Fn(&'a str) -> PolarsResult<Cow<'a, str>> + Copy,
    {
        self.try_apply_values_generic(f)
    }

    fn apply<F>(&'a self, f: F) -> Self
    where
        F: Fn(Option<&'a str>) -> Option<Cow<'a, str>> + Copy,
    {
        self.apply_generic(f)
    }

    fn apply_to_slice<F, T>(&'a self, f: F, slice: &mut [T])
    where
        F: Fn(Option<&'a str>, &T) -> T,
    {
        assert!(slice.len() >= self.len());

        let mut idx = 0;
        self.downcast_iter().for_each(|arr| {
            arr.into_iter().for_each(|opt_val| {
                // Safety:
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
        self.apply_values_generic(f)
    }

    fn try_apply<F>(&'a self, f: F) -> PolarsResult<Self>
    where
        F: Fn(&'a [u8]) -> PolarsResult<Cow<'a, [u8]>> + Copy,
    {
        self.try_apply_values_generic(f)
    }

    fn apply<F>(&'a self, f: F) -> Self
    where
        F: Fn(Option<&'a [u8]>) -> Option<Cow<'a, [u8]>> + Copy,
    {
        self.apply_generic(f)
    }

    fn apply_to_slice<F, T>(&'a self, f: F, slice: &mut [T])
    where
        F: Fn(Option<&'a [u8]>, &T) -> T,
    {
        assert!(slice.len() >= self.len());

        let mut idx = 0;
        self.downcast_iter().for_each(|arr| {
            arr.into_iter().for_each(|opt_val| {
                // Safety:
                // length asserted above
                let item = unsafe { slice.get_unchecked_mut(idx) };
                *item = f(opt_val, item);
                idx += 1;
            })
        });
    }
}

impl ChunkApplyKernel<BooleanArray> for BooleanChunked {
    fn apply_kernel(&self, f: &dyn Fn(&BooleanArray) -> ArrayRef) -> Self {
        let chunks = self.downcast_iter().map(f).collect();
        unsafe { Self::from_chunks(self.name(), chunks) }
    }

    fn apply_kernel_cast<S>(&self, f: &dyn Fn(&BooleanArray) -> ArrayRef) -> ChunkedArray<S>
    where
        S: PolarsDataType,
    {
        let chunks = self.downcast_iter().map(f).collect();
        unsafe { ChunkedArray::<S>::from_chunks(self.name(), chunks) }
    }
}

impl<T> ChunkApplyKernel<PrimitiveArray<T::Native>> for ChunkedArray<T>
where
    T: PolarsNumericType,
{
    fn apply_kernel(&self, f: &dyn Fn(&PrimitiveArray<T::Native>) -> ArrayRef) -> Self {
        self.apply_kernel_cast(&f)
    }
    fn apply_kernel_cast<S>(
        &self,
        f: &dyn Fn(&PrimitiveArray<T::Native>) -> ArrayRef,
    ) -> ChunkedArray<S>
    where
        S: PolarsDataType,
    {
        let chunks = self.downcast_iter().map(f).collect();
        unsafe { ChunkedArray::from_chunks(self.name(), chunks) }
    }
}

impl ChunkApplyKernel<Utf8ViewArray> for StringChunked {
    fn apply_kernel(&self, f: &dyn Fn(&Utf8ViewArray) -> ArrayRef) -> Self {
        self.apply_kernel_cast(&f)
    }

    fn apply_kernel_cast<S>(&self, f: &dyn Fn(&Utf8ViewArray) -> ArrayRef) -> ChunkedArray<S>
    where
        S: PolarsDataType,
    {
        let chunks = self.downcast_iter().map(f).collect();
        unsafe { ChunkedArray::from_chunks(self.name(), chunks) }
    }
}

impl ChunkApplyKernel<BinaryViewArray> for BinaryChunked {
    fn apply_kernel(&self, f: &dyn Fn(&BinaryViewArray) -> ArrayRef) -> Self {
        self.apply_kernel_cast(&f)
    }

    fn apply_kernel_cast<S>(&self, f: &dyn Fn(&BinaryViewArray) -> ArrayRef) -> ChunkedArray<S>
    where
        S: PolarsDataType,
    {
        let chunks = self.downcast_iter().map(f).collect();
        unsafe { ChunkedArray::from_chunks(self.name(), chunks) }
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
            if !self.has_validity() {
                self.into_no_null_iter()
                    .map(&mut function)
                    .collect_trusted()
            } else {
                self.into_iter()
                    .map(|opt_v| opt_v.map(&mut function))
                    .collect_trusted()
            }
        };
        if fast_explode {
            ca.set_fast_explode()
        }
        ca
    }

    fn try_apply<F>(&'a self, f: F) -> PolarsResult<Self>
    where
        F: Fn(Series) -> PolarsResult<Series> + Copy,
    {
        if self.is_empty() {
            return Ok(self.clone());
        }

        let mut fast_explode = true;
        let mut function = |s: Series| {
            let out = f(s);
            if let Ok(out) = &out {
                if out.is_empty() {
                    fast_explode = false;
                }
            }
            out
        };
        let ca: PolarsResult<ListChunked> = {
            if !self.has_validity() {
                self.into_no_null_iter().map(&mut function).collect()
            } else {
                self.into_iter()
                    .map(|opt_v| opt_v.map(&mut function).transpose())
                    .collect()
            }
        };
        let mut ca = ca?;
        if fast_explode {
            ca.set_fast_explode()
        }
        Ok(ca)
    }

    fn apply<F>(&'a self, f: F) -> Self
    where
        F: Fn(Option<Series>) -> Option<Series> + Copy,
    {
        if self.is_empty() {
            return self.clone();
        }
        self.into_iter().map(f).collect_trusted()
    }

    fn apply_to_slice<F, T>(&'a self, f: F, slice: &mut [T])
    where
        F: Fn(Option<Series>, &T) -> T,
    {
        assert!(slice.len() >= self.len());

        let mut idx = 0;
        self.downcast_iter().for_each(|arr| {
            arr.iter().for_each(|opt_val| {
                let opt_val = opt_val.map(|arrayref| Series::try_from(("", arrayref)).unwrap());

                // Safety:
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
        let mut ca: ObjectChunked<T> = self.into_iter().map(|opt_v| opt_v.map(f)).collect();
        ca.rename(self.name());
        ca
    }

    fn try_apply<F>(&'a self, _f: F) -> PolarsResult<Self>
    where
        F: Fn(&'a T) -> PolarsResult<T> + Copy,
    {
        todo!()
    }

    fn apply<F>(&'a self, f: F) -> Self
    where
        F: Fn(Option<&'a T>) -> Option<T> + Copy,
    {
        let mut ca: ObjectChunked<T> = self.into_iter().map(f).collect();
        ca.rename(self.name());
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
                // Safety:
                // length asserted above
                let item = unsafe { slice.get_unchecked_mut(idx) };
                *item = f(opt_val, item);
                idx += 1;
            })
        });
    }
}
