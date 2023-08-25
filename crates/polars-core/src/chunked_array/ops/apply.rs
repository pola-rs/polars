//! Implementations of the ChunkApply Trait.
use std::borrow::Cow;
use std::convert::TryFrom;
use std::error::Error;

use arrow::array::{BooleanArray, PrimitiveArray};
use arrow::bitmap::utils::{get_bit_unchecked, set_bit_unchecked};
use arrow::bitmap::Bitmap;
use arrow::trusted_len::TrustedLen;
use arrow::types::NativeType;
use polars_arrow::bitmap::unary_mut;

use crate::prelude::*;
use crate::series::IsSorted;
use crate::utils::CustomIterTools;

impl<T> ChunkedArray<T>
where
    T: PolarsDataType,
    Self: HasUnderlyingArray,
{
    pub fn apply_values_generic<'a, U, K, F>(&'a self, op: F) -> ChunkedArray<U>
    where
        U: PolarsDataType,
        F: FnMut(<<Self as HasUnderlyingArray>::ArrayT as StaticArray>::ValueT<'a>) -> K + Copy,
        K: ArrayFromElementIter,
        K::ArrayType: StaticallyMatchesPolarsType<U>,
    {
        let iter = self.downcast_iter().map(|arr| {
            let element_iter = arr.values_iter().map(op);
            let array = K::array_from_values_iter(element_iter);
            array.with_validity_typed(arr.validity().cloned())
        });

        ChunkedArray::from_chunk_iter(self.name(), iter)
    }

    pub fn try_apply_values_generic<'a, U, K, F, E>(&'a self, op: F) -> Result<ChunkedArray<U>, E>
    where
        U: PolarsDataType,
        F: FnMut(<<Self as HasUnderlyingArray>::ArrayT as StaticArray>::ValueT<'a>) -> Result<K, E>
            + Copy,
        K: ArrayFromElementIter,
        K::ArrayType: StaticallyMatchesPolarsType<U>,
        E: Error,
    {
        let iter = self.downcast_iter().map(|arr| {
            let element_iter = arr.values_iter().map(op);
            let array = K::try_array_from_values_iter(element_iter)?;
            Ok(array.with_validity_typed(arr.validity().cloned()))
        });

        ChunkedArray::try_from_chunk_iter(self.name(), iter)
    }

    pub fn try_apply_generic<'a, U, K, F, E>(&'a self, op: F) -> Result<ChunkedArray<U>, E>
    where
        U: PolarsDataType,
        F: FnMut(
                Option<<<Self as HasUnderlyingArray>::ArrayT as StaticArray>::ValueT<'a>>,
            ) -> Result<Option<K>, E>
            + Copy,
        K: ArrayFromElementIter,
        K::ArrayType: StaticallyMatchesPolarsType<U>,
        E: Error,
    {
        let iter = self.downcast_iter().map(|arr| {
            let element_iter = arr.iter().map(op);
            let array = K::try_array_from_iter(element_iter)?;
            Ok(array.with_validity_typed(arr.validity().cloned()))
        });

        ChunkedArray::try_from_chunk_iter(self.name(), iter)
    }

    pub fn apply_generic<'a, U, K, F>(&'a self, op: F) -> ChunkedArray<U>
    where
        U: PolarsDataType,
        F: FnMut(
                Option<<<Self as HasUnderlyingArray>::ArrayT as StaticArray>::ValueT<'a>>,
            ) -> Option<K>
            + Copy,
        K: ArrayFromElementIter,
        K::ArrayType: StaticallyMatchesPolarsType<U>,
    {
        let iter = self.downcast_iter().map(|arr| {
            let element_iter = arr.iter().map(op);
            K::array_from_iter(element_iter)
        });

        ChunkedArray::from_chunk_iter(self.name(), iter)
    }
}

fn collect_array<T: NativeType, I: TrustedLen<Item = T>>(
    iter: I,
    validity: Option<Bitmap>,
) -> PrimitiveArray<T> {
    PrimitiveArray::from_trusted_len_values_iter(iter).with_validity(validity)
}

macro_rules! try_apply {
    ($self:expr, $f:expr) => {{
        if !$self.has_validity() {
            $self.into_no_null_iter().map($f).collect()
        } else {
            $self
                .into_iter()
                .map(|opt_v| opt_v.map($f).transpose())
                .collect()
        }
    }};
}

macro_rules! apply {
    ($self:expr, $f:expr) => {{
        if !$self.has_validity() {
            $self.into_no_null_iter().map($f).collect_trusted()
        } else {
            $self
                .into_iter()
                .map(|opt_v| opt_v.map($f))
                .collect_trusted()
        }
    }};
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
            arrow::compute::arity::unary(arr, f, S::get_dtype().to_arrow())
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
                collect_array(slice.iter().copied().map(f), validity.cloned())
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

impl Utf8Chunked {
    pub fn apply_mut<'a, F>(&'a self, mut f: F) -> Self
    where
        F: FnMut(&'a str) -> &'a str,
    {
        use polars_arrow::array::utf8::Utf8FromIter;
        let chunks = self.downcast_iter().map(|arr| {
            let iter = arr.values_iter().map(&mut f);
            let value_size = (arr.get_values_size() as f64 * 1.3) as usize;
            let new = Utf8Array::<i64>::from_values_iter(iter, arr.len(), value_size);
            new.with_validity(arr.validity().cloned())
        });
        Utf8Chunked::from_chunk_iter(self.name(), chunks)
    }
}

impl BinaryChunked {
    pub fn apply_mut<'a, F>(&'a self, mut f: F) -> Self
    where
        F: FnMut(&'a [u8]) -> &'a [u8],
    {
        use polars_arrow::array::utf8::BinaryFromIter;
        let chunks = self.downcast_iter().map(|arr| {
            let iter = arr.values_iter().map(&mut f);
            let value_size = (arr.get_values_size() as f64 * 1.3) as usize;
            let new = BinaryArray::<i64>::from_values_iter(iter, arr.len(), value_size);
            new.with_validity(arr.validity().cloned())
        });
        BinaryChunked::from_chunk_iter(self.name(), chunks)
    }
}

impl<'a> ChunkApply<'a, &'a str> for Utf8Chunked {
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

impl ChunkApplyKernel<LargeStringArray> for Utf8Chunked {
    fn apply_kernel(&self, f: &dyn Fn(&LargeStringArray) -> ArrayRef) -> Self {
        self.apply_kernel_cast(&f)
    }

    fn apply_kernel_cast<S>(&self, f: &dyn Fn(&LargeStringArray) -> ArrayRef) -> ChunkedArray<S>
    where
        S: PolarsDataType,
    {
        let chunks = self.downcast_iter().map(f).collect();
        unsafe { ChunkedArray::from_chunks(self.name(), chunks) }
    }
}

impl ChunkApplyKernel<LargeBinaryArray> for BinaryChunked {
    fn apply_kernel(&self, f: &dyn Fn(&LargeBinaryArray) -> ArrayRef) -> Self {
        self.apply_kernel_cast(&f)
    }

    fn apply_kernel_cast<S>(&self, f: &dyn Fn(&LargeBinaryArray) -> ArrayRef) -> ChunkedArray<S>
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
        let mut ca: ListChunked = apply!(self, &mut function);
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
        let ca: PolarsResult<ListChunked> = try_apply!(self, &mut function);
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
