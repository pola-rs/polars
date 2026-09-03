//! Implementations of the ChunkApply Trait.
#![allow(unsafe_op_in_unsafe_fn)]
use std::borrow::Cow;

use polars_array::arrow::bridge::chunk_to_arrow;
use polars_array::arrow::import;

use crate::chunked_array::arity::{unary_elementwise, unary_elementwise_values};
use crate::chunked_array::cast::CastOptions;
use crate::prelude::*;
use crate::series::IsSorted;

impl<T> ChunkedArray<T>
where
    T: PolarsDataType,
{
    /// Applies a function only to the non-null elements, propagating nulls.
    pub fn apply_nonnull_values_generic<'a, U, K, F>(
        &'a self,
        // The chunks carry no logical type — the arrays of `polars-array` are physical storage
        // only — so the collect below builds them without one.
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
                let out: U::Array = arr.values_iter().map(&mut op).collect_arr();
                out.with_validity_broadcast_typed(arr.validity().map(|v| v.to_flat_or_scalar()))
            } else {
                let out: U::Array = arr.iter().map(|opt| opt.map(&mut op)).collect_arr();
                out.with_validity_broadcast_typed(arr.validity().map(|v| v.to_flat_or_scalar()))
            }
        });

        ChunkedArray::from_chunk_iter(self.name().clone(), iter)
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
                out.with_validity_broadcast_typed(arr.validity().map(|v| v.to_flat_or_scalar()))
            } else {
                let out: U::Array = arr
                    .iter()
                    .map(|opt| opt.map(&mut op).transpose())
                    .try_collect_arr()?;
                out.with_validity_broadcast_typed(arr.validity().map(|v| v.to_flat_or_scalar()))
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
                let mut mutarr = PlUtf8ViewArrayBuilder::with_capacity(arr.len());
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
                let mut mutarr = PlUtf8ViewArrayBuilder::with_capacity(arr.len());
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
    use arrow::Either::*;
    let chunks = chunks.into_iter().map(|arr| {
        // TODO(polars-array-scalar): the crossing to Arrow writes a scalar chunk out, where `f`
        // could be applied to the single value it stands for in `O(1)`.
        let owned_arr = chunk_to_arrow(
            arr.as_any()
                .downcast_ref::<PlPrimitiveArray<S::Native>>()
                .unwrap(),
        );
        // Make sure we have a single ref count coming in.
        drop(arr);

        let compute_immutable = |arr: &PrimitiveArray<S::Native>| {
            arrow::compute::arity::unary(
                arr,
                f,
                S::get_static_dtype().to_arrow(CompatLevel::newest()),
            )
        };

        let out = if owned_arr.values().is_sliced() {
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
        };
        import::primitive_from_arrow(&out)
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
                // Each chunk is mapped in whatever representation it is in: a scalar values
                // buffer holds one value standing for every element, which stays a single value.
                if let Some(values) = arr.flat_values_mut() {
                    match values.get_mut_slice() {
                        Some(slice) => slice.iter_mut().for_each(|v| *v = f(*v)),
                        // The buffer is shared with another array, so it cannot be written over.
                        None => {
                            *values = values
                                .as_slice()
                                .iter()
                                .copied()
                                .map(f)
                                .collect::<Vec<_>>()
                                .into()
                        },
                    }
                } else {
                    let value = arr
                        .scalar_values()
                        .expect("values buffer should be one of flat or scalar");
                    let validity = arr.validity().map(|v| v.to_flat_or_scalar());
                    *arr = PlPrimitiveArray::new_scalar(f(value), arr.len())
                        .with_validity_broadcast(validity);
                }
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
        // The values are read as slices, so the chunks are written out flat first where they are
        // not already.
        let ca = self.to_flat();
        let chunks =
            ca.data_views()
                .zip(ca.as_array().iter_validities())
                .map(|(slice, validity)| {
                    let arr: T::Array = slice.iter().copied().map(f).collect_arr();
                    arr.with_validity_broadcast_typed(validity.map(|v| v.to_flat_or_scalar()))
                });
        ChunkedArray::from_chunk_iter(self.name().clone(), chunks)
    }

    fn apply<F>(&'a self, f: F) -> Self
    where
        F: Fn(Option<T::Native>) -> Option<T::Native> + Copy,
    {
        let chunks = self.downcast_iter().map(|arr| {
            let out: T::Array = arr.iter().map(f).collect_arr();
            out
        });
        Self::from_chunk_iter(self.name().clone(), chunks)
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
        // Can just fully deduce behavior from two invocations. A chunk of one value repeated is
        // scalar, so the two constant branches are `O(1)` in memory.
        let constant = |value: bool| {
            let chunks = self
                .downcast_iter()
                .map(|arr| {
                    PlBooleanArray::new_scalar(value, arr.len())
                        .with_validity_broadcast(arr.validity().map(|v| v.to_flat_or_scalar()))
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
            let iter = arr.values_iter().map(&mut f);
            let new = PlUtf8ViewArray::arr_from_iter(iter);
            new.with_validity_broadcast(arr.validity().map(|v| v.to_flat_or_scalar()))
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
            let iter = arr.values_iter().map(&mut f);
            let new = PlBinaryViewArray::arr_from_iter(iter);
            new.with_validity_broadcast(arr.validity().map(|v| v.to_flat_or_scalar()))
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

impl ChunkApplyKernel<BooleanArray> for BooleanChunked {
    fn apply_kernel(&self, f: &dyn Fn(&BooleanArray) -> ArrayRef) -> Self {
        self.apply_kernel_cast(f)
    }

    fn apply_kernel_cast<S>(&self, f: &dyn Fn(&BooleanArray) -> ArrayRef) -> ChunkedArray<S>
    where
        S: PolarsDataType,
    {
        let chunks = apply_arrow_kernel(self, f);
        unsafe { ChunkedArray::<S>::from_chunks(self.name().clone(), chunks) }
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
        let chunks = apply_arrow_kernel(self, f);
        unsafe { ChunkedArray::from_chunks(self.name().clone(), chunks) }
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
        let chunks = apply_arrow_kernel(self, f);
        unsafe { ChunkedArray::from_chunks(self.name().clone(), chunks) }
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
        let chunks = apply_arrow_kernel(self, f);
        unsafe { ChunkedArray::from_chunks(self.name().clone(), chunks) }
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

        // The chunks carry no logical type, so the inner dtype is taken from this array.
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

/// Runs an Arrow kernel over every chunk of `ca`, handing the results back as chunks. A chunk in
/// the [`scalar`](polars_array::broadcast) representation is written out by the crossing.
fn apply_arrow_kernel<T: PolarsDataType>(
    ca: &ChunkedArray<T>,
    f: &dyn Fn(&<T::Array as ToArrow>::Arrow) -> ArrayRef,
) -> Vec<PlArrayRef>
where
    T::Array: ToArrow,
{
    ca.downcast_iter()
        .map(|arr| import::from_arrow(&*f(&chunk_to_arrow(arr))))
        .collect()
}
