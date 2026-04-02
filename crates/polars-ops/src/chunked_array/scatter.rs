#![allow(unsafe_op_in_unsafe_fn)]
use arrow::array::{Array, BinaryViewArrayGeneric, BooleanArray, PrimitiveArray, View, ViewType};
use polars_buffer::Buffer;
use polars_core::prelude::*;
use polars_core::series::IsSorted;
use polars_core::utils::arrow::bitmap::MutableBitmap;
use polars_core::utils::arrow::types::NativeType;
use polars_utils::index::check_bounds;

pub trait ChunkedSet<T: Copy> {
    /// Invariant for implementations: if the scatter() fails, typically because
    /// of bad indexes, then self should remain unmodified.
    fn scatter<V>(self, idx: &[IdxSize], values: V) -> PolarsResult<Series>
    where
        V: IntoIterator<Item = Option<T>>;
}

trait PolarsOpsNumericType: PolarsNumericType {}

impl PolarsOpsNumericType for UInt8Type {}
impl PolarsOpsNumericType for UInt16Type {}
impl PolarsOpsNumericType for UInt32Type {}
impl PolarsOpsNumericType for UInt64Type {}
#[cfg(feature = "dtype-u128")]
impl PolarsOpsNumericType for UInt128Type {}
impl PolarsOpsNumericType for Int8Type {}
impl PolarsOpsNumericType for Int16Type {}
impl PolarsOpsNumericType for Int32Type {}
impl PolarsOpsNumericType for Int64Type {}
#[cfg(feature = "dtype-i128")]
impl PolarsOpsNumericType for Int128Type {}
#[cfg(feature = "dtype-f16")]
impl PolarsOpsNumericType for Float16Type {}
impl PolarsOpsNumericType for Float32Type {}
impl PolarsOpsNumericType for Float64Type {}

unsafe fn scatter_primitive_impl<V, T: NativeType>(
    set_values: V,
    arr: &mut PrimitiveArray<T>,
    idx: &[IdxSize],
) where
    V: IntoIterator<Item = Option<T>>,
{
    let mut values_iter = set_values.into_iter();

    if let Some(validity) = arr.take_validity() {
        let mut mut_validity = validity.make_mut();
        arr.with_values_mut(|cur_values| {
            for (idx, val) in idx.iter().zip(&mut values_iter) {
                match val {
                    Some(value) => {
                        mut_validity.set_unchecked(*idx as usize, true);
                        *cur_values.get_unchecked_mut(*idx as usize) = value
                    },
                    None => mut_validity.set_unchecked(*idx as usize, false),
                }
            }
        });
        arr.set_validity(mut_validity.into())
    } else {
        let mut null_idx = vec![];
        arr.with_values_mut(|cur_values| {
            for (idx, val) in idx.iter().zip(values_iter) {
                match val {
                    Some(value) => *cur_values.get_unchecked_mut(*idx as usize) = value,
                    None => {
                        null_idx.push(*idx);
                    },
                }
            }
        });

        // Only make a validity bitmap when null values are set.
        if !null_idx.is_empty() {
            let mut validity = MutableBitmap::with_capacity(arr.len());
            validity.extend_constant(arr.len(), true);
            for idx in null_idx {
                validity.set_unchecked(idx as usize, false)
            }
            arr.set_validity(Some(validity.into()))
        }
    }
}

unsafe fn scatter_bool_impl<V>(set_values: V, arr: &mut BooleanArray, idx: &[IdxSize])
where
    V: IntoIterator<Item = Option<bool>>,
{
    let mut values_iter = set_values.into_iter();

    if let Some(validity) = arr.take_validity() {
        let mut mut_validity = validity.make_mut();
        arr.apply_values_mut(|cur_values| {
            for (idx, val) in idx.iter().zip(&mut values_iter) {
                match val {
                    Some(value) => {
                        mut_validity.set_unchecked(*idx as usize, true);
                        cur_values.set_unchecked(*idx as usize, value);
                    },
                    None => mut_validity.set_unchecked(*idx as usize, false),
                }
            }
        });
        arr.set_validity(mut_validity.into())
    } else {
        let mut null_idx = vec![];
        arr.apply_values_mut(|cur_values| {
            for (idx, val) in idx.iter().zip(values_iter) {
                match val {
                    Some(value) => cur_values.set_unchecked(*idx as usize, value),
                    None => {
                        null_idx.push(*idx);
                    },
                }
            }
        });

        // Only make a validity bitmap when null values are set.
        if !null_idx.is_empty() {
            let mut validity = MutableBitmap::with_capacity(arr.len());
            validity.extend_constant(arr.len(), true);
            for idx in null_idx {
                validity.set_unchecked(idx as usize, false)
            }
            arr.set_validity(Some(validity.into()))
        }
    }
}

unsafe fn scatter_binview_impl<'a, V, T: ViewType + ?Sized>(
    set_values: V,
    arr: &mut BinaryViewArrayGeneric<T>,
    idx: &[IdxSize],
) where
    V: IntoIterator<Item = Option<&'a T>>,
{
    let mut values_iter = set_values.into_iter();
    let buffer_offset = arr.data_buffers().len() as u32;
    let mut new_buffers = Vec::new();

    if let Some(validity) = arr.take_validity() {
        let mut mut_validity = validity.make_mut();
        arr.with_views_mut(|views| {
            for (idx, val) in idx.iter().zip(&mut values_iter) {
                if let Some(v) = val {
                    let view =
                        View::new_with_buffers(v.to_bytes(), buffer_offset, &mut new_buffers);
                    *views.get_unchecked_mut(*idx as usize) = view;
                    mut_validity.set_unchecked(*idx as usize, true);
                } else {
                    mut_validity.set_unchecked(*idx as usize, false);
                }
            }
        });
        arr.set_validity(mut_validity.into())
    } else {
        let mut null_idx = vec![];
        arr.with_views_mut(|views| {
            for (idx, val) in idx.iter().zip(values_iter) {
                if let Some(v) = val {
                    let view =
                        View::new_with_buffers(v.to_bytes(), buffer_offset, &mut new_buffers);
                    *views.get_unchecked_mut(*idx as usize) = view;
                } else {
                    null_idx.push(*idx);
                }
            }
        });

        // Only make a validity bitmap when null values are set.
        if !null_idx.is_empty() {
            let mut validity = MutableBitmap::with_capacity(arr.len());
            validity.extend_constant(arr.len(), true);
            for idx in null_idx {
                validity.set_unchecked(idx as usize, false)
            }
            arr.set_validity(Some(validity.into()))
        }
    }

    let mut buffers = Buffer::to_vec(core::mem::take(arr.data_buffers_mut()));
    buffers.extend(new_buffers.into_iter().map(Buffer::from));
    *arr.data_buffers_mut() = Buffer::from(buffers);
}

impl<T: PolarsOpsNumericType> ChunkedSet<T::Native> for &mut ChunkedArray<T> {
    fn scatter<V>(self, idx: &[IdxSize], values: V) -> PolarsResult<Series>
    where
        V: IntoIterator<Item = Option<T::Native>>,
    {
        check_bounds(idx, self.len() as IdxSize)?;
        let mut ca = std::mem::take(self);

        // SAFETY: we will not modify the length and we unset the sorted flag,
        // making sure to update the null count as well.
        unsafe {
            ca.rechunk_mut();
            let arr = ca.downcast_iter_mut().next().unwrap();
            scatter_primitive_impl(values, arr, idx);
            let null_count = arr.null_count();
            ca.set_sorted_flag(IsSorted::Not);
            ca.set_null_count(null_count);
        }

        Ok(ca.into_series())
    }
}

impl<'a> ChunkedSet<&'a [u8]> for &mut BinaryChunked {
    fn scatter<V>(self, idx: &[IdxSize], values: V) -> PolarsResult<Series>
    where
        V: IntoIterator<Item = Option<&'a [u8]>>,
    {
        check_bounds(idx, self.len() as IdxSize)?;
        let mut ca = std::mem::take(self);

        unsafe {
            ca.rechunk_mut();
            let arr = ca.downcast_iter_mut().next().unwrap();
            scatter_binview_impl(values, arr, idx);
            let null_count = arr.null_count();
            ca.set_sorted_flag(IsSorted::Not);
            ca.set_null_count(null_count);
        }

        Ok(ca.into_series())
    }
}

impl<'a> ChunkedSet<&'a str> for &mut StringChunked {
    fn scatter<V>(self, idx: &[IdxSize], values: V) -> PolarsResult<Series>
    where
        V: IntoIterator<Item = Option<&'a str>>,
    {
        check_bounds(idx, self.len() as IdxSize)?;
        let mut ca = std::mem::take(self);

        unsafe {
            ca.rechunk_mut();
            let arr = ca.downcast_iter_mut().next().unwrap();
            scatter_binview_impl(values, arr, idx);
            let null_count = arr.null_count();
            ca.set_sorted_flag(IsSorted::Not);
            ca.set_null_count(null_count);
        }

        Ok(ca.into_series())
    }
}
impl ChunkedSet<bool> for &mut BooleanChunked {
    fn scatter<V>(self, idx: &[IdxSize], values: V) -> PolarsResult<Series>
    where
        V: IntoIterator<Item = Option<bool>>,
    {
        check_bounds(idx, self.len() as IdxSize)?;
        let mut ca = std::mem::take(self);

        unsafe {
            ca.rechunk_mut();
            let arr = ca.downcast_iter_mut().next().unwrap();
            scatter_bool_impl(values, arr, idx);
            let null_count = arr.null_count();
            ca.set_sorted_flag(IsSorted::Not);
            ca.set_null_count(null_count);
        }

        Ok(ca.into_series())
    }
}
