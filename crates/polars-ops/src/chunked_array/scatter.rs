#![allow(unsafe_op_in_unsafe_fn)]
use arrow::array::View;
use polars_buffer::Buffer;
use polars_core::prelude::*;
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

/// Writes into the values of `arr` where it can, and copies them out first where it cannot.
unsafe fn with_values_mut<T: NativeType, F: FnOnce(&mut [T])>(arr: &mut PlPrimitiveArray<T>, f: F) {
    let length = arr.len();
    let Some(values) = arr.flat_values_mut() else {
        // A scalar chunk holds one slot standing for every element, so it is written out before
        // anything can be written into it — as a buffer of its own, since the array still holds
        // the one slot it was written out of. `to_flat` is no help here: an array of a *single*
        // element reads as scalar however it was built, so writing it out leaves it reading that
        // way too.
        let mut owned = match arr.scalar_value_ignore_validity() {
            Some(value) => vec![value; length],
            // Values that are neither flat nor scalar are no values at all.
            None => Vec::new(),
        };
        f(&mut owned);
        let validity = arr.validity().map(PlBitmap::from);
        // SAFETY: the buffer written out holds one slot per element, and the mask is the one the
        // array already carried.
        *arr = unsafe { PlPrimitiveArray::new_unchecked(Buffer::from(owned), length, validity) };
        return;
    };

    match values.get_mut_slice() {
        Some(slice) => f(slice),
        None => {
            // Something else reads these values, so they are copied before being written.
            let mut owned = values.as_slice().to_vec();
            f(&mut owned);
            *values = Buffer::from(owned);
        },
    }
}

unsafe fn scatter_primitive_impl<V, T: NativeType>(
    set_values: V,
    arr: &mut PlPrimitiveArray<T>,
    idx: &[IdxSize],
) where
    V: IntoIterator<Item = Option<T>>,
{
    let mut values_iter = set_values.into_iter();
    let length = arr.len();

    if let Some(validity) = arr.validity() {
        // A scalar mask stands for one bit per element, which `to_flat` resolves.
        let mut mut_validity = validity.to_flat().into_owned().make_mut();
        with_values_mut(arr, |cur_values| {
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
        arr.set_validity(Some(PlBitmap::from_bitmap(mut_validity.into())))
    } else {
        let mut null_idx = vec![];
        with_values_mut(arr, |cur_values| {
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
            let mut validity = MutableBitmap::with_capacity(length);
            validity.extend_constant(length, true);
            for idx in null_idx {
                validity.set_unchecked(idx as usize, false)
            }
            arr.set_validity(Some(PlBitmap::from_bitmap(validity.into())))
        }
    }
}

/// [`with_values_mut`] for booleans: writes into the values, copying them out where it cannot.
fn with_bool_values_mut<F: FnOnce(&mut MutableBitmap)>(arr: &mut PlBooleanArray, f: F) {
    let length = arr.len();

    let values = match arr.flat_values_mut() {
        Some(values) => std::mem::take(values),
        None => {
            // A scalar chunk holds one bit standing for every element, so it is written out before
            // anything can be written into it — as a bitmap of its own, for the reason given in
            // `with_values_mut`.
            let mut values = MutableBitmap::new();
            if let Some(value) = arr.scalar_value_ignore_validity() {
                values.extend_constant(length, value);
            }
            f(&mut values);
            let validity = arr.validity().map(PlBitmap::from);
            // SAFETY: the bitmap written out holds one bit per element — `f` may only write over
            // the bits, not add or drop any — and the mask is the one the array already carried.
            *arr = unsafe { PlBooleanArray::new_unchecked(values.into(), length, validity) };
            return;
        },
    };

    let mut values = values.make_mut();
    f(&mut values);
    assert_eq!(
        values.len(),
        length,
        "writing into the values of an array cannot change how many elements it has",
    );
    // The slot the values were taken out of is the one they go back into: nothing between the two
    // reads the array.
    *arr.flat_values_mut().unwrap() = values.into();
}

unsafe fn scatter_bool_impl<V>(set_values: V, arr: &mut PlBooleanArray, idx: &[IdxSize])
where
    V: IntoIterator<Item = Option<bool>>,
{
    let mut values_iter = set_values.into_iter();
    let length = arr.len();

    if let Some(validity) = arr.validity() {
        // A scalar mask stands for one bit per element, which `to_flat` resolves.
        let mut mut_validity = validity.to_flat().into_owned().make_mut();
        with_bool_values_mut(arr, |cur_values| {
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
        arr.set_validity(Some(PlBitmap::from_bitmap(mut_validity.into())))
    } else {
        let mut null_idx = vec![];
        with_bool_values_mut(arr, |cur_values| {
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
            let mut validity = MutableBitmap::with_capacity(length);
            validity.extend_constant(length, true);
            for idx in null_idx {
                validity.set_unchecked(idx as usize, false)
            }
            arr.set_validity(Some(PlBitmap::from_bitmap(validity.into())))
        }
    }
}

/// [`with_values_mut`] for views: writes into the views, copying them out where it cannot.
///
/// # Safety
/// Every view `f` leaves behind must read bytes that the array's buffers hold, or ones it pushed
/// onto the buffers it was handed.
unsafe fn with_views_mut<F>(arr: &mut PlBinaryViewArray, f: F)
where
    F: FnOnce(&mut [View], u32, &mut Vec<Vec<u8>>),
{
    let length = arr.len();
    let buffer_offset = arr.data_buffers().len() as u32;
    let mut new_buffers: Vec<Vec<u8>> = Vec::new();

    if arr.views_are_scalar() {
        // A scalar chunk holds one view standing for every element, so it is written out before
        // anything can be written into it. Written out as a buffer of its own, for the reason
        // given in `with_values_mut`.
        let mut owned = match arr.scalar_views() {
            Some(view) => vec![view; length],
            // Views that are neither flat nor scalar are no views at all.
            None => Vec::new(),
        };
        let validity = arr.validity().map(PlBitmap::from);
        let mut buffers = Buffer::to_vec(core::mem::take(unsafe { arr.data_buffers_mut() }));

        f(&mut owned, buffer_offset, &mut new_buffers);
        buffers.extend(new_buffers.into_iter().map(Buffer::from));

        // SAFETY: the buffer written out holds one view per element, and every view reads bytes
        // the buffers hold — the ones the array came with, or the ones just appended.
        *arr = unsafe {
            PlBinaryViewArray::new_unchecked(
                Buffer::from(owned),
                Buffer::from(buffers),
                length,
                validity,
            )
        };
        return;
    }

    {
        // SAFETY: the caller owes that `f` leaves every view reading the bytes the buffers hold
        // once the ones it appends are in them, which is what happens below.
        let views = unsafe { arr.flat_views_mut() }.unwrap();
        match views.get_mut_slice() {
            Some(slice) => f(slice, buffer_offset, &mut new_buffers),
            None => {
                // Something else reads these views, so they are copied before being written.
                let mut owned = views.as_slice().to_vec();
                f(&mut owned, buffer_offset, &mut new_buffers);
                *views = Buffer::from(owned);
            },
        }
    }

    // The views written above index the buffers past the ones the array already held, which is
    // what `buffer_offset` counted; appending them leaves every view that was already there
    // reading what it read.
    let mut buffers = Buffer::to_vec(core::mem::take(unsafe { arr.data_buffers_mut() }));
    buffers.extend(new_buffers.into_iter().map(Buffer::from));
    *unsafe { arr.data_buffers_mut() } = Buffer::from(buffers);
}

unsafe fn scatter_binview_impl<'a, V, T>(
    set_values: V,
    arr: &mut PlBinaryViewArray,
    idx: &[IdxSize],
) where
    V: IntoIterator<Item = Option<&'a T>>,
    T: AsRef<[u8]> + ?Sized + 'a,
{
    let mut values_iter = set_values.into_iter();
    let length = arr.len();

    if let Some(validity) = arr.validity() {
        // A scalar mask stands for one bit per element, which `to_flat` resolves.
        let mut mut_validity = validity.to_flat().into_owned().make_mut();
        with_views_mut(arr, |views, buffer_offset, new_buffers| {
            for (idx, val) in idx.iter().zip(&mut values_iter) {
                if let Some(v) = val {
                    let view = View::new_with_buffers(v.as_ref(), buffer_offset, new_buffers);
                    *views.get_unchecked_mut(*idx as usize) = view;
                    mut_validity.set_unchecked(*idx as usize, true);
                } else {
                    mut_validity.set_unchecked(*idx as usize, false);
                }
            }
        });
        arr.set_validity(Some(PlBitmap::from_bitmap(mut_validity.into())))
    } else {
        let mut null_idx = vec![];
        with_views_mut(arr, |views, buffer_offset, new_buffers| {
            for (idx, val) in idx.iter().zip(values_iter) {
                if let Some(v) = val {
                    let view = View::new_with_buffers(v.as_ref(), buffer_offset, new_buffers);
                    *views.get_unchecked_mut(*idx as usize) = view;
                } else {
                    null_idx.push(*idx);
                }
            }
        });

        // Only make a validity bitmap when null values are set.
        if !null_idx.is_empty() {
            let mut validity = MutableBitmap::with_capacity(length);
            validity.extend_constant(length, true);
            for idx in null_idx {
                validity.set_unchecked(idx as usize, false)
            }
            arr.set_validity(Some(PlBitmap::from_bitmap(validity.into())))
        }
    }
}

impl<T: PolarsOpsNumericType> ChunkedSet<T::Native> for &mut ChunkedArray<T> {
    fn scatter<V>(self, idx: &[IdxSize], values: V) -> PolarsResult<Series>
    where
        V: IntoIterator<Item = Option<T::Native>>,
    {
        check_bounds(idx, self.len() as IdxSize)?;
        let mut ca = std::mem::take(self);
        ca.rechunk_mut();
        let name = ca.name().clone();

        // Scattering writes a different value at each index it names, so a chunk that repeats a
        // single value cannot stay repeated: `with_values_mut` writes it out, once, on the way in.
        let mut arr = ca.downcast_into_iter().next().unwrap();

        unsafe { scatter_primitive_impl(values, &mut arr, idx) };

        let out = ChunkedArray::<T>::with_chunk(name, arr);
        Ok(out.into_series())
    }
}

impl<'a> ChunkedSet<&'a [u8]> for &mut BinaryChunked {
    fn scatter<V>(self, idx: &[IdxSize], values: V) -> PolarsResult<Series>
    where
        V: IntoIterator<Item = Option<&'a [u8]>>,
    {
        check_bounds(idx, self.len() as IdxSize)?;
        let mut ca = std::mem::take(self);
        ca.rechunk_mut();
        let name = ca.name().clone();

        // As above: a scatter writes a different view at each index it names, so a chunk that
        // repeats a single view is written out on the way in.
        let mut arr = ca.downcast_into_iter().next().unwrap();

        unsafe { scatter_binview_impl(values, &mut arr, idx) };

        let out = BinaryChunked::with_chunk(name, arr);
        Ok(out.into_series())
    }
}

impl<'a> ChunkedSet<&'a str> for &mut StringChunked {
    fn scatter<V>(self, idx: &[IdxSize], values: V) -> PolarsResult<Series>
    where
        V: IntoIterator<Item = Option<&'a str>>,
    {
        check_bounds(idx, self.len() as IdxSize)?;
        let mut ca = std::mem::take(self);
        ca.rechunk_mut();
        let name = ca.name().clone();

        // As above, a chunk that repeats a single view is written out on the way in.
        //
        // The strings are scattered into the array as the bytes they are, which is why it is the
        // binary view underneath that the kernel writes into.
        let mut arr = ca.downcast_into_iter().next().unwrap().into_binview();

        unsafe { scatter_binview_impl(values, &mut arr, idx) };

        // SAFETY: every element left in the array is either one that was already valid UTF-8 or
        // one of the `&str`s just written over it.
        let arr = unsafe { PlUtf8ViewArray::from_binview_unchecked(arr) };
        let out = StringChunked::with_chunk(name, arr);
        Ok(out.into_series())
    }
}
impl ChunkedSet<bool> for &mut BooleanChunked {
    fn scatter<V>(self, idx: &[IdxSize], values: V) -> PolarsResult<Series>
    where
        V: IntoIterator<Item = Option<bool>>,
    {
        check_bounds(idx, self.len() as IdxSize)?;
        let mut ca = std::mem::take(self);
        ca.rechunk_mut();
        let name = ca.name().clone();

        // As above, a chunk that repeats a single bit is written out on the way in.
        let mut arr = ca.downcast_into_iter().next().unwrap();

        unsafe { scatter_bool_impl(values, &mut arr, idx) };

        let out = BooleanChunked::with_chunk(name, arr);
        Ok(out.into_series())
    }
}
