#![allow(unsafe_op_in_unsafe_fn)]
use arrow::array::{BinaryViewArrayGeneric, BooleanArray, PrimitiveArray, View, ViewType};
use polars_array::arrow::bridge::{chunk_from_arrow, chunk_to_arrow};
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

/// Writes into the values of `arr` where they can be written into, and copies them out first
/// where they cannot — which is what Arrow's `with_values_mut` did, one level down.
unsafe fn with_values_mut<T: NativeType, F: FnOnce(&mut [T])>(arr: &mut PlPrimitiveArray<T>, f: F) {
    let length = arr.len();
    let Some(values) = arr.flat_values_mut() else {
        // A scalar chunk holds one slot standing for every element, so it is written out before
        // anything can be written into it.
        let mut values = arr.to_flat().into_owned().flat_values().unwrap().clone();
        let slice = values
            .get_mut_slice()
            .expect("a freshly written buffer is unshared");
        f(slice);
        let validity = arr.validity().map(PlBitmap::from);
        // SAFETY: the buffer written out holds one slot per element, and the mask is the one the
        // array already carried.
        *arr = unsafe { PlPrimitiveArray::new_unchecked(values, length, validity) };
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
        ca.rechunk_mut();
        let name = ca.name().clone();

        // TODO(polars-array-scalar): the kernel writes one slot per element, so a scalar chunk is
        // written out on the way in rather than the one value it stands for being set once.
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

        // TODO(polars-array-scalar): the scatter kernels are Arrow ones that write into the
        // backing buffers, so the chunk crosses over and back, a scalar one written out on the way.
        let chunk = ca.downcast_into_iter().next().unwrap();
        let mut arr = chunk_to_arrow(&chunk);
        // The chunk held the only other handle on those buffers; dropping it lets the kernel
        // write into them rather than copy them out.
        drop(chunk);

        unsafe { scatter_binview_impl(values, &mut arr, idx) };

        let out = BinaryChunked::with_chunk(name, chunk_from_arrow(&arr));
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

        // TODO(polars-array-scalar): the scatter kernels are Arrow ones that write into the
        // backing buffers, so the chunk crosses over and back, a scalar one written out on the way.
        let chunk = ca.downcast_into_iter().next().unwrap();
        let mut arr = chunk_to_arrow(&chunk);
        // The chunk held the only other handle on those buffers; dropping it lets the kernel
        // write into them rather than copy them out.
        drop(chunk);

        unsafe { scatter_binview_impl(values, &mut arr, idx) };

        let out = StringChunked::with_chunk(name, chunk_from_arrow(&arr));
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

        // TODO(polars-array-scalar): the scatter kernels are Arrow ones that write into the
        // backing buffers, so the chunk crosses over and back, a scalar one written out on the way.
        let chunk = ca.downcast_into_iter().next().unwrap();
        let mut arr = chunk_to_arrow(&chunk);
        // The chunk held the only other handle on those buffers; dropping it lets the kernel
        // write into them rather than copy them out.
        drop(chunk);

        unsafe { scatter_bool_impl(values, &mut arr, idx) };

        let out = BooleanChunked::with_chunk(name, chunk_from_arrow(&arr));
        Ok(out.into_series())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn idx(idx: &[u32]) -> Vec<IdxSize> {
        idx.iter().map(|i| *i as IdxSize).collect()
    }

    /// Writing into an array that has no mask only grows one when a null is actually set.
    #[test]
    fn scattering_into_a_fully_valid_primitive() {
        let mut ca = Int32Chunked::new("a".into(), &[1, 2, 3, 4]);
        let out = (&mut ca)
            .scatter(&idx(&[1, 3]), [Some(20), Some(40)])
            .unwrap();
        assert_eq!(
            out.i32().unwrap().iter().collect::<Vec<_>>(),
            [Some(1), Some(20), Some(3), Some(40)],
        );

        let mut ca = Int32Chunked::new("a".into(), &[1, 2, 3, 4]);
        let out = (&mut ca).scatter(&idx(&[0, 2]), [None, Some(30)]).unwrap();
        assert_eq!(
            out.i32().unwrap().iter().collect::<Vec<_>>(),
            [None, Some(2), Some(30), Some(4)],
        );
    }

    /// Writing into an array that already has a mask has to set bits both ways.
    #[test]
    fn scattering_over_an_existing_mask_sets_bits_both_ways() {
        let mut ca = Int32Chunked::new("a".into(), &[Some(1), None, Some(3), None]);
        let out = (&mut ca).scatter(&idx(&[1, 2]), [Some(20), None]).unwrap();
        assert_eq!(
            out.i32().unwrap().iter().collect::<Vec<_>>(),
            // index 1 was null and became valid; index 2 was valid and became null.
            [Some(1), Some(20), None, None],
        );
    }

    #[test]
    fn scattering_booleans_and_strings() {
        let mut ca = BooleanChunked::new("a".into(), &[true, false, true]);
        let out = (&mut ca)
            .scatter(&idx(&[0, 2]), [Some(false), None])
            .unwrap();
        assert_eq!(
            out.bool().unwrap().iter().collect::<Vec<_>>(),
            [Some(false), Some(false), None],
        );

        let mut ca = StringChunked::new("a".into(), &["a", "bb", "ccc"]);
        let out = (&mut ca).scatter(&idx(&[1]), [Some("zzzz")]).unwrap();
        assert_eq!(
            out.str().unwrap().iter().collect::<Vec<_>>(),
            [Some("a"), Some("zzzz"), Some("ccc")],
        );
    }

    /// Scatter writes into the values that are already there — it sets a handful of slots, so
    /// copying the whole buffer would turn an `O(idx)` write into an `O(len)` one.
    #[test]
    fn scattering_writes_into_the_existing_allocation() {
        let values_ptr = |s: &Series| {
            s.i32()
                .unwrap()
                .downcast_iter()
                .next()
                .unwrap()
                .flat_values()
                .unwrap()
                .as_slice()
                .as_ptr()
        };

        let mut ca = Int32Chunked::from_vec("a".into(), (0..64).collect());
        let before = ca
            .downcast_iter()
            .next()
            .unwrap()
            .flat_values()
            .unwrap()
            .as_slice()
            .as_ptr();

        let out = (&mut ca)
            .scatter(&idx(&[7, 40]), [Some(-7), Some(-40)])
            .unwrap();

        assert_eq!(values_ptr(&out), before);
        assert_eq!(out.i32().unwrap().get(7), Some(-7));
        assert_eq!(out.i32().unwrap().get(40), Some(-40));
        assert_eq!(out.i32().unwrap().get(8), Some(8));
    }

    /// The trait's stated invariant: a failed scatter leaves the array as it was.
    #[test]
    fn an_out_of_bounds_index_leaves_the_array_alone() {
        let mut ca = Int32Chunked::new("a".into(), &[1, 2, 3]);
        assert!((&mut ca).scatter(&idx(&[9]), [Some(90)]).is_err());
        assert_eq!(ca.iter().collect::<Vec<_>>(), [Some(1), Some(2), Some(3)]);
    }
}
