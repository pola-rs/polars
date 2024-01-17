//! Contains operators to filter arrays such as [`filter`].
use polars_error::PolarsResult;

use crate::array::growable::{make_growable, Growable};
use crate::array::*;
use crate::bitmap::utils::{BitChunkIterExact, BitChunksExact, SlicesIterator};
use crate::bitmap::{Bitmap, MutableBitmap};
use crate::chunk::Chunk;
use crate::datatypes::ArrowDataType;
use crate::types::simd::Simd;
use crate::types::{BitChunkOnes, NativeType};
use crate::with_match_primitive_type_full;

/// Function that can filter arbitrary arrays
pub type Filter<'a> = Box<dyn Fn(&dyn Array) -> Box<dyn Array> + 'a + Send + Sync>;

#[inline]
fn get_leading_ones(chunk: u64) -> u32 {
    if cfg!(target_endian = "little") {
        chunk.trailing_ones()
    } else {
        chunk.leading_ones()
    }
}

/// # Safety
/// This assumes that the `mask_chunks` contains a number of set/true items equal
/// to `filter_count`
unsafe fn nonnull_filter_impl<T, I>(values: &[T], mut mask_chunks: I, filter_count: usize) -> Vec<T>
where
    T: NativeType + Simd,
    I: BitChunkIterExact<u64>,
{
    let mut chunks = values.chunks_exact(64);
    let mut new = Vec::<T>::with_capacity(filter_count);
    let mut dst = new.as_mut_ptr();

    chunks
        .by_ref()
        .zip(mask_chunks.by_ref())
        .for_each(|(chunk, mask_chunk)| {
            let ones = mask_chunk.count_ones();
            let leading_ones = get_leading_ones(mask_chunk);

            if ones == leading_ones {
                let size = leading_ones as usize;
                unsafe {
                    std::ptr::copy(chunk.as_ptr(), dst, size);
                    dst = dst.add(size);
                }
                return;
            }

            let ones_iter = BitChunkOnes::from_known_count(mask_chunk, ones as usize);
            for pos in ones_iter {
                dst.write(*chunk.get_unchecked(pos));
                dst = dst.add(1);
            }
        });

    chunks
        .remainder()
        .iter()
        .zip(mask_chunks.remainder_iter())
        .for_each(|(value, b)| {
            if b {
                unsafe {
                    dst.write(*value);
                    dst = dst.add(1);
                };
            }
        });

    unsafe { new.set_len(filter_count) };
    new
}

/// # Safety
/// This assumes that the `mask_chunks` contains a number of set/true items equal
/// to `filter_count`
unsafe fn null_filter_impl<T, I>(
    values: &[T],
    validity: &Bitmap,
    mut mask_chunks: I,
    filter_count: usize,
) -> (Vec<T>, MutableBitmap)
where
    T: NativeType + Simd,
    I: BitChunkIterExact<u64>,
{
    let mut chunks = values.chunks_exact(64);

    let mut validity_chunks = validity.chunks::<u64>();

    let mut new = Vec::<T>::with_capacity(filter_count);
    let mut dst = new.as_mut_ptr();
    let mut new_validity = MutableBitmap::with_capacity(filter_count);

    chunks
        .by_ref()
        .zip(validity_chunks.by_ref())
        .zip(mask_chunks.by_ref())
        .for_each(|((chunk, validity_chunk), mask_chunk)| {
            let ones = mask_chunk.count_ones();
            let leading_ones = get_leading_ones(mask_chunk);

            if ones == leading_ones {
                let size = leading_ones as usize;
                unsafe {
                    std::ptr::copy(chunk.as_ptr(), dst, size);
                    dst = dst.add(size);

                    // safety: invariant offset + length <= slice.len()
                    new_validity.extend_from_slice_unchecked(
                        validity_chunk.to_ne_bytes().as_ref(),
                        0,
                        size,
                    );
                }
                return;
            }

            // this triggers a bitcount
            let ones_iter = BitChunkOnes::from_known_count(mask_chunk, ones as usize);
            for pos in ones_iter {
                dst.write(*chunk.get_unchecked(pos));
                dst = dst.add(1);
                new_validity.push_unchecked(validity_chunk & (1 << pos) > 0);
            }
        });

    chunks
        .remainder()
        .iter()
        .zip(validity_chunks.remainder_iter())
        .zip(mask_chunks.remainder_iter())
        .for_each(|((value, is_valid), is_selected)| {
            if is_selected {
                unsafe {
                    dst.write(*value);
                    dst = dst.add(1);
                    new_validity.push_unchecked(is_valid);
                };
            }
        });

    unsafe { new.set_len(filter_count) };
    (new, new_validity)
}

fn null_filter_simd<T: NativeType + Simd>(
    values: &[T],
    validity: &Bitmap,
    mask: &Bitmap,
) -> (Vec<T>, MutableBitmap) {
    assert_eq!(values.len(), mask.len());
    let filter_count = mask.len() - mask.unset_bits();

    let (slice, offset, length) = mask.as_slice();
    if offset == 0 {
        let mask_chunks = BitChunksExact::<u64>::new(slice, length);
        unsafe { null_filter_impl(values, validity, mask_chunks, filter_count) }
    } else {
        let mask_chunks = mask.chunks::<u64>();
        unsafe { null_filter_impl(values, validity, mask_chunks, filter_count) }
    }
}

fn nonnull_filter_simd<T: NativeType + Simd>(values: &[T], mask: &Bitmap) -> Vec<T> {
    assert_eq!(values.len(), mask.len());
    let filter_count = mask.len() - mask.unset_bits();

    let (slice, offset, length) = mask.as_slice();
    if offset == 0 {
        let mask_chunks = BitChunksExact::<u64>::new(slice, length);
        unsafe { nonnull_filter_impl(values, mask_chunks, filter_count) }
    } else {
        let mask_chunks = mask.chunks::<u64>();
        unsafe { nonnull_filter_impl(values, mask_chunks, filter_count) }
    }
}

fn filter_nonnull_primitive<T: NativeType + Simd>(
    array: &PrimitiveArray<T>,
    mask: &Bitmap,
) -> PrimitiveArray<T> {
    assert_eq!(array.len(), mask.len());

    if let Some(validity) = array.validity() {
        let (values, validity) = null_filter_simd(array.values(), validity, mask);
        PrimitiveArray::<T>::new(array.data_type().clone(), values.into(), validity.into())
    } else {
        let values = nonnull_filter_simd(array.values(), mask);
        PrimitiveArray::<T>::new(array.data_type().clone(), values.into(), None)
    }
}

fn filter_primitive<T: NativeType + Simd>(
    array: &PrimitiveArray<T>,
    mask: &BooleanArray,
) -> PrimitiveArray<T> {
    // todo: branch on mask.validity()
    filter_nonnull_primitive(array, mask.values())
}

fn filter_growable<'a>(growable: &mut impl Growable<'a>, chunks: &[(usize, usize)]) {
    chunks
        .iter()
        .for_each(|(start, len)| growable.extend(0, *start, *len));
}

/// Returns a prepared function optimized to filter multiple arrays.
/// Creating this function requires time, but using it is faster than [filter] when the
/// same filter needs to be applied to multiple arrays (e.g. a multiple columns).
pub fn build_filter(filter: &BooleanArray) -> PolarsResult<Filter> {
    let iter = SlicesIterator::new(filter.values());
    let filter_count = iter.slots();
    let chunks = iter.collect::<Vec<_>>();

    use crate::datatypes::PhysicalType::*;
    Ok(Box::new(move |array: &dyn Array| {
        match array.data_type().to_physical_type() {
            Primitive(primitive) => with_match_primitive_type_full!(primitive, |$T| {
                let array = array.as_any().downcast_ref().unwrap();
                let mut growable =
                    growable::GrowablePrimitive::<$T>::new(vec![array], false, filter_count);
                filter_growable(&mut growable, &chunks);
                let array: PrimitiveArray<$T> = growable.into();
                Box::new(array)
            }),
            LargeUtf8 => {
                let array = array.as_any().downcast_ref::<Utf8Array<i64>>().unwrap();
                let mut growable = growable::GrowableUtf8::new(vec![array], false, filter_count);
                filter_growable(&mut growable, &chunks);
                let array: Utf8Array<i64> = growable.into();
                Box::new(array)
            },
            _ => {
                let mut mutable = make_growable(&[array], false, filter_count);
                chunks
                    .iter()
                    .for_each(|(start, len)| mutable.extend(0, *start, *len));
                mutable.as_box()
            },
        }
    }))
}

pub fn filter(array: &dyn Array, filter: &BooleanArray) -> PolarsResult<Box<dyn Array>> {
    dbg!(&array);
    // The validities may be masking out `true` bits, making the filter operation
    // based on the values incorrect
    if let Some(validities) = filter.validity() {
        let values = filter.values();
        let new_values = values & validities;
        let filter = BooleanArray::new(ArrowDataType::Boolean, new_values, None);
        return crate::compute::filter::filter(array, &filter);
    }

    let false_count = filter.values().unset_bits();
    if false_count == filter.len() {
        assert_eq!(array.len(), filter.len());
        return Ok(new_empty_array(array.data_type().clone()));
    }
    if false_count == 0 {
        assert_eq!(array.len(), filter.len());
        return Ok(array.to_boxed());
    }

    use crate::datatypes::PhysicalType::*;
    match array.data_type().to_physical_type() {
        Primitive(primitive) => with_match_primitive_type_full!(primitive, |$T| {
            let array = array.as_any().downcast_ref().unwrap();
            Ok(Box::new(filter_primitive::<$T>(array, filter)))
        }),
        BinaryView => {
            let iter = SlicesIterator::new(filter.values());
            let mut mutable = growable::GrowableBinaryViewArray::new(
                vec![array.as_any().downcast_ref::<BinaryViewArray>().unwrap()], false, iter.slots()
            );
            unsafe {
                iter.for_each(|(start, len)| mutable.extend_unchecked(0, start, len));
            }
            Ok(mutable.as_box())
        }
        // Should go via BinaryView
        Utf8View => {
            unreachable!()
        }
        _ => {
            let iter = SlicesIterator::new(filter.values());
            let mut mutable = make_growable(&[array], false, iter.slots());
            iter.for_each(|(start, len)| mutable.extend(0, start, len));
            Ok(mutable.as_box())
        },
    }
}

/// Returns a new [Chunk] with arrays containing only values matching the filter.
/// This is a convenience function: filter multiple columns is embarrassingly parallel.
pub fn filter_chunk<A: AsRef<dyn Array>>(
    columns: &Chunk<A>,
    filter_values: &BooleanArray,
) -> PolarsResult<Chunk<Box<dyn Array>>> {
    let arrays = columns.arrays();

    let num_columns = arrays.len();

    let filtered_arrays = match num_columns {
        1 => {
            vec![filter(columns.arrays()[0].as_ref(), filter_values)?]
        },
        _ => {
            let filter = build_filter(filter_values)?;
            arrays.iter().map(|a| filter(a.as_ref())).collect()
        },
    };
    Chunk::try_new(filtered_arrays)
}
