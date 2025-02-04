use std::sync::Arc;

use hashbrown::hash_map::Entry;
use polars_error::{polars_bail, PolarsResult};
use polars_utils::aliases::{InitHashMaps, PlHashMap};
use polars_utils::itertools::Itertools;
use polars_utils::vec::PushUnchecked;

use crate::array::growable::make_growable;
use crate::array::*;
use crate::bitmap::{Bitmap, BitmapBuilder};
use crate::buffer::Buffer;
use crate::datatypes::PhysicalType;
use crate::offset::Offsets;
use crate::types::{NativeType, Offset};
use crate::with_match_primitive_type_full;

/// Concatenate multiple [`Array`] of the same type into a single [`Array`].
pub fn concatenate(arrays: &[&dyn Array]) -> PolarsResult<Box<dyn Array>> {
    if arrays.is_empty() {
        polars_bail!(InvalidOperation: "concat requires input of at least one array")
    }

    if arrays
        .iter()
        .any(|array| array.dtype() != arrays[0].dtype())
    {
        polars_bail!(InvalidOperation: "It is not possible to concatenate arrays of different data types.")
    }

    let lengths = arrays.iter().map(|array| array.len()).collect_vec();
    let capacity = lengths.iter().sum();

    let mut mutable = make_growable(arrays, false, capacity);

    for (i, len) in lengths.iter().enumerate() {
        // SAFETY: len is correct
        unsafe { mutable.extend(i, 0, *len) }
    }

    Ok(mutable.as_box())
}

fn len_null_count<A: AsRef<dyn Array>>(arrays: &[A]) -> (usize, usize) {
    let mut len = 0;
    let mut null_count = 0;
    for arr in arrays {
        let arr = arr.as_ref();
        len += arr.len();
        null_count += arr.null_count();
    }
    (len, null_count)
}

/// Concatenate the validities of multiple [Array]s into a single Bitmap.
pub fn concatenate_validities<A: AsRef<dyn Array>>(arrays: &[A]) -> Option<Bitmap> {
    let (len, null_count) = len_null_count(arrays);
    concatenate_validities_with_len_null_count(arrays, len, null_count)
}

fn concatenate_validities_with_len_null_count<A: AsRef<dyn Array>>(arrays: &[A], len: usize, null_count: usize) -> Option<Bitmap> {
    if null_count == 0 {
        return None;
    }

    let mut bitmap = BitmapBuilder::with_capacity(len);
    for arr in arrays {
        let arr = arr.as_ref();
        if arr.null_count() == arr.len() {
            bitmap.extend_constant(arr.len(), false);
        } else if arr.null_count() == 0 {
            bitmap.extend_constant(arr.len(), true);
        } else {
            bitmap.extend_from_bitmap(arr.validity().unwrap());
        }
    }
    bitmap.into_opt_validity()
}

/// Concatenate multiple [`Array`] of the same type into a single [`Array`].
/// 
/// # Safety
/// All arrays must be of the same dtype.
pub fn concatenate_unchecked<A: AsRef<dyn Array>>(arrays: &[A]) -> PolarsResult<Box<dyn Array>> {
    if arrays.is_empty() {
        polars_bail!(InvalidOperation: "concat requires input of at least one array")
    }
    
    if arrays.len() == 1 {
        return Ok(arrays[0].as_ref().to_boxed());
    }

    use PhysicalType::*;
    match arrays[0].as_ref().dtype().to_physical_type() {
        Null => Ok(Box::new(concatenate_null(arrays))),
        Boolean => Ok(Box::new(concatenate_bool(arrays))),
        Primitive(ptype) => {
            with_match_primitive_type_full!(ptype, |$T| {
                Ok(Box::new(concatenate_primitive::<$T, _>(arrays)))
            })
        }
        Binary => Ok(Box::new(concatenate_binary::<i32, _>(arrays)?)),
        LargeBinary => Ok(Box::new(concatenate_binary::<i64, _>(arrays)?)),
        Utf8 => Ok(Box::new(concatenate_utf8::<i32, _>(arrays)?)),
        LargeUtf8 => Ok(Box::new(concatenate_utf8::<i64, _>(arrays)?)),
        BinaryView => Ok(Box::new(concatenate_view::<[u8], _>(arrays))),
        Utf8View => Ok(Box::new(concatenate_view::<str, _>(arrays))),
        List => Ok(Box::new(concatenate_list::<i32, _>(arrays)?)),
        LargeList => Ok(Box::new(concatenate_list::<i64, _>(arrays)?)),
        FixedSizeBinary => Ok(Box::new(concatenate_fixed_size_binary(arrays)?)),
        FixedSizeList => Ok(Box::new(concatenate_fixed_size_list(arrays)?)),
        Struct => Ok(Box::new(concatenate_struct(arrays)?)),
        Union => unimplemented!(),
        Map => unimplemented!(),
        Dictionary(_) => unimplemented!(),
    }
}

fn concatenate_null<A: AsRef<dyn Array>>(arrays: &[A]) -> NullArray {
    let dtype = arrays[0].as_ref().dtype().clone();
    let total_len = arrays.iter().map(|arr| arr.as_ref().len()).sum();
    NullArray::new(dtype, total_len)
}

fn concatenate_bool<A: AsRef<dyn Array>>(arrays: &[A]) -> BooleanArray {
    let dtype = arrays[0].as_ref().dtype().clone();
    let (total_len, null_count) = len_null_count(arrays);
    let mut bitmap = BitmapBuilder::with_capacity(total_len);
    for arr in arrays {
        let arr: &BooleanArray = arr.as_ref().as_any().downcast_ref().unwrap();
        bitmap.extend_from_bitmap(arr.values());
    }
    let validity = concatenate_validities_with_len_null_count(arrays, total_len, null_count);
    BooleanArray::new(dtype, bitmap.freeze(), validity)
}

fn concatenate_primitive<T: NativeType, A: AsRef<dyn Array>>(arrays: &[A]) -> PrimitiveArray<T> {
    let dtype = arrays[0].as_ref().dtype().clone();
    let (total_len, null_count) = len_null_count(arrays);
    let mut out = Vec::with_capacity(total_len);
    for arr in arrays {
        let arr: &PrimitiveArray<T> = arr.as_ref().as_any().downcast_ref().unwrap();
        out.extend_from_slice(arr.values());
    }
    let validity = concatenate_validities_with_len_null_count(arrays, total_len, null_count);
    unsafe { PrimitiveArray::new_unchecked(dtype, Buffer::from(out), validity) }
}

fn concatenate_binary<O: Offset, A: AsRef<dyn Array>>(arrays: &[A]) -> PolarsResult<BinaryArray<O>> {
    let dtype = arrays[0].as_ref().dtype().clone();
    let (total_len, null_count) = len_null_count(arrays);
    let total_bytes = arrays.iter().map(|arr| {
        let arr: &BinaryArray<O> = arr.as_ref().as_any().downcast_ref().unwrap();
        arr.get_values_size()
    }).sum();

    let mut values = Vec::with_capacity(total_bytes);
    let mut offsets = Offsets::<O>::with_capacity(total_len);

    for arr in arrays {
        let arr: &BinaryArray<O> = arr.as_ref().as_any().downcast_ref().unwrap();
        let first_offset = arr.offsets().first().to_usize();
        let last_offset = arr.offsets().last().to_usize();
        values.extend_from_slice(&arr.values()[first_offset..last_offset]);
        for len in arr.offsets().lengths() {
            offsets.try_push(len)?;
        }
    }
    
    let validity = concatenate_validities_with_len_null_count(arrays, total_len, null_count);
    Ok(unsafe { BinaryArray::new(dtype, offsets.into(), values.into(), validity) })
}

fn concatenate_utf8<O: Offset, A: AsRef<dyn Array>>(arrays: &[A]) -> PolarsResult<Utf8Array<O>> {
    let dtype = arrays[0].as_ref().dtype().clone();
    let (total_len, null_count) = len_null_count(arrays);
    let total_bytes = arrays.iter().map(|arr| {
        let arr: &Utf8Array<O> = arr.as_ref().as_any().downcast_ref().unwrap();
        arr.get_values_size()
    }).sum();

    let mut bytes = Vec::with_capacity(total_bytes);
    let mut offsets = Offsets::<O>::with_capacity(total_len);

    for arr in arrays {
        let arr: &Utf8Array<O> = arr.as_ref().as_any().downcast_ref().unwrap();
        let first_offset = arr.offsets().first().to_usize();
        let last_offset = arr.offsets().last().to_usize();
        bytes.extend_from_slice(&arr.values()[first_offset..last_offset]);
        for len in arr.offsets().lengths() {
            offsets.try_push(len)?;
        }
    }
    
    let validity = concatenate_validities_with_len_null_count(arrays, total_len, null_count);
    Ok(unsafe { Utf8Array::new_unchecked(dtype, offsets.into(), bytes.into(), validity) })
}

fn concatenate_view<V: ViewType + ?Sized, A: AsRef<dyn Array>>(arrays: &[A]) -> BinaryViewArrayGeneric<V> {
    let dtype = arrays[0].as_ref().dtype().clone();
    let (total_len, null_count) = len_null_count(arrays);
    let first_arr: &BinaryViewArrayGeneric<V> = arrays[0].as_ref().as_any().downcast_ref().unwrap();
    let mut total_nondedup_buffers = first_arr.data_buffers().len();
    let all_same_bufs = arrays.iter().skip(1).all(|arr| {
        let arr: &BinaryViewArrayGeneric<V> = arr.as_ref().as_any().downcast_ref().unwrap();
        total_nondedup_buffers += arr.data_buffers().len();
        // Fat pointer equality, checks both start and length.
        std::ptr::eq(Arc::as_ptr(arr.data_buffers()), Arc::as_ptr(first_arr.data_buffers()))
    });
    
    let mut views = Vec::with_capacity(total_len);
    let validity = concatenate_validities_with_len_null_count(arrays, total_len, null_count);

    if all_same_bufs {
        let total_buffer_len = first_arr.total_buffer_len();
        let mut total_bytes_len = 0;
        let buffers = Arc::clone(&first_arr.data_buffers());
        for arr in arrays {
            let arr: &BinaryViewArrayGeneric<V> = arr.as_ref().as_any().downcast_ref().unwrap();
            views.extend_from_slice(arr.views());
            total_bytes_len += arr.total_bytes_len();
        }
        unsafe {
            BinaryViewArrayGeneric::new_unchecked(dtype, views.into(), buffers, validity, total_bytes_len, total_buffer_len)
        }
    } else {
        let mut total_buffers = Vec::with_capacity(total_nondedup_buffers);
        let mut total_bytes_len = 0;
        let mut total_buffer_len = 0;
        let mut buffers_dedup = PlHashMap::with_capacity(total_nondedup_buffers);
        let mut buf_idx_mapping = Vec::new();
        for arr in arrays {
            let arr: &BinaryViewArrayGeneric<V> = arr.as_ref().as_any().downcast_ref().unwrap();
            
            // Dedup the buffers and check if a translation is needed.
            buf_idx_mapping.clear();
            buf_idx_mapping.reserve(arr.data_buffers().len());
            let mut needs_translation = false;
            for (i, buffer) in arr.data_buffers().iter().enumerate_u32() {
                let buf_id = (buffer.as_slice().as_ptr(), buffer.len());
                let offset = match buffers_dedup.entry(buf_id) {
                    Entry::Occupied(o) => *o.get(),
                    Entry::Vacant(v) => {
                        let offset = total_buffers.len() as u32;
                        total_buffers.push(buffer.clone());
                        total_buffer_len += buffer.len();
                        v.insert(offset);
                        offset
                    },
                };
                
                needs_translation |= offset != i;
                buf_idx_mapping.push(offset);
            }

            if needs_translation {
                unsafe {
                    for mut view in arr.views().iter().copied() {
                        total_bytes_len += view.length as usize;
                        if view.length > 12 {
                            view.buffer_idx = *buf_idx_mapping.get_unchecked(view.buffer_idx as usize);
                        }
                        views.push_unchecked(view);
                    }
                }
            } else {
                total_bytes_len += arr.total_bytes_len();
                views.extend_from_slice(arr.views());
            }
        }

        let buffers = total_buffers.into_iter().collect();
        unsafe {
            BinaryViewArrayGeneric::new_unchecked(dtype, views.into(), buffers, validity, total_bytes_len, total_buffer_len)
        }
    }
}


fn concatenate_list<O: Offset, A: AsRef<dyn Array>>(arrays: &[A]) -> PolarsResult<ListArray<O>> {
    let dtype = arrays[0].as_ref().dtype().clone();
    let (total_len, null_count) = len_null_count(arrays);
    
    let mut num_sliced = 0;
    let mut offsets = Offsets::<O>::with_capacity(total_len);
    for arr in arrays {
        let arr: &ListArray<O> = arr.as_ref().as_any().downcast_ref().unwrap();
        for len in arr.offsets().lengths() {
            offsets.try_push(len)?;
        }
        let first_offset = arr.offsets().first().to_usize();
        let offset_range = arr.offsets().range().to_usize();
        num_sliced += (first_offset != 0 || offset_range != arr.len()) as usize;
    }
    
    let values = if num_sliced > 0 {
        let inner_sliced_arrays = arrays.iter().map(|arr| {
            let arr: &ListArray<O> = arr.as_ref().as_any().downcast_ref().unwrap();
            let first_offset = arr.offsets().first().to_usize();
            let offset_range = arr.offsets().range().to_usize();
            if first_offset != 0 || offset_range != arr.len() {
                arr.values().sliced(first_offset, offset_range)
            } else {
                arr.values().to_boxed()
            }
        }).collect_vec();
        concatenate_unchecked(&inner_sliced_arrays[..])?
    } else {
        let inner_arrays = arrays.iter().map(|arr|  {
            let arr: &ListArray<O> = arr.as_ref().as_any().downcast_ref().unwrap();
            &**arr.values()
        }).collect_vec();
        concatenate_unchecked(&inner_arrays)?
    };
    
    let validity = concatenate_validities_with_len_null_count(arrays, total_len, null_count);
    Ok(ListArray::new(dtype, offsets.into(), values, validity))
}

fn concatenate_fixed_size_binary<A: AsRef<dyn Array>>(arrays: &[A]) -> PolarsResult<FixedSizeBinaryArray> {
    let dtype = arrays[0].as_ref().dtype().clone();
    let (total_len, null_count) = len_null_count(arrays);
    let total_bytes = arrays.iter().map(|arr| {
        let arr: &FixedSizeBinaryArray = arr.as_ref().as_any().downcast_ref().unwrap();
        arr.values().len()
    }).sum();

    let mut bytes = Vec::with_capacity(total_bytes);
    for arr in arrays {
        let arr: &FixedSizeBinaryArray = arr.as_ref().as_any().downcast_ref().unwrap();
        bytes.extend_from_slice(arr.values());
    }

    let validity = concatenate_validities_with_len_null_count(arrays, total_len, null_count);
    Ok(FixedSizeBinaryArray::new(dtype, bytes.into(), validity))
}

fn concatenate_fixed_size_list<A: AsRef<dyn Array>>(arrays: &[A]) -> PolarsResult<FixedSizeListArray> {
    let dtype = arrays[0].as_ref().dtype().clone();
    let (total_len, null_count) = len_null_count(arrays);

    let inner_arrays = arrays.iter().map(|arr|  {
        let arr: &FixedSizeListArray = arr.as_ref().as_any().downcast_ref().unwrap();
        &**arr.values()
    }).collect_vec();
    let values = concatenate_unchecked(&inner_arrays)?;
    let validity = concatenate_validities_with_len_null_count(arrays, total_len, null_count);
    Ok(FixedSizeListArray::new(dtype, total_len, values, validity))
}

fn concatenate_struct<A: AsRef<dyn Array>>(arrays: &[A]) -> PolarsResult<StructArray> {
    let dtype = arrays[0].as_ref().dtype().clone();
    let (total_len, null_count) = len_null_count(arrays);
    
    let num_fields = arrays[0].as_ref().as_any().downcast_ref::<StructArray>().unwrap().values().len();
    
    let mut inner_arrays = Vec::with_capacity(arrays.len());
    let values = (0..num_fields).map(|f| {
        inner_arrays.clear();
        for arr in arrays {
            let arr: &StructArray = arr.as_ref().as_any().downcast_ref().unwrap();
            inner_arrays.push(&arr.values()[f]);
        }
        concatenate_unchecked(&inner_arrays)
    }).try_collect_vec()?;

    let validity = concatenate_validities_with_len_null_count(arrays, total_len, null_count);
    Ok(StructArray::new(dtype, total_len, values, validity))
}
