use bytemuck::Pod;
use arrow::array::{Array, BinaryArray, ListArray, PrimitiveArray};
use arrow::bitmap::{Bitmap, MutableBitmap};
use arrow::datatypes::{ArrowDataType, PhysicalType};
use arrow::ffi::mmap::slice_and_owner;
use arrow::offset::OffsetsBuffer;
use arrow::types::NativeType;
use arrow::with_match_primitive_type;
use polars_error::*;

pub fn primitive_to<T: NativeType, K: NativeType + Pod>(array: &PrimitiveArray<T>) -> PolarsResult<(PrimitiveArray<K>, usize)> {
    let values = array.values();

    let values_slice = bytemuck::try_cast_slice::<_, K>(values.as_slice()).map_err(|_| polars_err!(ComputeError: "cannot reinterpret {:?} into {:?}", K::PRIMITIVE, T::PRIMITIVE))?;
    let out = unsafe { slice_and_owner(values_slice, values.clone()) };

    if out.null_count() > 0 && std::mem::size_of::<K>() != std::mem::size_of::<T>() {
        let validity = out.validity().unwrap();
        let n = std::mem::size_of::<K>() / std::mem::size_of::<T>();

        let mut new_valid = MutableBitmap::with_capacity(validity.len() * n);

        for v in validity {
            new_valid.extend_constant(n, v)
        }

        Ok((out.with_validity(Some(new_valid.freeze())), n))
    } else {
        Ok((out, 1))
    }
}

pub fn list_to_binary(array: &ListArray<i64>) -> PolarsResult<BinaryArray<i64>>{
    let values = array.values();

    let (values, n) = match values.data_type().to_physical_type() {
        PhysicalType::Primitive(primitive) => {
            with_match_primitive_type!(primitive, |$T| {
                let values = values.as_any().downcast_ref::<PrimitiveArray<$T>>().unwrap();
                let (out, n) = primitive_to::<_, u8>(values)?;
                (out, n)
        })
        },
        _ => polars_bail!(ComputeError: "reinterpret only supported on primitive types")
    };
    let dtype = ArrowDataType::LargeBinary;

    let validity = if values.null_count() > 0 {
        let validity = (0..array.len()).map(|i| values.is_valid(i * n)).collect::<Bitmap>();
        match array.validity() {
            Some(v) => Some(v & &validity),
            None => Some(validity)
        }
    } else {
        array.validity().cloned()
    };
    let values = values.values().clone();

    Ok(if n > 1 {
        let n = n as i64;
        let offsets = array.offsets().iter().map(|v| *v * n).collect::<Vec<_>>();
        let offsets = unsafe { OffsetsBuffer::new_unchecked(offsets.into()) };

        BinaryArray::<i64>::new(
            dtype,
            offsets,
            values,
            validity
        )
    } else {
        BinaryArray::<i64>::new(
            dtype,
            array.offsets().clone(),
            values,
            validity
        )
    })
}