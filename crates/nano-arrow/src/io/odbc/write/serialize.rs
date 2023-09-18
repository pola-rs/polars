use api::buffers::{BinColumnWriter, TextColumnWriter};

use crate::array::*;
use crate::bitmap::Bitmap;
use crate::datatypes::DataType;
use crate::error::{Error, Result};
use crate::offset::Offset;
use crate::types::NativeType;

use super::super::api;
use super::super::api::buffers::NullableSliceMut;

/// Serializes an [`Array`] to [`api::buffers::AnyColumnViewMut`]
/// This operation is CPU-bounded
pub fn serialize(array: &dyn Array, column: &mut api::buffers::AnyColumnViewMut) -> Result<()> {
    match array.data_type() {
        DataType::Boolean => {
            if let api::buffers::AnyColumnViewMut::Bit(values) = column {
                bool(array.as_any().downcast_ref().unwrap(), values);
                Ok(())
            } else if let api::buffers::AnyColumnViewMut::NullableBit(values) = column {
                bool_optional(array.as_any().downcast_ref().unwrap(), values);
                Ok(())
            } else {
                Err(Error::nyi("serialize bool to non-bool ODBC"))
            }
        }
        DataType::Int16 => {
            if let api::buffers::AnyColumnViewMut::I16(values) = column {
                primitive(array.as_any().downcast_ref().unwrap(), values);
                Ok(())
            } else if let api::buffers::AnyColumnViewMut::NullableI16(values) = column {
                primitive_optional(array.as_any().downcast_ref().unwrap(), values);
                Ok(())
            } else {
                Err(Error::nyi("serialize i16 to non-i16 ODBC"))
            }
        }
        DataType::Int32 => {
            if let api::buffers::AnyColumnViewMut::I32(values) = column {
                primitive(array.as_any().downcast_ref().unwrap(), values);
                Ok(())
            } else if let api::buffers::AnyColumnViewMut::NullableI32(values) = column {
                primitive_optional(array.as_any().downcast_ref().unwrap(), values);
                Ok(())
            } else {
                Err(Error::nyi("serialize i32 to non-i32 ODBC"))
            }
        }
        DataType::Float32 => {
            if let api::buffers::AnyColumnViewMut::F32(values) = column {
                primitive(array.as_any().downcast_ref().unwrap(), values);
                Ok(())
            } else if let api::buffers::AnyColumnViewMut::NullableF32(values) = column {
                primitive_optional(array.as_any().downcast_ref().unwrap(), values);
                Ok(())
            } else {
                Err(Error::nyi("serialize f32 to non-f32 ODBC"))
            }
        }
        DataType::Float64 => {
            if let api::buffers::AnyColumnViewMut::F64(values) = column {
                primitive(array.as_any().downcast_ref().unwrap(), values);
                Ok(())
            } else if let api::buffers::AnyColumnViewMut::NullableF64(values) = column {
                primitive_optional(array.as_any().downcast_ref().unwrap(), values);
                Ok(())
            } else {
                Err(Error::nyi("serialize f64 to non-f64 ODBC"))
            }
        }
        DataType::Utf8 => {
            if let api::buffers::AnyColumnViewMut::Text(values) = column {
                utf8::<i32>(array.as_any().downcast_ref().unwrap(), values);
                Ok(())
            } else {
                Err(Error::nyi("serialize utf8 to non-text ODBC"))
            }
        }
        DataType::LargeUtf8 => {
            if let api::buffers::AnyColumnViewMut::Text(values) = column {
                utf8::<i64>(array.as_any().downcast_ref().unwrap(), values);
                Ok(())
            } else {
                Err(Error::nyi("serialize utf8 to non-text ODBC"))
            }
        }
        DataType::Binary => {
            if let api::buffers::AnyColumnViewMut::Binary(values) = column {
                binary::<i32>(array.as_any().downcast_ref().unwrap(), values);
                Ok(())
            } else {
                Err(Error::nyi("serialize utf8 to non-binary ODBC"))
            }
        }
        DataType::LargeBinary => {
            if let api::buffers::AnyColumnViewMut::Binary(values) = column {
                binary::<i64>(array.as_any().downcast_ref().unwrap(), values);
                Ok(())
            } else {
                Err(Error::nyi("serialize utf8 to non-text ODBC"))
            }
        }
        DataType::FixedSizeBinary(_) => {
            if let api::buffers::AnyColumnViewMut::Binary(values) = column {
                fixed_binary(array.as_any().downcast_ref().unwrap(), values);
                Ok(())
            } else {
                Err(Error::nyi("serialize fixed to non-binary ODBC"))
            }
        }
        other => Err(Error::nyi(format!("{other:?} to ODBC"))),
    }
}

fn bool(array: &BooleanArray, values: &mut [api::Bit]) {
    array
        .values()
        .iter()
        .zip(values.iter_mut())
        .for_each(|(from, to)| *to = api::Bit(from as u8));
}

fn bool_optional(array: &BooleanArray, values: &mut NullableSliceMut<api::Bit>) {
    let (values, indicators) = values.raw_values();
    array
        .values()
        .iter()
        .zip(values.iter_mut())
        .for_each(|(from, to)| *to = api::Bit(from as u8));
    write_validity(array.validity(), indicators);
}

fn primitive<T: NativeType>(array: &PrimitiveArray<T>, values: &mut [T]) {
    values.copy_from_slice(array.values())
}

fn write_validity(validity: Option<&Bitmap>, indicators: &mut [isize]) {
    if let Some(validity) = validity {
        indicators
            .iter_mut()
            .zip(validity.iter())
            .for_each(|(indicator, is_valid)| *indicator = if is_valid { 0 } else { -1 })
    } else {
        indicators.iter_mut().for_each(|x| *x = 0)
    }
}

fn primitive_optional<T: NativeType>(array: &PrimitiveArray<T>, values: &mut NullableSliceMut<T>) {
    let (values, indicators) = values.raw_values();
    values.copy_from_slice(array.values());
    write_validity(array.validity(), indicators);
}

fn fixed_binary(array: &FixedSizeBinaryArray, writer: &mut BinColumnWriter) {
    writer.set_max_len(array.size());
    writer.write(array.iter())
}

fn binary<O: Offset>(array: &BinaryArray<O>, writer: &mut BinColumnWriter) {
    let max_len = array
        .offsets()
        .buffer()
        .windows(2)
        .map(|x| (x[1] - x[0]).to_usize())
        .max()
        .unwrap_or(0);
    writer.set_max_len(max_len);
    writer.write(array.iter())
}

fn utf8<O: Offset>(array: &Utf8Array<O>, writer: &mut TextColumnWriter<u8>) {
    let max_len = array
        .offsets()
        .buffer()
        .windows(2)
        .map(|x| (x[1] - x[0]).to_usize())
        .max()
        .unwrap_or(0);
    writer.set_max_len(max_len);
    writer.write(array.iter().map(|x| x.map(|x| x.as_bytes())))
}
