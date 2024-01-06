#![allow(clippy::ptr_arg)] // false positive in clippy, see https://github.com/rust-lang/rust-clippy/issues/8463
use arrow_format::ipc;

use super::super::compression;
use super::super::endianness::is_native_little_endian;
use super::common::{pad_to_64, Compression};
use crate::array::*;
use crate::bitmap::Bitmap;
use crate::datatypes::PhysicalType;
use crate::offset::{Offset, OffsetsBuffer};
use crate::trusted_len::TrustedLen;
use crate::types::NativeType;
use crate::{match_integer_type, with_match_primitive_type_full};
mod binary;
mod binview;
mod boolean;
mod dictionary;
mod fixed_size_binary;
mod fixed_sized_list;
mod list;
mod map;
mod primitive;
mod struct_;
mod union;

use binary::*;
use binview::*;
use boolean::*;
pub(super) use dictionary::*;
use fixed_size_binary::*;
use fixed_sized_list::*;
use list::*;
use map::*;
use primitive::*;
use struct_::*;
use union::*;

/// Writes an [`Array`] to `arrow_data`
pub fn write(
    array: &dyn Array,
    buffers: &mut Vec<ipc::Buffer>,
    arrow_data: &mut Vec<u8>,
    nodes: &mut Vec<ipc::FieldNode>,
    offset: &mut i64,
    is_little_endian: bool,
    compression: Option<Compression>,
) {
    nodes.push(ipc::FieldNode {
        length: array.len() as i64,
        null_count: array.null_count() as i64,
    });
    use PhysicalType::*;
    match array.data_type().to_physical_type() {
        Null => (),
        Boolean => write_boolean(
            array.as_any().downcast_ref().unwrap(),
            buffers,
            arrow_data,
            offset,
            is_little_endian,
            compression,
        ),
        Primitive(primitive) => with_match_primitive_type_full!(primitive, |$T| {
            let array = array.as_any().downcast_ref().unwrap();
            write_primitive::<$T>(array, buffers, arrow_data, offset, is_little_endian, compression)
        }),
        Binary => write_binary::<i32>(
            array.as_any().downcast_ref().unwrap(),
            buffers,
            arrow_data,
            offset,
            is_little_endian,
            compression,
        ),
        LargeBinary => write_binary::<i64>(
            array.as_any().downcast_ref().unwrap(),
            buffers,
            arrow_data,
            offset,
            is_little_endian,
            compression,
        ),
        FixedSizeBinary => write_fixed_size_binary(
            array.as_any().downcast_ref().unwrap(),
            buffers,
            arrow_data,
            offset,
            is_little_endian,
            compression,
        ),
        Utf8 => write_utf8::<i32>(
            array.as_any().downcast_ref().unwrap(),
            buffers,
            arrow_data,
            offset,
            is_little_endian,
            compression,
        ),
        LargeUtf8 => write_utf8::<i64>(
            array.as_any().downcast_ref().unwrap(),
            buffers,
            arrow_data,
            offset,
            is_little_endian,
            compression,
        ),
        List => write_list::<i32>(
            array.as_any().downcast_ref().unwrap(),
            buffers,
            arrow_data,
            nodes,
            offset,
            is_little_endian,
            compression,
        ),
        LargeList => write_list::<i64>(
            array.as_any().downcast_ref().unwrap(),
            buffers,
            arrow_data,
            nodes,
            offset,
            is_little_endian,
            compression,
        ),
        FixedSizeList => write_fixed_size_list(
            array.as_any().downcast_ref().unwrap(),
            buffers,
            arrow_data,
            nodes,
            offset,
            is_little_endian,
            compression,
        ),
        Struct => write_struct(
            array.as_any().downcast_ref().unwrap(),
            buffers,
            arrow_data,
            nodes,
            offset,
            is_little_endian,
            compression,
        ),
        Dictionary(key_type) => match_integer_type!(key_type, |$T| {
            write_dictionary::<$T>(
                array.as_any().downcast_ref().unwrap(),
                buffers,
                arrow_data,
                nodes,
                offset,
                is_little_endian,
                compression,
                true,
            );
        }),
        Union => {
            write_union(
                array.as_any().downcast_ref().unwrap(),
                buffers,
                arrow_data,
                nodes,
                offset,
                is_little_endian,
                compression,
            );
        },
        Map => {
            write_map(
                array.as_any().downcast_ref().unwrap(),
                buffers,
                arrow_data,
                nodes,
                offset,
                is_little_endian,
                compression,
            );
        },
        Utf8View => write_binview(
            array.as_any().downcast_ref::<Utf8ViewArray>().unwrap(),
            buffers,
            arrow_data,
            offset,
            is_little_endian,
            compression,
        ),
        BinaryView => write_binview(
            array.as_any().downcast_ref::<BinaryViewArray>().unwrap(),
            buffers,
            arrow_data,
            offset,
            is_little_endian,
            compression,
        ),
    }
}

#[inline]
fn pad_buffer_to_64(buffer: &mut Vec<u8>, length: usize) {
    let pad_len = pad_to_64(length);
    for _ in 0..pad_len {
        buffer.push(0u8);
    }
}

/// writes `bytes` to `arrow_data` updating `buffers` and `offset` and guaranteeing a 8 byte boundary.
fn write_bytes(
    bytes: &[u8],
    buffers: &mut Vec<ipc::Buffer>,
    arrow_data: &mut Vec<u8>,
    offset: &mut i64,
    compression: Option<Compression>,
) {
    let start = arrow_data.len();
    if let Some(compression) = compression {
        arrow_data.extend_from_slice(&(bytes.len() as i64).to_le_bytes());
        match compression {
            Compression::LZ4 => {
                compression::compress_lz4(bytes, arrow_data).unwrap();
            },
            Compression::ZSTD => {
                compression::compress_zstd(bytes, arrow_data).unwrap();
            },
        }
    } else {
        arrow_data.extend_from_slice(bytes);
    };

    buffers.push(finish_buffer(arrow_data, start, offset));
}

fn write_bitmap(
    bitmap: Option<&Bitmap>,
    length: usize,
    buffers: &mut Vec<ipc::Buffer>,
    arrow_data: &mut Vec<u8>,
    offset: &mut i64,
    compression: Option<Compression>,
) {
    match bitmap {
        Some(bitmap) => {
            assert_eq!(bitmap.len(), length);
            let (slice, slice_offset, _) = bitmap.as_slice();
            if slice_offset != 0 {
                // case where we can't slice the bitmap as the offsets are not multiple of 8
                let bytes = Bitmap::from_trusted_len_iter(bitmap.iter());
                let (slice, _, _) = bytes.as_slice();
                write_bytes(slice, buffers, arrow_data, offset, compression)
            } else {
                write_bytes(slice, buffers, arrow_data, offset, compression)
            }
        },
        None => {
            buffers.push(ipc::Buffer {
                offset: *offset,
                length: 0,
            });
        },
    }
}

/// writes `bytes` to `arrow_data` updating `buffers` and `offset` and guaranteeing a 8 byte boundary.
fn write_buffer<T: NativeType>(
    buffer: &[T],
    buffers: &mut Vec<ipc::Buffer>,
    arrow_data: &mut Vec<u8>,
    offset: &mut i64,
    is_little_endian: bool,
    compression: Option<Compression>,
) {
    let start = arrow_data.len();
    if let Some(compression) = compression {
        _write_compressed_buffer(buffer, arrow_data, is_little_endian, compression);
    } else {
        _write_buffer(buffer, arrow_data, is_little_endian);
    };

    buffers.push(finish_buffer(arrow_data, start, offset));
}

#[inline]
fn _write_buffer_from_iter<T: NativeType, I: TrustedLen<Item = T>>(
    buffer: I,
    arrow_data: &mut Vec<u8>,
    is_little_endian: bool,
) {
    let len = buffer.size_hint().0;
    arrow_data.reserve(len * std::mem::size_of::<T>());
    if is_little_endian {
        buffer
            .map(|x| T::to_le_bytes(&x))
            .for_each(|x| arrow_data.extend_from_slice(x.as_ref()))
    } else {
        buffer
            .map(|x| T::to_be_bytes(&x))
            .for_each(|x| arrow_data.extend_from_slice(x.as_ref()))
    }
}

#[inline]
fn _write_compressed_buffer_from_iter<T: NativeType, I: TrustedLen<Item = T>>(
    buffer: I,
    arrow_data: &mut Vec<u8>,
    is_little_endian: bool,
    compression: Compression,
) {
    let len = buffer.size_hint().0;
    let mut swapped = Vec::with_capacity(len * std::mem::size_of::<T>());
    if is_little_endian {
        buffer
            .map(|x| T::to_le_bytes(&x))
            .for_each(|x| swapped.extend_from_slice(x.as_ref()));
    } else {
        buffer
            .map(|x| T::to_be_bytes(&x))
            .for_each(|x| swapped.extend_from_slice(x.as_ref()))
    };
    arrow_data.extend_from_slice(&(swapped.len() as i64).to_le_bytes());
    match compression {
        Compression::LZ4 => {
            compression::compress_lz4(&swapped, arrow_data).unwrap();
        },
        Compression::ZSTD => {
            compression::compress_zstd(&swapped, arrow_data).unwrap();
        },
    }
}

fn _write_buffer<T: NativeType>(buffer: &[T], arrow_data: &mut Vec<u8>, is_little_endian: bool) {
    if is_little_endian == is_native_little_endian() {
        // in native endianness we can use the bytes directly.
        let buffer = bytemuck::cast_slice(buffer);
        arrow_data.extend_from_slice(buffer);
    } else {
        _write_buffer_from_iter(buffer.iter().copied(), arrow_data, is_little_endian)
    }
}

fn _write_compressed_buffer<T: NativeType>(
    buffer: &[T],
    arrow_data: &mut Vec<u8>,
    is_little_endian: bool,
    compression: Compression,
) {
    if is_little_endian == is_native_little_endian() {
        let bytes = bytemuck::cast_slice(buffer);
        arrow_data.extend_from_slice(&(bytes.len() as i64).to_le_bytes());
        match compression {
            Compression::LZ4 => {
                compression::compress_lz4(bytes, arrow_data).unwrap();
            },
            Compression::ZSTD => {
                compression::compress_zstd(bytes, arrow_data).unwrap();
            },
        }
    } else {
        todo!()
    }
}

/// writes `bytes` to `arrow_data` updating `buffers` and `offset` and guaranteeing a 8 byte boundary.
#[inline]
fn write_buffer_from_iter<T: NativeType, I: TrustedLen<Item = T>>(
    buffer: I,
    buffers: &mut Vec<ipc::Buffer>,
    arrow_data: &mut Vec<u8>,
    offset: &mut i64,
    is_little_endian: bool,
    compression: Option<Compression>,
) {
    let start = arrow_data.len();

    if let Some(compression) = compression {
        _write_compressed_buffer_from_iter(buffer, arrow_data, is_little_endian, compression);
    } else {
        _write_buffer_from_iter(buffer, arrow_data, is_little_endian);
    }

    buffers.push(finish_buffer(arrow_data, start, offset));
}

fn finish_buffer(arrow_data: &mut Vec<u8>, start: usize, offset: &mut i64) -> ipc::Buffer {
    let buffer_len = (arrow_data.len() - start) as i64;

    pad_buffer_to_64(arrow_data, arrow_data.len() - start);
    let total_len = (arrow_data.len() - start) as i64;

    let buffer = ipc::Buffer {
        offset: *offset,
        length: buffer_len,
    };
    *offset += total_len;
    buffer
}
