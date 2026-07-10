mod primitive;

use std::borrow::Cow;

use arrow_format::ipc;
use arrow_format::ipc::FieldNode;
use polars_buffer::Buffer;
use polars_error::PolarsResult;
use polars_utils::scratch_vec::ScratchVec;
use primitive::{write_native_type_buffer, write_primitive};

use crate::array::{
    Array, BinaryArray, BinaryViewArray, BinaryViewArrayGeneric, BooleanArray, DictionaryArray,
    FixedSizeBinaryArray, FixedSizeListArray, ListArray, MapArray, PrimitiveArray, StructArray,
    UnionArray, Utf8Array, Utf8ViewArray, ViewType,
};
use crate::bitmap::Bitmap;
use crate::io::ipc::compression;
use crate::io::ipc::write::Compression;
use crate::io::ipc::write2::array::primitive::write_native_type_iter;
use crate::io::write_owned::WriteBytesOwned;
use crate::offset::OffsetsBuffer;
use crate::types::Offset;
use crate::{match_integer_type, with_match_primitive_type_full};

pub struct IpcBatchSerializationContext<'a> {
    pub ipc_message: &'a mut dyn WriteBytesOwned,
    pub arrow_data: &'a mut dyn WriteBytesOwned,
    pub field_nodes: Vec<FieldNode>,
    pub buffers: Vec<ipc::Buffer>,
    pub variadic_buffer_counts: Vec<i64>,
    pub compression: Option<Compression>,
    pub bytes_scratch: ScratchVec<u8>,
}

impl<'a> IpcBatchSerializationContext<'a> {
    pub fn new(
        ipc_message: &'a mut dyn WriteBytesOwned,
        arrow_data: &'a mut dyn WriteBytesOwned,
        compression: Option<Compression>,
    ) -> Self {
        Self {
            ipc_message,
            arrow_data,
            field_nodes: vec![],
            buffers: vec![],
            variadic_buffer_counts: vec![],
            compression,
            bytes_scratch: ScratchVec::default(),
        }
    }
}

pub fn write_array(
    ctx: &mut IpcBatchSerializationContext<'_>,
    array: &dyn Array,
) -> PolarsResult<()> {
    ctx.field_nodes.push(FieldNode {
        length: array.len() as i64,
        null_count: array.null_count() as i64,
    });

    use crate::datatypes::PhysicalType::*;

    match array.dtype().to_physical_type() {
        Null => (),
        Boolean => {
            let array: &BooleanArray = array.as_any().downcast_ref().unwrap();
            write_bitmap(ctx, array.validity())?;
            write_bitmap(ctx, Some(array.values()))?;
        },
        Primitive(primitive) => with_match_primitive_type_full!(primitive, |$T| {
            let array: &PrimitiveArray<$T> = array.as_any().downcast_ref().unwrap();
            write_primitive(ctx, array)?;
        }),
        Binary => {
            let array: &BinaryArray<i32> = array.as_any().downcast_ref().unwrap();
            write_binary_offset(ctx, array.validity(), array.offsets(), array.values())?;
        },
        LargeBinary => {
            let array: &BinaryArray<i64> = array.as_any().downcast_ref().unwrap();
            write_binary_offset(ctx, array.validity(), array.offsets(), array.values())?;
        },
        Utf8 => {
            let array: &Utf8Array<i32> = array.as_any().downcast_ref().unwrap();
            write_binary_offset(ctx, array.validity(), array.offsets(), array.values())?;
        },
        LargeUtf8 => {
            let array: &Utf8Array<i64> = array.as_any().downcast_ref().unwrap();
            write_binary_offset(ctx, array.validity(), array.offsets(), array.values())?;
        },
        FixedSizeBinary => {
            let array: &FixedSizeBinaryArray = array.as_any().downcast_ref().unwrap();
            write_bitmap(ctx, array.validity())?;
            write_bytes(ctx, BufferOrSlice::Buffer(array.values()))?;
        },
        Utf8View => {
            let array: &Utf8ViewArray = array.as_any().downcast_ref().unwrap();
            write_binview(ctx, array)?;
        },
        BinaryView => {
            let array: &BinaryViewArray = array.as_any().downcast_ref().unwrap();
            write_binview(ctx, array)?;
        },
        List => {
            let array: &ListArray<i32> = array.as_any().downcast_ref().unwrap();
            write_list(ctx, array)?;
        },
        LargeList => {
            let array: &ListArray<i64> = array.as_any().downcast_ref().unwrap();
            write_list(ctx, array)?;
        },
        FixedSizeList => {
            let array: &FixedSizeListArray = array.as_any().downcast_ref().unwrap();
            write_bitmap(ctx, array.validity())?;
            write_array(ctx, array.values().as_ref())?;
        },
        Struct => {
            let array: &StructArray = array.as_any().downcast_ref().unwrap();
            write_bitmap(ctx, array.validity())?;

            for field_arr in array.values() {
                write_array(ctx, field_arr.as_ref())?;
            }
        },
        Dictionary(key_type) => match_integer_type!(key_type, |$T| {
            let array: &DictionaryArray<$T> = array.as_any().downcast_ref().unwrap();
            let keys_array: &PrimitiveArray<$T> = array.keys().as_any().downcast_ref().unwrap();
            write_primitive::<$T>(ctx, keys_array)?
        }),
        Map => {
            let array: &MapArray = array.as_any().downcast_ref().unwrap();
            write_bitmap(ctx, array.validity())?;
            write_offsets(ctx, array.offsets())?;
            write_array(
                ctx,
                array
                    .field()
                    .clone()
                    .sliced(
                        *array.offsets().first() as usize,
                        array.offsets().range() as usize,
                    )
                    .as_ref(),
            )?;
        },
        Union => {
            let array: &UnionArray = array.as_any().downcast_ref().unwrap();
            write_native_type_buffer(ctx, array.types())?;
            if let Some(offsets) = array.offsets() {
                write_native_type_buffer(ctx, offsets)?;
            }
            for field_arr in array.fields() {
                write_array(ctx, field_arr.as_ref())?;
            }
        },
    }

    Ok(())
}

fn pad_and_finish_ipc_buffer(
    arrow_data: &mut dyn WriteBytesOwned,
    buffer_start: usize,
) -> PolarsResult<ipc::Buffer> {
    let unpadded_len = arrow_data.len() - buffer_start;

    if let Some(padding) = unpadded_len
        .checked_next_multiple_of(64)
        .map(|x| x - unpadded_len)
        && padding != 0
    {
        arrow_data.write_all_owned(&Buffer::zeroed(padding))?;
    }

    Ok(ipc::Buffer {
        offset: buffer_start as i64,
        length: unpadded_len as i64,
    })
}

fn write_bytes(
    ctx: &mut IpcBatchSerializationContext<'_>,
    bytes: BufferOrSlice<'_>,
) -> PolarsResult<()> {
    let start_offset = ctx.arrow_data.len();

    if let Some(compression) = ctx.compression {
        ctx.arrow_data
            .write_all(&i64::to_le_bytes(bytes.len() as _))?;

        match compression {
            Compression::LZ4 => {
                compression::compress_lz4(&bytes, ctx.arrow_data.as_io_write())?;
            },
            Compression::ZSTD(level) => {
                compression::compress_zstd(&bytes, ctx.arrow_data.as_io_write(), level)?;
            },
        }
    } else {
        match bytes {
            BufferOrSlice::Buffer(buffer) => ctx.arrow_data.write_all_owned(buffer)?,
            BufferOrSlice::Slice(slice) => ctx.arrow_data.write_all(slice)?,
        }
    };

    ctx.buffers
        .push(pad_and_finish_ipc_buffer(ctx.arrow_data, start_offset)?);

    Ok(())
}

fn write_bitmap(
    ctx: &mut IpcBatchSerializationContext<'_>,
    bitmap: Option<&Bitmap>,
) -> PolarsResult<()> {
    match bitmap {
        Some(bitmap) => {
            let buf = match bitmap.as_buffer() {
                (buf, 0, _) => buf,
                _ => {
                    let bytes = Bitmap::from_trusted_len_iter(bitmap.iter());
                    let (buf, 0, _) = bytes.as_buffer() else {
                        panic!()
                    };
                    buf
                },
            };

            write_bytes(ctx, BufferOrSlice::Buffer(&buf))?;
        },
        None => {
            ctx.buffers.push(ipc::Buffer {
                offset: ctx.arrow_data.len() as i64,
                length: 0,
            });
        },
    }

    Ok(())
}

fn write_offsets<O: Offset>(
    ctx: &mut IpcBatchSerializationContext<'_>,
    offsets: &OffsetsBuffer<O>,
) -> PolarsResult<()> {
    let offsets = offsets.buffer();
    let first_offset = offsets[0];

    if first_offset == O::default() {
        write_native_type_buffer(ctx, offsets)?;
    } else {
        write_native_type_iter(ctx, offsets.iter().map(|x| *x - first_offset))?;
    }

    Ok(())
}

fn write_binary_offset<O: Offset>(
    ctx: &mut IpcBatchSerializationContext<'_>,
    validity: Option<&Bitmap>,
    offsets: &OffsetsBuffer<O>,
    values: &Buffer<u8>,
) -> PolarsResult<()> {
    write_bitmap(ctx, validity)?;
    write_offsets(ctx, offsets)?;
    write_bytes(
        ctx,
        BufferOrSlice::Buffer(
            &values
                .clone()
                .sliced(offsets.first().to_usize()..offsets.last().to_usize()),
        ),
    )
}

fn write_binview<T: ViewType + ?Sized>(
    ctx: &mut IpcBatchSerializationContext<'_>,
    array: &BinaryViewArrayGeneric<T>,
) -> PolarsResult<()> {
    let array = gc_bin_view(array);
    ctx.variadic_buffer_counts
        .push(array.data_buffers().len() as i64);

    write_bitmap(ctx, array.validity())?;
    write_native_type_buffer(ctx, array.views())?;

    for data in array.data_buffers().as_ref() {
        write_bytes(ctx, BufferOrSlice::Buffer(data))?;
    }

    Ok(())
}

fn write_list<O: Offset>(
    ctx: &mut IpcBatchSerializationContext<'_>,
    array: &ListArray<O>,
) -> PolarsResult<()> {
    write_bitmap(ctx, array.validity())?;
    write_offsets(ctx, array.offsets())?;
    write_array(
        ctx,
        array
            .values()
            .clone()
            .sliced(
                array.offsets().first().to_usize(),
                array.offsets().range().to_usize(),
            )
            .as_ref(),
    )
}

fn gc_bin_view<'a, T: ViewType + ?Sized>(
    array: &'a BinaryViewArrayGeneric<T>,
) -> Cow<'a, BinaryViewArrayGeneric<T>> {
    let bytes_len = array.total_bytes_len();
    let buffer_len = array.total_buffer_len();
    let extra_len = buffer_len.saturating_sub(bytes_len);
    if extra_len < bytes_len.min(1024) {
        // We can afford some tiny waste.
        Cow::Borrowed(array)
    } else {
        // Force GC it.
        Cow::Owned(array.clone().gc())
    }
}

#[derive(Debug)]
pub enum BufferOrSlice<'a> {
    Buffer(&'a Buffer<u8>),
    Slice(&'a [u8]),
}

impl<'a> std::ops::Deref for BufferOrSlice<'a> {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        match self {
            Self::Buffer(buf) => buf.as_slice(),
            Self::Slice(slice) => slice,
        }
    }
}
