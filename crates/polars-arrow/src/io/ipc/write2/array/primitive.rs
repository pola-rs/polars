use polars_buffer::Buffer;
use polars_error::PolarsResult;

use crate::array::PrimitiveArray;
use crate::io::ipc::compression;
use crate::io::ipc::endianness::is_native_little_endian;
use crate::io::ipc::write::Compression;
use crate::io::ipc::write2::array::{
    BufferOrSlice, IpcBatchSerializationContext, pad_and_finish_ipc_buffer, write_bitmap,
    write_bytes,
};
use crate::types::NativeType;

pub(super) fn write_primitive<T: NativeType>(
    ctx: &mut IpcBatchSerializationContext<'_>,
    array: &PrimitiveArray<T>,
) -> PolarsResult<()> {
    write_bitmap(ctx, array.validity())?;
    write_native_type_buffer(ctx, array.values())
}

pub(super) fn write_native_type_buffer<T: NativeType>(
    ctx: &mut IpcBatchSerializationContext<'_>,
    buffer: &Buffer<T>,
) -> PolarsResult<()> {
    if is_native_little_endian() {
        write_bytes(
            ctx,
            BufferOrSlice::Buffer(&buffer.clone().try_transmute().unwrap()),
        )
    } else {
        write_native_type_iter(ctx, buffer.iter().copied())
    }
}

pub(super) fn write_native_type_iter<T: NativeType, I: ExactSizeIterator<Item = T>>(
    ctx: &mut IpcBatchSerializationContext<'_>,
    iter: I,
) -> PolarsResult<()> {
    let start_offset = ctx.arrow_data.len();
    let mut iter = iter.map(|v| T::to_le_bytes(&v));

    if let Some(compression) = ctx.compression {
        if !is_native_little_endian() {
            unimplemented!();
        }

        let bytes_scratch = ctx.bytes_scratch.get();
        bytes_scratch.reserve_exact(std::mem::size_of::<T>() * iter.len());

        iter.for_each(|v| {
            bytes_scratch.extend_from_slice(v.as_ref());
        });

        ctx.arrow_data
            .write_all(&(bytes_scratch.len() as i64).to_le_bytes())?;

        match compression {
            Compression::LZ4 => {
                compression::compress_lz4(bytes_scratch, ctx.arrow_data.as_io_write())?;
            },
            Compression::ZSTD(level) => {
                compression::compress_zstd(bytes_scratch, ctx.arrow_data.as_io_write(), level)?;
            },
        }
    } else {
        iter.try_for_each(|v| ctx.arrow_data.write_all(v.as_ref()))?
    }

    ctx.buffers
        .push(pad_and_finish_ipc_buffer(ctx.arrow_data, start_offset)?);

    Ok(())
}
