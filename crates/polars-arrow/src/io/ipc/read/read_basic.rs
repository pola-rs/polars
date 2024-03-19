use std::collections::VecDeque;
use std::io::{Read, Seek, SeekFrom};

use polars_error::{polars_bail, polars_err, PolarsResult};

use super::super::compression;
use super::super::endianness::is_native_little_endian;
use super::{Compression, IpcBuffer, Node, OutOfSpecKind};
use crate::bitmap::Bitmap;
use crate::buffer::Buffer;
use crate::types::NativeType;

fn read_swapped<T: NativeType, R: Read + Seek>(
    reader: &mut R,
    length: usize,
    buffer: &mut Vec<T>,
    is_little_endian: bool,
) -> PolarsResult<()> {
    // slow case where we must reverse bits
    let mut slice = vec![0u8; length * std::mem::size_of::<T>()];
    reader.read_exact(&mut slice)?;

    let chunks = slice.chunks_exact(std::mem::size_of::<T>());
    if !is_little_endian {
        // machine is little endian, file is big endian
        buffer
            .as_mut_slice()
            .iter_mut()
            .zip(chunks)
            .try_for_each(|(slot, chunk)| {
                let a: T::Bytes = match chunk.try_into() {
                    Ok(a) => a,
                    Err(_) => unreachable!(),
                };
                *slot = T::from_be_bytes(a);
                PolarsResult::Ok(())
            })?;
    } else {
        // machine is big endian, file is little endian
        polars_bail!(ComputeError:
            "Reading little endian files from big endian machines",
        )
    }
    Ok(())
}

fn read_uncompressed_bytes<R: Read + Seek>(
    reader: &mut R,
    buffer_length: usize,
    is_little_endian: bool,
) -> PolarsResult<Vec<u8>> {
    if is_native_little_endian() == is_little_endian {
        let mut buffer = Vec::with_capacity(buffer_length);
        let _ = reader
            .take(buffer_length as u64)
            .read_to_end(&mut buffer)
            .unwrap();
        Ok(buffer)
    } else {
        unreachable!()
    }
}

fn read_uncompressed_buffer<T: NativeType, R: Read + Seek>(
    reader: &mut R,
    buffer_length: usize,
    length: usize,
    is_little_endian: bool,
) -> PolarsResult<Vec<T>> {
    let required_number_of_bytes = length.saturating_mul(std::mem::size_of::<T>());
    if required_number_of_bytes > buffer_length {
        polars_bail!(
            oos = OutOfSpecKind::InvalidBuffer {
                length,
                type_name: std::any::type_name::<T>(),
                required_number_of_bytes,
                buffer_length,
            }
        );
    }

    // it is undefined behavior to call read_exact on un-initialized, https://doc.rust-lang.org/std/io/trait.Read.html#tymethod.read
    // see also https://github.com/MaikKlein/ash/issues/354#issue-781730580
    let mut buffer = vec![T::default(); length];

    if is_native_little_endian() == is_little_endian {
        // fast case where we can just copy the contents
        let slice = bytemuck::cast_slice_mut(&mut buffer);
        reader.read_exact(slice)?;
    } else {
        read_swapped(reader, length, &mut buffer, is_little_endian)?;
    }
    Ok(buffer)
}

fn read_compressed_buffer<T: NativeType, R: Read + Seek>(
    reader: &mut R,
    buffer_length: usize,
    output_length: Option<usize>,
    is_little_endian: bool,
    compression: Compression,
    scratch: &mut Vec<u8>,
) -> PolarsResult<Vec<T>> {
    if output_length == Some(0) {
        return Ok(vec![]);
    }

    if is_little_endian != is_native_little_endian() {
        polars_bail!(ComputeError:
            "Reading compressed and big endian IPC".to_string(),
        )
    }

    // decompress first
    scratch.clear();
    scratch.try_reserve(buffer_length)?;
    reader
        .by_ref()
        .take(buffer_length as u64)
        .read_to_end(scratch)?;

    let length = output_length
        .unwrap_or_else(|| i64::from_le_bytes(scratch[..8].try_into().unwrap()) as usize);

    // It is undefined behavior to call read_exact on un-initialized, https://doc.rust-lang.org/std/io/trait.Read.html#tymethod.read
    // see also https://github.com/MaikKlein/ash/issues/354#issue-781730580
    let mut buffer = vec![T::default(); length];

    let out_slice = bytemuck::cast_slice_mut(&mut buffer);

    let compression = compression
        .codec()
        .map_err(|err| polars_err!(oos = OutOfSpecKind::InvalidFlatbufferCompression(err)))?;

    match compression {
        arrow_format::ipc::CompressionType::Lz4Frame => {
            compression::decompress_lz4(&scratch[8..], out_slice)?;
        },
        arrow_format::ipc::CompressionType::Zstd => {
            compression::decompress_zstd(&scratch[8..], out_slice)?;
        },
    }
    Ok(buffer)
}

fn read_compressed_bytes<R: Read + Seek>(
    reader: &mut R,
    buffer_length: usize,
    is_little_endian: bool,
    compression: Compression,
    scratch: &mut Vec<u8>,
) -> PolarsResult<Vec<u8>> {
    read_compressed_buffer::<u8, _>(
        reader,
        buffer_length,
        None,
        is_little_endian,
        compression,
        scratch,
    )
}

pub fn read_bytes<R: Read + Seek>(
    buf: &mut VecDeque<IpcBuffer>,
    reader: &mut R,
    block_offset: u64,
    is_little_endian: bool,
    compression: Option<Compression>,
    scratch: &mut Vec<u8>,
) -> PolarsResult<Buffer<u8>> {
    let buf = buf
        .pop_front()
        .ok_or_else(|| polars_err!(oos = OutOfSpecKind::ExpectedBuffer))?;

    let offset: u64 = buf
        .offset()
        .try_into()
        .map_err(|_| polars_err!(oos = OutOfSpecKind::NegativeFooterLength))?;

    let buffer_length: usize = buf
        .length()
        .try_into()
        .map_err(|_| polars_err!(oos = OutOfSpecKind::NegativeFooterLength))?;

    reader.seek(SeekFrom::Start(block_offset + offset))?;

    if let Some(compression) = compression {
        Ok(read_compressed_bytes(
            reader,
            buffer_length,
            is_little_endian,
            compression,
            scratch,
        )?
        .into())
    } else {
        Ok(read_uncompressed_bytes(reader, buffer_length, is_little_endian)?.into())
    }
}

pub fn read_buffer<T: NativeType, R: Read + Seek>(
    buf: &mut VecDeque<IpcBuffer>,
    length: usize, // in slots
    reader: &mut R,
    block_offset: u64,
    is_little_endian: bool,
    compression: Option<Compression>,
    scratch: &mut Vec<u8>,
) -> PolarsResult<Buffer<T>> {
    let buf = buf
        .pop_front()
        .ok_or_else(|| polars_err!(oos = OutOfSpecKind::ExpectedBuffer))?;

    let offset: u64 = buf
        .offset()
        .try_into()
        .map_err(|_| polars_err!(oos = OutOfSpecKind::NegativeFooterLength))?;

    let buffer_length: usize = buf
        .length()
        .try_into()
        .map_err(|_| polars_err!(oos = OutOfSpecKind::NegativeFooterLength))?;

    reader.seek(SeekFrom::Start(block_offset + offset))?;

    if let Some(compression) = compression {
        Ok(read_compressed_buffer(
            reader,
            buffer_length,
            Some(length),
            is_little_endian,
            compression,
            scratch,
        )?
        .into())
    } else {
        Ok(read_uncompressed_buffer(reader, buffer_length, length, is_little_endian)?.into())
    }
}

fn read_uncompressed_bitmap<R: Read + Seek>(
    length: usize,
    bytes: usize,
    reader: &mut R,
) -> PolarsResult<Vec<u8>> {
    if length > bytes * 8 {
        polars_bail!(
            oos = OutOfSpecKind::InvalidBitmap {
                length,
                number_of_bits: bytes * 8,
            }
        )
    }

    let mut buffer = vec![];
    buffer.try_reserve(bytes)?;
    reader
        .by_ref()
        .take(bytes as u64)
        .read_to_end(&mut buffer)?;

    Ok(buffer)
}

fn read_compressed_bitmap<R: Read + Seek>(
    length: usize,
    bytes: usize,
    compression: Compression,
    reader: &mut R,
    scratch: &mut Vec<u8>,
) -> PolarsResult<Vec<u8>> {
    let mut buffer = vec![0; (length + 7) / 8];

    scratch.clear();
    scratch.try_reserve(bytes)?;
    reader.by_ref().take(bytes as u64).read_to_end(scratch)?;

    let compression = compression
        .codec()
        .map_err(|err| polars_err!(oos = OutOfSpecKind::InvalidFlatbufferCompression(err)))?;

    match compression {
        arrow_format::ipc::CompressionType::Lz4Frame => {
            compression::decompress_lz4(&scratch[8..], &mut buffer)?;
        },
        arrow_format::ipc::CompressionType::Zstd => {
            compression::decompress_zstd(&scratch[8..], &mut buffer)?;
        },
    }
    Ok(buffer)
}

pub fn read_bitmap<R: Read + Seek>(
    buf: &mut VecDeque<IpcBuffer>,
    length: usize,
    reader: &mut R,
    block_offset: u64,
    _: bool,
    compression: Option<Compression>,
    scratch: &mut Vec<u8>,
) -> PolarsResult<Bitmap> {
    let buf = buf
        .pop_front()
        .ok_or_else(|| polars_err!(oos = OutOfSpecKind::ExpectedBuffer))?;

    let offset: u64 = buf
        .offset()
        .try_into()
        .map_err(|_| polars_err!(oos = OutOfSpecKind::NegativeFooterLength))?;

    let bytes: usize = buf
        .length()
        .try_into()
        .map_err(|_| polars_err!(oos = OutOfSpecKind::NegativeFooterLength))?;

    reader.seek(SeekFrom::Start(block_offset + offset))?;

    let buffer = if let Some(compression) = compression {
        read_compressed_bitmap(length, bytes, compression, reader, scratch)
    } else {
        read_uncompressed_bitmap(length, bytes, reader)
    }?;

    Bitmap::try_new(buffer, length)
}

#[allow(clippy::too_many_arguments)]
pub fn read_validity<R: Read + Seek>(
    buffers: &mut VecDeque<IpcBuffer>,
    field_node: Node,
    reader: &mut R,
    block_offset: u64,
    is_little_endian: bool,
    compression: Option<Compression>,
    limit: Option<usize>,
    scratch: &mut Vec<u8>,
) -> PolarsResult<Option<Bitmap>> {
    let length: usize = field_node
        .length()
        .try_into()
        .map_err(|_| polars_err!(oos = OutOfSpecKind::NegativeFooterLength))?;
    let length = limit.map(|limit| limit.min(length)).unwrap_or(length);

    Ok(if field_node.null_count() > 0 {
        Some(read_bitmap(
            buffers,
            length,
            reader,
            block_offset,
            is_little_endian,
            compression,
            scratch,
        )?)
    } else {
        let _ = buffers
            .pop_front()
            .ok_or_else(|| polars_err!(oos = OutOfSpecKind::ExpectedBuffer))?;
        None
    })
}
