use std::collections::VecDeque;
use std::io::{Read, Seek, SeekFrom};

use polars_buffer::Buffer;
use polars_error::{PolarsResult, polars_bail, polars_ensure, polars_err};

use super::super::compression;
use super::super::endianness::is_native_little_endian;
use super::{Compression, IpcBuffer, Node, OutOfSpecKind};
use crate::bitmap::Bitmap;
use crate::types::NativeType;

fn read_swapped<T: NativeType, R: Read + Seek>(
    reader: &mut R,
    length: usize,
    buffer: &mut Vec<T>,
    is_little_endian: bool,
) -> PolarsResult<()> {
    // Slow case where we must reverse bits.
    #[expect(clippy::slow_vector_initialization)] // Avoid alloc_zeroed, leads to syscall.
    let mut slice = Vec::new();
    slice.resize(length * size_of::<T>(), 0);
    reader.read_exact(&mut slice)?;

    let chunks = slice.chunks_exact(size_of::<T>());
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

        polars_ensure!(buffer.len() == buffer_length, ComputeError: "Malformed IPC file: expected compressed buffer of len {buffer_length}, got {}", buffer.len());

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
    let required_number_of_bytes = length.saturating_mul(size_of::<T>());
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
    // Upper bound for the number of rows to be returned.
    row_limit: Option<usize>,
    is_little_endian: bool,
    compression: Compression,
    scratch: &mut Vec<u8>,
) -> PolarsResult<Vec<T>> {
    if row_limit == Some(0) {
        return Ok(vec![]);
    }

    if is_little_endian != is_native_little_endian() {
        polars_bail!(ComputeError:
            "Reading compressed and big endian IPC",
        )
    }

    // Decompress first.
    scratch.clear();
    scratch.try_reserve(buffer_length)?;
    reader
        .by_ref()
        .take(buffer_length as u64)
        .read_to_end(scratch)?;

    polars_ensure!(scratch.len() == buffer_length, ComputeError: "Malformed IPC file: expected compressed buffer of len {buffer_length}, got {}", scratch.len());

    let decompressed_len_field = i64::from_le_bytes(scratch[..8].try_into().unwrap());
    let decompressed_bytes: usize = if decompressed_len_field == -1 {
        buffer_length - 8
    } else {
        decompressed_len_field.try_into().map_err(|_| {
            polars_err!(ComputeError: "Malformed IPC file: got invalid decompressed length {decompressed_len_field}")
        })?
    };

    polars_ensure!(decompressed_bytes.is_multiple_of(size_of::<T>()),
            ComputeError: "Malformed IPC file: got decompressed buffer length which is not a multiple of the data type");
    let n_rows_in_array = decompressed_bytes / size_of::<T>();

    if decompressed_len_field == -1 {
        return Ok(bytemuck::cast_slice(&scratch[8..]).to_vec());
    }

    // It is undefined behavior to call read_exact on un-initialized, https://doc.rust-lang.org/std/io/trait.Read.html#tymethod.read
    // see also https://github.com/MaikKlein/ash/issues/354#issue-781730580

    let n_rows_exact = row_limit
        .map(|limit| std::cmp::min(limit, n_rows_in_array))
        .unwrap_or(n_rows_in_array);

    let mut buffer = vec![T::default(); n_rows_exact];
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
    row_limit: usize,
    bytes: usize,
    reader: &mut R,
) -> PolarsResult<Vec<u8>> {
    if row_limit > bytes * 8 {
        polars_bail!(
            oos = OutOfSpecKind::InvalidBitmap {
                length: row_limit,
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

    polars_ensure!(buffer.len() == bytes, ComputeError: "Malformed IPC file: expected compressed buffer of len {bytes}, got {}", buffer.len());

    Ok(buffer)
}

fn read_compressed_bitmap<R: Read + Seek>(
    row_limit: usize,
    bytes: usize,
    compression: Compression,
    reader: &mut R,
    scratch: &mut Vec<u8>,
) -> PolarsResult<Vec<u8>> {
    scratch.clear();
    scratch.try_reserve(bytes)?;
    reader.by_ref().take(bytes as u64).read_to_end(scratch)?;
    if scratch.len() != bytes {
        polars_bail!(ComputeError: "Malformed IPC file: expected compressed buffer of len {bytes}, got {}", scratch.len());
    }

    let decompressed_len_field = i64::from_le_bytes(scratch[..8].try_into().unwrap());
    let decompressed_bytes: usize = if decompressed_len_field == -1 {
        scratch.len() - 8
    } else {
        decompressed_len_field.try_into().map_err(|_| {
            polars_err!(ComputeError: "Malformed IPC file: got invalid decompressed length {decompressed_len_field}")
        })?
    };

    // In addition to the slicing use case, we allow for excess bytes in untruncated buffers,
    // see https://github.com/pola-rs/polars/issues/26126
    // and https://github.com/apache/arrow/issues/48883
    polars_ensure!(decompressed_bytes >= row_limit.div_ceil(8),
        ComputeError: "Malformed IPC file: got unexpected decompressed output length {decompressed_bytes}, expected {}", row_limit.div_ceil(8));

    if decompressed_len_field == -1 {
        return Ok(bytemuck::cast_slice(&scratch[8..]).to_vec());
    }

    #[expect(clippy::slow_vector_initialization)] // Avoid alloc_zeroed, leads to syscall.
    let mut buffer = Vec::new();
    buffer.resize(decompressed_bytes, 0);

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
    row_limit: usize,
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
        read_compressed_bitmap(row_limit, bytes, compression, reader, scratch)
    } else {
        read_uncompressed_bitmap(row_limit, bytes, reader)
    }?;

    Bitmap::try_new(buffer, row_limit)
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
    let row_limit = limit.map(|limit| limit.min(length)).unwrap_or(length);

    Ok(if field_node.null_count() > 0 {
        Some(read_bitmap(
            buffers,
            row_limit,
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
