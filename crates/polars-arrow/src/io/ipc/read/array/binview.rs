use std::io::{Read, Seek};

use polars_buffer::Buffer;
use polars_error::polars_err;
use polars_utils::bool::UnsafeBool;

use super::super::read_basic::*;
use super::*;
use crate::array::{ArrayRef, BinaryViewArrayGeneric, View, ViewType};

#[allow(clippy::too_many_arguments)]
pub fn read_binview<T: ViewType + ?Sized, R: Read + Seek>(
    field_nodes: &mut VecDeque<Node>,
    variadic_buffer_counts: &mut VecDeque<usize>,
    dtype: ArrowDataType,
    buffers: &mut VecDeque<IpcBuffer>,
    reader: &mut R,
    block_offset: u64,
    is_little_endian: bool,
    compression: Option<Compression>,
    limit: Option<usize>,
    scratch: &mut Vec<u8>,
    checked: UnsafeBool,
) -> PolarsResult<ArrayRef> {
    let field_node = try_get_field_node(field_nodes, &dtype)?;

    let validity = read_validity(
        buffers,
        field_node,
        reader,
        block_offset,
        is_little_endian,
        compression,
        limit,
        scratch,
    )?;

    let length = try_get_array_length(field_node, limit)?;
    let views: Buffer<View> = read_buffer(
        buffers,
        length,
        reader,
        block_offset,
        is_little_endian,
        compression,
        scratch,
    )?;

    let n_variadic = variadic_buffer_counts.pop_front().ok_or_else(
        || polars_err!(ComputeError: "IPC: unable to fetch the variadic buffers\n\nThe file or stream is corrupted.")
    )?;

    let variadic_buffers = (0..n_variadic)
        .map(|_| {
            read_bytes(
                buffers,
                reader,
                block_offset,
                is_little_endian,
                compression,
                scratch,
            )
        })
        .collect::<PolarsResult<Vec<Buffer<u8>>>>()?;

    if *checked {
        BinaryViewArrayGeneric::<T>::try_new(dtype, views, Buffer::from(variadic_buffers), validity)
            .map(|arr| arr.boxed())
    } else {
        unsafe {
            Ok(BinaryViewArrayGeneric::<T>::new_unchecked_unknown_md(
                dtype,
                views,
                Buffer::from(variadic_buffers),
                validity,
                None,
            )
            .boxed())
        }
    }
}

pub fn skip_binview(
    field_nodes: &mut VecDeque<Node>,
    buffers: &mut VecDeque<IpcBuffer>,
    variadic_buffer_counts: &mut VecDeque<usize>,
) -> PolarsResult<()> {
    let _ = field_nodes.pop_front().ok_or_else(|| {
        polars_err!(
            oos = "IPC: unable to fetch the field for utf8. The file or stream is corrupted."
        )
    })?;

    let _ = buffers
        .pop_front()
        .ok_or_else(|| polars_err!(oos = "IPC: missing validity buffer."))?;

    let _ = buffers
        .pop_front()
        .ok_or_else(|| polars_err!(oos = "IPC: missing views buffer."))?;

    let n_variadic = variadic_buffer_counts.pop_front().ok_or_else(
        || polars_err!(ComputeError: "IPC: unable to fetch the variadic buffers\n\nThe file or stream is corrupted.")
    )?;

    for _ in 0..n_variadic {
        let _ = buffers
            .pop_front()
            .ok_or_else(|| polars_err!(oos = "IPC: missing variadic buffer"))?;
    }
    Ok(())
}
