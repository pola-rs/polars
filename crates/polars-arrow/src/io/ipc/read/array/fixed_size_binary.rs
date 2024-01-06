use std::collections::VecDeque;
use std::io::{Read, Seek};

use polars_error::{polars_err, PolarsResult};

use super::super::read_basic::*;
use super::super::{Compression, IpcBuffer, Node};
use crate::array::FixedSizeBinaryArray;
use crate::datatypes::ArrowDataType;
use crate::io::ipc::read::array::{try_get_array_length, try_get_field_node};

#[allow(clippy::too_many_arguments)]
pub fn read_fixed_size_binary<R: Read + Seek>(
    field_nodes: &mut VecDeque<Node>,
    data_type: ArrowDataType,
    buffers: &mut VecDeque<IpcBuffer>,
    reader: &mut R,
    block_offset: u64,
    is_little_endian: bool,
    compression: Option<Compression>,
    limit: Option<usize>,
    scratch: &mut Vec<u8>,
) -> PolarsResult<FixedSizeBinaryArray> {
    let field_node = try_get_field_node(field_nodes, &data_type)?;

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

    let length = length.saturating_mul(FixedSizeBinaryArray::maybe_get_size(&data_type)?);
    let values = read_buffer(
        buffers,
        length,
        reader,
        block_offset,
        is_little_endian,
        compression,
        scratch,
    )?;

    FixedSizeBinaryArray::try_new(data_type, values, validity)
}

pub fn skip_fixed_size_binary(
    field_nodes: &mut VecDeque<Node>,
    buffers: &mut VecDeque<IpcBuffer>,
) -> PolarsResult<()> {
    let _ = field_nodes.pop_front().ok_or_else(|| {
        polars_err!(oos =
            "IPC: unable to fetch the field for fixed-size binary. The file or stream is corrupted."
        )
    })?;

    let _ = buffers
        .pop_front()
        .ok_or_else(|| polars_err!(oos = "IPC: missing validity buffer."))?;
    let _ = buffers
        .pop_front()
        .ok_or_else(|| polars_err!(oos = "IPC: missing values buffer."))?;
    Ok(())
}
