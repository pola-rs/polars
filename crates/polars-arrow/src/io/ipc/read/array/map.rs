use std::collections::VecDeque;
use std::io::{Read, Seek};

use polars_error::{polars_err, PolarsResult};

use super::super::super::IpcField;
use super::super::deserialize::{read, skip};
use super::super::read_basic::*;
use super::super::{Compression, Dictionaries, IpcBuffer, Node, Version};
use crate::array::MapArray;
use crate::buffer::Buffer;
use crate::datatypes::ArrowDataType;
use crate::io::ipc::read::array::{try_get_array_length, try_get_field_node};

#[allow(clippy::too_many_arguments)]
pub fn read_map<R: Read + Seek>(
    field_nodes: &mut VecDeque<Node>,
    variadic_buffer_counts: &mut VecDeque<usize>,
    data_type: ArrowDataType,
    ipc_field: &IpcField,
    buffers: &mut VecDeque<IpcBuffer>,
    reader: &mut R,
    dictionaries: &Dictionaries,
    block_offset: u64,
    is_little_endian: bool,
    compression: Option<Compression>,
    limit: Option<usize>,
    version: Version,
    scratch: &mut Vec<u8>,
) -> PolarsResult<MapArray> {
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

    let offsets = read_buffer::<i32, _>(
        buffers,
        1 + length,
        reader,
        block_offset,
        is_little_endian,
        compression,
        scratch,
    )
    // Older versions of the IPC format sometimes do not report an offset
    .or_else(|_| PolarsResult::Ok(Buffer::<i32>::from(vec![0i32])))?;

    let field = MapArray::get_field(&data_type);

    let last_offset: usize = offsets.last().copied().unwrap() as usize;

    let field = read(
        field_nodes,
        variadic_buffer_counts,
        field,
        &ipc_field.fields[0],
        buffers,
        reader,
        dictionaries,
        block_offset,
        is_little_endian,
        compression,
        Some(last_offset),
        version,
        scratch,
    )?;
    MapArray::try_new(data_type, offsets.try_into()?, field, validity)
}

pub fn skip_map(
    field_nodes: &mut VecDeque<Node>,
    data_type: &ArrowDataType,
    buffers: &mut VecDeque<IpcBuffer>,
    variadic_buffer_counts: &mut VecDeque<usize>,
) -> PolarsResult<()> {
    let _ = field_nodes.pop_front().ok_or_else(|| {
        polars_err!(
            oos = "IPC: unable to fetch the field for map. The file or stream is corrupted."
        )
    })?;

    let _ = buffers
        .pop_front()
        .ok_or_else(|| polars_err!(oos = "IPC: missing validity buffer."))?;
    let _ = buffers
        .pop_front()
        .ok_or_else(|| polars_err!(oos = "IPC: missing offsets buffer."))?;

    let data_type = MapArray::get_field(data_type).data_type();

    skip(field_nodes, data_type, buffers, variadic_buffer_counts)
}
