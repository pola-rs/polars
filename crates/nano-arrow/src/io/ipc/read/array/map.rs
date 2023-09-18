use std::collections::VecDeque;
use std::io::{Read, Seek};

use crate::array::MapArray;
use crate::buffer::Buffer;
use crate::datatypes::DataType;
use crate::error::{Error, Result};

use super::super::super::IpcField;
use super::super::deserialize::{read, skip};
use super::super::read_basic::*;
use super::super::{Compression, Dictionaries, IpcBuffer, Node, OutOfSpecKind, Version};

#[allow(clippy::too_many_arguments)]
pub fn read_map<R: Read + Seek>(
    field_nodes: &mut VecDeque<Node>,
    data_type: DataType,
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
) -> Result<MapArray> {
    let field_node = field_nodes.pop_front().ok_or_else(|| {
        Error::oos(format!(
            "IPC: unable to fetch the field for {data_type:?}. The file or stream is corrupted."
        ))
    })?;

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

    let length: usize = field_node
        .length()
        .try_into()
        .map_err(|_| Error::from(OutOfSpecKind::NegativeFooterLength))?;
    let length = limit.map(|limit| limit.min(length)).unwrap_or(length);

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
    .or_else(|_| Result::Ok(Buffer::<i32>::from(vec![0i32])))?;

    let field = MapArray::get_field(&data_type);

    let last_offset: usize = offsets.last().copied().unwrap() as usize;

    let field = read(
        field_nodes,
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
    data_type: &DataType,
    buffers: &mut VecDeque<IpcBuffer>,
) -> Result<()> {
    let _ = field_nodes.pop_front().ok_or_else(|| {
        Error::oos("IPC: unable to fetch the field for map. The file or stream is corrupted.")
    })?;

    let _ = buffers
        .pop_front()
        .ok_or_else(|| Error::oos("IPC: missing validity buffer."))?;
    let _ = buffers
        .pop_front()
        .ok_or_else(|| Error::oos("IPC: missing offsets buffer."))?;

    let data_type = MapArray::get_field(data_type).data_type();

    skip(field_nodes, data_type, buffers)
}
