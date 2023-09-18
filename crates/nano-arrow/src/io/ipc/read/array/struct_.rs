use std::collections::VecDeque;
use std::io::{Read, Seek};

use crate::array::StructArray;
use crate::datatypes::DataType;
use crate::error::{Error, Result};

use super::super::super::IpcField;
use super::super::deserialize::{read, skip};
use super::super::read_basic::*;
use super::super::{Compression, Dictionaries, IpcBuffer, Node, Version};

#[allow(clippy::too_many_arguments)]
pub fn read_struct<R: Read + Seek>(
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
) -> Result<StructArray> {
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

    let fields = StructArray::get_fields(&data_type);

    let values = fields
        .iter()
        .zip(ipc_field.fields.iter())
        .map(|(field, ipc_field)| {
            read(
                field_nodes,
                field,
                ipc_field,
                buffers,
                reader,
                dictionaries,
                block_offset,
                is_little_endian,
                compression,
                limit,
                version,
                scratch,
            )
        })
        .collect::<Result<Vec<_>>>()?;

    StructArray::try_new(data_type, values, validity)
}

pub fn skip_struct(
    field_nodes: &mut VecDeque<Node>,
    data_type: &DataType,
    buffers: &mut VecDeque<IpcBuffer>,
) -> Result<()> {
    let _ = field_nodes.pop_front().ok_or_else(|| {
        Error::oos("IPC: unable to fetch the field for struct. The file or stream is corrupted.")
    })?;

    let _ = buffers
        .pop_front()
        .ok_or_else(|| Error::oos("IPC: missing validity buffer."))?;

    let fields = StructArray::get_fields(data_type);

    fields
        .iter()
        .try_for_each(|field| skip(field_nodes, field.data_type(), buffers))
}
