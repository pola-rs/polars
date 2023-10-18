use std::collections::VecDeque;
use std::io::{Read, Seek};

use polars_error::{polars_err, PolarsResult};

use super::super::super::IpcField;
use super::super::deserialize::{read, skip};
use super::super::read_basic::*;
use super::super::{Compression, Dictionaries, IpcBuffer, Node, OutOfSpecKind, Version};
use crate::array::UnionArray;
use crate::datatypes::DataType;
use crate::datatypes::UnionMode::Dense;

#[allow(clippy::too_many_arguments)]
pub fn read_union<R: Read + Seek>(
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
) -> PolarsResult<UnionArray> {
    let field_node = field_nodes.pop_front().ok_or_else(|| {
        polars_err!(ComputeError:
            "IPC: unable to fetch the field for `{data_type:?}`\n\nThe file or stream is corrupted."
        )
    })?;

    if version != Version::V5 {
        let _ = buffers
            .pop_front()
            .ok_or_else(|| polars_err!(oos = "IPC: missing validity buffer"))?;
    };

    let length: usize = field_node
        .length()
        .try_into()
        .map_err(|_| polars_err!(oos = OutOfSpecKind::NegativeFooterLength))?;
    let length = limit.map(|limit| limit.min(length)).unwrap_or(length);

    let types = read_buffer(
        buffers,
        length,
        reader,
        block_offset,
        is_little_endian,
        compression,
        scratch,
    )?;

    let offsets = if let DataType::Union(_, _, mode) = data_type {
        if !mode.is_sparse() {
            Some(read_buffer(
                buffers,
                length,
                reader,
                block_offset,
                is_little_endian,
                compression,
                scratch,
            )?)
        } else {
            None
        }
    } else {
        unreachable!()
    };

    let fields = UnionArray::get_fields(&data_type);

    let fields = fields
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
                None,
                version,
                scratch,
            )
        })
        .collect::<PolarsResult<Vec<_>>>()?;

    UnionArray::try_new(data_type, types, fields, offsets)
}

pub fn skip_union(
    field_nodes: &mut VecDeque<Node>,
    data_type: &DataType,
    buffers: &mut VecDeque<IpcBuffer>,
) -> PolarsResult<()> {
    let _ = field_nodes.pop_front().ok_or_else(|| {
        polars_err!(
            oos = "IPC: unable to fetch the field for struct\n\nThe file or stream is corrupted."
        )
    })?;

    let _ = buffers
        .pop_front()
        .ok_or_else(|| polars_err!(oos = "IPC: missing validity buffer"))?;
    if let DataType::Union(_, _, Dense) = data_type {
        let _ = buffers
            .pop_front()
            .ok_or_else(|| polars_err!(oos = "IPC: missing offsets buffer"))?;
    } else {
        unreachable!()
    };

    let fields = UnionArray::get_fields(data_type);

    fields
        .iter()
        .try_for_each(|field| skip(field_nodes, field.data_type(), buffers))
}
