use std::collections::VecDeque;

use polars_error::{PolarsResult, polars_ensure, polars_err};
use polars_utils::bool::UnsafeBool;

use super::super::super::IpcField;
use super::super::deserialize::{read, read_pruned, skip};
use super::super::read_basic::*;
use super::super::{Compression, Dictionaries, IpcBuffer, Node, Version};
use super::try_get_array_length;
use crate::array::StructArray;
use crate::datatypes::{ArrowDataType, Field};
use crate::io::ipc::read::ReadSeek;
use crate::io::ipc::read::array::try_get_field_node;

#[allow(clippy::too_many_arguments)]
pub fn read_struct(
    field_nodes: &mut VecDeque<Node>,
    variadic_buffer_counts: &mut VecDeque<usize>,
    dtype: ArrowDataType,
    ipc_field: &IpcField,
    buffers: &mut VecDeque<IpcBuffer>,
    reader: &mut dyn ReadSeek,
    dictionaries: &Dictionaries,
    block_offset: u64,
    is_little_endian: bool,
    compression: Option<Compression>,
    limit: Option<usize>,
    version: Version,
    scratch: &mut Vec<u8>,
    checked: UnsafeBool,
) -> PolarsResult<StructArray> {
    let field_node = try_get_field_node(field_nodes, &dtype)?;
    let length = try_get_array_length(field_node, limit)?;

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

    let fields = StructArray::get_fields(&dtype);

    let values = fields
        .iter()
        .zip(ipc_field.fields.iter())
        .map(|(field, ipc_field)| {
            read(
                field_nodes,
                variadic_buffer_counts,
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
                checked,
            )
        })
        .collect::<PolarsResult<Vec<_>>>()?;

    StructArray::try_new(dtype, length, values, validity)
}

/// Reads a struct column as `pruned_dtype`, a struct dtype with some of the fields of `field`'s
/// struct dtype, in the same order (and recursively so for struct fields): the buffers of the
/// other fields are skipped.
#[allow(clippy::too_many_arguments)]
pub fn read_struct_pruned(
    field_nodes: &mut VecDeque<Node>,
    variadic_buffer_counts: &mut VecDeque<usize>,
    field: &Field,
    pruned_dtype: &ArrowDataType,
    ipc_field: &IpcField,
    buffers: &mut VecDeque<IpcBuffer>,
    reader: &mut dyn ReadSeek,
    dictionaries: &Dictionaries,
    block_offset: u64,
    is_little_endian: bool,
    compression: Option<Compression>,
    limit: Option<usize>,
    version: Version,
    scratch: &mut Vec<u8>,
    checked: UnsafeBool,
) -> PolarsResult<StructArray> {
    let field_node = try_get_field_node(field_nodes, &field.dtype)?;
    let length = try_get_array_length(field_node, limit)?;

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

    let fields = StructArray::get_fields(&field.dtype);
    let mut pruned_fields = StructArray::get_fields(pruned_dtype).iter().peekable();
    let mut values = Vec::with_capacity(pruned_fields.len());

    for (child, ipc_child) in fields.iter().zip(ipc_field.fields.iter()) {
        match pruned_fields.next_if(|pruned| pruned.name == child.name) {
            Some(pruned) => values.push(read_pruned(
                field_nodes,
                variadic_buffer_counts,
                child,
                &pruned.dtype,
                ipc_child,
                buffers,
                reader,
                dictionaries,
                block_offset,
                is_little_endian,
                compression,
                limit,
                version,
                scratch,
                checked,
            )?),
            None => skip(field_nodes, child.dtype(), buffers, variadic_buffer_counts)?,
        }
    }

    polars_ensure!(
        pruned_fields.next().is_none(),
        ComputeError: "IPC: the fields to read of struct '{}' are not a subsequence of its fields",
        field.name
    );

    StructArray::try_new(pruned_dtype.clone(), length, values, validity)
}

pub fn skip_struct(
    field_nodes: &mut VecDeque<Node>,
    dtype: &ArrowDataType,
    buffers: &mut VecDeque<IpcBuffer>,
    variadic_buffer_counts: &mut VecDeque<usize>,
) -> PolarsResult<()> {
    let _ = field_nodes.pop_front().ok_or_else(|| {
        polars_err!(
            oos = "IPC: unable to fetch the field for struct. The file or stream is corrupted."
        )
    })?;

    let _ = buffers
        .pop_front()
        .ok_or_else(|| polars_err!(oos = "IPC: missing validity buffer."))?;

    let fields = StructArray::get_fields(dtype);

    fields
        .iter()
        .try_for_each(|field| skip(field_nodes, field.dtype(), buffers, variadic_buffer_counts))
}
