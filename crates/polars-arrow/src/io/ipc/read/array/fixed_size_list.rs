use std::collections::VecDeque;

use polars_error::{PolarsResult, polars_ensure, polars_err};
use polars_utils::bool::UnsafeBool;

use super::super::super::IpcField;
use super::super::deserialize::{read_pruned, skip};
use super::super::read_basic::*;
use super::super::{Compression, Dictionaries, IpcBuffer, Node, ReadSeek, Version};
use crate::array::FixedSizeListArray;
use crate::datatypes::ArrowDataType;
use crate::io::ipc::read::array::try_get_field_node;

#[allow(clippy::too_many_arguments)]
pub fn read_fixed_size_list(
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
) -> PolarsResult<FixedSizeListArray> {
    read_fixed_size_list_pruned(
        field_nodes,
        variadic_buffer_counts,
        &dtype.clone(),
        dtype,
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
}

/// Reads a fixed-size list of dtype `file_dtype` as `dtype`, whose element has a pruned dtype
/// (see [`read_pruned`]).
#[allow(clippy::too_many_arguments)]
pub fn read_fixed_size_list_pruned(
    field_nodes: &mut VecDeque<Node>,
    variadic_buffer_counts: &mut VecDeque<usize>,
    file_dtype: &ArrowDataType,
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
) -> PolarsResult<FixedSizeListArray> {
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

    let (field, size) = FixedSizeListArray::get_child_and_size(file_dtype);
    polars_ensure!(size > 0, nyi = "Cannot read zero sized arrays from IPC");
    let pruned_dtype = FixedSizeListArray::get_child_and_size(&dtype).0.dtype();

    let limit = limit.map(|x| x.saturating_mul(size));

    let values = read_pruned(
        field_nodes,
        variadic_buffer_counts,
        field,
        pruned_dtype,
        &ipc_field.fields[0],
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
    )?;
    FixedSizeListArray::try_new(dtype, values.len() / size, values, validity)
}

pub fn skip_fixed_size_list(
    field_nodes: &mut VecDeque<Node>,
    dtype: &ArrowDataType,
    buffers: &mut VecDeque<IpcBuffer>,
    variadic_buffer_counts: &mut VecDeque<usize>,
) -> PolarsResult<()> {
    let _ = field_nodes.pop_front().ok_or_else(|| {
        polars_err!(oos =
            "IPC: unable to fetch the field for fixed-size list. The file or stream is corrupted."
        )
    })?;

    let _ = buffers
        .pop_front()
        .ok_or_else(|| polars_err!(oos = "IPC: missing validity buffer."))?;

    let (field, _) = FixedSizeListArray::get_child_and_size(dtype);

    skip(field_nodes, field.dtype(), buffers, variadic_buffer_counts)
}
