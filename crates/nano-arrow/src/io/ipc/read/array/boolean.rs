use std::collections::VecDeque;
use std::io::{Read, Seek};

use crate::array::BooleanArray;
use crate::datatypes::DataType;
use crate::error::{Error, Result};

use super::super::read_basic::*;
use super::super::{Compression, IpcBuffer, Node, OutOfSpecKind};

#[allow(clippy::too_many_arguments)]
pub fn read_boolean<R: Read + Seek>(
    field_nodes: &mut VecDeque<Node>,
    data_type: DataType,
    buffers: &mut VecDeque<IpcBuffer>,
    reader: &mut R,
    block_offset: u64,
    is_little_endian: bool,
    compression: Option<Compression>,
    limit: Option<usize>,
    scratch: &mut Vec<u8>,
) -> Result<BooleanArray> {
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

    let values = read_bitmap(
        buffers,
        length,
        reader,
        block_offset,
        is_little_endian,
        compression,
        scratch,
    )?;
    BooleanArray::try_new(data_type, values, validity)
}

pub fn skip_boolean(
    field_nodes: &mut VecDeque<Node>,
    buffers: &mut VecDeque<IpcBuffer>,
) -> Result<()> {
    let _ = field_nodes.pop_front().ok_or_else(|| {
        Error::oos("IPC: unable to fetch the field for boolean. The file or stream is corrupted.")
    })?;

    let _ = buffers
        .pop_front()
        .ok_or_else(|| Error::oos("IPC: missing validity buffer."))?;
    let _ = buffers
        .pop_front()
        .ok_or_else(|| Error::oos("IPC: missing values buffer."))?;
    Ok(())
}
