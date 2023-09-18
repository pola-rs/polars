use std::collections::VecDeque;
use std::io::{Read, Seek};

use crate::array::Utf8Array;
use crate::buffer::Buffer;
use crate::datatypes::DataType;
use crate::error::{Error, Result};
use crate::offset::Offset;

use super::super::read_basic::*;
use super::super::{Compression, IpcBuffer, Node, OutOfSpecKind};

#[allow(clippy::too_many_arguments)]
pub fn read_utf8<O: Offset, R: Read + Seek>(
    field_nodes: &mut VecDeque<Node>,
    data_type: DataType,
    buffers: &mut VecDeque<IpcBuffer>,
    reader: &mut R,
    block_offset: u64,
    is_little_endian: bool,
    compression: Option<Compression>,
    limit: Option<usize>,
    scratch: &mut Vec<u8>,
) -> Result<Utf8Array<O>> {
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

    let offsets: Buffer<O> = read_buffer(
        buffers,
        1 + length,
        reader,
        block_offset,
        is_little_endian,
        compression,
        scratch,
    )
    // Older versions of the IPC format sometimes do not report an offset
    .or_else(|_| Result::Ok(Buffer::<O>::from(vec![O::default()])))?;

    let last_offset = offsets.last().unwrap().to_usize();
    let values = read_buffer(
        buffers,
        last_offset,
        reader,
        block_offset,
        is_little_endian,
        compression,
        scratch,
    )?;

    Utf8Array::<O>::try_new(data_type, offsets.try_into()?, values, validity)
}

pub fn skip_utf8(
    field_nodes: &mut VecDeque<Node>,
    buffers: &mut VecDeque<IpcBuffer>,
) -> Result<()> {
    let _ = field_nodes.pop_front().ok_or_else(|| {
        Error::oos("IPC: unable to fetch the field for utf8. The file or stream is corrupted.")
    })?;

    let _ = buffers
        .pop_front()
        .ok_or_else(|| Error::oos("IPC: missing validity buffer."))?;
    let _ = buffers
        .pop_front()
        .ok_or_else(|| Error::oos("IPC: missing offsets buffer."))?;
    let _ = buffers
        .pop_front()
        .ok_or_else(|| Error::oos("IPC: missing values buffer."))?;
    Ok(())
}
