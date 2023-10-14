use std::collections::VecDeque;
use std::convert::TryInto;
use std::io::{Read, Seek};

use polars_error::{polars_err, PolarsResult};

use super::super::read_basic::*;
use super::super::{Compression, IpcBuffer, Node, OutOfSpecKind};
use crate::array::PrimitiveArray;
use crate::datatypes::DataType;
use crate::types::NativeType;

#[allow(clippy::too_many_arguments)]
pub fn read_primitive<T: NativeType, R: Read + Seek>(
    field_nodes: &mut VecDeque<Node>,
    data_type: DataType,
    buffers: &mut VecDeque<IpcBuffer>,
    reader: &mut R,
    block_offset: u64,
    is_little_endian: bool,
    compression: Option<Compression>,
    limit: Option<usize>,
    scratch: &mut Vec<u8>,
) -> PolarsResult<PrimitiveArray<T>>
where
    Vec<u8>: TryInto<T::Bytes>,
{
    let field_node = field_nodes.pop_front().ok_or_else(|| {
        polars_err!(oos =
            "IPC: unable to fetch the field for {data_type:?}. The file or stream is corrupted."
        )
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
        .map_err(|_| polars_err!(oos = OutOfSpecKind::NegativeFooterLength))?;
    let length = limit.map(|limit| limit.min(length)).unwrap_or(length);

    let values = read_buffer(
        buffers,
        length,
        reader,
        block_offset,
        is_little_endian,
        compression,
        scratch,
    )?;
    PrimitiveArray::<T>::try_new(data_type, values, validity)
}

pub fn skip_primitive(
    field_nodes: &mut VecDeque<Node>,
    buffers: &mut VecDeque<IpcBuffer>,
) -> PolarsResult<()> {
    let _ = field_nodes.pop_front().ok_or_else(|| {
        polars_err!(
            oos = "IPC: unable to fetch the field for primitive. The file or stream is corrupted."
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
