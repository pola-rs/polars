use std::collections::VecDeque;
use std::io::{Read, Seek};
use crate::array::ViewType;
use polars_error::{polars_err, PolarsResult};

use super::super::read_basic::*;
use super::*;
use crate::array::BinaryViewArrayGeneric;
use crate::buffer::Buffer;
use crate::datatypes::ArrowDataType;
use crate::offset::Offset;

pub fn read_binview<T: ViewType + ?Sized, R: Read + Seek>(
    field_nodes: &mut VecDeque<Node>,
    data_type: ArrowDataType,
    buffers: &mut VecDeque<IpcBuffer>,
    reader: &mut R,
    block_offset: u64,
    is_little_endian: bool,
    compression: Option<Compression>,
    limit: Option<usize>,
    scratch: &mut Vec<u8>
) -> PolarsResult<BinaryViewArrayGeneric<T>> {
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
    let views: Buffer<u128> = read_buffer(buffers,
        length,
        reader,
        block_offset,
        is_little_endian,
        compression,
        scratch
    )?;

    todo!()
}