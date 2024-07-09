use std::collections::VecDeque;
use std::io::{Read, Seek};

use arrow_format::ipc::{BodyCompressionRef, MetadataVersion};
use polars_error::PolarsResult;

use super::array::*;
use super::{Dictionaries, IpcBuffer, Node};
use crate::array::*;
use crate::datatypes::{ArrowDataType, Field, PhysicalType};
use crate::io::ipc::IpcField;
use crate::{match_integer_type, with_match_primitive_type_full};

#[allow(clippy::too_many_arguments)]
pub fn read<R: Read + Seek>(
    field_nodes: &mut VecDeque<Node>,
    variadic_buffer_counts: &mut VecDeque<usize>,
    field: &Field,
    ipc_field: &IpcField,
    buffers: &mut VecDeque<IpcBuffer>,
    reader: &mut R,
    dictionaries: &Dictionaries,
    block_offset: u64,
    is_little_endian: bool,
    compression: Option<BodyCompressionRef>,
    limit: Option<usize>,
    version: MetadataVersion,
    scratch: &mut Vec<u8>,
) -> PolarsResult<Box<dyn Array>> {
    use PhysicalType::*;
    let data_type = field.data_type.clone();

    match data_type.to_physical_type() {
        Null => read_null(field_nodes, data_type, limit).map(|x| x.boxed()),
        Boolean => read_boolean(
            field_nodes,
            data_type,
            buffers,
            reader,
            block_offset,
            is_little_endian,
            compression,
            limit,
            scratch,
        )
        .map(|x| x.boxed()),
        Primitive(primitive) => with_match_primitive_type_full!(primitive, |$T| {
            read_primitive::<$T, _>(
                field_nodes,
                data_type,
                buffers,
                reader,
                block_offset,
                is_little_endian,
                compression,
                limit,
                scratch,
            )
            .map(|x| x.boxed())
        }),
        Binary => read_binary::<i32, _>(
            field_nodes,
            data_type,
            buffers,
            reader,
            block_offset,
            is_little_endian,
            compression,
            limit,
            scratch,
        )
        .map(|x| x.boxed()),
        LargeBinary => read_binary::<i64, _>(
            field_nodes,
            data_type,
            buffers,
            reader,
            block_offset,
            is_little_endian,
            compression,
            limit,
            scratch,
        )
        .map(|x| x.boxed()),
        FixedSizeBinary => read_fixed_size_binary(
            field_nodes,
            data_type,
            buffers,
            reader,
            block_offset,
            is_little_endian,
            compression,
            limit,
            scratch,
        )
        .map(|x| x.boxed()),
        Utf8 => read_utf8::<i32, _>(
            field_nodes,
            data_type,
            buffers,
            reader,
            block_offset,
            is_little_endian,
            compression,
            limit,
            scratch,
        )
        .map(|x| x.boxed()),
        LargeUtf8 => read_utf8::<i64, _>(
            field_nodes,
            data_type,
            buffers,
            reader,
            block_offset,
            is_little_endian,
            compression,
            limit,
            scratch,
        )
        .map(|x| x.boxed()),
        List => read_list::<i32, _>(
            field_nodes,
            variadic_buffer_counts,
            data_type,
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
        .map(|x| x.boxed()),
        LargeList => read_list::<i64, _>(
            field_nodes,
            variadic_buffer_counts,
            data_type,
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
        .map(|x| x.boxed()),
        FixedSizeList => read_fixed_size_list(
            field_nodes,
            variadic_buffer_counts,
            data_type,
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
        .map(|x| x.boxed()),
        Struct => read_struct(
            field_nodes,
            variadic_buffer_counts,
            data_type,
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
        .map(|x| x.boxed()),
        Dictionary(key_type) => {
            match_integer_type!(key_type, |$T| {
                read_dictionary::<$T, _>(
                    field_nodes,
                    data_type,
                    ipc_field.dictionary_id,
                    buffers,
                    reader,
                    dictionaries,
                    block_offset,
                    compression,
                    limit,
                    is_little_endian,
                    scratch,
                )
                .map(|x| x.boxed())
            })
        },
        Union => read_union(
            field_nodes,
            variadic_buffer_counts,
            data_type,
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
        .map(|x| x.boxed()),
        Map => read_map(
            field_nodes,
            variadic_buffer_counts,
            data_type,
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
        .map(|x| x.boxed()),
        Utf8View => read_binview::<str, _>(
            field_nodes,
            variadic_buffer_counts,
            data_type,
            buffers,
            reader,
            block_offset,
            is_little_endian,
            compression,
            limit,
            scratch,
        ),
        BinaryView => read_binview::<[u8], _>(
            field_nodes,
            variadic_buffer_counts,
            data_type,
            buffers,
            reader,
            block_offset,
            is_little_endian,
            compression,
            limit,
            scratch,
        ),
    }
}

pub fn skip(
    field_nodes: &mut VecDeque<Node>,
    data_type: &ArrowDataType,
    buffers: &mut VecDeque<IpcBuffer>,
    variadic_buffer_counts: &mut VecDeque<usize>,
) -> PolarsResult<()> {
    use PhysicalType::*;
    match data_type.to_physical_type() {
        Null => skip_null(field_nodes),
        Boolean => skip_boolean(field_nodes, buffers),
        Primitive(_) => skip_primitive(field_nodes, buffers),
        LargeBinary | Binary => skip_binary(field_nodes, buffers),
        LargeUtf8 | Utf8 => skip_utf8(field_nodes, buffers),
        FixedSizeBinary => skip_fixed_size_binary(field_nodes, buffers),
        List => skip_list::<i32>(field_nodes, data_type, buffers, variadic_buffer_counts),
        LargeList => skip_list::<i64>(field_nodes, data_type, buffers, variadic_buffer_counts),
        FixedSizeList => {
            skip_fixed_size_list(field_nodes, data_type, buffers, variadic_buffer_counts)
        },
        Struct => skip_struct(field_nodes, data_type, buffers, variadic_buffer_counts),
        Dictionary(_) => skip_dictionary(field_nodes, buffers),
        Union => skip_union(field_nodes, data_type, buffers, variadic_buffer_counts),
        Map => skip_map(field_nodes, data_type, buffers, variadic_buffer_counts),
        BinaryView | Utf8View => skip_binview(field_nodes, buffers, variadic_buffer_counts),
    }
}
