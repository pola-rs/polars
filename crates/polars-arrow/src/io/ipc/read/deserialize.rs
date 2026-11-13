use std::collections::VecDeque;

use polars_arrow_format::ipc::{BodyCompressionRef, MetadataVersion};
use polars_error::{PolarsResult, polars_bail};
use polars_utils::bool::UnsafeBool;

use super::array::*;
use super::{Dictionaries, IpcBuffer, Node};
use crate::array::*;
use crate::datatypes::{ArrowDataType, Field, PhysicalType};
use crate::io::ipc::IpcField;
use crate::io::ipc::read::ReadSeek;
use crate::types::{PrimitiveType, months_days_ns};
use crate::{match_integer_type, with_match_primitive_type_full};

#[allow(clippy::too_many_arguments)]
pub fn read(
    field_nodes: &mut VecDeque<Node>,
    variadic_buffer_counts: &mut VecDeque<usize>,
    field: &Field,
    ipc_field: &IpcField,
    buffers: &mut VecDeque<IpcBuffer>,
    reader: &mut dyn ReadSeek,
    dictionaries: &Dictionaries,
    block_offset: u64,
    is_little_endian: bool,
    compression: Option<BodyCompressionRef>,
    limit: Option<usize>,
    version: MetadataVersion,
    scratch: &mut Vec<u8>,
    checked: UnsafeBool,
) -> PolarsResult<Box<dyn Array>> {
    use PhysicalType::*;
    let dtype = field.dtype.clone();

    match dtype.to_physical_type() {
        Null => read_null(field_nodes, dtype, limit).map(|x| x.boxed()),
        Boolean => read_boolean(
            field_nodes,
            dtype,
            buffers,
            reader,
            block_offset,
            is_little_endian,
            compression,
            limit,
            scratch,
        )
        .map(|x| x.boxed()),
        Primitive(PrimitiveType::MonthDayNano) => read_primitive::<months_days_ns>(
            field_nodes,
            dtype,
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
            read_primitive::<$T>(
                field_nodes,
                dtype,
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
        Binary => read_binary::<i32>(
            field_nodes,
            dtype,
            buffers,
            reader,
            block_offset,
            is_little_endian,
            compression,
            limit,
            scratch,
            checked,
        )
        .map(|x| x.boxed()),
        LargeBinary => read_binary::<i64>(
            field_nodes,
            dtype,
            buffers,
            reader,
            block_offset,
            is_little_endian,
            compression,
            limit,
            scratch,
            checked,
        )
        .map(|x| x.boxed()),
        FixedSizeBinary => read_fixed_size_binary(
            field_nodes,
            dtype,
            buffers,
            reader,
            block_offset,
            is_little_endian,
            compression,
            limit,
            scratch,
        )
        .map(|x| x.boxed()),
        Utf8 => read_utf8::<i32>(
            field_nodes,
            dtype,
            buffers,
            reader,
            block_offset,
            is_little_endian,
            compression,
            limit,
            scratch,
            checked,
        )
        .map(|x| x.boxed()),
        LargeUtf8 => read_utf8::<i64>(
            field_nodes,
            dtype,
            buffers,
            reader,
            block_offset,
            is_little_endian,
            compression,
            limit,
            scratch,
            checked,
        )
        .map(|x| x.boxed()),
        List => read_list::<i32>(
            field_nodes,
            variadic_buffer_counts,
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
        .map(|x| x.boxed()),
        LargeList => read_list::<i64>(
            field_nodes,
            variadic_buffer_counts,
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
        .map(|x| x.boxed()),
        FixedSizeList => read_fixed_size_list(
            field_nodes,
            variadic_buffer_counts,
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
        .map(|x| x.boxed()),
        Struct => read_struct(
            field_nodes,
            variadic_buffer_counts,
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
        .map(|x| x.boxed()),
        Dictionary(key_type) => {
            match_integer_type!(key_type, |$T| {
                read_dictionary::<$T>(
                    field_nodes,
                    dtype,
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
        .map(|x| x.boxed()),
        Map => read_map(
            field_nodes,
            variadic_buffer_counts,
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
        .map(|x| x.boxed()),
        Utf8View => read_binview::<str>(
            field_nodes,
            variadic_buffer_counts,
            dtype,
            buffers,
            reader,
            block_offset,
            is_little_endian,
            compression,
            limit,
            scratch,
            checked,
        ),
        BinaryView => read_binview::<[u8]>(
            field_nodes,
            variadic_buffer_counts,
            dtype,
            buffers,
            reader,
            block_offset,
            is_little_endian,
            compression,
            limit,
            scratch,
            checked,
        ),
    }
}

/// Reads `field` as `pruned_dtype`: its own dtype, or for a struct a struct dtype with some of
/// its fields (see [`read_struct_pruned`]), or for a list a list of such an element.
#[allow(clippy::too_many_arguments)]
pub fn read_pruned(
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
    compression: Option<BodyCompressionRef>,
    limit: Option<usize>,
    version: MetadataVersion,
    scratch: &mut Vec<u8>,
    checked: UnsafeBool,
) -> PolarsResult<Box<dyn Array>> {
    if &field.dtype == pruned_dtype {
        return read(
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
        );
    }

    match (&field.dtype, pruned_dtype) {
        (ArrowDataType::List(_), ArrowDataType::List(_)) => read_list_pruned::<i32>(
            field_nodes,
            variadic_buffer_counts,
            &field.dtype,
            pruned_dtype.clone(),
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
        .map(|x| x.boxed()),
        (ArrowDataType::LargeList(_), ArrowDataType::LargeList(_)) => read_list_pruned::<i64>(
            field_nodes,
            variadic_buffer_counts,
            &field.dtype,
            pruned_dtype.clone(),
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
        .map(|x| x.boxed()),
        (ArrowDataType::FixedSizeList(_, a), ArrowDataType::FixedSizeList(_, b)) if a == b => {
            read_fixed_size_list_pruned(
                field_nodes,
                variadic_buffer_counts,
                &field.dtype,
                pruned_dtype.clone(),
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
            .map(|x| x.boxed())
        },
        (ArrowDataType::Struct(_), ArrowDataType::Struct(_)) => read_struct_pruned(
            field_nodes,
            variadic_buffer_counts,
            field,
            pruned_dtype,
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
        .map(|x| x.boxed()),
        _ => polars_bail!(
            ComputeError: "IPC: cannot read field '{}' of dtype {:?} as {:?}",
            field.name, field.dtype, pruned_dtype
        ),
    }
}

pub fn skip(
    field_nodes: &mut VecDeque<Node>,
    dtype: &ArrowDataType,
    buffers: &mut VecDeque<IpcBuffer>,
    variadic_buffer_counts: &mut VecDeque<usize>,
) -> PolarsResult<()> {
    use PhysicalType::*;
    match dtype.to_physical_type() {
        Null => skip_null(field_nodes),
        Boolean => skip_boolean(field_nodes, buffers),
        Primitive(_) => skip_primitive(field_nodes, buffers),
        LargeBinary | Binary => skip_binary(field_nodes, buffers),
        LargeUtf8 | Utf8 => skip_utf8(field_nodes, buffers),
        FixedSizeBinary => skip_fixed_size_binary(field_nodes, buffers),
        List => skip_list::<i32>(field_nodes, dtype, buffers, variadic_buffer_counts),
        LargeList => skip_list::<i64>(field_nodes, dtype, buffers, variadic_buffer_counts),
        FixedSizeList => skip_fixed_size_list(field_nodes, dtype, buffers, variadic_buffer_counts),
        Struct => skip_struct(field_nodes, dtype, buffers, variadic_buffer_counts),
        Dictionary(_) => skip_dictionary(field_nodes, buffers),
        Union => skip_union(field_nodes, dtype, buffers, variadic_buffer_counts),
        Map => skip_map(field_nodes, dtype, buffers, variadic_buffer_counts),
        BinaryView | Utf8View => skip_binview(field_nodes, buffers, variadic_buffer_counts),
    }
}
