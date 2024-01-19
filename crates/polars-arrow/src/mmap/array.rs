use std::collections::VecDeque;
use std::sync::Arc;

use polars_error::{polars_bail, polars_err, PolarsResult};

use crate::array::{Array, DictionaryKey, FixedSizeListArray, ListArray, StructArray};
use crate::datatypes::ArrowDataType;
use crate::ffi::mmap::create_array;
use crate::ffi::{export_array_to_c, try_from, ArrowArray, InternalArrowArray};
use crate::io::ipc::read::{Dictionaries, IpcBuffer, Node, OutOfSpecKind};
use crate::io::ipc::IpcField;
use crate::offset::Offset;
use crate::types::NativeType;
use crate::{match_integer_type, with_match_primitive_type};

fn get_buffer_bounds(buffers: &mut VecDeque<IpcBuffer>) -> PolarsResult<(usize, usize)> {
    let buffer = buffers.pop_front().ok_or_else(
        || polars_err!(ComputeError: "out-of-spec {:?}", OutOfSpecKind::ExpectedBuffer),
    )?;

    let offset: usize = buffer.offset().try_into().map_err(
        |_| polars_err!(ComputeError: "out-of-spec {:?}", OutOfSpecKind::NegativeFooterLength),
    )?;

    let length: usize = buffer.length().try_into().map_err(
        |_| polars_err!(ComputeError: "out-of-spec {:?}", OutOfSpecKind::NegativeFooterLength),
    )?;

    Ok((offset, length))
}

fn get_buffer<'a, T: NativeType>(
    data: &'a [u8],
    block_offset: usize,
    buffers: &mut VecDeque<IpcBuffer>,
    num_rows: usize,
) -> PolarsResult<&'a [u8]> {
    let (offset, length) = get_buffer_bounds(buffers)?;

    // verify that they are in-bounds
    let values = data
        .get(block_offset + offset..block_offset + offset + length)
        .ok_or_else(|| polars_err!(ComputeError: "buffer out of bounds"))?;

    // validate alignment
    let v: &[T] = bytemuck::try_cast_slice(values)
        .map_err(|_| polars_err!(ComputeError: "buffer not aligned for mmap"))?;

    if v.len() < num_rows {
        polars_bail!(ComputeError: "buffer's length is too small in mmap",
        )
    }

    Ok(values)
}

fn get_validity<'a>(
    data: &'a [u8],
    block_offset: usize,
    buffers: &mut VecDeque<IpcBuffer>,
    null_count: usize,
) -> PolarsResult<Option<&'a [u8]>> {
    let validity = get_buffer_bounds(buffers)?;
    let (offset, length) = validity;

    Ok(if null_count > 0 {
        // verify that they are in-bounds and get its pointer
        Some(
            data.get(block_offset + offset..block_offset + offset + length)
                .ok_or_else(|| polars_err!(ComputeError: "buffer out of bounds"))?,
        )
    } else {
        None
    })
}

fn get_num_rows_and_null_count(node: &Node) -> PolarsResult<(usize, usize)> {
    let num_rows: usize = node
        .length()
        .try_into()
        .map_err(|_| polars_err!(oos = OutOfSpecKind::NegativeFooterLength))?;

    let null_count: usize = node
        .null_count()
        .try_into()
        .map_err(|_| polars_err!(oos = OutOfSpecKind::NegativeFooterLength))?;
    Ok((num_rows, null_count))
}

fn mmap_binary<O: Offset, T: AsRef<[u8]>>(
    data: Arc<T>,
    node: &Node,
    block_offset: usize,
    buffers: &mut VecDeque<IpcBuffer>,
) -> PolarsResult<ArrowArray> {
    let (num_rows, null_count) = get_num_rows_and_null_count(node)?;
    let data_ref = data.as_ref().as_ref();

    let validity = get_validity(data_ref, block_offset, buffers, null_count)?.map(|x| x.as_ptr());

    let offsets = get_buffer::<O>(data_ref, block_offset, buffers, num_rows + 1)?.as_ptr();
    let values = get_buffer::<u8>(data_ref, block_offset, buffers, 0)?.as_ptr();

    // NOTE: offsets and values invariants are _not_ validated
    Ok(unsafe {
        create_array(
            data,
            num_rows,
            null_count,
            [validity, Some(offsets), Some(values)].into_iter(),
            [].into_iter(),
            None,
            None,
        )
    })
}

fn mmap_binview<T: AsRef<[u8]>>(
    data: Arc<T>,
    node: &Node,
    block_offset: usize,
    buffers: &mut VecDeque<IpcBuffer>,
) -> PolarsResult<ArrowArray> {
    let (num_rows, null_count) = get_num_rows_and_null_count(node)?;
    let data_ref = data.as_ref().as_ref();

    let validity = get_validity(data_ref, block_offset, buffers, null_count)?.map(|x| x.as_ptr());

    let views = get_buffer::<u128>(data_ref, block_offset, buffers, num_rows + 1)?.as_ptr();
    todo!()
    // buffers

    //
    // // NOTE: offsets and values invariants are _not_ validated
    // Ok(unsafe {
    //     create_array(
    //         data,
    //         num_rows,
    //         null_count,
    //         [validity, Some(offsets), Some(values)].into_iter(),
    //         [].into_iter(),
    //         None,
    //         None,
    //     )
    // })
}

fn mmap_fixed_size_binary<T: AsRef<[u8]>>(
    data: Arc<T>,
    node: &Node,
    block_offset: usize,
    buffers: &mut VecDeque<IpcBuffer>,
    data_type: &ArrowDataType,
) -> PolarsResult<ArrowArray> {
    let bytes_per_row = if let ArrowDataType::FixedSizeBinary(bytes_per_row) = data_type {
        bytes_per_row
    } else {
        polars_bail!(ComputeError: "out-of-spec {:?}", OutOfSpecKind::InvalidDataType);
    };
    let (num_rows, null_count) = get_num_rows_and_null_count(node)?;

    let data_ref = data.as_ref().as_ref();

    let validity = get_validity(data_ref, block_offset, buffers, null_count)?.map(|x| x.as_ptr());
    let values =
        get_buffer::<u8>(data_ref, block_offset, buffers, num_rows * bytes_per_row)?.as_ptr();

    Ok(unsafe {
        create_array(
            data,
            num_rows,
            null_count,
            [validity, Some(values)].into_iter(),
            [].into_iter(),
            None,
            None,
        )
    })
}

fn mmap_null<T: AsRef<[u8]>>(
    data: Arc<T>,
    node: &Node,
    _block_offset: usize,
    _buffers: &mut VecDeque<IpcBuffer>,
) -> PolarsResult<ArrowArray> {
    let (num_rows, null_count) = get_num_rows_and_null_count(node)?;

    Ok(unsafe {
        create_array(
            data,
            num_rows,
            null_count,
            [].into_iter(),
            [].into_iter(),
            None,
            None,
        )
    })
}

fn mmap_boolean<T: AsRef<[u8]>>(
    data: Arc<T>,
    node: &Node,
    block_offset: usize,
    buffers: &mut VecDeque<IpcBuffer>,
) -> PolarsResult<ArrowArray> {
    let (num_rows, null_count) = get_num_rows_and_null_count(node)?;

    let data_ref = data.as_ref().as_ref();

    let validity = get_validity(data_ref, block_offset, buffers, null_count)?.map(|x| x.as_ptr());

    let values = get_buffer_bounds(buffers)?;
    let (offset, length) = values;

    // verify that they are in-bounds and get its pointer
    let values = data_ref[block_offset + offset..block_offset + offset + length].as_ptr();

    Ok(unsafe {
        create_array(
            data,
            num_rows,
            null_count,
            [validity, Some(values)].into_iter(),
            [].into_iter(),
            None,
            None,
        )
    })
}

fn mmap_primitive<P: NativeType, T: AsRef<[u8]>>(
    data: Arc<T>,
    node: &Node,
    block_offset: usize,
    buffers: &mut VecDeque<IpcBuffer>,
) -> PolarsResult<ArrowArray> {
    let data_ref = data.as_ref().as_ref();
    let (num_rows, null_count) = get_num_rows_and_null_count(node)?;

    let validity = get_validity(data_ref, block_offset, buffers, null_count)?.map(|x| x.as_ptr());

    let values = get_buffer::<P>(data_ref, block_offset, buffers, num_rows)?.as_ptr();

    Ok(unsafe {
        create_array(
            data,
            num_rows,
            null_count,
            [validity, Some(values)].into_iter(),
            [].into_iter(),
            None,
            None,
        )
    })
}

#[allow(clippy::too_many_arguments)]
fn mmap_list<O: Offset, T: AsRef<[u8]>>(
    data: Arc<T>,
    node: &Node,
    block_offset: usize,
    data_type: &ArrowDataType,
    ipc_field: &IpcField,
    dictionaries: &Dictionaries,
    field_nodes: &mut VecDeque<Node>,
    variadic_buffer_counts: &mut VecDeque<usize>,
    buffers: &mut VecDeque<IpcBuffer>,
) -> PolarsResult<ArrowArray> {
    let child = ListArray::<O>::try_get_child(data_type)?.data_type();
    let (num_rows, null_count) = get_num_rows_and_null_count(node)?;

    let data_ref = data.as_ref().as_ref();

    let validity = get_validity(data_ref, block_offset, buffers, null_count)?.map(|x| x.as_ptr());

    let offsets = get_buffer::<O>(data_ref, block_offset, buffers, num_rows + 1)?.as_ptr();

    let values = get_array(
        data.clone(),
        block_offset,
        child,
        &ipc_field.fields[0],
        dictionaries,
        field_nodes,
        variadic_buffer_counts,
        buffers,
    )?;

    // NOTE: offsets and values invariants are _not_ validated
    Ok(unsafe {
        create_array(
            data,
            num_rows,
            null_count,
            [validity, Some(offsets)].into_iter(),
            [values].into_iter(),
            None,
            None,
        )
    })
}

#[allow(clippy::too_many_arguments)]
fn mmap_fixed_size_list<T: AsRef<[u8]>>(
    data: Arc<T>,
    node: &Node,
    block_offset: usize,
    data_type: &ArrowDataType,
    ipc_field: &IpcField,
    dictionaries: &Dictionaries,
    field_nodes: &mut VecDeque<Node>,
    variadic_buffer_counts: &mut VecDeque<usize>,
    buffers: &mut VecDeque<IpcBuffer>,
) -> PolarsResult<ArrowArray> {
    let child = FixedSizeListArray::try_child_and_size(data_type)?
        .0
        .data_type();
    let (num_rows, null_count) = get_num_rows_and_null_count(node)?;

    let data_ref = data.as_ref().as_ref();

    let validity = get_validity(data_ref, block_offset, buffers, null_count)?.map(|x| x.as_ptr());

    let values = get_array(
        data.clone(),
        block_offset,
        child,
        &ipc_field.fields[0],
        dictionaries,
        field_nodes,
        variadic_buffer_counts,
        buffers,
    )?;

    Ok(unsafe {
        create_array(
            data,
            num_rows,
            null_count,
            [validity].into_iter(),
            [values].into_iter(),
            None,
            None,
        )
    })
}

#[allow(clippy::too_many_arguments)]
fn mmap_struct<T: AsRef<[u8]>>(
    data: Arc<T>,
    node: &Node,
    block_offset: usize,
    data_type: &ArrowDataType,
    ipc_field: &IpcField,
    dictionaries: &Dictionaries,
    field_nodes: &mut VecDeque<Node>,
    variadic_buffer_counts: &mut VecDeque<usize>,
    buffers: &mut VecDeque<IpcBuffer>,
) -> PolarsResult<ArrowArray> {
    let children = StructArray::try_get_fields(data_type)?;
    let (num_rows, null_count) = get_num_rows_and_null_count(node)?;

    let data_ref = data.as_ref().as_ref();

    let validity = get_validity(data_ref, block_offset, buffers, null_count)?.map(|x| x.as_ptr());

    let values = children
        .iter()
        .map(|f| &f.data_type)
        .zip(ipc_field.fields.iter())
        .map(|(child, ipc)| {
            get_array(
                data.clone(),
                block_offset,
                child,
                ipc,
                dictionaries,
                field_nodes,
                variadic_buffer_counts,
                buffers,
            )
        })
        .collect::<PolarsResult<Vec<_>>>()?;

    Ok(unsafe {
        create_array(
            data,
            num_rows,
            null_count,
            [validity].into_iter(),
            values.into_iter(),
            None,
            None,
        )
    })
}

#[allow(clippy::too_many_arguments)]
fn mmap_dict<K: DictionaryKey, T: AsRef<[u8]>>(
    data: Arc<T>,
    node: &Node,
    block_offset: usize,
    _: &ArrowDataType,
    ipc_field: &IpcField,
    dictionaries: &Dictionaries,
    _: &mut VecDeque<Node>,
    buffers: &mut VecDeque<IpcBuffer>,
) -> PolarsResult<ArrowArray> {
    let (num_rows, null_count) = get_num_rows_and_null_count(node)?;

    let data_ref = data.as_ref().as_ref();

    let dictionary = dictionaries
        .get(&ipc_field.dictionary_id.unwrap())
        .ok_or_else(|| polars_err!(ComputeError: "out-of-spec: missing dictionary"))?
        .clone();

    let validity = get_validity(data_ref, block_offset, buffers, null_count)?.map(|x| x.as_ptr());

    let values = get_buffer::<K>(data_ref, block_offset, buffers, num_rows)?.as_ptr();

    Ok(unsafe {
        create_array(
            data,
            num_rows,
            null_count,
            [validity, Some(values)].into_iter(),
            [].into_iter(),
            Some(export_array_to_c(dictionary)),
            None,
        )
    })
}

fn get_array<T: AsRef<[u8]>>(
    data: Arc<T>,
    block_offset: usize,
    data_type: &ArrowDataType,
    ipc_field: &IpcField,
    dictionaries: &Dictionaries,
    field_nodes: &mut VecDeque<Node>,
    variadic_buffer_counts: &mut VecDeque<usize>,
    buffers: &mut VecDeque<IpcBuffer>,
) -> PolarsResult<ArrowArray> {
    use crate::datatypes::PhysicalType::*;
    let node = field_nodes.pop_front().ok_or_else(
        || polars_err!(ComputeError: "out-of-spec: {:?}", OutOfSpecKind::ExpectedBuffer),
    )?;

    match data_type.to_physical_type() {
        Null => mmap_null(data, &node, block_offset, buffers),
        Boolean => mmap_boolean(data, &node, block_offset, buffers),
        Primitive(p) => with_match_primitive_type!(p, |$T| {
            mmap_primitive::<$T, _>(data, &node, block_offset, buffers)
        }),
        Utf8 | Binary => mmap_binary::<i32, _>(data, &node, block_offset, buffers),
        FixedSizeBinary => mmap_fixed_size_binary(data, &node, block_offset, buffers, data_type),
        LargeBinary | LargeUtf8 => mmap_binary::<i64, _>(data, &node, block_offset, buffers),
        List => mmap_list::<i32, _>(
            data,
            &node,
            block_offset,
            data_type,
            ipc_field,
            dictionaries,
            field_nodes,
            variadic_buffer_counts,
            buffers,
        ),
        LargeList => mmap_list::<i64, _>(
            data,
            &node,
            block_offset,
            data_type,
            ipc_field,
            dictionaries,
            field_nodes,
            variadic_buffer_counts,
            buffers,
        ),
        FixedSizeList => mmap_fixed_size_list(
            data,
            &node,
            block_offset,
            data_type,
            ipc_field,
            dictionaries,
            field_nodes,
            variadic_buffer_counts,
            buffers,
        ),
        Struct => mmap_struct(
            data,
            &node,
            block_offset,
            data_type,
            ipc_field,
            dictionaries,
            field_nodes,
            variadic_buffer_counts,
            buffers,
        ),
        Dictionary(key_type) => match_integer_type!(key_type, |$T| {
            mmap_dict::<$T, _>(
                data,
                &node,
                block_offset,
                data_type,
                ipc_field,
                dictionaries,
                field_nodes,
                buffers,
            )
        }),
        _ => todo!(),
    }
}

/// Maps a memory region to an [`Array`].
pub(crate) unsafe fn mmap<T: AsRef<[u8]>>(
    data: Arc<T>,
    block_offset: usize,
    data_type: ArrowDataType,
    ipc_field: &IpcField,
    dictionaries: &Dictionaries,
    field_nodes: &mut VecDeque<Node>,
    variadic_buffer_counts: &mut VecDeque<usize>,
    buffers: &mut VecDeque<IpcBuffer>,
) -> PolarsResult<Box<dyn Array>> {
    let array = get_array(
        data,
        block_offset,
        &data_type,
        ipc_field,
        dictionaries,
        field_nodes,
        variadic_buffer_counts,
        buffers,
    )?;
    // The unsafety comes from the fact that `array` is not necessarily valid -
    // the IPC file may be corrupted (e.g. invalid offsets or non-utf8 data)
    unsafe { try_from(InternalArrowArray::new(array, data_type)) }
}
