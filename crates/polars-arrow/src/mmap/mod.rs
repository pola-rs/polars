#![allow(unsafe_op_in_unsafe_fn)]
//! Memory maps regions defined on the IPC format into [`Array`].
use std::collections::VecDeque;
use std::sync::Arc;

mod array;

use arrow_format::ipc::planus::ReadAsRoot;
use arrow_format::ipc::{Block, DictionaryBatchRef, MessageRef, RecordBatchRef};
use polars_error::{PolarsResult, polars_bail, polars_err, to_compute_err};
use polars_utils::pl_str::PlSmallStr;

use crate::array::Array;
use crate::datatypes::{ArrowDataType, ArrowSchema, Field};
use crate::io::ipc::read::file::{get_dictionary_batch, get_record_batch};
use crate::io::ipc::read::{
    Dictionaries, FileMetadata, IpcBuffer, Node, OutOfSpecKind, first_dict_field,
};
use crate::io::ipc::{CONTINUATION_MARKER, IpcField};
use crate::record_batch::RecordBatchT;

fn read_message(
    mut bytes: &[u8],
    block: arrow_format::ipc::Block,
) -> PolarsResult<(MessageRef, usize)> {
    let offset: usize = block.offset.try_into().map_err(
        |_err| polars_err!(ComputeError: "out-of-spec {:?}", OutOfSpecKind::NegativeFooterLength),
    )?;

    let block_length: usize = block.meta_data_length.try_into().map_err(
        |_err| polars_err!(ComputeError: "out-of-spec {:?}", OutOfSpecKind::NegativeFooterLength),
    )?;

    bytes = &bytes[offset..];
    let mut message_length = bytes[..4].try_into().unwrap();
    bytes = &bytes[4..];

    if message_length == CONTINUATION_MARKER {
        // continuation marker encountered, read message next
        message_length = bytes[..4].try_into().unwrap();
        bytes = &bytes[4..];
    };

    let message_length: usize = i32::from_le_bytes(message_length).try_into().map_err(
        |_err| polars_err!(ComputeError: "out-of-spec {:?}", OutOfSpecKind::NegativeFooterLength),
    )?;

    let message = arrow_format::ipc::MessageRef::read_as_root(&bytes[..message_length])
        .map_err(|err| polars_err!(ComputeError: "out-of-spec {:?}", OutOfSpecKind::InvalidFlatbufferMessage(err)))?;

    Ok((message, offset + block_length))
}

fn get_buffers_nodes(batch: RecordBatchRef) -> PolarsResult<(VecDeque<IpcBuffer>, VecDeque<Node>)> {
    let compression = batch.compression().map_err(to_compute_err)?;
    if compression.is_some() {
        polars_bail!(ComputeError: "memory_map can only be done on uncompressed IPC files")
    }

    let buffers = batch
        .buffers()
        .map_err(|err| polars_err!(ComputeError: "out-of-spec {:?}", OutOfSpecKind::InvalidFlatbufferBuffers(err)))?
        .ok_or_else(|| polars_err!(ComputeError: "out-of-spec {:?}", OutOfSpecKind::MissingMessageBuffers))?;
    let buffers = buffers.iter().collect::<VecDeque<_>>();

    let field_nodes = batch
        .nodes()
        .map_err(|err| polars_err!(ComputeError: "out-of-spec {:?}", OutOfSpecKind::InvalidFlatbufferNodes(err)))?
        .ok_or_else(|| polars_err!(ComputeError: "out-of-spec {:?}", OutOfSpecKind::MissingMessageNodes))?;
    let field_nodes = field_nodes.iter().collect::<VecDeque<_>>();

    Ok((buffers, field_nodes))
}

pub(crate) unsafe fn mmap_record<T: AsRef<[u8]>>(
    fields: &ArrowSchema,
    ipc_fields: &[IpcField],
    data: Arc<T>,
    batch: RecordBatchRef,
    offset: usize,
    dictionaries: &Dictionaries,
) -> PolarsResult<RecordBatchT<Box<dyn Array>>> {
    let (mut buffers, mut field_nodes) = get_buffers_nodes(batch)?;
    let mut variadic_buffer_counts = batch
        .variadic_buffer_counts()
        .map_err(|err| polars_err!(oos = OutOfSpecKind::InvalidFlatbufferRecordBatches(err)))?
        .map(|v| v.iter().map(|v| v as usize).collect::<VecDeque<usize>>())
        .unwrap_or_else(VecDeque::new);

    let length = batch
        .length()
        .map_err(|_| polars_err!(oos = OutOfSpecKind::MissingData))
        .unwrap()
        .try_into()
        .map_err(|_| polars_err!(oos = OutOfSpecKind::NegativeFooterLength))?;

    fields
        .iter_values()
        .map(|f| &f.dtype)
        .cloned()
        .zip(ipc_fields)
        .map(|(dtype, ipc_field)| {
            array::mmap(
                data.clone(),
                offset,
                dtype,
                ipc_field,
                dictionaries,
                &mut field_nodes,
                &mut variadic_buffer_counts,
                &mut buffers,
            )
        })
        .collect::<PolarsResult<_>>()
        .and_then(|arr| {
            RecordBatchT::try_new(
                length,
                Arc::new(fields.iter_values().cloned().collect()),
                arr,
            )
        })
}

/// Memory maps an record batch from an IPC file into a [`RecordBatchT`].
/// # Errors
/// This function errors when:
/// * The IPC file is not valid
/// * the buffers on the file are un-aligned with their corresponding data. This can happen when:
///     * the file was written with 8-bit alignment
///     * the file contains type decimal 128 or 256
/// # Safety
/// The caller must ensure that `data` contains a valid buffers, for example:
/// * Offsets in variable-sized containers must be in-bounds and increasing
/// * Utf8 data is valid
pub unsafe fn mmap_unchecked<T: AsRef<[u8]>>(
    metadata: &FileMetadata,
    dictionaries: &Dictionaries,
    data: Arc<T>,
    chunk: usize,
) -> PolarsResult<RecordBatchT<Box<dyn Array>>> {
    let block = metadata.blocks[chunk];

    let (message, offset) = read_message(data.as_ref().as_ref(), block)?;
    let batch = get_record_batch(message)?;
    mmap_record(
        &metadata.schema,
        &metadata.ipc_schema.fields,
        data.clone(),
        batch,
        offset,
        dictionaries,
    )
}

unsafe fn mmap_dictionary<T: AsRef<[u8]>>(
    schema: &ArrowSchema,
    ipc_fields: &[IpcField],
    data: Arc<T>,
    block: Block,
    dictionaries: &mut Dictionaries,
) -> PolarsResult<()> {
    let (message, offset) = read_message(data.as_ref().as_ref(), block)?;
    let batch = get_dictionary_batch(&message)?;
    mmap_dictionary_from_batch(schema, ipc_fields, &data, batch, dictionaries, offset)
}

pub(crate) unsafe fn mmap_dictionary_from_batch<T: AsRef<[u8]>>(
    schema: &ArrowSchema,
    ipc_fields: &[IpcField],
    data: &Arc<T>,
    batch: DictionaryBatchRef,
    dictionaries: &mut Dictionaries,
    offset: usize,
) -> PolarsResult<()> {
    let id = batch
        .id()
        .map_err(|err| polars_err!(ComputeError: "out-of-spec {:?}", OutOfSpecKind::InvalidFlatbufferId(err)))?;
    let (first_field, first_ipc_field) = first_dict_field(id, schema, ipc_fields)?;

    let batch = batch
        .data()
        .map_err(|err| polars_err!(ComputeError: "out-of-spec {:?}", OutOfSpecKind::InvalidFlatbufferData(err)))?
        .ok_or_else(|| polars_err!(ComputeError: "out-of-spec {:?}", OutOfSpecKind::MissingData))?;

    let value_type = if let ArrowDataType::Dictionary(_, value_type, _) =
        first_field.dtype.to_logical_type()
    {
        value_type.as_ref()
    } else {
        polars_bail!(ComputeError: "out-of-spec {:?}", OutOfSpecKind::InvalidIdDataType {requested_id: id} )
    };

    // Make a fake schema for the dictionary batch.
    let field = Field::new(PlSmallStr::EMPTY, value_type.clone(), false);

    let chunk = mmap_record(
        &std::iter::once((field.name.clone(), field)).collect(),
        &[first_ipc_field.clone()],
        data.clone(),
        batch,
        offset,
        dictionaries,
    )?;

    dictionaries.insert(id, chunk.into_arrays().pop().unwrap());

    Ok(())
}

/// Memory maps dictionaries from an IPC file into
/// # Safety
/// The caller must ensure that `data` contains a valid buffers, for example:
/// * Offsets in variable-sized containers must be in-bounds and increasing
/// * Utf8 data is valid
pub unsafe fn mmap_dictionaries_unchecked<T: AsRef<[u8]>>(
    metadata: &FileMetadata,
    data: Arc<T>,
) -> PolarsResult<Dictionaries> {
    mmap_dictionaries_unchecked2(
        metadata.schema.as_ref(),
        &metadata.ipc_schema.fields,
        metadata.dictionaries.as_ref(),
        data,
    )
}

pub(crate) unsafe fn mmap_dictionaries_unchecked2<T: AsRef<[u8]>>(
    schema: &ArrowSchema,
    ipc_fields: &[IpcField],
    dictionaries: Option<&Vec<arrow_format::ipc::Block>>,
    data: Arc<T>,
) -> PolarsResult<Dictionaries> {
    let blocks = if let Some(blocks) = &dictionaries {
        blocks
    } else {
        return Ok(Default::default());
    };

    let mut dictionaries = Default::default();

    blocks.iter().cloned().try_for_each(|block| {
        mmap_dictionary(schema, ipc_fields, data.clone(), block, &mut dictionaries)
    })?;
    Ok(dictionaries)
}
