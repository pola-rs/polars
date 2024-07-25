use std::borrow::{Borrow, Cow};

use arrow_format::ipc::planus::Builder;
use polars_error::{polars_bail, polars_err, PolarsResult};

use super::super::IpcField;
use super::{write, write_dictionary};
use crate::array::*;
use crate::datatypes::*;
use crate::io::ipc::endianness::is_native_little_endian;
use crate::io::ipc::read::Dictionaries;
use crate::legacy::prelude::LargeListArray;
use crate::match_integer_type;
use crate::record_batch::RecordBatchT;

/// Compression codec
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Compression {
    /// LZ4 (framed)
    LZ4,
    /// ZSTD
    ZSTD,
}

/// Options declaring the behaviour of writing to IPC
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct WriteOptions {
    /// Whether the buffers should be compressed and which codec to use.
    /// Note: to use compression the crate must be compiled with feature `io_ipc_compression`.
    pub compression: Option<Compression>,
}

fn encode_dictionary(
    field: &IpcField,
    array: &dyn Array,
    options: &WriteOptions,
    dictionary_tracker: &mut DictionaryTracker,
    encoded_dictionaries: &mut Vec<EncodedData>,
) -> PolarsResult<()> {
    use PhysicalType::*;
    match array.data_type().to_physical_type() {
        Utf8 | LargeUtf8 | Binary | LargeBinary | Primitive(_) | Boolean | Null
        | FixedSizeBinary | BinaryView | Utf8View => Ok(()),
        Dictionary(key_type) => match_integer_type!(key_type, |$T| {
            let dict_id = field.dictionary_id
                .ok_or_else(|| polars_err!(InvalidOperation: "Dictionaries must have an associated id"))?;

            let emit = dictionary_tracker.insert(dict_id, array)?;

            let array = array.as_any().downcast_ref::<DictionaryArray<$T>>().unwrap();
            let values = array.values();
            encode_dictionary(field,
                values.as_ref(),
                options,
                dictionary_tracker,
                encoded_dictionaries
            )?;

            if emit {
                encoded_dictionaries.push(dictionary_batch_to_bytes::<$T>(
                    dict_id,
                    array,
                    options,
                    is_native_little_endian(),
                ));
            };
            Ok(())
        }),
        Struct => {
            let array = array.as_any().downcast_ref::<StructArray>().unwrap();
            let fields = field.fields.as_slice();
            if array.fields().len() != fields.len() {
                polars_bail!(InvalidOperation:
                    "The number of fields in a struct must equal the number of children in IpcField".to_string(),
                );
            }
            fields
                .iter()
                .zip(array.values().iter())
                .try_for_each(|(field, values)| {
                    encode_dictionary(
                        field,
                        values.as_ref(),
                        options,
                        dictionary_tracker,
                        encoded_dictionaries,
                    )
                })
        },
        List => {
            let values = array
                .as_any()
                .downcast_ref::<ListArray<i32>>()
                .unwrap()
                .values();
            let field = &field.fields[0]; // todo: error instead
            encode_dictionary(
                field,
                values.as_ref(),
                options,
                dictionary_tracker,
                encoded_dictionaries,
            )
        },
        LargeList => {
            let values = array
                .as_any()
                .downcast_ref::<ListArray<i64>>()
                .unwrap()
                .values();
            let field = &field.fields[0]; // todo: error instead
            encode_dictionary(
                field,
                values.as_ref(),
                options,
                dictionary_tracker,
                encoded_dictionaries,
            )
        },
        FixedSizeList => {
            let values = array
                .as_any()
                .downcast_ref::<FixedSizeListArray>()
                .unwrap()
                .values();
            let field = &field.fields[0]; // todo: error instead
            encode_dictionary(
                field,
                values.as_ref(),
                options,
                dictionary_tracker,
                encoded_dictionaries,
            )
        },
        Union => {
            let values = array
                .as_any()
                .downcast_ref::<UnionArray>()
                .unwrap()
                .fields();
            let fields = &field.fields[..]; // todo: error instead
            if values.len() != fields.len() {
                polars_bail!(InvalidOperation:
                    "The number of fields in a union must equal the number of children in IpcField"
                );
            }
            fields
                .iter()
                .zip(values.iter())
                .try_for_each(|(field, values)| {
                    encode_dictionary(
                        field,
                        values.as_ref(),
                        options,
                        dictionary_tracker,
                        encoded_dictionaries,
                    )
                })
        },
        Map => {
            let values = array.as_any().downcast_ref::<MapArray>().unwrap().field();
            let field = &field.fields[0]; // todo: error instead
            encode_dictionary(
                field,
                values.as_ref(),
                options,
                dictionary_tracker,
                encoded_dictionaries,
            )
        },
    }
}

pub fn encode_chunk(
    chunk: &RecordBatchT<Box<dyn Array>>,
    fields: &[IpcField],
    dictionary_tracker: &mut DictionaryTracker,
    options: &WriteOptions,
) -> PolarsResult<(Vec<EncodedData>, EncodedData)> {
    let mut encoded_message = EncodedData::default();
    let encoded_dictionaries = encode_chunk_amortized(
        chunk,
        fields,
        dictionary_tracker,
        options,
        &mut encoded_message,
    )?;
    Ok((encoded_dictionaries, encoded_message))
}

// Amortizes `EncodedData` allocation.
pub fn encode_chunk_amortized(
    chunk: &RecordBatchT<Box<dyn Array>>,
    fields: &[IpcField],
    dictionary_tracker: &mut DictionaryTracker,
    options: &WriteOptions,
    encoded_message: &mut EncodedData,
) -> PolarsResult<Vec<EncodedData>> {
    let mut encoded_dictionaries = vec![];

    for (field, array) in fields.iter().zip(chunk.as_ref()) {
        encode_dictionary(
            field,
            array.as_ref(),
            options,
            dictionary_tracker,
            &mut encoded_dictionaries,
        )?;
    }

    chunk_to_bytes_amortized(chunk, options, encoded_message);

    Ok(encoded_dictionaries)
}

fn serialize_compression(
    compression: Option<Compression>,
) -> Option<Box<arrow_format::ipc::BodyCompression>> {
    if let Some(compression) = compression {
        let codec = match compression {
            Compression::LZ4 => arrow_format::ipc::CompressionType::Lz4Frame,
            Compression::ZSTD => arrow_format::ipc::CompressionType::Zstd,
        };
        Some(Box::new(arrow_format::ipc::BodyCompression {
            codec,
            method: arrow_format::ipc::BodyCompressionMethod::Buffer,
        }))
    } else {
        None
    }
}

fn set_variadic_buffer_counts(counts: &mut Vec<i64>, array: &dyn Array) {
    match array.data_type() {
        ArrowDataType::Utf8View => {
            let array = array.as_any().downcast_ref::<Utf8ViewArray>().unwrap();
            counts.push(array.data_buffers().len() as i64);
        },
        ArrowDataType::BinaryView => {
            let array = array.as_any().downcast_ref::<BinaryViewArray>().unwrap();
            counts.push(array.data_buffers().len() as i64);
        },
        ArrowDataType::Struct(_) => {
            let array = array.as_any().downcast_ref::<StructArray>().unwrap();
            for array in array.values() {
                set_variadic_buffer_counts(counts, array.as_ref())
            }
        },
        ArrowDataType::LargeList(_) => {
            let array = array.as_any().downcast_ref::<LargeListArray>().unwrap();
            set_variadic_buffer_counts(counts, array.values().as_ref())
        },
        ArrowDataType::FixedSizeList(_, _) => {
            let array = array.as_any().downcast_ref::<FixedSizeListArray>().unwrap();
            set_variadic_buffer_counts(counts, array.values().as_ref())
        },
        ArrowDataType::Dictionary(_, _, _) => {
            let array = array
                .as_any()
                .downcast_ref::<DictionaryArray<u32>>()
                .unwrap();
            set_variadic_buffer_counts(counts, array.values().as_ref())
        },
        _ => (),
    }
}

fn gc_bin_view<'a, T: ViewType + ?Sized>(
    arr: &'a Box<dyn Array>,
    concrete_arr: &'a BinaryViewArrayGeneric<T>,
) -> Cow<'a, Box<dyn Array>> {
    let bytes_len = concrete_arr.total_bytes_len();
    let buffer_len = concrete_arr.total_buffer_len();
    let extra_len = buffer_len.saturating_sub(bytes_len);
    if extra_len < bytes_len.min(1024) {
        // We can afford some tiny waste.
        Cow::Borrowed(arr)
    } else {
        // Force GC it.
        Cow::Owned(concrete_arr.clone().gc().boxed())
    }
}

/// Write [`RecordBatchT`] into two sets of bytes, one for the header (ipc::Schema::Message) and the
/// other for the batch's data
fn chunk_to_bytes_amortized(
    chunk: &RecordBatchT<Box<dyn Array>>,
    options: &WriteOptions,
    encoded_message: &mut EncodedData,
) {
    let mut nodes: Vec<arrow_format::ipc::FieldNode> = vec![];
    let mut buffers: Vec<arrow_format::ipc::Buffer> = vec![];
    let mut arrow_data = std::mem::take(&mut encoded_message.arrow_data);
    arrow_data.clear();

    let mut offset = 0;
    let mut variadic_buffer_counts = vec![];
    for array in chunk.arrays() {
        // We don't want to write all buffers in sliced arrays.
        let array = match array.data_type() {
            ArrowDataType::BinaryView => {
                let concrete_arr = array.as_any().downcast_ref::<BinaryViewArray>().unwrap();
                gc_bin_view(array, concrete_arr)
            },
            ArrowDataType::Utf8View => {
                let concrete_arr = array.as_any().downcast_ref::<Utf8ViewArray>().unwrap();
                gc_bin_view(array, concrete_arr)
            },
            _ => Cow::Borrowed(array),
        };
        let array = array.as_ref().as_ref();

        set_variadic_buffer_counts(&mut variadic_buffer_counts, array);

        write(
            array,
            &mut buffers,
            &mut arrow_data,
            &mut nodes,
            &mut offset,
            is_native_little_endian(),
            options.compression,
        )
    }

    let variadic_buffer_counts = if variadic_buffer_counts.is_empty() {
        None
    } else {
        Some(variadic_buffer_counts)
    };

    let compression = serialize_compression(options.compression);

    let message = arrow_format::ipc::Message {
        version: arrow_format::ipc::MetadataVersion::V5,
        header: Some(arrow_format::ipc::MessageHeader::RecordBatch(Box::new(
            arrow_format::ipc::RecordBatch {
                length: chunk.len() as i64,
                nodes: Some(nodes),
                buffers: Some(buffers),
                compression,
                variadic_buffer_counts,
            },
        ))),
        body_length: arrow_data.len() as i64,
        custom_metadata: None,
    };

    let mut builder = Builder::new();
    let ipc_message = builder.finish(&message, None);
    encoded_message.ipc_message = ipc_message.to_vec();
    encoded_message.arrow_data = arrow_data
}

/// Write dictionary values into two sets of bytes, one for the header (ipc::Schema::Message) and the
/// other for the data
fn dictionary_batch_to_bytes<K: DictionaryKey>(
    dict_id: i64,
    array: &DictionaryArray<K>,
    options: &WriteOptions,
    is_little_endian: bool,
) -> EncodedData {
    let mut nodes: Vec<arrow_format::ipc::FieldNode> = vec![];
    let mut buffers: Vec<arrow_format::ipc::Buffer> = vec![];
    let mut arrow_data: Vec<u8> = vec![];
    let mut variadic_buffer_counts = vec![];
    set_variadic_buffer_counts(&mut variadic_buffer_counts, array.values().as_ref());

    let variadic_buffer_counts = if variadic_buffer_counts.is_empty() {
        None
    } else {
        Some(variadic_buffer_counts)
    };

    let length = write_dictionary(
        array,
        &mut buffers,
        &mut arrow_data,
        &mut nodes,
        &mut 0,
        is_little_endian,
        options.compression,
        false,
    );

    let compression = serialize_compression(options.compression);

    let message = arrow_format::ipc::Message {
        version: arrow_format::ipc::MetadataVersion::V5,
        header: Some(arrow_format::ipc::MessageHeader::DictionaryBatch(Box::new(
            arrow_format::ipc::DictionaryBatch {
                id: dict_id,
                data: Some(Box::new(arrow_format::ipc::RecordBatch {
                    length: length as i64,
                    nodes: Some(nodes),
                    buffers: Some(buffers),
                    compression,
                    variadic_buffer_counts,
                })),
                is_delta: false,
            },
        ))),
        body_length: arrow_data.len() as i64,
        custom_metadata: None,
    };

    let mut builder = Builder::new();
    let ipc_message = builder.finish(&message, None);

    EncodedData {
        ipc_message: ipc_message.to_vec(),
        arrow_data,
    }
}

/// Keeps track of dictionaries that have been written, to avoid emitting the same dictionary
/// multiple times. Can optionally error if an update to an existing dictionary is attempted, which
/// isn't allowed in the `FileWriter`.
pub struct DictionaryTracker {
    pub dictionaries: Dictionaries,
    pub cannot_replace: bool,
}

impl DictionaryTracker {
    /// Keep track of the dictionary with the given ID and values. Behavior:
    ///
    /// * If this ID has been written already and has the same data, return `Ok(false)` to indicate
    ///   that the dictionary was not actually inserted (because it's already been seen).
    /// * If this ID has been written already but with different data, and this tracker is
    ///   configured to return an error, return an error.
    /// * If the tracker has not been configured to error on replacement or this dictionary
    ///   has never been seen before, return `Ok(true)` to indicate that the dictionary was just
    ///   inserted.
    pub fn insert(&mut self, dict_id: i64, array: &dyn Array) -> PolarsResult<bool> {
        let values = match array.data_type() {
            ArrowDataType::Dictionary(key_type, _, _) => {
                match_integer_type!(key_type, |$T| {
                    let array = array
                        .as_any()
                        .downcast_ref::<DictionaryArray<$T>>()
                        .unwrap();
                    array.values()
                })
            },
            _ => unreachable!(),
        };

        // If a dictionary with this id was already emitted, check if it was the same.
        if let Some(last) = self.dictionaries.get(&dict_id) {
            if last.as_ref() == values.as_ref() {
                // Same dictionary values => no need to emit it again
                return Ok(false);
            } else if self.cannot_replace {
                polars_bail!(InvalidOperation:
                    "Dictionary replacement detected when writing IPC file format. \
                     Arrow IPC files only support a single dictionary for a given field \
                     across all batches."
                );
            }
        };

        self.dictionaries.insert(dict_id, values.clone());
        Ok(true)
    }
}

/// Stores the encoded data, which is an ipc::Schema::Message, and optional Arrow data
#[derive(Debug, Default)]
pub struct EncodedData {
    /// An encoded ipc::Schema::Message
    pub ipc_message: Vec<u8>,
    /// Arrow buffers to be written, should be an empty vec for schema messages
    pub arrow_data: Vec<u8>,
}

/// Calculate an 8-byte boundary and return the number of bytes needed to pad to 8 bytes
#[inline]
pub(crate) fn pad_to_64(len: usize) -> usize {
    ((len + 63) & !63) - len
}

/// An array [`RecordBatchT`] with optional accompanying IPC fields.
#[derive(Debug, Clone, PartialEq)]
pub struct Record<'a> {
    columns: Cow<'a, RecordBatchT<Box<dyn Array>>>,
    fields: Option<Cow<'a, [IpcField]>>,
}

impl<'a> Record<'a> {
    /// Get the IPC fields for this record.
    pub fn fields(&self) -> Option<&[IpcField]> {
        self.fields.as_deref()
    }

    /// Get the Arrow columns in this record.
    pub fn columns(&self) -> &RecordBatchT<Box<dyn Array>> {
        self.columns.borrow()
    }
}

impl From<RecordBatchT<Box<dyn Array>>> for Record<'static> {
    fn from(columns: RecordBatchT<Box<dyn Array>>) -> Self {
        Self {
            columns: Cow::Owned(columns),
            fields: None,
        }
    }
}

impl<'a, F> From<(RecordBatchT<Box<dyn Array>>, Option<F>)> for Record<'a>
where
    F: Into<Cow<'a, [IpcField]>>,
{
    fn from((columns, fields): (RecordBatchT<Box<dyn Array>>, Option<F>)) -> Self {
        Self {
            columns: Cow::Owned(columns),
            fields: fields.map(|f| f.into()),
        }
    }
}

impl<'a, F> From<(&'a RecordBatchT<Box<dyn Array>>, Option<F>)> for Record<'a>
where
    F: Into<Cow<'a, [IpcField]>>,
{
    fn from((columns, fields): (&'a RecordBatchT<Box<dyn Array>>, Option<F>)) -> Self {
        Self {
            columns: Cow::Borrowed(columns),
            fields: fields.map(|f| f.into()),
        }
    }
}
