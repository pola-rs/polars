// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! Arrow IPC File and Stream Readers
//!
//! The `FileReader` and `StreamReader` have similar interfaces,
//! however the `FileReader` expects a reader that supports `Seek`ing

use std::collections::HashMap;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::sync::Arc;

use crate::array::*;
use crate::buffer::Buffer;
use crate::compute::cast;
use crate::datatypes::{DataType, Field, IntervalUnit, Schema, SchemaRef};
use crate::error::{ArrowError, Result};
use crate::ipc;
use crate::record_batch::{RecordBatch, RecordBatchReader};
use DataType::*;

const CONTINUATION_MARKER: u32 = 0xffff_ffff;

/// Read a buffer based on offset and length
fn read_buffer(buf: &ipc::Buffer, a_data: &[u8]) -> Buffer {
    let start_offset = buf.offset() as usize;
    let end_offset = start_offset + buf.length() as usize;
    let buf_data = &a_data[start_offset..end_offset];
    Buffer::from(&buf_data)
}

/// Coordinates reading arrays based on data types.
///
/// Notes:
/// * In the IPC format, null buffers are always set, but may be empty. We discard them if an array has 0 nulls
/// * Numeric values inside list arrays are often stored as 64-bit values regardless of their data type size.
///   We thus:
///     - check if the bit width of non-64-bit numbers is 64, and
///     - read the buffer as 64-bit (signed integer or float), and
///     - cast the 64-bit array to the appropriate data type
fn create_array(
    nodes: &[ipc::FieldNode],
    data_type: &DataType,
    data: &[u8],
    buffers: &[ipc::Buffer],
    dictionaries: &[Option<ArrayRef>],
    mut node_index: usize,
    mut buffer_index: usize,
) -> (ArrayRef, usize, usize) {
    use DataType::*;
    let array = match data_type {
        Utf8 | Binary => {
            let array = create_primitive_array(
                &nodes[node_index],
                data_type,
                buffers[buffer_index..buffer_index + 3]
                    .iter()
                    .map(|buf| read_buffer(buf, data))
                    .collect(),
            );
            node_index += 1;
            buffer_index += 3;
            array
        }
        FixedSizeBinary(_) => {
            let array = create_primitive_array(
                &nodes[node_index],
                data_type,
                buffers[buffer_index..buffer_index + 2]
                    .iter()
                    .map(|buf| read_buffer(buf, data))
                    .collect(),
            );
            node_index += 1;
            buffer_index += 2;
            array
        }
        List(ref list_data_type) => {
            let list_node = &nodes[node_index];
            let list_buffers: Vec<Buffer> = buffers[buffer_index..buffer_index + 2]
                .iter()
                .map(|buf| read_buffer(buf, data))
                .collect();
            node_index += 1;
            buffer_index += 2;
            let triple = create_array(
                nodes,
                list_data_type,
                data,
                buffers,
                dictionaries,
                node_index,
                buffer_index,
            );
            node_index = triple.1;
            buffer_index = triple.2;

            create_list_array(list_node, data_type, &list_buffers[..], triple.0)
        }
        FixedSizeList(ref list_data_type, _) => {
            let list_node = &nodes[node_index];
            let list_buffers: Vec<Buffer> = buffers[buffer_index..=buffer_index]
                .iter()
                .map(|buf| read_buffer(buf, data))
                .collect();
            node_index += 1;
            buffer_index += 1;
            let triple = create_array(
                nodes,
                list_data_type,
                data,
                buffers,
                dictionaries,
                node_index,
                buffer_index,
            );
            node_index = triple.1;
            buffer_index = triple.2;

            create_list_array(list_node, data_type, &list_buffers[..], triple.0)
        }
        Struct(struct_fields) => {
            let struct_node = &nodes[node_index];
            let null_buffer: Buffer = read_buffer(&buffers[buffer_index], data);
            node_index += 1;
            buffer_index += 1;

            // read the arrays for each field
            let mut struct_arrays = vec![];
            // TODO investigate whether just knowing the number of buffers could
            // still work
            for struct_field in struct_fields {
                let triple = create_array(
                    nodes,
                    struct_field.data_type(),
                    data,
                    buffers,
                    dictionaries,
                    node_index,
                    buffer_index,
                );
                node_index = triple.1;
                buffer_index = triple.2;
                struct_arrays.push((struct_field.clone(), triple.0));
            }
            let null_count = struct_node.null_count() as usize;
            let struct_array = if null_count > 0 {
                // create struct array from fields, arrays and null data
                StructArray::from((
                    struct_arrays,
                    null_buffer,
                    struct_node.null_count() as usize,
                ))
            } else {
                StructArray::from(struct_arrays)
            };
            Arc::new(struct_array)
        }
        // Create dictionary array from RecordBatch
        Dictionary(_, _) => {
            let index_node = &nodes[node_index];
            let index_buffers: Vec<Buffer> = buffers[buffer_index..buffer_index + 2]
                .iter()
                .map(|buf| read_buffer(buf, data))
                .collect();
            let value_array = dictionaries[node_index].clone().unwrap();
            node_index += 1;
            buffer_index += 2;

            create_dictionary_array(
                index_node,
                data_type,
                &index_buffers[..],
                value_array,
            )
        }
        Null => {
            let length = nodes[node_index].length() as usize;
            let data = ArrayData::builder(data_type.clone())
                .len(length)
                .offset(0)
                .build();
            node_index += 1;
            // no buffer increases
            make_array(data)
        }
        _ => {
            let array = create_primitive_array(
                &nodes[node_index],
                data_type,
                buffers[buffer_index..buffer_index + 2]
                    .iter()
                    .map(|buf| read_buffer(buf, data))
                    .collect(),
            );
            node_index += 1;
            buffer_index += 2;
            array
        }
    };
    (array, node_index, buffer_index)
}

/// Reads the correct number of buffers based on data type and null_count, and creates a
/// primitive array ref
fn create_primitive_array(
    field_node: &ipc::FieldNode,
    data_type: &DataType,
    buffers: Vec<Buffer>,
) -> ArrayRef {
    let length = field_node.length() as usize;
    let null_count = field_node.null_count() as usize;
    let array_data = match data_type {
        Utf8 | Binary => {
            // read 3 buffers
            let mut builder = ArrayData::builder(data_type.clone())
                .len(length)
                .buffers(buffers[1..3].to_vec())
                .offset(0);
            if null_count > 0 {
                builder = builder
                    .null_count(null_count)
                    .null_bit_buffer(buffers[0].clone())
            }
            builder.build()
        }
        FixedSizeBinary(_) => {
            // read 3 buffers
            let mut builder = ArrayData::builder(data_type.clone())
                .len(length)
                .buffers(buffers[1..2].to_vec())
                .offset(0);
            if null_count > 0 {
                builder = builder
                    .null_count(null_count)
                    .null_bit_buffer(buffers[0].clone())
            }
            builder.build()
        }
        Int8
        | Int16
        | Int32
        | UInt8
        | UInt16
        | UInt32
        | Time32(_)
        | Date32(_)
        | Interval(IntervalUnit::YearMonth) => {
            if buffers[1].len() / 8 == length {
                // interpret as a signed i64, and cast appropriately
                let mut builder = ArrayData::builder(DataType::Int64)
                    .len(length)
                    .buffers(buffers[1..].to_vec())
                    .offset(0);
                if null_count > 0 {
                    builder = builder
                        .null_count(null_count)
                        .null_bit_buffer(buffers[0].clone())
                }
                let values = Arc::new(Int64Array::from(builder.build())) as ArrayRef;
                // this cast is infallible, the unwrap is safe
                let casted = cast(&values, data_type).unwrap();
                casted.data()
            } else {
                let mut builder = ArrayData::builder(data_type.clone())
                    .len(length)
                    .buffers(buffers[1..].to_vec())
                    .offset(0);
                if null_count > 0 {
                    builder = builder
                        .null_count(null_count)
                        .null_bit_buffer(buffers[0].clone())
                }
                builder.build()
            }
        }
        Float32 => {
            if buffers[1].len() / 8 == length {
                // interpret as a f64, and cast appropriately
                let mut builder = ArrayData::builder(DataType::Float64)
                    .len(length)
                    .buffers(buffers[1..].to_vec())
                    .offset(0);
                if null_count > 0 {
                    builder = builder
                        .null_count(null_count)
                        .null_bit_buffer(buffers[0].clone())
                }
                let values = Arc::new(Float64Array::from(builder.build())) as ArrayRef;
                // this cast is infallible, the unwrap is safe
                let casted = cast(&values, data_type).unwrap();
                casted.data()
            } else {
                let mut builder = ArrayData::builder(data_type.clone())
                    .len(length)
                    .buffers(buffers[1..].to_vec())
                    .offset(0);
                if null_count > 0 {
                    builder = builder
                        .null_count(null_count)
                        .null_bit_buffer(buffers[0].clone())
                }
                builder.build()
            }
        }
        Boolean
        | Int64
        | UInt64
        | Float64
        | Time64(_)
        | Timestamp(_, _)
        | Date64(_)
        | Duration(_)
        | Interval(IntervalUnit::DayTime) => {
            let mut builder = ArrayData::builder(data_type.clone())
                .len(length)
                .buffers(buffers[1..].to_vec())
                .offset(0);
            if null_count > 0 {
                builder = builder
                    .null_count(null_count)
                    .null_bit_buffer(buffers[0].clone())
            }
            builder.build()
        }
        t => panic!("Data type {:?} either unsupported or not primitive", t),
    };

    make_array(array_data)
}

/// Reads the correct number of buffers based on list type and null_count, and creates a
/// list array ref
fn create_list_array(
    field_node: &ipc::FieldNode,
    data_type: &DataType,
    buffers: &[Buffer],
    child_array: ArrayRef,
) -> ArrayRef {
    if let DataType::List(_) = *data_type {
        let null_count = field_node.null_count() as usize;
        let mut builder = ArrayData::builder(data_type.clone())
            .len(field_node.length() as usize)
            .buffers(buffers[1..2].to_vec())
            .offset(0)
            .child_data(vec![child_array.data()]);
        if null_count > 0 {
            builder = builder
                .null_count(null_count)
                .null_bit_buffer(buffers[0].clone())
        }
        make_array(builder.build())
    } else if let DataType::FixedSizeList(_, _) = *data_type {
        let null_count = field_node.null_count() as usize;
        let mut builder = ArrayData::builder(data_type.clone())
            .len(field_node.length() as usize)
            .buffers(buffers[1..1].to_vec())
            .offset(0)
            .child_data(vec![child_array.data()]);
        if null_count > 0 {
            builder = builder
                .null_count(null_count)
                .null_bit_buffer(buffers[0].clone())
        }
        make_array(builder.build())
    } else {
        panic!("Cannot create list array from {:?}", data_type)
    }
}

/// Reads the correct number of buffers based on list type and null_count, and creates a
/// list array ref
fn create_dictionary_array(
    field_node: &ipc::FieldNode,
    data_type: &DataType,
    buffers: &[Buffer],
    value_array: ArrayRef,
) -> ArrayRef {
    if let DataType::Dictionary(_, _) = *data_type {
        let null_count = field_node.null_count() as usize;
        let mut builder = ArrayData::builder(data_type.clone())
            .len(field_node.length() as usize)
            .buffers(buffers[1..2].to_vec())
            .offset(0)
            .child_data(vec![value_array.data()]);
        if null_count > 0 {
            builder = builder
                .null_count(null_count)
                .null_bit_buffer(buffers[0].clone())
        }
        make_array(builder.build())
    } else {
        unreachable!("Cannot create dictionary array from {:?}", data_type)
    }
}

/// Creates a record batch from binary data using the `ipc::RecordBatch` indexes and the `Schema`
pub(crate) fn read_record_batch(
    buf: &[u8],
    batch: ipc::RecordBatch,
    schema: Arc<Schema>,
    dictionaries: &[Option<ArrayRef>],
) -> Result<Option<RecordBatch>> {
    let buffers = batch.buffers().ok_or_else(|| {
        ArrowError::IoError("Unable to get buffers from IPC RecordBatch".to_string())
    })?;
    let field_nodes = batch.nodes().ok_or_else(|| {
        ArrowError::IoError("Unable to get field nodes from IPC RecordBatch".to_string())
    })?;
    // keep track of buffer and node index, the functions that create arrays mutate these
    let mut buffer_index = 0;
    let mut node_index = 0;
    let mut arrays = vec![];

    // keep track of index as lists require more than one node
    for field in schema.fields() {
        let triple = create_array(
            field_nodes,
            field.data_type(),
            &buf,
            buffers,
            dictionaries,
            node_index,
            buffer_index,
        );
        node_index = triple.1;
        buffer_index = triple.2;
        arrays.push(triple.0);
    }

    RecordBatch::try_new(schema, arrays).map(|batch| Some(batch))
}

// Linear search for the first dictionary field with a dictionary id.
fn find_dictionary_field(ipc_schema: &ipc::Schema, id: i64) -> Option<usize> {
    let fields = ipc_schema.fields().unwrap();
    for i in 0..fields.len() {
        let field: ipc::Field = fields.get(i);
        if let Some(dictionary) = field.dictionary() {
            if dictionary.id() == id {
                return Some(i);
            }
        }
    }
    None
}

/// Arrow File reader
pub struct FileReader<R: Read + Seek> {
    /// Buffered file reader that supports reading and seeking
    reader: BufReader<R>,

    /// The schema that is read from the file header
    schema: Arc<Schema>,

    /// The blocks in the file
    ///
    /// A block indicates the regions in the file to read to get data
    blocks: Vec<ipc::Block>,

    /// A counter to keep track of the current block that should be read
    current_block: usize,

    /// The total number of blocks, which may contain record batches and other types
    total_blocks: usize,

    /// Optional dictionaries for each schema field.
    ///
    /// Dictionaries may be appended to in the streaming format.
    dictionaries_by_field: Vec<Option<ArrayRef>>,
}

impl<R: Read + Seek> FileReader<R> {
    /// Try to create a new file reader
    ///
    /// Returns errors if the file does not meet the Arrow Format header and footer
    /// requirements
    pub fn try_new(reader: R) -> Result<Self> {
        let mut reader = BufReader::new(reader);
        // check if header and footer contain correct magic bytes
        let mut magic_buffer: [u8; 6] = [0; 6];
        reader.read_exact(&mut magic_buffer)?;
        if magic_buffer != super::ARROW_MAGIC {
            return Err(ArrowError::IoError(
                "Arrow file does not contain correct header".to_string(),
            ));
        }
        reader.seek(SeekFrom::End(-6))?;
        reader.read_exact(&mut magic_buffer)?;
        if magic_buffer != super::ARROW_MAGIC {
            return Err(ArrowError::IoError(
                "Arrow file does not contain correct footer".to_string(),
            ));
        }

        // what does the footer contain?
        let mut footer_size: [u8; 4] = [0; 4];
        reader.seek(SeekFrom::End(-10))?;
        reader.read_exact(&mut footer_size)?;
        let footer_len = u32::from_le_bytes(footer_size);

        // read footer
        let mut footer_data = vec![0; footer_len as usize];
        reader.seek(SeekFrom::End(-10 - footer_len as i64))?;
        reader.read_exact(&mut footer_data)?;
        let footer = ipc::get_root_as_footer(&footer_data[..]);

        let blocks = footer.recordBatches().ok_or_else(|| {
            ArrowError::IoError(
                "Unable to get record batches from IPC Footer".to_string(),
            )
        })?;

        let total_blocks = blocks.len();

        let ipc_schema = footer.schema().unwrap();
        let schema = ipc::convert::fb_to_schema(ipc_schema);

        // Create an array of optional dictionary value arrays, one per field.
        let mut dictionaries_by_field = vec![None; schema.fields().len()];
        for block in footer.dictionaries().unwrap() {
            // read length from end of offset
            let meta_len = block.metaDataLength() - 4;

            let mut block_data = vec![0; meta_len as usize];
            reader.seek(SeekFrom::Start(block.offset() as u64 + 4))?;
            reader.read_exact(&mut block_data)?;

            let message = ipc::get_root_as_message(&block_data[..]);

            match message.header_type() {
                ipc::MessageHeader::DictionaryBatch => {
                    let batch = message.header_as_dictionary_batch().unwrap();

                    // read the block that makes up the dictionary batch into a buffer
                    let mut buf = vec![0; block.bodyLength() as usize];
                    reader.seek(SeekFrom::Start(
                        block.offset() as u64 + block.metaDataLength() as u64,
                    ))?;
                    reader.read_exact(&mut buf)?;

                    if batch.isDelta() {
                        panic!("delta dictionary batches not supported");
                    }

                    let id = batch.id();

                    // As the dictionary batch does not contain the type of the
                    // values array, we need to retieve this from the schema.
                    let first_field = find_dictionary_field(&ipc_schema, id)
                        .expect("dictionary id not found in shchema");

                    // Get an array representing this dictionary's values.
                    let dictionary_values: ArrayRef =
                        match schema.field(first_field).data_type() {
                            DataType::Dictionary(_, ref value_type) => {
                                // Make a fake schema for the dictionary batch.
                                let schema = Schema {
                                    fields: vec![Field::new(
                                        "",
                                        value_type.as_ref().clone(),
                                        false,
                                    )],
                                    metadata: HashMap::new(),
                                };
                                // Read a single column
                                let record_batch = read_record_batch(
                                    &buf,
                                    batch.data().unwrap(),
                                    Arc::new(schema),
                                    &dictionaries_by_field,
                                )?
                                .unwrap();
                                Some(record_batch.column(0).clone())
                            }
                            _ => None,
                        }
                        .expect("dictionary id not found in schema");

                    // for all fields with this dictionary id, update the dictionaries vector
                    // in the reader. Note that a dictionary batch may be shared between many fields.
                    // We don't currently record the isOrdered field. This could be general
                    // attributes of arrays.
                    let fields = ipc_schema.fields().unwrap();
                    for (i, field) in fields.iter().enumerate() {
                        if let Some(dictionary) = field.dictionary() {
                            if dictionary.id() == id {
                                // Add (possibly multiple) array refs to the dictionaries array.
                                dictionaries_by_field[i] =
                                    Some(dictionary_values.clone());
                            }
                        }
                    }
                }
                _ => panic!("Expecting DictionaryBatch in dictionary blocks."),
            };
        }

        Ok(Self {
            reader,
            schema: Arc::new(schema),
            blocks: blocks.to_vec(),
            current_block: 0,
            total_blocks,
            dictionaries_by_field,
        })
    }

    /// Return the number of batches in the file
    pub fn num_batches(&self) -> usize {
        self.total_blocks
    }

    /// Return the schema of the file
    pub fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }

    /// Read the next record batch
    pub fn next(&mut self) -> Result<Option<RecordBatch>> {
        // get current block
        if self.current_block < self.total_blocks {
            let block = self.blocks[self.current_block];
            self.current_block += 1;

            // read length from end of offset
            let meta_len = block.metaDataLength() - 4;

            let mut block_data = vec![0; meta_len as usize];
            self.reader
                .seek(SeekFrom::Start(block.offset() as u64 + 4))?;
            self.reader.read_exact(&mut block_data)?;

            let message = ipc::get_root_as_message(&block_data[..]);

            match message.header_type() {
                ipc::MessageHeader::Schema => Err(ArrowError::IoError(
                    "Not expecting a schema when messages are read".to_string(),
                )),
                ipc::MessageHeader::RecordBatch => {
                    let batch = message.header_as_record_batch().ok_or_else(|| {
                        ArrowError::IoError(
                            "Unable to read IPC message as record batch".to_string(),
                        )
                    })?;
                    // read the block that makes up the record batch into a buffer
                    let mut buf = vec![0; block.bodyLength() as usize];
                    self.reader.seek(SeekFrom::Start(
                        block.offset() as u64 + block.metaDataLength() as u64,
                    ))?;
                    self.reader.read_exact(&mut buf)?;

                    read_record_batch(
                        &buf,
                        batch,
                        self.schema(),
                        &self.dictionaries_by_field,
                    )
                }
                ipc::MessageHeader::NONE => {
                    Ok(None)
                }
                t => Err(ArrowError::IoError(format!(
                    "Reading types other than record batches not yet supported, unable to read {:?}", t
                ))),
            }
        } else {
            Ok(None)
        }
    }

    /// Read a specific record batch
    ///
    /// Sets the current block to the index, allowing random reads
    pub fn set_index(&mut self, index: usize) -> Result<()> {
        if index >= self.total_blocks {
            Err(ArrowError::IoError(format!(
                "Cannot set batch to index {} from {} total batches",
                index, self.total_blocks
            )))
        } else {
            self.current_block = index;
            Ok(())
        }
    }
}

impl<R: Read + Seek> RecordBatchReader for FileReader<R> {
    fn schema(&mut self) -> SchemaRef {
        self.schema.clone()
    }

    fn next_batch(&mut self) -> Result<Option<RecordBatch>> {
        self.next()
    }
}

/// Arrow Stream reader
pub struct StreamReader<R: Read> {
    /// Buffered stream reader
    reader: BufReader<R>,
    /// The schema that is read from the stream's first message
    schema: Arc<Schema>,
    /// An indicator of whether the strewam is complete.
    ///
    /// This value is set to `true` the first time the reader's `next()` returns `None`.
    finished: bool,

    /// Optional dictionaries for each schema field.
    ///
    /// Dictionaries may be appended to in the streaming format.
    dictionaries_by_field: Vec<Option<ArrayRef>>,
}

impl<R: Read> StreamReader<R> {
    /// Try to create a new stream reader
    ///
    /// The first message in the stream is the schema, the reader will fail if it does not
    /// encounter a schema.
    /// To check if the reader is done, use `is_finished(self)`
    pub fn try_new(reader: R) -> Result<Self> {
        let mut reader = BufReader::new(reader);
        // determine metadata length
        let mut meta_size: [u8; 4] = [0; 4];
        reader.read_exact(&mut meta_size)?;
        let meta_len = {
            let meta_len = u32::from_le_bytes(meta_size);

            // If a continuation marker is encountered, skip over it and read
            // the size from the next four bytes.
            if meta_len == CONTINUATION_MARKER {
                reader.read_exact(&mut meta_size)?;
                u32::from_le_bytes(meta_size)
            } else {
                meta_len
            }
        };

        let mut meta_buffer = vec![0; meta_len as usize];
        reader.read_exact(&mut meta_buffer)?;

        let vecs = &meta_buffer.to_vec();
        let message = ipc::get_root_as_message(vecs);
        // message header is a Schema, so read it
        let ipc_schema: ipc::Schema = message.header_as_schema().ok_or_else(|| {
            ArrowError::IoError("Unable to read IPC message as schema".to_string())
        })?;
        let schema = ipc::convert::fb_to_schema(ipc_schema);

        // Create an array of optional dictionary value arrays, one per field.
        let dictionaries_by_field = vec![None; schema.fields().len()];

        Ok(Self {
            reader,
            schema: Arc::new(schema),
            finished: false,
            dictionaries_by_field,
        })
    }

    /// Return the schema of the stream
    pub fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }

    /// Read the next record batch
    pub fn next(&mut self) -> Result<Option<RecordBatch>> {
        if self.finished {
            return Ok(None);
        }
        // determine metadata length
        let mut meta_size: [u8; 4] = [0; 4];

        match self.reader.read_exact(&mut meta_size) {
            Ok(()) => (),
            Err(e) => {
                return if e.kind() == std::io::ErrorKind::UnexpectedEof {
                    // Handle EOF without the "0xFFFFFFFF 0x00000000"
                    // valid according to:
                    // https://arrow.apache.org/docs/format/Columnar.html#ipc-streaming-format
                    self.finished = true;
                    Ok(None)
                } else {
                    Err(ArrowError::from(e))
                };
            }
        }

        let meta_len = {
            let meta_len = u32::from_le_bytes(meta_size);

            // If a continuation marker is encountered, skip over it and read
            // the size from the next four bytes.
            if meta_len == CONTINUATION_MARKER {
                self.reader.read_exact(&mut meta_size)?;
                u32::from_le_bytes(meta_size)
            } else {
                meta_len
            }
        };

        if meta_len == 0 {
            // the stream has ended, mark the reader as finished
            self.finished = true;
            return Ok(None);
        }

        let mut meta_buffer = vec![0; meta_len as usize];
        self.reader.read_exact(&mut meta_buffer)?;

        let vecs = &meta_buffer.to_vec();
        let message = ipc::get_root_as_message(vecs);

        match message.header_type() {
            ipc::MessageHeader::Schema => Err(ArrowError::IoError(
                "Not expecting a schema when messages are read".to_string(),
            )),
            ipc::MessageHeader::RecordBatch => {
                let batch = message.header_as_record_batch().ok_or_else(|| {
                    ArrowError::IoError(
                        "Unable to read IPC message as record batch".to_string(),
                    )
                })?;
                // read the block that makes up the record batch into a buffer
                let mut buf = vec![0; message.bodyLength() as usize];
                self.reader.read_exact(&mut buf)?;

                read_record_batch(&buf, batch, self.schema(), &self.dictionaries_by_field)
            }
            ipc::MessageHeader::NONE => {
                Ok(None)
            }
            t => Err(ArrowError::IoError(
                format!("Reading types other than record batches not yet supported, unable to read {:?} ", t)
            )),
        }
    }

    /// Check if the stream is finished
    pub fn is_finished(&self) -> bool {
        self.finished
    }
}

impl<R: Read> RecordBatchReader for StreamReader<R> {
    fn schema(&mut self) -> SchemaRef {
        self.schema.clone()
    }

    fn next_batch(&mut self) -> Result<Option<RecordBatch>> {
        self.next()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use flate2::read::GzDecoder;

    use crate::util::integration_util::*;
    use std::env;
    use std::fs::File;

    #[test]
    fn read_generated_files() {
        let testdata = env::var("ARROW_TEST_DATA").expect("ARROW_TEST_DATA not defined");
        // the test is repetitive, thus we can read all supported files at once
        let paths = vec![
            "generated_interval",
            "generated_datetime",
            "generated_dictionary",
            "generated_nested",
            "generated_primitive_no_batches",
            "generated_primitive_zerolength",
            "generated_primitive",
        ];
        paths.iter().for_each(|path| {
            let file = File::open(format!(
                "{}/some-ipc-stream/integration/0.14.1/{}.arrow_file",
                testdata, path
            ))
            .unwrap();

            let mut reader = FileReader::try_new(file).unwrap();

            // read expected JSON output
            let arrow_json = read_gzip_json(path);
            assert!(arrow_json.equals_reader(&mut reader));
        });
    }

    #[test]
    fn read_generated_streams() {
        let testdata = env::var("ARROW_TEST_DATA").expect("ARROW_TEST_DATA not defined");
        // the test is repetitive, thus we can read all supported files at once
        let paths = vec![
            "generated_interval",
            "generated_datetime",
            // "generated_dictionary",
            "generated_nested",
            "generated_primitive_no_batches",
            "generated_primitive_zerolength",
            "generated_primitive",
        ];
        paths.iter().for_each(|path| {
            let file = File::open(format!(
                "{}/some-ipc-stream/integration/0.14.1/{}.stream",
                testdata, path
            ))
            .unwrap();

            let mut reader = StreamReader::try_new(file).unwrap();

            // read expected JSON output
            let arrow_json = read_gzip_json(path);
            assert!(arrow_json.equals_reader(&mut reader));
            // the next batch must be empty
            assert!(reader.next().unwrap().is_none());
            // the stream must indicate that it's finished
            assert!(reader.is_finished());
        });
    }

    /// Read gzipped JSON file
    fn read_gzip_json(path: &str) -> ArrowJson {
        let testdata = env::var("ARROW_TEST_DATA").expect("ARROW_TEST_DATA not defined");
        let file = File::open(format!(
            "{}/some-ipc-stream/integration/0.14.1/{}.json.gz",
            testdata, path
        ))
        .unwrap();
        let mut gz = GzDecoder::new(&file);
        let mut s = String::new();
        gz.read_to_string(&mut s).unwrap();
        // convert to Arrow JSON
        let arrow_json: ArrowJson = serde_json::from_str(&s).unwrap();
        arrow_json
    }
}
