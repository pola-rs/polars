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

//! Arrow IPC File and Stream Writers
//!
//! The `FileWriter` and `StreamWriter` have similar interfaces,
//! however the `FileWriter` expects a reader that supports `Seek`ing

use std::io::{BufWriter, Write};

use flatbuffers::FlatBufferBuilder;

use crate::array::ArrayDataRef;
use crate::buffer::{Buffer, MutableBuffer};
use crate::datatypes::*;
use crate::error::{ArrowError, Result};
use crate::ipc;
use crate::record_batch::RecordBatch;
use crate::util::bit_util;

pub struct FileWriter<W: Write> {
    /// The object to write to
    writer: BufWriter<W>,
    /// A reference to the schema, used in validating record batches
    schema: Schema,
    /// The number of bytes between each block of bytes, as an offset for random access
    block_offsets: usize,
    /// Dictionary blocks that will be written as part of the IPC footer
    dictionary_blocks: Vec<ipc::Block>,
    /// Record blocks that will be written as part of the IPC footer
    record_blocks: Vec<ipc::Block>,
    /// Whether the writer footer has been written, and the writer is finished
    finished: bool,
}

impl<W: Write> FileWriter<W> {
    /// Try create a new writer, with the schema written as part of the header
    pub fn try_new(writer: W, schema: &Schema) -> Result<Self> {
        let mut writer = BufWriter::new(writer);
        // write magic to header
        writer.write_all(&super::ARROW_MAGIC[..])?;
        // create an 8-byte boundary after the header
        writer.write_all(&[0, 0])?;
        // write the schema, set the written bytes to the schema + header
        let written = write_schema(&mut writer, schema)? + 8;
        Ok(Self {
            writer,
            schema: schema.clone(),
            block_offsets: written,
            dictionary_blocks: vec![],
            record_blocks: vec![],
            finished: false,
        })
    }

    /// Write a record batch to the file
    pub fn write(&mut self, batch: &RecordBatch) -> Result<()> {
        if self.finished {
            return Err(ArrowError::IoError(
                "Cannot write record batch to file writer as it is closed".to_string(),
            ));
        }
        let (meta, data) = write_record_batch(&mut self.writer, batch, false)?;
        // add a record block for the footer
        self.record_blocks.push(ipc::Block::new(
            self.block_offsets as i64,
            (meta as i32) + 4,
            data as i64,
        ));
        self.block_offsets += meta + data;
        Ok(())
    }

    /// Write footer and closing tag, then mark the writer as done
    pub fn finish(&mut self) -> Result<()> {
        let mut fbb = FlatBufferBuilder::new();
        let dictionaries = fbb.create_vector(&self.dictionary_blocks);
        let record_batches = fbb.create_vector(&self.record_blocks);
        // TODO: this is duplicated as we otherwise mutably borrow twice
        let schema = {
            let mut fields = vec![];
            for field in self.schema.fields() {
                let fb_field_name = fbb.create_string(field.name().as_str());
                let (ipc_type_type, ipc_type, ipc_children) =
                    ipc::convert::get_fb_field_type(field.data_type(), &mut fbb);
                let mut field_builder = ipc::FieldBuilder::new(&mut fbb);
                field_builder.add_name(fb_field_name);
                field_builder.add_type_type(ipc_type_type);
                field_builder.add_nullable(field.is_nullable());
                match ipc_children {
                    None => {}
                    Some(children) => field_builder.add_children(children),
                };
                field_builder.add_type_(ipc_type);
                fields.push(field_builder.finish());
            }

            let mut custom_metadata = vec![];
            for (k, v) in self.schema.metadata() {
                let fb_key_name = fbb.create_string(k.as_str());
                let fb_val_name = fbb.create_string(v.as_str());

                let mut kv_builder = ipc::KeyValueBuilder::new(&mut fbb);
                kv_builder.add_key(fb_key_name);
                kv_builder.add_value(fb_val_name);
                custom_metadata.push(kv_builder.finish());
            }

            let fb_field_list = fbb.create_vector(&fields);
            let fb_metadata_list = fbb.create_vector(&custom_metadata);

            let mut builder = ipc::SchemaBuilder::new(&mut fbb);
            builder.add_fields(fb_field_list);
            builder.add_custom_metadata(fb_metadata_list);
            builder.finish()
        };
        let root = {
            let mut footer_builder = ipc::FooterBuilder::new(&mut fbb);
            footer_builder.add_version(ipc::MetadataVersion::V4);
            footer_builder.add_schema(schema);
            footer_builder.add_dictionaries(dictionaries);
            footer_builder.add_recordBatches(record_batches);
            footer_builder.finish()
        };
        fbb.finish(root, None);
        write_padded_data(&mut self.writer, fbb.finished_data(), WriteDataType::Footer)?;
        self.writer.write_all(&super::ARROW_MAGIC)?;
        self.writer.flush()?;
        self.finished = true;

        Ok(())
    }
}

/// Finish the file if it is not 'finished' when it goes out of scope
impl<W: Write> Drop for FileWriter<W> {
    fn drop(&mut self) {
        if !self.finished {
            self.finish().unwrap();
        }
    }
}

pub struct StreamWriter<W: Write> {
    /// The object to write to
    writer: BufWriter<W>,
    /// A reference to the schema, used in validating record batches
    schema: Schema,
    /// Whether the writer footer has been written, and the writer is finished
    finished: bool,
}

impl<W: Write> StreamWriter<W> {
    /// Try create a new writer, with the schema written as part of the header
    pub fn try_new(writer: W, schema: &Schema) -> Result<Self> {
        let mut writer = BufWriter::new(writer);
        // write the schema, set the written bytes to the schema
        write_schema(&mut writer, schema)?;
        Ok(Self {
            writer,
            schema: schema.clone(),
            finished: false,
        })
    }

    /// Write a record batch to the stream
    pub fn write(&mut self, batch: &RecordBatch) -> Result<()> {
        if self.finished {
            return Err(ArrowError::IoError(
                "Cannot write record batch to stream writer as it is closed".to_string(),
            ));
        }
        write_record_batch(&mut self.writer, batch, true)?;
        Ok(())
    }

    /// Write continuation bytes, and mark the stream as done
    pub fn finish(&mut self) -> Result<()> {
        self.writer.write_all(&[255u8, 255, 255, 255])?;
        self.writer.write_all(&[0u8, 0, 0, 0])?;
        self.writer.flush()?;

        self.finished = true;

        Ok(())
    }
}

/// Finish the stream if it is not 'finished' when it goes out of scope
impl<W: Write> Drop for StreamWriter<W> {
    fn drop(&mut self) {
        if !self.finished {
            self.finish().unwrap();
        }
    }
}

pub(crate) fn schema_to_bytes(schema: &Schema) -> Vec<u8> {
    let mut fbb = FlatBufferBuilder::new();
    let schema = {
        let fb = ipc::convert::schema_to_fb_offset(&mut fbb, schema);
        fb.as_union_value()
    };

    let mut message = ipc::MessageBuilder::new(&mut fbb);
    message.add_version(ipc::MetadataVersion::V4);
    message.add_header_type(ipc::MessageHeader::Schema);
    message.add_bodyLength(0);
    message.add_header(schema);
    // TODO: custom metadata
    let data = message.finish();
    fbb.finish(data, None);

    let data = fbb.finished_data();
    data.to_vec()
}

/// Convert the schema to its IPC representation, and write it to the `writer`
fn write_schema<R: Write>(writer: &mut BufWriter<R>, schema: &Schema) -> Result<usize> {
    let data = schema_to_bytes(schema);
    write_padded_data(writer, &data[..], WriteDataType::Header)
}

/// The message type being written. This determines whether to write the data length or not.
/// Data length is written before the header, after the footer, and never for the body.
#[derive(PartialEq)]
enum WriteDataType {
    Header,
    Body,
    Footer,
}

/// Write a slice of data to the writer, ensuring that it is padded to 8 bytes
fn write_padded_data<R: Write>(
    writer: &mut BufWriter<R>,
    data: &[u8],
    data_type: WriteDataType,
) -> Result<usize> {
    let len = data.len() as u32;
    let pad_len = pad_to_8(len) as u32;
    let total_len = len + pad_len;
    // write data length
    if data_type == WriteDataType::Header {
        writer.write_all(&total_len.to_le_bytes()[..])?;
    }
    // write flatbuffer data
    writer.write_all(data)?;
    if pad_len > 0 {
        writer.write_all(&vec![0u8; pad_len as usize][..])?;
    }
    if data_type == WriteDataType::Footer {
        writer.write_all(&total_len.to_le_bytes()[..])?;
    }
    writer.flush()?;
    Ok(total_len as usize)
}

/// Write a `RecordBatch` into a tuple of bytes, one for the header (ipc::Message) and the other for the batch's data
pub(crate) fn record_batch_to_bytes(batch: &RecordBatch) -> (Vec<u8>, Vec<u8>) {
    let mut fbb = FlatBufferBuilder::new();

    let mut nodes: Vec<ipc::FieldNode> = vec![];
    let mut buffers: Vec<ipc::Buffer> = vec![];
    let mut arrow_data: Vec<u8> = vec![];
    let mut offset = 0;
    for array in batch.columns() {
        let array_data = array.data();
        offset = write_array_data(
            &array_data,
            &mut buffers,
            &mut arrow_data,
            &mut nodes,
            offset,
            array.len(),
            array.null_count(),
        );
    }

    // write data
    let buffers = fbb.create_vector(&buffers);
    let nodes = fbb.create_vector(&nodes);

    let root = {
        let mut batch_builder = ipc::RecordBatchBuilder::new(&mut fbb);
        batch_builder.add_length(batch.num_rows() as i64);
        batch_builder.add_nodes(nodes);
        batch_builder.add_buffers(buffers);
        let b = batch_builder.finish();
        b.as_union_value()
    };
    // create an ipc::Message
    let mut message = ipc::MessageBuilder::new(&mut fbb);
    message.add_version(ipc::MetadataVersion::V4);
    message.add_header_type(ipc::MessageHeader::RecordBatch);
    message.add_bodyLength(arrow_data.len() as i64);
    message.add_header(root);
    let root = message.finish();
    fbb.finish(root, None);
    let finished_data = fbb.finished_data();

    (finished_data.to_vec(), arrow_data)
}

/// Write a record batch to the writer, writing the message size before the message
/// if the record batch is being written to a stream
fn write_record_batch<R: Write>(
    writer: &mut BufWriter<R>,
    batch: &RecordBatch,
    is_stream: bool,
) -> Result<(usize, usize)> {
    let (meta_data, arrow_data) = record_batch_to_bytes(batch);
    // write the length of data if writing to stream
    if is_stream {
        let total_len: u32 = meta_data.len() as u32;
        writer.write_all(&total_len.to_le_bytes()[..])?;
    }
    let meta_written = write_padded_data(writer, &meta_data[..], WriteDataType::Body)?;
    let arrow_data_written =
        write_padded_data(writer, &arrow_data[..], WriteDataType::Body)?;
    Ok((meta_written, arrow_data_written))
}

/// Write array data to a vector of bytes
fn write_array_data(
    array_data: &ArrayDataRef,
    mut buffers: &mut Vec<ipc::Buffer>,
    mut arrow_data: &mut Vec<u8>,
    mut nodes: &mut Vec<ipc::FieldNode>,
    offset: i64,
    num_rows: usize,
    null_count: usize,
) -> i64 {
    let mut offset = offset;
    nodes.push(ipc::FieldNode::new(num_rows as i64, null_count as i64));
    // NullArray does not have any buffers, thus the null buffer is not generated
    if array_data.data_type() != &DataType::Null {
        // write null buffer if exists
        let null_buffer = match array_data.null_buffer() {
            None => {
                // create a buffer and fill it with valid bits
                let num_bytes = bit_util::ceil(num_rows, 8);
                let buffer = MutableBuffer::new(num_bytes);
                let buffer = buffer.with_bitset(num_bytes, true);
                buffer.freeze()
            }
            Some(buffer) => buffer.clone(),
        };

        offset = write_buffer(&null_buffer, &mut buffers, &mut arrow_data, offset);
    }

    array_data.buffers().iter().for_each(|buffer| {
        offset = write_buffer(buffer, &mut buffers, &mut arrow_data, offset);
    });

    // recursively write out nested structures
    array_data.child_data().iter().for_each(|data_ref| {
        // write the nested data (e.g list data)
        offset = write_array_data(
            data_ref,
            &mut buffers,
            &mut arrow_data,
            &mut nodes,
            offset,
            data_ref.len(),
            data_ref.null_count(),
        );
    });
    offset
}

/// Write a buffer to a vector of bytes, and add its ipc Buffer to a vector
fn write_buffer(
    buffer: &Buffer,
    buffers: &mut Vec<ipc::Buffer>,
    arrow_data: &mut Vec<u8>,
    offset: i64,
) -> i64 {
    let len = buffer.len();
    let pad_len = pad_to_8(len as u32);
    let total_len: i64 = (len + pad_len) as i64;
    // assert_eq!(len % 8, 0, "Buffer width not a multiple of 8 bytes");
    buffers.push(ipc::Buffer::new(offset, total_len));
    arrow_data.extend_from_slice(buffer.data());
    arrow_data.extend_from_slice(&vec![0u8; pad_len][..]);
    offset + total_len
}

/// Calculate an 8-byte boundary and return the number of bytes needed to pad to 8 bytes
fn pad_to_8<'a>(len: u32) -> usize {
    match len % 8 {
        0 => 0 as usize,
        v => 8 - v as usize,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use flate2::read::GzDecoder;

    use crate::array::*;
    use crate::datatypes::Field;
    use crate::ipc::reader::*;
    use crate::util::integration_util::*;
    use std::env;
    use std::fs::File;
    use std::io::Read;
    use std::sync::Arc;

    #[test]
    fn test_write_file() {
        let schema = Schema::new(vec![Field::new("field1", DataType::UInt32, false)]);
        let values: Vec<Option<u32>> = vec![
            Some(999),
            None,
            Some(235),
            Some(123),
            None,
            None,
            None,
            None,
            None,
        ];
        let array1 = UInt32Array::from(values);
        let batch = RecordBatch::try_new(
            Arc::new(schema.clone()),
            vec![Arc::new(array1) as ArrayRef],
        )
        .unwrap();
        {
            let file = File::create("target/debug/testdata/some.arrow_file").unwrap();
            let mut writer = FileWriter::try_new(file, &schema).unwrap();

            writer.write(&batch).unwrap();
            // this is inside a block to test the implicit finishing of the file on `Drop`
        }

        {
            let file = File::open(format!("target/debug/testdata/{}.arrow_file", "some"))
                .unwrap();
            let mut reader = FileReader::try_new(file).unwrap();
            while let Ok(Some(read_batch)) = reader.next() {
                read_batch
                    .columns()
                    .iter()
                    .zip(batch.columns())
                    .for_each(|(a, b)| {
                        assert_eq!(a.data_type(), b.data_type());
                        assert_eq!(a.len(), b.len());
                        assert_eq!(a.null_count(), b.null_count());
                    });
            }
        }
    }

    #[test]
    fn test_write_null_file() {
        let schema = Schema::new(vec![
            Field::new("nulls", DataType::Null, true),
            Field::new("int32s", DataType::Int32, false),
            Field::new("nulls2", DataType::Null, false),
            Field::new("f64s", DataType::Float64, false),
        ]);
        let array1 = NullArray::new(32);
        let array2 = Int32Array::from(vec![1; 32]);
        let array3 = NullArray::new(32);
        let array4 = Float64Array::from(vec![std::f64::NAN; 32]);
        let batch = RecordBatch::try_new(
            Arc::new(schema.clone()),
            vec![
                Arc::new(array1) as ArrayRef,
                Arc::new(array2) as ArrayRef,
                Arc::new(array3) as ArrayRef,
                Arc::new(array4) as ArrayRef,
            ],
        )
        .unwrap();
        {
            let file = File::create("target/debug/testdata/nulls.arrow_file").unwrap();
            let mut writer = FileWriter::try_new(file, &schema).unwrap();

            writer.write(&batch).unwrap();
            // this is inside a block to test the implicit finishing of the file on `Drop`
        }

        {
            let file = File::open("target/debug/testdata/nulls.arrow_file").unwrap();
            let mut reader = FileReader::try_new(file).unwrap();
            while let Ok(Some(read_batch)) = reader.next() {
                read_batch
                    .columns()
                    .iter()
                    .zip(batch.columns())
                    .for_each(|(a, b)| {
                        assert_eq!(a.data_type(), b.data_type());
                        assert_eq!(a.len(), b.len());
                        assert_eq!(a.null_count(), b.null_count());
                    });
            }
        }
    }

    #[test]
    fn read_and_rewrite_generated_files() {
        let testdata = env::var("ARROW_TEST_DATA").expect("ARROW_TEST_DATA not defined");
        // the test is repetitive, thus we can read all supported files at once
        let paths = vec![
            "generated_interval",
            "generated_datetime",
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

            // read and rewrite the file to a temp location
            {
                let file =
                    File::create(format!("target/debug/testdata/{}.arrow_file", path))
                        .unwrap();
                let mut writer = FileWriter::try_new(file, &reader.schema()).unwrap();
                while let Ok(Some(batch)) = reader.next() {
                    writer.write(&batch).unwrap();
                }
                writer.finish().unwrap();
            }

            let file =
                File::open(format!("target/debug/testdata/{}.arrow_file", path)).unwrap();
            let mut reader = FileReader::try_new(file).unwrap();

            // read expected JSON output
            let arrow_json = read_gzip_json(path);
            assert!(arrow_json.equals_reader(&mut reader));
        });
    }

    #[test]
    fn read_and_rewrite_generated_streams() {
        let testdata = env::var("ARROW_TEST_DATA").expect("ARROW_TEST_DATA not defined");
        // the test is repetitive, thus we can read all supported files at once
        let paths = vec![
            "generated_interval",
            "generated_datetime",
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

            // read and rewrite the stream to a temp location
            {
                let file = File::create(format!("target/debug/testdata/{}.stream", path))
                    .unwrap();
                let mut writer = StreamWriter::try_new(file, &reader.schema()).unwrap();
                while let Ok(Some(batch)) = reader.next() {
                    writer.write(&batch).unwrap();
                }
                writer.finish().unwrap();
            }

            let file =
                File::open(format!("target/debug/testdata/{}.stream", path)).unwrap();
            let mut reader = StreamReader::try_new(file).unwrap();

            // read expected JSON output
            let arrow_json = read_gzip_json(path);
            assert!(arrow_json.equals_reader(&mut reader));
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
