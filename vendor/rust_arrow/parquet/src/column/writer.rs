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

//! Contains column writer API.

use std::{cmp, collections::VecDeque, rc::Rc};

use crate::basic::{Compression, Encoding, PageType, Type};
use crate::column::page::{CompressedPage, Page, PageWriteSpec, PageWriter};
use crate::compression::{create_codec, Codec};
use crate::data_type::*;
use crate::encodings::{
    encoding::{get_encoder, DictEncoder, Encoder},
    levels::{max_buffer_size, LevelEncoder},
};
use crate::errors::{ParquetError, Result};
use crate::file::{
    metadata::ColumnChunkMetaData,
    properties::{WriterProperties, WriterPropertiesPtr, WriterVersion},
};
use crate::schema::types::ColumnDescPtr;
use crate::util::memory::{ByteBufferPtr, MemTracker};

/// Column writer for a Parquet type.
pub enum ColumnWriter {
    BoolColumnWriter(ColumnWriterImpl<BoolType>),
    Int32ColumnWriter(ColumnWriterImpl<Int32Type>),
    Int64ColumnWriter(ColumnWriterImpl<Int64Type>),
    Int96ColumnWriter(ColumnWriterImpl<Int96Type>),
    FloatColumnWriter(ColumnWriterImpl<FloatType>),
    DoubleColumnWriter(ColumnWriterImpl<DoubleType>),
    ByteArrayColumnWriter(ColumnWriterImpl<ByteArrayType>),
    FixedLenByteArrayColumnWriter(ColumnWriterImpl<FixedLenByteArrayType>),
}

/// Gets a specific column writer corresponding to column descriptor `descr`.
pub fn get_column_writer(
    descr: ColumnDescPtr,
    props: WriterPropertiesPtr,
    page_writer: Box<PageWriter>,
) -> ColumnWriter {
    match descr.physical_type() {
        Type::BOOLEAN => ColumnWriter::BoolColumnWriter(ColumnWriterImpl::new(
            descr,
            props,
            page_writer,
        )),
        Type::INT32 => ColumnWriter::Int32ColumnWriter(ColumnWriterImpl::new(
            descr,
            props,
            page_writer,
        )),
        Type::INT64 => ColumnWriter::Int64ColumnWriter(ColumnWriterImpl::new(
            descr,
            props,
            page_writer,
        )),
        Type::INT96 => ColumnWriter::Int96ColumnWriter(ColumnWriterImpl::new(
            descr,
            props,
            page_writer,
        )),
        Type::FLOAT => ColumnWriter::FloatColumnWriter(ColumnWriterImpl::new(
            descr,
            props,
            page_writer,
        )),
        Type::DOUBLE => ColumnWriter::DoubleColumnWriter(ColumnWriterImpl::new(
            descr,
            props,
            page_writer,
        )),
        Type::BYTE_ARRAY => ColumnWriter::ByteArrayColumnWriter(ColumnWriterImpl::new(
            descr,
            props,
            page_writer,
        )),
        Type::FIXED_LEN_BYTE_ARRAY => ColumnWriter::FixedLenByteArrayColumnWriter(
            ColumnWriterImpl::new(descr, props, page_writer),
        ),
    }
}

/// Gets a typed column writer for the specific type `T`, by "up-casting" `col_writer` of
/// non-generic type to a generic column writer type `ColumnWriterImpl`.
///
/// Panics if actual enum value for `col_writer` does not match the type `T`.
pub fn get_typed_column_writer<T: DataType>(
    col_writer: ColumnWriter,
) -> ColumnWriterImpl<T> {
    T::get_column_writer(col_writer).unwrap_or_else(|| {
        panic!(
            "Failed to convert column writer into a typed column writer for `{}` type",
            T::get_physical_type()
        )
    })
}

/// Similar to `get_typed_column_writer` but returns a reference.
pub fn get_typed_column_writer_ref<T: DataType>(
    col_writer: &ColumnWriter,
) -> &ColumnWriterImpl<T> {
    T::get_column_writer_ref(col_writer).unwrap_or_else(|| {
        panic!(
            "Failed to convert column writer into a typed column writer for `{}` type",
            T::get_physical_type()
        )
    })
}

/// Similar to `get_typed_column_writer` but returns a reference.
pub fn get_typed_column_writer_mut<T: DataType>(
    col_writer: &mut ColumnWriter,
) -> &mut ColumnWriterImpl<T> {
    T::get_column_writer_mut(col_writer).unwrap_or_else(|| {
        panic!(
            "Failed to convert column writer into a typed column writer for `{}` type",
            T::get_physical_type()
        )
    })
}

/// Typed column writer for a primitive column.
pub struct ColumnWriterImpl<T: DataType> {
    // Column writer properties
    descr: ColumnDescPtr,
    props: WriterPropertiesPtr,
    page_writer: Box<PageWriter>,
    has_dictionary: bool,
    dict_encoder: Option<DictEncoder<T>>,
    encoder: Box<Encoder<T>>,
    codec: Compression,
    compressor: Option<Box<Codec>>,
    // Metrics per page
    num_buffered_values: u32,
    num_buffered_encoded_values: u32,
    num_buffered_rows: u32,
    // Metrics per column writer
    total_bytes_written: u64,
    total_rows_written: u64,
    total_uncompressed_size: u64,
    total_compressed_size: u64,
    total_num_values: u64,
    dictionary_page_offset: Option<u64>,
    data_page_offset: Option<u64>,
    // Reused buffers
    def_levels_sink: Vec<i16>,
    rep_levels_sink: Vec<i16>,
    data_pages: VecDeque<CompressedPage>,
}

impl<T: DataType> ColumnWriterImpl<T> {
    pub fn new(
        descr: ColumnDescPtr,
        props: WriterPropertiesPtr,
        page_writer: Box<PageWriter>,
    ) -> Self {
        let codec = props.compression(descr.path());
        let compressor = create_codec(codec).unwrap();

        // Optionally set dictionary encoder.
        let dict_encoder = if props.dictionary_enabled(descr.path())
            && Self::has_dictionary_support(&props)
        {
            Some(DictEncoder::new(descr.clone(), Rc::new(MemTracker::new())))
        } else {
            None
        };

        // Whether or not this column writer has a dictionary encoding.
        let has_dictionary = dict_encoder.is_some();

        // Set either main encoder or fallback encoder.
        let fallback_encoder = get_encoder(
            descr.clone(),
            props
                .encoding(descr.path())
                .unwrap_or(Self::fallback_encoding(&props)),
            Rc::new(MemTracker::new()),
        )
        .unwrap();

        Self {
            descr,
            props,
            page_writer,
            has_dictionary,
            dict_encoder,
            encoder: fallback_encoder,
            codec,
            compressor,
            num_buffered_values: 0,
            num_buffered_encoded_values: 0,
            num_buffered_rows: 0,
            total_bytes_written: 0,
            total_rows_written: 0,
            total_uncompressed_size: 0,
            total_compressed_size: 0,
            total_num_values: 0,
            dictionary_page_offset: None,
            data_page_offset: None,
            def_levels_sink: vec![],
            rep_levels_sink: vec![],
            data_pages: VecDeque::new(),
        }
    }

    /// Writes batch of values, definition levels and repetition levels.
    /// Returns number of values processed (written).
    ///
    /// If definition and repetition levels are provided, we write fully those levels and
    /// select how many values to write (this number will be returned), since number of
    /// actual written values may be smaller than provided values.
    ///
    /// If only values are provided, then all values are written and the length of
    /// of the values buffer is returned.
    ///
    /// Definition and/or repetition levels can be omitted, if values are
    /// non-nullable and/or non-repeated.
    pub fn write_batch(
        &mut self,
        values: &[T::T],
        def_levels: Option<&[i16]>,
        rep_levels: Option<&[i16]>,
    ) -> Result<usize> {
        // We check for DataPage limits only after we have inserted the values. If a user
        // writes a large number of values, the DataPage size can be well above the limit.
        //
        // The purpose of this chunking is to bound this. Even if a user writes large
        // number of values, the chunking will ensure that we add data page at a
        // reasonable pagesize limit.

        // TODO: find out why we don't account for size of levels when we estimate page
        // size.

        // Find out the minimal length to prevent index out of bound errors.
        let mut min_len = values.len();
        if let Some(levels) = def_levels {
            min_len = cmp::min(min_len, levels.len());
        }
        if let Some(levels) = rep_levels {
            min_len = cmp::min(min_len, levels.len());
        }

        // Find out number of batches to process.
        let write_batch_size = self.props.write_batch_size();
        let num_batches = min_len / write_batch_size;

        let mut values_offset = 0;
        let mut levels_offset = 0;

        for _ in 0..num_batches {
            values_offset += self.write_mini_batch(
                &values[values_offset..values_offset + write_batch_size],
                def_levels.map(|lv| &lv[levels_offset..levels_offset + write_batch_size]),
                rep_levels.map(|lv| &lv[levels_offset..levels_offset + write_batch_size]),
            )?;
            levels_offset += write_batch_size;
        }

        values_offset += self.write_mini_batch(
            &values[values_offset..],
            def_levels.map(|lv| &lv[levels_offset..]),
            rep_levels.map(|lv| &lv[levels_offset..]),
        )?;

        // Return total number of values processed.
        Ok(values_offset)
    }

    /// Returns total number of bytes written by this column writer so far.
    /// This value is also returned when column writer is closed.
    pub fn get_total_bytes_written(&self) -> u64 {
        self.total_bytes_written
    }

    /// Returns total number of rows written by this column writer so far.
    /// This value is also returned when column writer is closed.
    pub fn get_total_rows_written(&self) -> u64 {
        self.total_rows_written
    }

    /// Finalises writes and closes the column writer.
    /// Returns total bytes written, total rows written and column chunk metadata.
    pub fn close(mut self) -> Result<(u64, u64, ColumnChunkMetaData)> {
        if self.dict_encoder.is_some() {
            self.write_dictionary_page()?;
        }
        self.flush_data_pages()?;
        let metadata = self.write_column_metadata()?;
        self.dict_encoder = None;
        self.page_writer.close()?;

        Ok((self.total_bytes_written, self.total_rows_written, metadata))
    }

    /// Writes mini batch of values, definition and repetition levels.
    /// This allows fine-grained processing of values and maintaining a reasonable
    /// page size.
    fn write_mini_batch(
        &mut self,
        values: &[T::T],
        def_levels: Option<&[i16]>,
        rep_levels: Option<&[i16]>,
    ) -> Result<usize> {
        let num_values;
        let mut values_to_write = 0;

        // Check if number of definition levels is the same as number of repetition
        // levels.
        if def_levels.is_some() && rep_levels.is_some() {
            let def = def_levels.unwrap();
            let rep = rep_levels.unwrap();
            if def.len() != rep.len() {
                return Err(general_err!(
                    "Inconsistent length of definition and repetition levels: {} != {}",
                    def.len(),
                    rep.len()
                ));
            }
        }

        // Process definition levels and determine how many values to write.
        if self.descr.max_def_level() > 0 {
            if def_levels.is_none() {
                return Err(general_err!(
                    "Definition levels are required, because max definition level = {}",
                    self.descr.max_def_level()
                ));
            }

            let levels = def_levels.unwrap();
            num_values = levels.len();
            for &level in levels {
                values_to_write += (level == self.descr.max_def_level()) as usize;
            }

            self.write_definition_levels(levels);
        } else {
            values_to_write = values.len();
            num_values = values_to_write;
        }

        // Process repetition levels and determine how many rows we are about to process.
        if self.descr.max_rep_level() > 0 {
            // A row could contain more than one value.
            if rep_levels.is_none() {
                return Err(general_err!(
                    "Repetition levels are required, because max repetition level = {}",
                    self.descr.max_rep_level()
                ));
            }

            // Count the occasions where we start a new row
            let levels = rep_levels.unwrap();
            for &level in levels {
                self.num_buffered_rows += (level == 0) as u32
            }

            self.write_repetition_levels(levels);
        } else {
            // Each value is exactly one row.
            // Equals to the number of values, we count nulls as well.
            self.num_buffered_rows += num_values as u32;
        }

        // Check that we have enough values to write.
        if values.len() < values_to_write {
            return Err(general_err!(
                "Expected to write {} values, but have only {}",
                values_to_write,
                values.len()
            ));
        }

        // TODO: update page statistics

        self.write_values(&values[0..values_to_write])?;

        self.num_buffered_values += num_values as u32;
        self.num_buffered_encoded_values += values_to_write as u32;

        if self.should_add_data_page() {
            self.add_data_page()?;
        }

        if self.should_dict_fallback() {
            self.dict_fallback()?;
        }

        Ok(values_to_write)
    }

    #[inline]
    fn write_definition_levels(&mut self, def_levels: &[i16]) {
        self.def_levels_sink.extend_from_slice(def_levels);
    }

    #[inline]
    fn write_repetition_levels(&mut self, rep_levels: &[i16]) {
        self.rep_levels_sink.extend_from_slice(rep_levels);
    }

    #[inline]
    fn write_values(&mut self, values: &[T::T]) -> Result<()> {
        match self.dict_encoder {
            Some(ref mut encoder) => encoder.put(values),
            None => self.encoder.put(values),
        }
    }

    /// Returns true if we need to fall back to non-dictionary encoding.
    ///
    /// We can only fall back if dictionary encoder is set and we have exceeded dictionary
    /// size.
    #[inline]
    fn should_dict_fallback(&self) -> bool {
        match self.dict_encoder {
            Some(ref encoder) => {
                encoder.dict_encoded_size() >= self.props.dictionary_pagesize_limit()
            }
            None => false,
        }
    }

    /// Returns true if there is enough data for a data page, false otherwise.
    #[inline]
    fn should_add_data_page(&self) -> bool {
        match self.dict_encoder {
            Some(ref encoder) => {
                encoder.estimated_data_encoded_size() >= self.props.data_pagesize_limit()
            }
            None => {
                self.encoder.estimated_data_encoded_size()
                    >= self.props.data_pagesize_limit()
            }
        }
    }

    /// Performs dictionary fallback.
    /// Prepares and writes dictionary and all data pages into page writer.
    fn dict_fallback(&mut self) -> Result<()> {
        // At this point we know that we need to fall back.
        self.write_dictionary_page()?;
        self.flush_data_pages()?;
        self.dict_encoder = None;
        Ok(())
    }

    /// Adds data page.
    /// Data page is either buffered in case of dictionary encoding or written directly.
    fn add_data_page(&mut self) -> Result<()> {
        // Extract encoded values
        let value_bytes = match self.dict_encoder {
            Some(ref mut encoder) => encoder.write_indices()?,
            None => self.encoder.flush_buffer()?,
        };

        // Select encoding based on current encoder and writer version (v1 or v2).
        let encoding = if self.dict_encoder.is_some() {
            self.props.dictionary_data_page_encoding()
        } else {
            self.encoder.encoding()
        };

        let max_def_level = self.descr.max_def_level();
        let max_rep_level = self.descr.max_rep_level();

        let compressed_page = match self.props.writer_version() {
            WriterVersion::PARQUET_1_0 => {
                let mut buffer = vec![];

                if max_rep_level > 0 {
                    buffer.extend_from_slice(
                        &self.encode_levels_v1(
                            Encoding::RLE,
                            &self.rep_levels_sink[..],
                            max_rep_level,
                        )?[..],
                    );
                }

                if max_def_level > 0 {
                    buffer.extend_from_slice(
                        &self.encode_levels_v1(
                            Encoding::RLE,
                            &self.def_levels_sink[..],
                            max_def_level,
                        )?[..],
                    );
                }

                buffer.extend_from_slice(value_bytes.data());
                let uncompressed_size = buffer.len();

                if let Some(ref mut cmpr) = self.compressor {
                    let mut compressed_buf = Vec::with_capacity(value_bytes.data().len());
                    cmpr.compress(&buffer[..], &mut compressed_buf)?;
                    buffer = compressed_buf;
                }

                let data_page = Page::DataPage {
                    buf: ByteBufferPtr::new(buffer),
                    num_values: self.num_buffered_values,
                    encoding,
                    def_level_encoding: Encoding::RLE,
                    rep_level_encoding: Encoding::RLE,
                    // TODO: process statistics
                    statistics: None,
                };

                CompressedPage::new(data_page, uncompressed_size)
            }
            WriterVersion::PARQUET_2_0 => {
                let mut rep_levels_byte_len = 0;
                let mut def_levels_byte_len = 0;
                let mut buffer = vec![];

                if max_rep_level > 0 {
                    let levels =
                        self.encode_levels_v2(&self.rep_levels_sink[..], max_rep_level)?;
                    rep_levels_byte_len = levels.len();
                    buffer.extend_from_slice(&levels[..]);
                }

                if max_def_level > 0 {
                    let levels =
                        self.encode_levels_v2(&self.def_levels_sink[..], max_def_level)?;
                    def_levels_byte_len = levels.len();
                    buffer.extend_from_slice(&levels[..]);
                }

                let uncompressed_size =
                    rep_levels_byte_len + def_levels_byte_len + value_bytes.len();

                // Data Page v2 compresses values only.
                match self.compressor {
                    Some(ref mut cmpr) => {
                        let mut compressed_buf =
                            Vec::with_capacity(value_bytes.data().len());
                        cmpr.compress(value_bytes.data(), &mut compressed_buf)?;
                        buffer.extend_from_slice(&compressed_buf[..]);
                    }
                    None => {
                        buffer.extend_from_slice(value_bytes.data());
                    }
                }

                let data_page = Page::DataPageV2 {
                    buf: ByteBufferPtr::new(buffer),
                    num_values: self.num_buffered_values,
                    encoding,
                    num_nulls: self.num_buffered_values
                        - self.num_buffered_encoded_values,
                    num_rows: self.num_buffered_rows,
                    def_levels_byte_len: def_levels_byte_len as u32,
                    rep_levels_byte_len: rep_levels_byte_len as u32,
                    is_compressed: self.compressor.is_some(),
                    // TODO: process statistics
                    statistics: None,
                };

                CompressedPage::new(data_page, uncompressed_size)
            }
        };

        // Check if we need to buffer data page or flush it to the sink directly.
        if self.dict_encoder.is_some() {
            self.data_pages.push_back(compressed_page);
        } else {
            self.write_data_page(compressed_page)?;
        }

        // Update total number of rows.
        self.total_rows_written += self.num_buffered_rows as u64;

        // Reset state.
        self.rep_levels_sink.clear();
        self.def_levels_sink.clear();
        self.num_buffered_values = 0;
        self.num_buffered_encoded_values = 0;
        self.num_buffered_rows = 0;

        Ok(())
    }

    /// Finalises any outstanding data pages and flushes buffered data pages from
    /// dictionary encoding into underlying sink.
    #[inline]
    fn flush_data_pages(&mut self) -> Result<()> {
        // Write all outstanding data to a new page.
        if self.num_buffered_values > 0 {
            self.add_data_page()?;
        }

        while let Some(page) = self.data_pages.pop_front() {
            self.write_data_page(page)?;
        }

        Ok(())
    }

    /// Assembles and writes column chunk metadata.
    fn write_column_metadata(&mut self) -> Result<ColumnChunkMetaData> {
        let total_compressed_size = self.total_compressed_size as i64;
        let total_uncompressed_size = self.total_uncompressed_size as i64;
        let num_values = self.total_num_values as i64;
        let dict_page_offset = self.dictionary_page_offset.map(|v| v as i64);
        // If data page offset is not set, then no pages have been written
        let data_page_offset = self.data_page_offset.unwrap_or(0) as i64;

        let file_offset;
        let mut encodings = Vec::new();

        if self.has_dictionary {
            assert!(dict_page_offset.is_some(), "Dictionary offset is not set");
            file_offset = dict_page_offset.unwrap() + total_compressed_size;
            // NOTE: This should be in sync with writing dictionary pages.
            encodings.push(self.props.dictionary_page_encoding());
            encodings.push(self.props.dictionary_data_page_encoding());
            // Fallback to alternative encoding, add it to the list.
            if self.dict_encoder.is_none() {
                encodings.push(self.encoder.encoding());
            }
        } else {
            file_offset = data_page_offset + total_compressed_size;
            encodings.push(self.encoder.encoding());
        }
        // We use only RLE level encoding for data page v1 and data page v2.
        encodings.push(Encoding::RLE);

        let metadata = ColumnChunkMetaData::builder(self.descr.clone())
            .set_compression(self.codec)
            .set_encodings(encodings)
            .set_file_offset(file_offset)
            .set_total_compressed_size(total_compressed_size)
            .set_total_uncompressed_size(total_uncompressed_size)
            .set_num_values(num_values)
            .set_data_page_offset(data_page_offset)
            .set_dictionary_page_offset(dict_page_offset)
            .build()?;

        self.page_writer.write_metadata(&metadata)?;

        Ok(metadata)
    }

    /// Encodes definition or repetition levels for Data Page v1.
    #[inline]
    fn encode_levels_v1(
        &self,
        encoding: Encoding,
        levels: &[i16],
        max_level: i16,
    ) -> Result<Vec<u8>> {
        let size = max_buffer_size(encoding, max_level, levels.len());
        let mut encoder = LevelEncoder::v1(encoding, max_level, vec![0; size]);
        encoder.put(&levels)?;
        encoder.consume()
    }

    /// Encodes definition or repetition levels for Data Page v2.
    /// Encoding is always RLE.
    #[inline]
    fn encode_levels_v2(&self, levels: &[i16], max_level: i16) -> Result<Vec<u8>> {
        let size = max_buffer_size(Encoding::RLE, max_level, levels.len());
        let mut encoder = LevelEncoder::v2(max_level, vec![0; size]);
        encoder.put(&levels)?;
        encoder.consume()
    }

    /// Writes compressed data page into underlying sink and updates global metrics.
    #[inline]
    fn write_data_page(&mut self, page: CompressedPage) -> Result<()> {
        let page_spec = self.page_writer.write_page(page)?;
        self.update_metrics_for_page(page_spec);
        Ok(())
    }

    /// Writes dictionary page into underlying sink.
    #[inline]
    fn write_dictionary_page(&mut self) -> Result<()> {
        if self.dict_encoder.is_none() {
            return Err(general_err!("Dictionary encoder is not set"));
        }

        let compressed_page = {
            let encoder = self.dict_encoder.as_ref().unwrap();
            let is_sorted = encoder.is_sorted();
            let num_values = encoder.num_entries();
            let mut values_buf = encoder.write_dict()?;
            let uncompressed_size = values_buf.len();

            if let Some(ref mut cmpr) = self.compressor {
                let mut output_buf = Vec::with_capacity(uncompressed_size);
                cmpr.compress(values_buf.data(), &mut output_buf)?;
                values_buf = ByteBufferPtr::new(output_buf);
            }

            let dict_page = Page::DictionaryPage {
                buf: values_buf,
                num_values: num_values as u32,
                encoding: self.props.dictionary_page_encoding(),
                is_sorted,
            };
            CompressedPage::new(dict_page, uncompressed_size)
        };

        let page_spec = self.page_writer.write_page(compressed_page)?;
        self.update_metrics_for_page(page_spec);
        Ok(())
    }

    /// Updates column writer metrics with each page metadata.
    #[inline]
    fn update_metrics_for_page(&mut self, page_spec: PageWriteSpec) {
        self.total_uncompressed_size += page_spec.uncompressed_size as u64;
        self.total_compressed_size += page_spec.compressed_size as u64;
        self.total_num_values += page_spec.num_values as u64;
        self.total_bytes_written += page_spec.bytes_written;

        match page_spec.page_type {
            PageType::DATA_PAGE | PageType::DATA_PAGE_V2 => {
                if self.data_page_offset.is_none() {
                    self.data_page_offset = Some(page_spec.offset);
                }
            }
            PageType::DICTIONARY_PAGE => {
                assert!(
                    self.dictionary_page_offset.is_none(),
                    "Dictionary offset is already set"
                );
                self.dictionary_page_offset = Some(page_spec.offset);
            }
            _ => {}
        }
    }

    /// Returns reference to the underlying page writer.
    /// This method is intended to use in tests only.
    fn get_page_writer_ref(&self) -> &Box<PageWriter> {
        &self.page_writer
    }
}

// ----------------------------------------------------------------------
// Encoding support for column writer.
// This mirrors parquet-mr default encodings for writes. See:
// https://github.com/apache/parquet-mr/blob/master/parquet-column/src/main/java/org/apache/parquet/column/values/factory/DefaultV1ValuesWriterFactory.java
// https://github.com/apache/parquet-mr/blob/master/parquet-column/src/main/java/org/apache/parquet/column/values/factory/DefaultV2ValuesWriterFactory.java

/// Trait to define default encoding for types, including whether or not the type
/// supports dictionary encoding.
trait EncodingWriteSupport {
    /// Returns encoding for a column when no other encoding is provided in writer
    /// properties.
    fn fallback_encoding(props: &WriterProperties) -> Encoding;

    /// Returns true if dictionary is supported for column writer, false otherwise.
    fn has_dictionary_support(props: &WriterProperties) -> bool;
}

// Basic implementation, always falls back to PLAIN and supports dictionary.
impl<T: DataType> EncodingWriteSupport for ColumnWriterImpl<T> {
    default fn fallback_encoding(_props: &WriterProperties) -> Encoding {
        Encoding::PLAIN
    }

    default fn has_dictionary_support(_props: &WriterProperties) -> bool {
        true
    }
}

impl EncodingWriteSupport for ColumnWriterImpl<BoolType> {
    fn fallback_encoding(props: &WriterProperties) -> Encoding {
        match props.writer_version() {
            WriterVersion::PARQUET_1_0 => Encoding::PLAIN,
            WriterVersion::PARQUET_2_0 => Encoding::RLE,
        }
    }

    // Boolean column does not support dictionary encoding and should fall back to
    // whatever fallback encoding is defined.
    fn has_dictionary_support(_props: &WriterProperties) -> bool {
        false
    }
}

impl EncodingWriteSupport for ColumnWriterImpl<Int32Type> {
    fn fallback_encoding(props: &WriterProperties) -> Encoding {
        match props.writer_version() {
            WriterVersion::PARQUET_1_0 => Encoding::PLAIN,
            WriterVersion::PARQUET_2_0 => Encoding::DELTA_BINARY_PACKED,
        }
    }
}

impl EncodingWriteSupport for ColumnWriterImpl<Int64Type> {
    fn fallback_encoding(props: &WriterProperties) -> Encoding {
        match props.writer_version() {
            WriterVersion::PARQUET_1_0 => Encoding::PLAIN,
            WriterVersion::PARQUET_2_0 => Encoding::DELTA_BINARY_PACKED,
        }
    }
}

impl EncodingWriteSupport for ColumnWriterImpl<ByteArrayType> {
    fn fallback_encoding(props: &WriterProperties) -> Encoding {
        match props.writer_version() {
            WriterVersion::PARQUET_1_0 => Encoding::PLAIN,
            WriterVersion::PARQUET_2_0 => Encoding::DELTA_BYTE_ARRAY,
        }
    }
}

impl EncodingWriteSupport for ColumnWriterImpl<FixedLenByteArrayType> {
    fn fallback_encoding(props: &WriterProperties) -> Encoding {
        match props.writer_version() {
            WriterVersion::PARQUET_1_0 => Encoding::PLAIN,
            WriterVersion::PARQUET_2_0 => Encoding::DELTA_BYTE_ARRAY,
        }
    }

    fn has_dictionary_support(props: &WriterProperties) -> bool {
        match props.writer_version() {
            // Dictionary encoding was not enabled in PARQUET 1.0
            WriterVersion::PARQUET_1_0 => false,
            WriterVersion::PARQUET_2_0 => true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use rand::distributions::uniform::SampleUniform;

    use crate::column::{
        page::PageReader,
        reader::{get_column_reader, get_typed_column_reader, ColumnReaderImpl},
    };
    use crate::file::{
        properties::WriterProperties, reader::SerializedPageReader,
        writer::SerializedPageWriter,
    };
    use crate::schema::types::{ColumnDescriptor, ColumnPath, Type as SchemaType};
    use crate::util::{
        io::{FileSink, FileSource},
        test_common::{get_temp_file, random_numbers_range},
    };

    #[test]
    fn test_column_writer_inconsistent_def_rep_length() {
        let page_writer = get_test_page_writer();
        let props = Rc::new(WriterProperties::builder().build());
        let mut writer = get_test_column_writer::<Int32Type>(page_writer, 1, 1, props);
        let res = writer.write_batch(&[1, 2, 3, 4], Some(&[1, 1, 1]), Some(&[0, 0]));
        assert!(res.is_err());
        if let Err(err) = res {
            assert_eq!(
                format!("{}", err),
                "Parquet error: Inconsistent length of definition and repetition levels: 3 != 2"
            );
        }
    }

    #[test]
    fn test_column_writer_invalid_def_levels() {
        let page_writer = get_test_page_writer();
        let props = Rc::new(WriterProperties::builder().build());
        let mut writer = get_test_column_writer::<Int32Type>(page_writer, 1, 0, props);
        let res = writer.write_batch(&[1, 2, 3, 4], None, None);
        assert!(res.is_err());
        if let Err(err) = res {
            assert_eq!(
                format!("{}", err),
                "Parquet error: Definition levels are required, because max definition level = 1"
            );
        }
    }

    #[test]
    fn test_column_writer_invalid_rep_levels() {
        let page_writer = get_test_page_writer();
        let props = Rc::new(WriterProperties::builder().build());
        let mut writer = get_test_column_writer::<Int32Type>(page_writer, 0, 1, props);
        let res = writer.write_batch(&[1, 2, 3, 4], None, None);
        assert!(res.is_err());
        if let Err(err) = res {
            assert_eq!(
                format!("{}", err),
                "Parquet error: Repetition levels are required, because max repetition level = 1"
            );
        }
    }

    #[test]
    fn test_column_writer_not_enough_values_to_write() {
        let page_writer = get_test_page_writer();
        let props = Rc::new(WriterProperties::builder().build());
        let mut writer = get_test_column_writer::<Int32Type>(page_writer, 1, 0, props);
        let res = writer.write_batch(&[1, 2], Some(&[1, 1, 1, 1]), None);
        assert!(res.is_err());
        if let Err(err) = res {
            assert_eq!(
                format!("{}", err),
                "Parquet error: Expected to write 4 values, but have only 2"
            );
        }
    }

    #[test]
    #[should_panic(expected = "Dictionary offset is already set")]
    fn test_column_writer_write_only_one_dictionary_page() {
        let page_writer = get_test_page_writer();
        let props = Rc::new(WriterProperties::builder().build());
        let mut writer = get_test_column_writer::<Int32Type>(page_writer, 0, 0, props);
        writer.write_batch(&[1, 2, 3, 4], None, None).unwrap();
        // First page should be correctly written.
        let res = writer.write_dictionary_page();
        assert!(res.is_ok());
        writer.write_dictionary_page().unwrap();
    }

    #[test]
    fn test_column_writer_error_when_writing_disabled_dictionary() {
        let page_writer = get_test_page_writer();
        let props = Rc::new(
            WriterProperties::builder()
                .set_dictionary_enabled(false)
                .build(),
        );
        let mut writer = get_test_column_writer::<Int32Type>(page_writer, 0, 0, props);
        writer.write_batch(&[1, 2, 3, 4], None, None).unwrap();
        let res = writer.write_dictionary_page();
        assert!(res.is_err());
        if let Err(err) = res {
            assert_eq!(
                format!("{}", err),
                "Parquet error: Dictionary encoder is not set"
            );
        }
    }

    #[test]
    fn test_column_writer_boolean_type_does_not_support_dictionary() {
        let page_writer = get_test_page_writer();
        let props = Rc::new(
            WriterProperties::builder()
                .set_dictionary_enabled(true)
                .build(),
        );
        let mut writer = get_test_column_writer::<BoolType>(page_writer, 0, 0, props);
        writer
            .write_batch(&[true, false, true, false], None, None)
            .unwrap();

        let (bytes_written, rows_written, metadata) = writer.close().unwrap();
        // PlainEncoder uses bit writer to write boolean values, which all fit into 1
        // byte.
        assert_eq!(bytes_written, 1);
        assert_eq!(rows_written, 4);
        assert_eq!(metadata.encodings(), &vec![Encoding::PLAIN, Encoding::RLE]);
        assert_eq!(metadata.num_values(), 4); // just values
        assert_eq!(metadata.dictionary_page_offset(), None);
    }

    #[test]
    fn test_column_writer_default_encoding_support_bool() {
        check_encoding_write_support::<BoolType>(
            WriterVersion::PARQUET_1_0,
            true,
            &[true, false],
            None,
            &[Encoding::PLAIN, Encoding::RLE],
        );
        check_encoding_write_support::<BoolType>(
            WriterVersion::PARQUET_1_0,
            false,
            &[true, false],
            None,
            &[Encoding::PLAIN, Encoding::RLE],
        );
        check_encoding_write_support::<BoolType>(
            WriterVersion::PARQUET_2_0,
            true,
            &[true, false],
            None,
            &[Encoding::RLE, Encoding::RLE],
        );
        check_encoding_write_support::<BoolType>(
            WriterVersion::PARQUET_2_0,
            false,
            &[true, false],
            None,
            &[Encoding::RLE, Encoding::RLE],
        );
    }

    #[test]
    fn test_column_writer_default_encoding_support_int32() {
        check_encoding_write_support::<Int32Type>(
            WriterVersion::PARQUET_1_0,
            true,
            &[1, 2],
            Some(0),
            &[Encoding::PLAIN, Encoding::RLE_DICTIONARY, Encoding::RLE],
        );
        check_encoding_write_support::<Int32Type>(
            WriterVersion::PARQUET_1_0,
            false,
            &[1, 2],
            None,
            &[Encoding::PLAIN, Encoding::RLE],
        );
        check_encoding_write_support::<Int32Type>(
            WriterVersion::PARQUET_2_0,
            true,
            &[1, 2],
            Some(0),
            &[Encoding::PLAIN, Encoding::RLE_DICTIONARY, Encoding::RLE],
        );
        check_encoding_write_support::<Int32Type>(
            WriterVersion::PARQUET_2_0,
            false,
            &[1, 2],
            None,
            &[Encoding::DELTA_BINARY_PACKED, Encoding::RLE],
        );
    }

    #[test]
    fn test_column_writer_default_encoding_support_int64() {
        check_encoding_write_support::<Int64Type>(
            WriterVersion::PARQUET_1_0,
            true,
            &[1, 2],
            Some(0),
            &[Encoding::PLAIN, Encoding::RLE_DICTIONARY, Encoding::RLE],
        );
        check_encoding_write_support::<Int64Type>(
            WriterVersion::PARQUET_1_0,
            false,
            &[1, 2],
            None,
            &[Encoding::PLAIN, Encoding::RLE],
        );
        check_encoding_write_support::<Int64Type>(
            WriterVersion::PARQUET_2_0,
            true,
            &[1, 2],
            Some(0),
            &[Encoding::PLAIN, Encoding::RLE_DICTIONARY, Encoding::RLE],
        );
        check_encoding_write_support::<Int64Type>(
            WriterVersion::PARQUET_2_0,
            false,
            &[1, 2],
            None,
            &[Encoding::DELTA_BINARY_PACKED, Encoding::RLE],
        );
    }

    #[test]
    fn test_column_writer_default_encoding_support_int96() {
        check_encoding_write_support::<Int96Type>(
            WriterVersion::PARQUET_1_0,
            true,
            &[Int96::from(vec![1, 2, 3])],
            Some(0),
            &[Encoding::PLAIN, Encoding::RLE_DICTIONARY, Encoding::RLE],
        );
        check_encoding_write_support::<Int96Type>(
            WriterVersion::PARQUET_1_0,
            false,
            &[Int96::from(vec![1, 2, 3])],
            None,
            &[Encoding::PLAIN, Encoding::RLE],
        );
        check_encoding_write_support::<Int96Type>(
            WriterVersion::PARQUET_2_0,
            true,
            &[Int96::from(vec![1, 2, 3])],
            Some(0),
            &[Encoding::PLAIN, Encoding::RLE_DICTIONARY, Encoding::RLE],
        );
        check_encoding_write_support::<Int96Type>(
            WriterVersion::PARQUET_2_0,
            false,
            &[Int96::from(vec![1, 2, 3])],
            None,
            &[Encoding::PLAIN, Encoding::RLE],
        );
    }

    #[test]
    fn test_column_writer_default_encoding_support_float() {
        check_encoding_write_support::<FloatType>(
            WriterVersion::PARQUET_1_0,
            true,
            &[1.0, 2.0],
            Some(0),
            &[Encoding::PLAIN, Encoding::RLE_DICTIONARY, Encoding::RLE],
        );
        check_encoding_write_support::<FloatType>(
            WriterVersion::PARQUET_1_0,
            false,
            &[1.0, 2.0],
            None,
            &[Encoding::PLAIN, Encoding::RLE],
        );
        check_encoding_write_support::<FloatType>(
            WriterVersion::PARQUET_2_0,
            true,
            &[1.0, 2.0],
            Some(0),
            &[Encoding::PLAIN, Encoding::RLE_DICTIONARY, Encoding::RLE],
        );
        check_encoding_write_support::<FloatType>(
            WriterVersion::PARQUET_2_0,
            false,
            &[1.0, 2.0],
            None,
            &[Encoding::PLAIN, Encoding::RLE],
        );
    }

    #[test]
    fn test_column_writer_default_encoding_support_double() {
        check_encoding_write_support::<DoubleType>(
            WriterVersion::PARQUET_1_0,
            true,
            &[1.0, 2.0],
            Some(0),
            &[Encoding::PLAIN, Encoding::RLE_DICTIONARY, Encoding::RLE],
        );
        check_encoding_write_support::<DoubleType>(
            WriterVersion::PARQUET_1_0,
            false,
            &[1.0, 2.0],
            None,
            &[Encoding::PLAIN, Encoding::RLE],
        );
        check_encoding_write_support::<DoubleType>(
            WriterVersion::PARQUET_2_0,
            true,
            &[1.0, 2.0],
            Some(0),
            &[Encoding::PLAIN, Encoding::RLE_DICTIONARY, Encoding::RLE],
        );
        check_encoding_write_support::<DoubleType>(
            WriterVersion::PARQUET_2_0,
            false,
            &[1.0, 2.0],
            None,
            &[Encoding::PLAIN, Encoding::RLE],
        );
    }

    #[test]
    fn test_column_writer_default_encoding_support_byte_array() {
        check_encoding_write_support::<ByteArrayType>(
            WriterVersion::PARQUET_1_0,
            true,
            &[ByteArray::from(vec![1u8])],
            Some(0),
            &[Encoding::PLAIN, Encoding::RLE_DICTIONARY, Encoding::RLE],
        );
        check_encoding_write_support::<ByteArrayType>(
            WriterVersion::PARQUET_1_0,
            false,
            &[ByteArray::from(vec![1u8])],
            None,
            &[Encoding::PLAIN, Encoding::RLE],
        );
        check_encoding_write_support::<ByteArrayType>(
            WriterVersion::PARQUET_2_0,
            true,
            &[ByteArray::from(vec![1u8])],
            Some(0),
            &[Encoding::PLAIN, Encoding::RLE_DICTIONARY, Encoding::RLE],
        );
        check_encoding_write_support::<ByteArrayType>(
            WriterVersion::PARQUET_2_0,
            false,
            &[ByteArray::from(vec![1u8])],
            None,
            &[Encoding::DELTA_BYTE_ARRAY, Encoding::RLE],
        );
    }

    #[test]
    fn test_column_writer_default_encoding_support_fixed_len_byte_array() {
        check_encoding_write_support::<FixedLenByteArrayType>(
            WriterVersion::PARQUET_1_0,
            true,
            &[ByteArray::from(vec![1u8])],
            None,
            &[Encoding::PLAIN, Encoding::RLE],
        );
        check_encoding_write_support::<FixedLenByteArrayType>(
            WriterVersion::PARQUET_1_0,
            false,
            &[ByteArray::from(vec![1u8])],
            None,
            &[Encoding::PLAIN, Encoding::RLE],
        );
        check_encoding_write_support::<FixedLenByteArrayType>(
            WriterVersion::PARQUET_2_0,
            true,
            &[ByteArray::from(vec![1u8])],
            Some(0),
            &[Encoding::PLAIN, Encoding::RLE_DICTIONARY, Encoding::RLE],
        );
        check_encoding_write_support::<FixedLenByteArrayType>(
            WriterVersion::PARQUET_2_0,
            false,
            &[ByteArray::from(vec![1u8])],
            None,
            &[Encoding::DELTA_BYTE_ARRAY, Encoding::RLE],
        );
    }

    #[test]
    fn test_column_writer_check_metadata() {
        let page_writer = get_test_page_writer();
        let props = Rc::new(WriterProperties::builder().build());
        let mut writer = get_test_column_writer::<Int32Type>(page_writer, 0, 0, props);
        writer.write_batch(&[1, 2, 3, 4], None, None).unwrap();

        let (bytes_written, rows_written, metadata) = writer.close().unwrap();
        assert_eq!(bytes_written, 20);
        assert_eq!(rows_written, 4);
        assert_eq!(
            metadata.encodings(),
            &vec![Encoding::PLAIN, Encoding::RLE_DICTIONARY, Encoding::RLE]
        );
        assert_eq!(metadata.num_values(), 8); // dictionary + value indexes
        assert_eq!(metadata.compressed_size(), 20);
        assert_eq!(metadata.uncompressed_size(), 20);
        assert_eq!(metadata.data_page_offset(), 0);
        assert_eq!(metadata.dictionary_page_offset(), Some(0));
    }

    #[test]
    fn test_column_writer_empty_column_roundtrip() {
        let props = WriterProperties::builder().build();
        column_roundtrip::<Int32Type>("test_col_writer_rnd_1", props, &[], None, None);
    }

    #[test]
    fn test_column_writer_non_nullable_values_roundtrip() {
        let props = WriterProperties::builder().build();
        column_roundtrip_random::<Int32Type>(
            "test_col_writer_rnd_2",
            props,
            1024,
            std::i32::MIN,
            std::i32::MAX,
            0,
            0,
        );
    }

    #[test]
    fn test_column_writer_nullable_non_repeated_values_roundtrip() {
        let props = WriterProperties::builder().build();
        column_roundtrip_random::<Int32Type>(
            "test_column_writer_nullable_non_repeated_values_roundtrip",
            props,
            1024,
            std::i32::MIN,
            std::i32::MAX,
            10,
            0,
        );
    }

    #[test]
    fn test_column_writer_nullable_repeated_values_roundtrip() {
        let props = WriterProperties::builder().build();
        column_roundtrip_random::<Int32Type>(
            "test_col_writer_rnd_3",
            props,
            1024,
            std::i32::MIN,
            std::i32::MAX,
            10,
            10,
        );
    }

    #[test]
    fn test_column_writer_dictionary_fallback_small_data_page() {
        let props = WriterProperties::builder()
            .set_dictionary_pagesize_limit(32)
            .set_data_pagesize_limit(32)
            .build();
        column_roundtrip_random::<Int32Type>(
            "test_col_writer_rnd_4",
            props,
            1024,
            std::i32::MIN,
            std::i32::MAX,
            10,
            10,
        );
    }

    #[test]
    fn test_column_writer_small_write_batch_size() {
        for i in vec![1, 2, 5, 10, 11, 1023] {
            let props = WriterProperties::builder().set_write_batch_size(i).build();

            column_roundtrip_random::<Int32Type>(
                "test_col_writer_rnd_5",
                props,
                1024,
                std::i32::MIN,
                std::i32::MAX,
                10,
                10,
            );
        }
    }

    #[test]
    fn test_column_writer_dictionary_disabled_v1() {
        let props = WriterProperties::builder()
            .set_writer_version(WriterVersion::PARQUET_1_0)
            .set_dictionary_enabled(false)
            .build();
        column_roundtrip_random::<Int32Type>(
            "test_col_writer_rnd_6",
            props,
            1024,
            std::i32::MIN,
            std::i32::MAX,
            10,
            10,
        );
    }

    #[test]
    fn test_column_writer_dictionary_disabled_v2() {
        let props = WriterProperties::builder()
            .set_writer_version(WriterVersion::PARQUET_2_0)
            .set_dictionary_enabled(false)
            .build();
        column_roundtrip_random::<Int32Type>(
            "test_col_writer_rnd_7",
            props,
            1024,
            std::i32::MIN,
            std::i32::MAX,
            10,
            10,
        );
    }

    #[test]
    fn test_column_writer_compression_v1() {
        let props = WriterProperties::builder()
            .set_writer_version(WriterVersion::PARQUET_1_0)
            .set_compression(Compression::SNAPPY)
            .build();
        column_roundtrip_random::<Int32Type>(
            "test_col_writer_rnd_8",
            props,
            2048,
            std::i32::MIN,
            std::i32::MAX,
            10,
            10,
        );
    }

    #[test]
    fn test_column_writer_compression_v2() {
        let props = WriterProperties::builder()
            .set_writer_version(WriterVersion::PARQUET_2_0)
            .set_compression(Compression::SNAPPY)
            .build();
        column_roundtrip_random::<Int32Type>(
            "test_col_writer_rnd_9",
            props,
            2048,
            std::i32::MIN,
            std::i32::MAX,
            10,
            10,
        );
    }

    #[test]
    fn test_column_writer_add_data_pages_with_dict() {
        // ARROW-5129: Test verifies that we add data page in case of dictionary encoding
        // and no fallback occurred so far.
        let file = get_temp_file("test_column_writer_add_data_pages_with_dict", &[]);
        let sink = FileSink::new(&file);
        let page_writer = Box::new(SerializedPageWriter::new(sink));
        let props = Rc::new(
            WriterProperties::builder()
                .set_data_pagesize_limit(15) // actually each page will have size 15-18 bytes
                .set_write_batch_size(3) // write 3 values at a time
                .build(),
        );
        let data = &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let mut writer = get_test_column_writer::<Int32Type>(page_writer, 0, 0, props);
        writer.write_batch(data, None, None).unwrap();
        let (bytes_written, _, _) = writer.close().unwrap();

        // Read pages and check the sequence
        let source = FileSource::new(&file, 0, bytes_written as usize);
        let mut page_reader = Box::new(
            SerializedPageReader::new(
                source,
                data.len() as i64,
                Compression::UNCOMPRESSED,
                Int32Type::get_physical_type(),
            )
            .unwrap(),
        );
        let mut res = Vec::new();
        while let Some(page) = page_reader.get_next_page().unwrap() {
            res.push((page.page_type(), page.num_values()));
        }
        assert_eq!(
            res,
            vec![
                (PageType::DICTIONARY_PAGE, 10),
                (PageType::DATA_PAGE, 3),
                (PageType::DATA_PAGE, 3),
                (PageType::DATA_PAGE, 3),
                (PageType::DATA_PAGE, 1)
            ]
        );
    }

    /// Performs write-read roundtrip with randomly generated values and levels.
    /// `max_size` is maximum number of values or levels (if `max_def_level` > 0) to write
    /// for a column.
    fn column_roundtrip_random<'a, T: DataType>(
        file_name: &'a str,
        props: WriterProperties,
        max_size: usize,
        min_value: T::T,
        max_value: T::T,
        max_def_level: i16,
        max_rep_level: i16,
    ) where
        T::T: PartialOrd + SampleUniform + Copy,
    {
        let mut num_values: usize = 0;

        let mut buf: Vec<i16> = Vec::new();
        let def_levels = if max_def_level > 0 {
            random_numbers_range(max_size, 0, max_def_level + 1, &mut buf);
            for &dl in &buf[..] {
                if dl == max_def_level {
                    num_values += 1;
                }
            }
            Some(&buf[..])
        } else {
            num_values = max_size;
            None
        };

        let mut buf: Vec<i16> = Vec::new();
        let rep_levels = if max_rep_level > 0 {
            random_numbers_range(max_size, 0, max_rep_level + 1, &mut buf);
            Some(&buf[..])
        } else {
            None
        };

        let mut values: Vec<T::T> = Vec::new();
        random_numbers_range(num_values, min_value, max_value, &mut values);

        column_roundtrip::<T>(file_name, props, &values[..], def_levels, rep_levels);
    }

    /// Performs write-read roundtrip and asserts written values and levels.
    fn column_roundtrip<'a, T: DataType>(
        file_name: &'a str,
        props: WriterProperties,
        values: &[T::T],
        def_levels: Option<&[i16]>,
        rep_levels: Option<&[i16]>,
    ) {
        let file = get_temp_file(file_name, &[]);
        let sink = FileSink::new(&file);
        let page_writer = Box::new(SerializedPageWriter::new(sink));

        let max_def_level = match def_levels {
            Some(buf) => *buf.iter().max().unwrap_or(&0i16),
            None => 0i16,
        };

        let max_rep_level = match rep_levels {
            Some(buf) => *buf.iter().max().unwrap_or(&0i16),
            None => 0i16,
        };

        let mut max_batch_size = values.len();
        if let Some(levels) = def_levels {
            max_batch_size = cmp::max(max_batch_size, levels.len());
        }
        if let Some(levels) = rep_levels {
            max_batch_size = cmp::max(max_batch_size, levels.len());
        }

        let mut writer = get_test_column_writer::<T>(
            page_writer,
            max_def_level,
            max_rep_level,
            Rc::new(props),
        );

        let values_written = writer.write_batch(values, def_levels, rep_levels).unwrap();
        assert_eq!(values_written, values.len());
        let (bytes_written, rows_written, column_metadata) = writer.close().unwrap();

        let source = FileSource::new(&file, 0, bytes_written as usize);
        let page_reader = Box::new(
            SerializedPageReader::new(
                source,
                column_metadata.num_values(),
                column_metadata.compression(),
                T::get_physical_type(),
            )
            .unwrap(),
        );
        let reader =
            get_test_column_reader::<T>(page_reader, max_def_level, max_rep_level);

        let mut actual_values = vec![T::T::default(); max_batch_size];
        let mut actual_def_levels = def_levels.map(|_| vec![0i16; max_batch_size]);
        let mut actual_rep_levels = rep_levels.map(|_| vec![0i16; max_batch_size]);

        let (values_read, levels_read) = read_fully(
            reader,
            max_batch_size,
            actual_def_levels.as_mut(),
            actual_rep_levels.as_mut(),
            actual_values.as_mut_slice(),
        );

        // Assert values, definition and repetition levels.

        assert_eq!(&actual_values[..values_read], values);
        match actual_def_levels {
            Some(ref vec) => assert_eq!(Some(&vec[..levels_read]), def_levels),
            None => assert_eq!(None, def_levels),
        }
        match actual_rep_levels {
            Some(ref vec) => assert_eq!(Some(&vec[..levels_read]), rep_levels),
            None => assert_eq!(None, rep_levels),
        }

        // Assert written rows.

        if let Some(levels) = actual_rep_levels {
            let mut actual_rows_written = 0;
            for l in levels {
                if l == 0 {
                    actual_rows_written += 1;
                }
            }
            assert_eq!(actual_rows_written, rows_written);
        } else if actual_def_levels.is_some() {
            assert_eq!(levels_read as u64, rows_written);
        } else {
            assert_eq!(values_read as u64, rows_written);
        }
    }

    /// Performs write of provided values and returns column metadata of those values.
    /// Used to test encoding support for column writer.
    fn column_write_and_get_metadata<T: DataType>(
        props: WriterProperties,
        values: &[T::T],
    ) -> ColumnChunkMetaData {
        let page_writer = get_test_page_writer();
        let props = Rc::new(props);
        let mut writer = get_test_column_writer::<T>(page_writer, 0, 0, props);
        writer.write_batch(values, None, None).unwrap();
        let (_, _, metadata) = writer.close().unwrap();
        metadata
    }

    // Function to use in tests for EncodingWriteSupport. This checks that dictionary
    // offset and encodings to make sure that column writer uses provided by trait
    // encodings.
    fn check_encoding_write_support<T: DataType>(
        version: WriterVersion,
        dict_enabled: bool,
        data: &[T::T],
        dictionary_page_offset: Option<i64>,
        encodings: &[Encoding],
    ) {
        let props = WriterProperties::builder()
            .set_writer_version(version)
            .set_dictionary_enabled(dict_enabled)
            .build();
        let meta = column_write_and_get_metadata::<T>(props, data);
        assert_eq!(meta.dictionary_page_offset(), dictionary_page_offset);
        assert_eq!(meta.encodings(), &encodings);
    }

    /// Reads one batch of data, considering that batch is large enough to capture all of
    /// the values and levels.
    fn read_fully<T: DataType>(
        mut reader: ColumnReaderImpl<T>,
        batch_size: usize,
        mut def_levels: Option<&mut Vec<i16>>,
        mut rep_levels: Option<&mut Vec<i16>>,
        values: &mut [T::T],
    ) -> (usize, usize) {
        let actual_def_levels = def_levels.as_mut().map(|vec| &mut vec[..]);
        let actual_rep_levels = rep_levels.as_mut().map(|vec| &mut vec[..]);
        reader
            .read_batch(batch_size, actual_def_levels, actual_rep_levels, values)
            .unwrap()
    }

    /// Returns column writer.
    fn get_test_column_writer<T: DataType>(
        page_writer: Box<PageWriter>,
        max_def_level: i16,
        max_rep_level: i16,
        props: WriterPropertiesPtr,
    ) -> ColumnWriterImpl<T> {
        let descr = Rc::new(get_test_column_descr::<T>(max_def_level, max_rep_level));
        let column_writer = get_column_writer(descr, props, page_writer);
        get_typed_column_writer::<T>(column_writer)
    }

    /// Returns column reader.
    fn get_test_column_reader<T: DataType>(
        page_reader: Box<PageReader>,
        max_def_level: i16,
        max_rep_level: i16,
    ) -> ColumnReaderImpl<T> {
        let descr = Rc::new(get_test_column_descr::<T>(max_def_level, max_rep_level));
        let column_reader = get_column_reader(descr, page_reader);
        get_typed_column_reader::<T>(column_reader)
    }

    /// Returns descriptor for primitive column.
    fn get_test_column_descr<T: DataType>(
        max_def_level: i16,
        max_rep_level: i16,
    ) -> ColumnDescriptor {
        let path = ColumnPath::from("col");
        let tpe = SchemaType::primitive_type_builder("col", T::get_physical_type())
            // length is set for "encoding support" tests for FIXED_LEN_BYTE_ARRAY type,
            // it should be no-op for other types
            .with_length(1)
            .build()
            .unwrap();
        ColumnDescriptor::new(Rc::new(tpe), None, max_def_level, max_rep_level, path)
    }

    /// Returns page writer that collects pages without serializing them.
    fn get_test_page_writer() -> Box<PageWriter> {
        Box::new(TestPageWriter {})
    }

    struct TestPageWriter {}

    impl PageWriter for TestPageWriter {
        fn write_page(&mut self, page: CompressedPage) -> Result<PageWriteSpec> {
            let mut res = PageWriteSpec::new();
            res.page_type = page.page_type();
            res.uncompressed_size = page.uncompressed_size();
            res.compressed_size = page.compressed_size();
            res.num_values = page.num_values();
            res.offset = 0;
            res.bytes_written = page.data().len() as u64;
            Ok(res)
        }

        fn write_metadata(&mut self, _metadata: &ColumnChunkMetaData) -> Result<()> {
            Ok(())
        }

        fn close(&mut self) -> Result<()> {
            Ok(())
        }
    }
}
