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

//! Contains all supported encoders for Parquet.

use std::{cmp, io::Write, marker::PhantomData, mem};

use crate::basic::*;
use crate::data_type::*;
use crate::encodings::rle::RleEncoder;
use crate::errors::{ParquetError, Result};
use crate::schema::types::ColumnDescPtr;
use crate::util::{
    bit_util::{self, log2, num_required_bits, BitWriter},
    hash_util,
    memory::{Buffer, ByteBuffer, ByteBufferPtr, MemTrackerPtr},
};

// ----------------------------------------------------------------------
// Encoders

/// An Parquet encoder for the data type `T`.
///
/// Currently this allocates internal buffers for the encoded values. After done putting
/// values, caller should call `flush_buffer()` to get an immutable buffer pointer.
pub trait Encoder<T: DataType> {
    /// Encodes data from `values`.
    fn put(&mut self, values: &[T::T]) -> Result<()>;

    /// Encodes data from `values`, which contains spaces for null values, that is
    /// identified by `valid_bits`.
    ///
    /// Returns the number of non-null values encoded.
    fn put_spaced(&mut self, values: &[T::T], valid_bits: &[u8]) -> Result<usize> {
        let num_values = values.len();
        let mut buffer = Vec::with_capacity(num_values);
        // TODO: this is pretty inefficient. Revisit in future.
        for i in 0..num_values {
            if bit_util::get_bit(valid_bits, i) {
                buffer.push(values[i].clone());
            }
        }
        self.put(&buffer[..])?;
        Ok(buffer.len())
    }

    /// Returns the encoding type of this encoder.
    fn encoding(&self) -> Encoding;

    /// Returns an estimate of the encoded data, in bytes.
    /// Method call must be O(1).
    fn estimated_data_encoded_size(&self) -> usize;

    /// Flushes the underlying byte buffer that's being processed by this encoder, and
    /// return the immutable copy of it. This will also reset the internal state.
    fn flush_buffer(&mut self) -> Result<ByteBufferPtr>;
}

/// Gets a encoder for the particular data type `T` and encoding `encoding`. Memory usage
/// for the encoder instance is tracked by `mem_tracker`.
pub fn get_encoder<T: DataType>(
    desc: ColumnDescPtr,
    encoding: Encoding,
    mem_tracker: MemTrackerPtr,
) -> Result<Box<Encoder<T>>> {
    let encoder: Box<Encoder<T>> = match encoding {
        Encoding::PLAIN => Box::new(PlainEncoder::new(desc, mem_tracker, vec![])),
        Encoding::RLE_DICTIONARY | Encoding::PLAIN_DICTIONARY => {
            return Err(general_err!(
                "Cannot initialize this encoding through this function"
            ));
        }
        Encoding::RLE => Box::new(RleValueEncoder::new()),
        Encoding::DELTA_BINARY_PACKED => Box::new(DeltaBitPackEncoder::new()),
        Encoding::DELTA_LENGTH_BYTE_ARRAY => Box::new(DeltaLengthByteArrayEncoder::new()),
        Encoding::DELTA_BYTE_ARRAY => Box::new(DeltaByteArrayEncoder::new()),
        e => return Err(nyi_err!("Encoding {} is not supported", e)),
    };
    Ok(encoder)
}

// ----------------------------------------------------------------------
// Plain encoding

/// Plain encoding that supports all types.
/// Values are encoded back to back.
/// The plain encoding is used whenever a more efficient encoding can not be used.
/// It stores the data in the following format:
/// - BOOLEAN - 1 bit per value, 0 is false; 1 is true.
/// - INT32 - 4 bytes per value, stored as little-endian.
/// - INT64 - 8 bytes per value, stored as little-endian.
/// - FLOAT - 4 bytes per value, stored as IEEE little-endian.
/// - DOUBLE - 8 bytes per value, stored as IEEE little-endian.
/// - BYTE_ARRAY - 4 byte length stored as little endian, followed by bytes.
/// - FIXED_LEN_BYTE_ARRAY - just the bytes are stored.
pub struct PlainEncoder<T: DataType> {
    buffer: ByteBuffer,
    bit_writer: BitWriter,
    desc: ColumnDescPtr,
    _phantom: PhantomData<T>,
}

impl<T: DataType> PlainEncoder<T> {
    /// Creates new plain encoder.
    pub fn new(desc: ColumnDescPtr, mem_tracker: MemTrackerPtr, vec: Vec<u8>) -> Self {
        let mut byte_buffer = ByteBuffer::new().with_mem_tracker(mem_tracker);
        byte_buffer.set_data(vec);
        Self {
            buffer: byte_buffer,
            bit_writer: BitWriter::new(256),
            desc,
            _phantom: PhantomData,
        }
    }
}

impl<T: DataType> Encoder<T> for PlainEncoder<T> {
    fn encoding(&self) -> Encoding {
        Encoding::PLAIN
    }

    fn estimated_data_encoded_size(&self) -> usize {
        self.buffer.size() + self.bit_writer.bytes_written()
    }

    #[inline]
    default fn flush_buffer(&mut self) -> Result<ByteBufferPtr> {
        self.buffer.write(self.bit_writer.flush_buffer())?;
        self.buffer.flush()?;
        self.bit_writer.clear();

        Ok(self.buffer.consume())
    }

    default fn put(&mut self, _values: &[T::T]) -> Result<()> {
        unreachable!()
    }
}

impl<T: SliceAsBytesDataType> Encoder<T> for PlainEncoder<T>
where
    T::T: SliceAsBytes,
{
    default fn put(&mut self, values: &[T::T]) -> Result<()> {
        let bytes = T::T::slice_as_bytes(values);
        self.buffer.write(bytes)?;
        Ok(())
    }
}

impl Encoder<BoolType> for PlainEncoder<BoolType> {
    fn put(&mut self, values: &[bool]) -> Result<()> {
        for v in values {
            self.bit_writer.put_value(*v as u64, 1);
        }
        Ok(())
    }
}

impl Encoder<Int96Type> for PlainEncoder<Int96Type> {
    fn put(&mut self, values: &[Int96]) -> Result<()> {
        for v in values {
            self.buffer.write(v.as_bytes())?;
        }
        self.buffer.flush()?;
        Ok(())
    }
}

impl Encoder<ByteArrayType> for PlainEncoder<ByteArrayType> {
    fn put(&mut self, values: &[ByteArray]) -> Result<()> {
        for v in values {
            self.buffer.write(&(v.len().to_le() as u32).as_bytes())?;
            self.buffer.write(v.data())?;
        }
        self.buffer.flush()?;
        Ok(())
    }
}

impl Encoder<FixedLenByteArrayType> for PlainEncoder<FixedLenByteArrayType> {
    fn put(&mut self, values: &[ByteArray]) -> Result<()> {
        for v in values {
            self.buffer.write(v.data())?;
        }
        self.buffer.flush()?;
        Ok(())
    }
}

// ----------------------------------------------------------------------
// Dictionary encoding

const INITIAL_HASH_TABLE_SIZE: usize = 1024;
const MAX_HASH_LOAD: f32 = 0.7;
const HASH_SLOT_EMPTY: i32 = -1;

/// Dictionary encoder.
/// The dictionary encoding builds a dictionary of values encountered in a given column.
/// The dictionary page is written first, before the data pages of the column chunk.
///
/// Dictionary page format: the entries in the dictionary - in dictionary order -
/// using the plain encoding.
///
/// Data page format: the bit width used to encode the entry ids stored as 1 byte
/// (max bit width = 32), followed by the values encoded using RLE/Bit packed described
/// above (with the given bit width).
pub struct DictEncoder<T: DataType> {
    // Descriptor for the column to be encoded.
    desc: ColumnDescPtr,

    // Size of the table. **Must be** a power of 2.
    hash_table_size: usize,

    // Store `hash_table_size` - 1, so that `j & mod_bitmask` is equivalent to
    // `j % hash_table_size`, but uses far fewer CPU cycles.
    mod_bitmask: u32,

    // Stores indices which map (many-to-one) to the values in the `uniques` array.
    // Here we are using fix-sized array with linear probing.
    // A slot with `HASH_SLOT_EMPTY` indicates the slot is not currently occupied.
    hash_slots: Buffer<i32>,

    // Indices that have not yet be written out by `write_indices()`.
    buffered_indices: Buffer<i32>,

    // The unique observed values.
    uniques: Buffer<T::T>,

    // Size in bytes needed to encode this dictionary.
    uniques_size_in_bytes: usize,

    // Tracking memory usage for the various data structures in this struct.
    mem_tracker: MemTrackerPtr,
}

impl<T: DataType> DictEncoder<T> {
    /// Creates new dictionary encoder.
    pub fn new(desc: ColumnDescPtr, mem_tracker: MemTrackerPtr) -> Self {
        let mut slots = Buffer::new().with_mem_tracker(mem_tracker.clone());
        slots.resize(INITIAL_HASH_TABLE_SIZE, -1);
        Self {
            desc,
            hash_table_size: INITIAL_HASH_TABLE_SIZE,
            mod_bitmask: (INITIAL_HASH_TABLE_SIZE - 1) as u32,
            hash_slots: slots,
            buffered_indices: Buffer::new().with_mem_tracker(mem_tracker.clone()),
            uniques: Buffer::new().with_mem_tracker(mem_tracker.clone()),
            uniques_size_in_bytes: 0,
            mem_tracker,
        }
    }

    /// Returns true if dictionary entries are sorted, false otherwise.
    #[inline]
    pub fn is_sorted(&self) -> bool {
        // Sorting is not supported currently.
        false
    }

    /// Returns number of unique values (keys) in the dictionary.
    pub fn num_entries(&self) -> usize {
        self.uniques.size()
    }

    /// Returns size of unique values (keys) in the dictionary, in bytes.
    pub fn dict_encoded_size(&self) -> usize {
        self.uniques_size_in_bytes
    }

    /// Writes out the dictionary values with PLAIN encoding in a byte buffer, and return
    /// the result.
    #[inline]
    pub fn write_dict(&self) -> Result<ByteBufferPtr> {
        let mut plain_encoder =
            PlainEncoder::<T>::new(self.desc.clone(), self.mem_tracker.clone(), vec![]);
        plain_encoder.put(self.uniques.data())?;
        plain_encoder.flush_buffer()
    }

    /// Writes out the dictionary values with RLE encoding in a byte buffer, and return
    /// the result.
    #[inline]
    pub fn write_indices(&mut self) -> Result<ByteBufferPtr> {
        // TODO: the caller should allocate the buffer
        let buffer_len = self.estimated_data_encoded_size();
        let mut buffer: Vec<u8> = vec![0; buffer_len as usize];
        buffer[0] = self.bit_width() as u8;
        self.mem_tracker.alloc(buffer.capacity() as i64);

        // Write bit width in the first byte
        buffer.write((self.bit_width() as u8).as_bytes())?;
        let mut encoder = RleEncoder::new_from_buf(self.bit_width(), buffer, 1);
        for index in self.buffered_indices.data() {
            if !encoder.put(*index as u64)? {
                return Err(general_err!("Encoder doesn't have enough space"));
            }
        }
        self.buffered_indices.clear();
        Ok(ByteBufferPtr::new(encoder.consume()?))
    }

    #[inline]
    fn put_one(&mut self, value: &T::T) -> Result<()> {
        let mut j = (hash_util::hash(value, 0) & self.mod_bitmask) as usize;
        let mut index = self.hash_slots[j];

        while index != HASH_SLOT_EMPTY && self.uniques[index as usize] != *value {
            j += 1;
            if j == self.hash_table_size {
                j = 0;
            }
            index = self.hash_slots[j];
        }

        if index == HASH_SLOT_EMPTY {
            index = self.uniques.size() as i32;
            self.hash_slots[j] = index;
            self.add_dict_key(value.clone());

            if self.uniques.size()
                > (self.hash_table_size as f32 * MAX_HASH_LOAD) as usize
            {
                self.double_table_size();
            }
        }

        self.buffered_indices.push(index);
        Ok(())
    }

    #[inline]
    fn add_dict_key(&mut self, value: T::T) {
        self.uniques_size_in_bytes += self.get_encoded_size(&value);
        self.uniques.push(value);
    }

    #[inline]
    fn bit_width(&self) -> u8 {
        let num_entries = self.uniques.size();
        if num_entries == 0 {
            0
        } else if num_entries == 1 {
            1
        } else {
            log2(num_entries as u64) as u8
        }
    }

    #[inline]
    fn double_table_size(&mut self) {
        let new_size = self.hash_table_size * 2;
        let mut new_hash_slots = Buffer::new().with_mem_tracker(self.mem_tracker.clone());
        new_hash_slots.resize(new_size, HASH_SLOT_EMPTY);
        for i in 0..self.hash_table_size {
            let index = self.hash_slots[i];
            if index == HASH_SLOT_EMPTY {
                continue;
            }
            let value = &self.uniques[index as usize];
            let mut j = (hash_util::hash(value, 0) & ((new_size - 1) as u32)) as usize;
            let mut slot = new_hash_slots[j];
            while slot != HASH_SLOT_EMPTY && self.uniques[slot as usize] != *value {
                j += 1;
                if j == new_size {
                    j = 0;
                }
                slot = new_hash_slots[j];
            }

            new_hash_slots[j] = index;
        }

        self.hash_table_size = new_size;
        self.mod_bitmask = (new_size - 1) as u32;
        mem::replace(&mut self.hash_slots, new_hash_slots);
    }
}

impl<T: DataType> Encoder<T> for DictEncoder<T> {
    #[inline]
    fn put(&mut self, values: &[T::T]) -> Result<()> {
        for i in values {
            self.put_one(&i)?
        }
        Ok(())
    }

    #[inline]
    fn encoding(&self) -> Encoding {
        Encoding::PLAIN_DICTIONARY
    }

    #[inline]
    fn estimated_data_encoded_size(&self) -> usize {
        let bit_width = self.bit_width();
        1 + RleEncoder::min_buffer_size(bit_width)
            + RleEncoder::max_buffer_size(bit_width, self.buffered_indices.size())
    }

    #[inline]
    fn flush_buffer(&mut self) -> Result<ByteBufferPtr> {
        self.write_indices()
    }
}

/// Provides encoded size for a data type.
/// This is a workaround to calculate dictionary size in bytes.
trait DictEncodedSize<T: DataType> {
    fn get_encoded_size(&self, value: &T::T) -> usize;
}

impl<T: DataType> DictEncodedSize<T> for DictEncoder<T> {
    #[inline]
    default fn get_encoded_size(&self, _: &T::T) -> usize {
        mem::size_of::<T::T>()
    }
}

impl DictEncodedSize<ByteArrayType> for DictEncoder<ByteArrayType> {
    #[inline]
    fn get_encoded_size(&self, value: &ByteArray) -> usize {
        mem::size_of::<u32>() + value.len()
    }
}

impl DictEncodedSize<FixedLenByteArrayType> for DictEncoder<FixedLenByteArrayType> {
    #[inline]
    fn get_encoded_size(&self, _value: &ByteArray) -> usize {
        self.desc.type_length() as usize
    }
}

// ----------------------------------------------------------------------
// RLE encoding

const DEFAULT_RLE_BUFFER_LEN: usize = 1024;

/// RLE/Bit-Packing hybrid encoding for values.
/// Currently is used only for data pages v2 and supports boolean types.
pub struct RleValueEncoder<T: DataType> {
    // Buffer with raw values that we collect,
    // when flushing buffer they are encoded using RLE encoder
    encoder: Option<RleEncoder>,
    _phantom: PhantomData<T>,
}

impl<T: DataType> RleValueEncoder<T> {
    /// Creates new rle value encoder.
    pub fn new() -> Self {
        Self {
            encoder: None,
            _phantom: PhantomData,
        }
    }
}

impl<T: DataType> Encoder<T> for RleValueEncoder<T> {
    #[inline]
    default fn put(&mut self, _values: &[T::T]) -> Result<()> {
        panic!("RleValueEncoder only supports BoolType");
    }

    fn encoding(&self) -> Encoding {
        Encoding::RLE
    }

    #[inline]
    default fn estimated_data_encoded_size(&self) -> usize {
        match self.encoder {
            Some(ref enc) => enc.len(),
            None => 0,
        }
    }

    #[inline]
    default fn flush_buffer(&mut self) -> Result<ByteBufferPtr> {
        panic!("RleValueEncoder only supports BoolType");
    }
}

impl Encoder<BoolType> for RleValueEncoder<BoolType> {
    #[inline]
    default fn put(&mut self, values: &[bool]) -> Result<()> {
        if self.encoder.is_none() {
            self.encoder = Some(RleEncoder::new(1, DEFAULT_RLE_BUFFER_LEN));
        }
        let rle_encoder = self.encoder.as_mut().unwrap();
        for value in values {
            if !rle_encoder.put(*value as u64)? {
                return Err(general_err!("RLE buffer is full"));
            }
        }
        Ok(())
    }

    #[inline]
    fn flush_buffer(&mut self) -> Result<ByteBufferPtr> {
        assert!(
            self.encoder.is_some(),
            "RLE value encoder is not initialized"
        );
        let rle_encoder = self.encoder.as_mut().unwrap();

        // Flush all encoder buffers and raw values
        let encoded_data = {
            let buf = rle_encoder.flush_buffer()?;

            // Note that buf does not have any offset, all data is encoded bytes
            let len = (buf.len() as i32).to_le();
            let len_bytes = len.as_bytes();
            let mut encoded_data = Vec::new();
            encoded_data.extend_from_slice(len_bytes);
            encoded_data.extend_from_slice(buf);
            encoded_data
        };
        // Reset rle encoder for the next batch
        rle_encoder.clear();

        Ok(ByteBufferPtr::new(encoded_data))
    }
}

// ----------------------------------------------------------------------
// DELTA_BINARY_PACKED encoding

const MAX_PAGE_HEADER_WRITER_SIZE: usize = 32;
const MAX_BIT_WRITER_SIZE: usize = 10 * 1024 * 1024;
const DEFAULT_BLOCK_SIZE: usize = 128;
const DEFAULT_NUM_MINI_BLOCKS: usize = 4;

/// Delta bit packed encoder.
/// Consists of a header followed by blocks of delta encoded values binary packed.
///
/// Delta-binary-packing:
/// ```shell
///   [page-header] [block 1], [block 2], ... [block N]
/// ```
///
/// Each page header consists of:
/// ```shell
///   [block size] [number of miniblocks in a block] [total value count] [first value]
/// ```
///
/// Each block consists of:
/// ```shell
///   [min delta] [list of bitwidths of miniblocks] [miniblocks]
/// ```
///
/// Current implementation writes values in `put` method, multiple calls to `put` to
/// existing block or start new block if block size is exceeded. Calling `flush_buffer`
/// writes out all data and resets internal state, including page header.
///
/// Supports only INT32 and INT64.
pub struct DeltaBitPackEncoder<T: DataType> {
    page_header_writer: BitWriter,
    bit_writer: BitWriter,
    total_values: usize,
    first_value: i64,
    current_value: i64,
    block_size: usize,
    mini_block_size: usize,
    num_mini_blocks: usize,
    values_in_block: usize,
    deltas: Vec<i64>,
    _phantom: PhantomData<T>,
}

impl<T: DataType> DeltaBitPackEncoder<T> {
    /// Creates new delta bit packed encoder.
    pub fn new() -> Self {
        let block_size = DEFAULT_BLOCK_SIZE;
        let num_mini_blocks = DEFAULT_NUM_MINI_BLOCKS;
        let mini_block_size = block_size / num_mini_blocks;
        assert!(mini_block_size % 8 == 0);
        Self::assert_supported_type();

        DeltaBitPackEncoder {
            page_header_writer: BitWriter::new(MAX_PAGE_HEADER_WRITER_SIZE),
            bit_writer: BitWriter::new(MAX_BIT_WRITER_SIZE),
            total_values: 0,
            first_value: 0,
            current_value: 0, // current value to keep adding deltas
            block_size,       // can write fewer values than block size for last block
            mini_block_size,
            num_mini_blocks,
            values_in_block: 0, // will be at most block_size
            deltas: vec![0; block_size],
            _phantom: PhantomData,
        }
    }

    /// Writes page header for blocks, this method is invoked when we are done encoding
    /// values. It is also okay to encode when no values have been provided
    fn write_page_header(&mut self) {
        // We ignore the result of each 'put' operation, because
        // MAX_PAGE_HEADER_WRITER_SIZE is chosen to fit all header values and
        // guarantees that writes will not fail.

        // Write the size of each block
        self.page_header_writer.put_vlq_int(self.block_size as u64);
        // Write the number of mini blocks
        self.page_header_writer
            .put_vlq_int(self.num_mini_blocks as u64);
        // Write the number of all values (including non-encoded first value)
        self.page_header_writer
            .put_vlq_int(self.total_values as u64);
        // Write first value
        self.page_header_writer.put_zigzag_vlq_int(self.first_value);
    }

    // Write current delta buffer (<= 'block size' values) into bit writer
    fn flush_block_values(&mut self) -> Result<()> {
        if self.values_in_block == 0 {
            return Ok(());
        }

        let mut min_delta = i64::max_value();
        for i in 0..self.values_in_block {
            min_delta = cmp::min(min_delta, self.deltas[i]);
        }

        // Write min delta
        self.bit_writer.put_zigzag_vlq_int(min_delta);

        // Slice to store bit width for each mini block
        let offset = self.bit_writer.skip(self.num_mini_blocks)?;

        for i in 0..self.num_mini_blocks {
            // Find how many values we need to encode - either block size or whatever
            // values left
            let n = cmp::min(self.mini_block_size, self.values_in_block);
            if n == 0 {
                break;
            }

            // Compute the max delta in current mini block
            let mut max_delta = i64::min_value();
            for j in 0..n {
                max_delta =
                    cmp::max(max_delta, self.deltas[i * self.mini_block_size + j]);
            }

            // Compute bit width to store (max_delta - min_delta)
            let bit_width = num_required_bits(self.subtract_u64(max_delta, min_delta));
            self.bit_writer.write_at(offset + i, bit_width as u8);

            // Encode values in current mini block using min_delta and bit_width
            for j in 0..n {
                let packed_value = self
                    .subtract_u64(self.deltas[i * self.mini_block_size + j], min_delta);
                self.bit_writer.put_value(packed_value, bit_width);
            }

            // Pad the last block (n < mini_block_size)
            for _ in n..self.mini_block_size {
                self.bit_writer.put_value(0, bit_width);
            }

            self.values_in_block -= n;
        }

        assert!(
            self.values_in_block == 0,
            "Expected 0 values in block, found {}",
            self.values_in_block
        );
        Ok(())
    }
}

// Implementation is shared between Int32Type and Int64Type,
// see `DeltaBitPackEncoderConversion` below for specifics.
impl<T: DataType> Encoder<T> for DeltaBitPackEncoder<T> {
    fn put(&mut self, values: &[T::T]) -> Result<()> {
        if values.is_empty() {
            return Ok(());
        }

        let mut idx;
        // Define values to encode, initialize state
        if self.total_values == 0 {
            self.first_value = self.as_i64(values, 0);
            self.current_value = self.first_value;
            idx = 1;
        } else {
            idx = 0;
        }
        // Add all values (including first value)
        self.total_values += values.len();

        // Write block
        while idx < values.len() {
            let value = self.as_i64(values, idx);
            self.deltas[self.values_in_block] = self.subtract(value, self.current_value);
            self.current_value = value;
            idx += 1;
            self.values_in_block += 1;
            if self.values_in_block == self.block_size {
                self.flush_block_values()?;
            }
        }
        Ok(())
    }

    fn encoding(&self) -> Encoding {
        Encoding::DELTA_BINARY_PACKED
    }

    fn estimated_data_encoded_size(&self) -> usize {
        self.bit_writer.bytes_written()
    }

    fn flush_buffer(&mut self) -> Result<ByteBufferPtr> {
        // Write remaining values
        self.flush_block_values()?;
        // Write page header with total values
        self.write_page_header();

        let mut buffer = ByteBuffer::new();
        buffer.write(self.page_header_writer.flush_buffer())?;
        buffer.write(self.bit_writer.flush_buffer())?;
        buffer.flush()?;

        // Reset state
        self.page_header_writer.clear();
        self.bit_writer.clear();
        self.total_values = 0;
        self.first_value = 0;
        self.current_value = 0;
        self.values_in_block = 0;

        Ok(buffer.consume())
    }
}

/// Helper trait to define specific conversions and subtractions when computing deltas
trait DeltaBitPackEncoderConversion<T: DataType> {
    // Method should panic if type is not supported, otherwise no-op
    fn assert_supported_type();

    fn as_i64(&self, values: &[T::T], index: usize) -> i64;

    fn subtract(&self, left: i64, right: i64) -> i64;

    fn subtract_u64(&self, left: i64, right: i64) -> u64;
}

impl<T: DataType> DeltaBitPackEncoderConversion<T> for DeltaBitPackEncoder<T> {
    #[inline]
    default fn assert_supported_type() {
        panic!("DeltaBitPackDecoder only supports Int32Type and Int64Type");
    }

    #[inline]
    default fn as_i64(&self, _values: &[T::T], _index: usize) -> i64 {
        0
    }

    #[inline]
    default fn subtract(&self, _left: i64, _right: i64) -> i64 {
        0
    }

    #[inline]
    default fn subtract_u64(&self, _left: i64, _right: i64) -> u64 {
        0
    }
}

impl DeltaBitPackEncoderConversion<Int32Type> for DeltaBitPackEncoder<Int32Type> {
    #[inline]
    fn assert_supported_type() {
        // no-op: supported type
    }

    #[inline]
    fn as_i64(&self, values: &[i32], index: usize) -> i64 {
        values[index] as i64
    }

    #[inline]
    fn subtract(&self, left: i64, right: i64) -> i64 {
        // It is okay for values to overflow, wrapping_sub wrapping around at the boundary
        (left as i32).wrapping_sub(right as i32) as i64
    }

    #[inline]
    fn subtract_u64(&self, left: i64, right: i64) -> u64 {
        // Conversion of i32 -> u32 -> u64 is to avoid non-zero left most bytes in int
        // representation
        (left as i32).wrapping_sub(right as i32) as u32 as u64
    }
}

impl DeltaBitPackEncoderConversion<Int64Type> for DeltaBitPackEncoder<Int64Type> {
    #[inline]
    fn assert_supported_type() {
        // no-op: supported type
    }

    #[inline]
    fn as_i64(&self, values: &[i64], index: usize) -> i64 {
        values[index]
    }

    #[inline]
    fn subtract(&self, left: i64, right: i64) -> i64 {
        // It is okay for values to overflow, wrapping_sub wrapping around at the boundary
        left.wrapping_sub(right)
    }

    #[inline]
    fn subtract_u64(&self, left: i64, right: i64) -> u64 {
        left.wrapping_sub(right) as u64
    }
}

// ----------------------------------------------------------------------
// DELTA_LENGTH_BYTE_ARRAY encoding

/// Encoding for byte arrays to separate the length values and the data.
/// The lengths are encoded using DELTA_BINARY_PACKED encoding, data is
/// stored as raw bytes.
pub struct DeltaLengthByteArrayEncoder<T: DataType> {
    // length encoder
    len_encoder: DeltaBitPackEncoder<Int32Type>,
    // byte array data
    data: Vec<ByteArray>,
    // data size in bytes of encoded values
    encoded_size: usize,
    _phantom: PhantomData<T>,
}

impl<T: DataType> DeltaLengthByteArrayEncoder<T> {
    /// Creates new delta length byte array encoder.
    pub fn new() -> Self {
        Self {
            len_encoder: DeltaBitPackEncoder::new(),
            data: vec![],
            encoded_size: 0,
            _phantom: PhantomData,
        }
    }
}

impl<T: DataType> Encoder<T> for DeltaLengthByteArrayEncoder<T> {
    default fn put(&mut self, _values: &[T::T]) -> Result<()> {
        panic!("DeltaLengthByteArrayEncoder only supports ByteArrayType");
    }

    fn encoding(&self) -> Encoding {
        Encoding::DELTA_LENGTH_BYTE_ARRAY
    }

    fn estimated_data_encoded_size(&self) -> usize {
        self.len_encoder.estimated_data_encoded_size() + self.encoded_size
    }

    default fn flush_buffer(&mut self) -> Result<ByteBufferPtr> {
        panic!("DeltaLengthByteArrayEncoder only supports ByteArrayType");
    }
}

impl Encoder<ByteArrayType> for DeltaLengthByteArrayEncoder<ByteArrayType> {
    fn put(&mut self, values: &[ByteArray]) -> Result<()> {
        let lengths: Vec<i32> = values
            .iter()
            .map(|byte_array| byte_array.len() as i32)
            .collect();
        self.len_encoder.put(&lengths)?;
        for byte_array in values {
            self.encoded_size += byte_array.len();
            self.data.push(byte_array.clone());
        }
        Ok(())
    }

    fn flush_buffer(&mut self) -> Result<ByteBufferPtr> {
        let mut total_bytes = vec![];
        let lengths = self.len_encoder.flush_buffer()?;
        total_bytes.extend_from_slice(lengths.data());
        self.data.iter().for_each(|byte_array| {
            total_bytes.extend_from_slice(byte_array.data());
        });
        self.data.clear();
        self.encoded_size = 0;
        Ok(ByteBufferPtr::new(total_bytes))
    }
}

// ----------------------------------------------------------------------
// DELTA_BYTE_ARRAY encoding

/// Encoding for byte arrays, prefix lengths are encoded using DELTA_BINARY_PACKED
/// encoding, followed by suffixes with DELTA_LENGTH_BYTE_ARRAY encoding.
pub struct DeltaByteArrayEncoder<T: DataType> {
    prefix_len_encoder: DeltaBitPackEncoder<Int32Type>,
    suffix_writer: DeltaLengthByteArrayEncoder<T>,
    previous: Vec<u8>,
    _phantom: PhantomData<T>,
}

impl<T: DataType> DeltaByteArrayEncoder<T> {
    /// Creates new delta byte array encoder.
    pub fn new() -> Self {
        Self {
            prefix_len_encoder: DeltaBitPackEncoder::<Int32Type>::new(),
            suffix_writer: DeltaLengthByteArrayEncoder::<T>::new(),
            previous: vec![],
            _phantom: PhantomData,
        }
    }
}

impl<T: DataType> Encoder<T> for DeltaByteArrayEncoder<T> {
    default fn put(&mut self, _values: &[T::T]) -> Result<()> {
        panic!(
            "DeltaByteArrayEncoder only supports ByteArrayType and FixedLenByteArrayType"
        );
    }

    fn encoding(&self) -> Encoding {
        Encoding::DELTA_BYTE_ARRAY
    }

    fn estimated_data_encoded_size(&self) -> usize {
        self.prefix_len_encoder.estimated_data_encoded_size()
            + self.suffix_writer.estimated_data_encoded_size()
    }

    default fn flush_buffer(&mut self) -> Result<ByteBufferPtr> {
        panic!(
            "DeltaByteArrayEncoder only supports ByteArrayType and FixedLenByteArrayType"
        );
    }
}

impl Encoder<ByteArrayType> for DeltaByteArrayEncoder<ByteArrayType> {
    fn put(&mut self, values: &[ByteArray]) -> Result<()> {
        let mut prefix_lengths: Vec<i32> = vec![];
        let mut suffixes: Vec<ByteArray> = vec![];

        for byte_array in values {
            let current = byte_array.data();
            // Maximum prefix length that is shared between previous value and current
            // value
            let prefix_len = cmp::min(self.previous.len(), current.len());
            let mut match_len = 0;
            while match_len < prefix_len && self.previous[match_len] == current[match_len]
            {
                match_len += 1;
            }
            prefix_lengths.push(match_len as i32);
            suffixes.push(byte_array.slice(match_len, byte_array.len() - match_len));
            // Update previous for the next prefix
            self.previous.clear();
            self.previous.extend_from_slice(current);
        }
        self.prefix_len_encoder.put(&prefix_lengths)?;
        self.suffix_writer.put(&suffixes)?;
        Ok(())
    }

    fn flush_buffer(&mut self) -> Result<ByteBufferPtr> {
        // TODO: investigate if we can merge lengths and suffixes
        // without copying data into new vector.
        let mut total_bytes = vec![];
        // Insert lengths ...
        let lengths = self.prefix_len_encoder.flush_buffer()?;
        total_bytes.extend_from_slice(lengths.data());
        // ... followed by suffixes
        let suffixes = self.suffix_writer.flush_buffer()?;
        total_bytes.extend_from_slice(suffixes.data());

        self.previous.clear();
        Ok(ByteBufferPtr::new(total_bytes))
    }
}

impl Encoder<FixedLenByteArrayType> for DeltaByteArrayEncoder<FixedLenByteArrayType> {
    fn put(&mut self, values: &[ByteArray]) -> Result<()> {
        let s: &mut DeltaByteArrayEncoder<ByteArrayType> =
            unsafe { mem::transmute(self) };
        s.put(values)
    }

    fn flush_buffer(&mut self) -> Result<ByteBufferPtr> {
        let s: &mut DeltaByteArrayEncoder<ByteArrayType> =
            unsafe { mem::transmute(self) };
        s.flush_buffer()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::rc::Rc;

    use crate::decoding::{get_decoder, Decoder, DictDecoder, PlainDecoder};
    use crate::schema::types::{
        ColumnDescPtr, ColumnDescriptor, ColumnPath, Type as SchemaType,
    };
    use crate::util::{
        memory::MemTracker,
        test_common::{random_bytes, RandGen},
    };

    const TEST_SET_SIZE: usize = 1024;

    #[test]
    fn test_get_encoders() {
        // supported encodings
        create_and_check_encoder::<Int32Type>(Encoding::PLAIN, None);
        create_and_check_encoder::<Int32Type>(Encoding::DELTA_BINARY_PACKED, None);
        create_and_check_encoder::<Int32Type>(Encoding::DELTA_LENGTH_BYTE_ARRAY, None);
        create_and_check_encoder::<Int32Type>(Encoding::DELTA_BYTE_ARRAY, None);
        create_and_check_encoder::<BoolType>(Encoding::RLE, None);

        // error when initializing
        create_and_check_encoder::<Int32Type>(
            Encoding::RLE_DICTIONARY,
            Some(general_err!(
                "Cannot initialize this encoding through this function"
            )),
        );
        create_and_check_encoder::<Int32Type>(
            Encoding::PLAIN_DICTIONARY,
            Some(general_err!(
                "Cannot initialize this encoding through this function"
            )),
        );

        // unsupported
        create_and_check_encoder::<Int32Type>(
            Encoding::BIT_PACKED,
            Some(nyi_err!("Encoding BIT_PACKED is not supported")),
        );
    }

    #[test]
    fn test_bool() {
        BoolType::test(Encoding::PLAIN, TEST_SET_SIZE, -1);
        BoolType::test(Encoding::PLAIN_DICTIONARY, TEST_SET_SIZE, -1);
        BoolType::test(Encoding::RLE, TEST_SET_SIZE, -1);
    }

    #[test]
    fn test_i32() {
        Int32Type::test(Encoding::PLAIN, TEST_SET_SIZE, -1);
        Int32Type::test(Encoding::PLAIN_DICTIONARY, TEST_SET_SIZE, -1);
        Int32Type::test(Encoding::DELTA_BINARY_PACKED, TEST_SET_SIZE, -1);
    }

    #[test]
    fn test_i64() {
        Int64Type::test(Encoding::PLAIN, TEST_SET_SIZE, -1);
        Int64Type::test(Encoding::PLAIN_DICTIONARY, TEST_SET_SIZE, -1);
        Int64Type::test(Encoding::DELTA_BINARY_PACKED, TEST_SET_SIZE, -1);
    }

    #[test]
    fn test_i96() {
        Int96Type::test(Encoding::PLAIN, TEST_SET_SIZE, -1);
        Int96Type::test(Encoding::PLAIN_DICTIONARY, TEST_SET_SIZE, -1);
    }

    #[test]
    fn test_float() {
        FloatType::test(Encoding::PLAIN, TEST_SET_SIZE, -1);
        FloatType::test(Encoding::PLAIN_DICTIONARY, TEST_SET_SIZE, -1);
    }

    #[test]
    fn test_double() {
        DoubleType::test(Encoding::PLAIN, TEST_SET_SIZE, -1);
        DoubleType::test(Encoding::PLAIN_DICTIONARY, TEST_SET_SIZE, -1);
    }

    #[test]
    fn test_byte_array() {
        ByteArrayType::test(Encoding::PLAIN, TEST_SET_SIZE, -1);
        ByteArrayType::test(Encoding::PLAIN_DICTIONARY, TEST_SET_SIZE, -1);
        ByteArrayType::test(Encoding::DELTA_LENGTH_BYTE_ARRAY, TEST_SET_SIZE, -1);
        ByteArrayType::test(Encoding::DELTA_BYTE_ARRAY, TEST_SET_SIZE, -1);
    }

    #[test]
    fn test_fixed_lenbyte_array() {
        FixedLenByteArrayType::test(Encoding::PLAIN, TEST_SET_SIZE, 100);
        FixedLenByteArrayType::test(Encoding::PLAIN_DICTIONARY, TEST_SET_SIZE, 100);
        FixedLenByteArrayType::test(Encoding::DELTA_BYTE_ARRAY, TEST_SET_SIZE, 100);
    }

    #[test]
    fn test_dict_encoded_size() {
        fn run_test<T: DataType>(
            type_length: i32,
            values: &[T::T],
            expected_size: usize,
        ) {
            let mut encoder = create_test_dict_encoder::<T>(type_length);
            assert_eq!(encoder.dict_encoded_size(), 0);
            encoder.put(values).unwrap();
            assert_eq!(encoder.dict_encoded_size(), expected_size);
            // We do not reset encoded size of the dictionary keys after flush_buffer
            encoder.flush_buffer().unwrap();
            assert_eq!(encoder.dict_encoded_size(), expected_size);
        }

        // Only 2 variations of values 1 byte each
        run_test::<BoolType>(-1, &[true, false, true, false, true], 2);
        run_test::<Int32Type>(-1, &[1i32, 2i32, 3i32, 4i32, 5i32], 20);
        run_test::<Int64Type>(-1, &[1i64, 2i64, 3i64, 4i64, 5i64], 40);
        run_test::<FloatType>(-1, &[1f32, 2f32, 3f32, 4f32, 5f32], 20);
        run_test::<DoubleType>(-1, &[1f64, 2f64, 3f64, 4f64, 5f64], 40);
        // Int96: len + reference
        run_test::<Int96Type>(
            -1,
            &[Int96::from(vec![1, 2, 3]), Int96::from(vec![2, 3, 4])],
            32,
        );
        run_test::<ByteArrayType>(
            -1,
            &[ByteArray::from("abcd"), ByteArray::from("efj")],
            15,
        );
        run_test::<FixedLenByteArrayType>(
            2,
            &[ByteArray::from("ab"), ByteArray::from("bc")],
            4,
        );
    }

    #[test]
    fn test_estimated_data_encoded_size() {
        fn run_test<T: DataType>(
            encoding: Encoding,
            type_length: i32,
            values: &[T::T],
            initial_size: usize,
            max_size: usize,
            flush_size: usize,
        ) {
            let mut encoder = match encoding {
                Encoding::PLAIN_DICTIONARY | Encoding::RLE_DICTIONARY => {
                    Box::new(create_test_dict_encoder::<T>(type_length))
                }
                _ => create_test_encoder::<T>(type_length, encoding),
            };
            assert_eq!(encoder.estimated_data_encoded_size(), initial_size);

            encoder.put(values).unwrap();
            assert_eq!(encoder.estimated_data_encoded_size(), max_size);

            encoder.flush_buffer().unwrap();
            assert_eq!(encoder.estimated_data_encoded_size(), flush_size);
        }

        // PLAIN
        run_test::<Int32Type>(Encoding::PLAIN, -1, &vec![123; 1024], 0, 4096, 0);

        // DICTIONARY
        // NOTE: The final size is almost the same because the dictionary entries are
        // preserved after encoded values have been written.
        run_test::<Int32Type>(Encoding::RLE_DICTIONARY, -1, &vec![123, 1024], 11, 68, 66);

        // DELTA_BINARY_PACKED
        run_test::<Int32Type>(
            Encoding::DELTA_BINARY_PACKED,
            -1,
            &vec![123; 1024],
            0,
            35,
            0,
        );

        // RLE
        let mut values = vec![];
        values.extend_from_slice(&vec![true; 16]);
        values.extend_from_slice(&vec![false; 16]);
        run_test::<BoolType>(Encoding::RLE, -1, &values, 0, 2, 0);

        // DELTA_LENGTH_BYTE_ARRAY
        run_test::<ByteArrayType>(
            Encoding::DELTA_LENGTH_BYTE_ARRAY,
            -1,
            &[ByteArray::from("ab"), ByteArray::from("abc")],
            0,
            5, // only value bytes, length encoder is not flushed yet
            0,
        );

        // DELTA_BYTE_ARRAY
        run_test::<ByteArrayType>(
            Encoding::DELTA_BYTE_ARRAY,
            -1,
            &[ByteArray::from("ab"), ByteArray::from("abc")],
            0,
            3, // only suffix bytes, length encoder is not flushed yet
            0,
        );
    }

    // See: https://github.com/sunchao/parquet-rs/issues/47
    #[test]
    fn test_issue_47() {
        let mut encoder =
            create_test_encoder::<ByteArrayType>(0, Encoding::DELTA_BYTE_ARRAY);
        let mut decoder =
            create_test_decoder::<ByteArrayType>(0, Encoding::DELTA_BYTE_ARRAY);

        let mut input = vec![];
        input.push(ByteArray::from("aa"));
        input.push(ByteArray::from("aaa"));
        input.push(ByteArray::from("aa"));
        input.push(ByteArray::from("aaa"));
        let mut output = vec![ByteArray::default(); input.len()];

        let mut result =
            put_and_get(&mut encoder, &mut decoder, &input[..2], &mut output[..2]);
        assert!(
            result.is_ok(),
            "first put_and_get() failed with: {}",
            result.unwrap_err()
        );
        result = put_and_get(&mut encoder, &mut decoder, &input[2..], &mut output[2..]);
        assert!(
            result.is_ok(),
            "second put_and_get() failed with: {}",
            result.unwrap_err()
        );
        assert_eq!(output, input);
    }

    trait EncodingTester<T: DataType> {
        fn test(enc: Encoding, total: usize, type_length: i32) {
            let result = match enc {
                Encoding::PLAIN_DICTIONARY | Encoding::RLE_DICTIONARY => {
                    Self::test_dict_internal(total, type_length)
                }
                enc @ _ => Self::test_internal(enc, total, type_length),
            };

            assert!(
                result.is_ok(),
                "Expected result to be OK but got err:\n {}",
                result.unwrap_err()
            );
        }

        fn test_internal(enc: Encoding, total: usize, type_length: i32) -> Result<()>;

        fn test_dict_internal(total: usize, type_length: i32) -> Result<()>;
    }

    impl<T: DataType> EncodingTester<T> for T {
        fn test_internal(enc: Encoding, total: usize, type_length: i32) -> Result<()> {
            let mut encoder = create_test_encoder::<T>(type_length, enc);
            let mut decoder = create_test_decoder::<T>(type_length, enc);
            let mut values = <T as RandGen<T>>::gen_vec(type_length, total);
            let mut result_data = vec![T::T::default(); total];

            // Test put/get spaced.
            let num_bytes = bit_util::ceil(total as i64, 8);
            let valid_bits = random_bytes(num_bytes as usize);
            let values_written = encoder.put_spaced(&values[..], &valid_bits[..])?;
            let data = encoder.flush_buffer()?;
            decoder.set_data(data, values_written)?;
            let _ = decoder.get_spaced(
                &mut result_data[..],
                values.len() - values_written,
                &valid_bits[..],
            )?;

            // Check equality
            for i in 0..total {
                if bit_util::get_bit(&valid_bits[..], i) {
                    assert_eq!(result_data[i], values[i]);
                } else {
                    assert_eq!(result_data[i], T::T::default());
                }
            }

            let mut actual_total = put_and_get(
                &mut encoder,
                &mut decoder,
                &values[..],
                &mut result_data[..],
            )?;
            assert_eq!(actual_total, total);
            assert_eq!(result_data, values);

            // Encode more data after flush and test with decoder

            values = <T as RandGen<T>>::gen_vec(type_length, total);
            actual_total = put_and_get(
                &mut encoder,
                &mut decoder,
                &values[..],
                &mut result_data[..],
            )?;
            assert_eq!(actual_total, total);
            assert_eq!(result_data, values);

            Ok(())
        }

        fn test_dict_internal(total: usize, type_length: i32) -> Result<()> {
            let mut encoder = create_test_dict_encoder::<T>(type_length);
            let mut values = <T as RandGen<T>>::gen_vec(type_length, total);
            encoder.put(&values[..])?;

            let mut data = encoder.flush_buffer()?;
            let mut decoder = create_test_dict_decoder::<T>();
            let mut dict_decoder = PlainDecoder::<T>::new(type_length);
            dict_decoder.set_data(encoder.write_dict()?, encoder.num_entries())?;
            decoder.set_dict(Box::new(dict_decoder))?;
            let mut result_data = vec![T::T::default(); total];
            decoder.set_data(data, total)?;
            let mut actual_total = decoder.get(&mut result_data)?;

            assert_eq!(actual_total, total);
            assert_eq!(result_data, values);

            // Encode more data after flush and test with decoder

            values = <T as RandGen<T>>::gen_vec(type_length, total);
            encoder.put(&values[..])?;
            data = encoder.flush_buffer()?;

            let mut dict_decoder = PlainDecoder::<T>::new(type_length);
            dict_decoder.set_data(encoder.write_dict()?, encoder.num_entries())?;
            decoder.set_dict(Box::new(dict_decoder))?;
            decoder.set_data(data, total)?;
            actual_total = decoder.get(&mut result_data)?;

            assert_eq!(actual_total, total);
            assert_eq!(result_data, values);

            Ok(())
        }
    }

    fn put_and_get<T: DataType>(
        encoder: &mut Box<Encoder<T>>,
        decoder: &mut Box<Decoder<T>>,
        input: &[T::T],
        output: &mut [T::T],
    ) -> Result<usize> {
        encoder.put(input)?;
        let data = encoder.flush_buffer()?;
        decoder.set_data(data, input.len())?;
        decoder.get(output)
    }

    fn create_and_check_encoder<T: DataType>(
        encoding: Encoding,
        err: Option<ParquetError>,
    ) {
        let descr = create_test_col_desc_ptr(-1, T::get_physical_type());
        let mem_tracker = Rc::new(MemTracker::new());
        let encoder = get_encoder::<T>(descr, encoding, mem_tracker);
        match err {
            Some(parquet_error) => {
                assert!(encoder.is_err());
                assert_eq!(encoder.err().unwrap(), parquet_error);
            }
            None => {
                assert!(encoder.is_ok());
                assert_eq!(encoder.unwrap().encoding(), encoding);
            }
        }
    }

    // Creates test column descriptor.
    fn create_test_col_desc_ptr(type_len: i32, t: Type) -> ColumnDescPtr {
        let ty = SchemaType::primitive_type_builder("t", t)
            .with_length(type_len)
            .build()
            .unwrap();
        Rc::new(ColumnDescriptor::new(
            Rc::new(ty),
            None,
            0,
            0,
            ColumnPath::new(vec![]),
        ))
    }

    fn create_test_encoder<T: DataType>(type_len: i32, enc: Encoding) -> Box<Encoder<T>> {
        let desc = create_test_col_desc_ptr(type_len, T::get_physical_type());
        let mem_tracker = Rc::new(MemTracker::new());
        get_encoder(desc, enc, mem_tracker).unwrap()
    }

    fn create_test_decoder<T: DataType>(type_len: i32, enc: Encoding) -> Box<Decoder<T>> {
        let desc = create_test_col_desc_ptr(type_len, T::get_physical_type());
        get_decoder(desc, enc).unwrap()
    }

    fn create_test_dict_encoder<T: DataType>(type_len: i32) -> DictEncoder<T> {
        let desc = create_test_col_desc_ptr(type_len, T::get_physical_type());
        let mem_tracker = Rc::new(MemTracker::new());
        DictEncoder::<T>::new(desc, mem_tracker)
    }

    fn create_test_dict_decoder<T: DataType>() -> DictDecoder<T> {
        DictDecoder::<T>::new()
    }
}
