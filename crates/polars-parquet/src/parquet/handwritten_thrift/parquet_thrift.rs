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

// Verbatim arrow-rs 57.0 port (see module docs in `mod.rs`). Polars wires only
// the read-side primitives into the file-metadata decoder; the write-side
// primitives + `ThriftReadInputProtocol` + `ReadThrift` macros stay byte-identical
// to arrow-rs for future re-ports but are unused in this build. Allow the
// resulting dead-code / private-interfaces lints at the module boundary so the
// source body itself remains an unmodified copy of arrow-rs.
#![allow(dead_code, private_interfaces)]

//! Structs used for encoding and decoding Parquet Thrift objects.
//!
//! These include:
//! * [`ThriftCompactInputProtocol`]: Trait implemented by Thrift decoders.
//!     * [`ThriftSliceInputProtocol`]: Thrift decoder that takes a slice of bytes as input.
//!     * [`ThriftReadInputProtocol`]: Thrift decoder that takes a [`Read`] as input.
//! * [`ReadThrift`]: Trait implemented by serializable objects.
//! * [`ThriftCompactOutputProtocol`]: Thrift encoder.
//! * [`WriteThrift`]: Trait implemented by serializable objects.
//! * [`WriteThriftField`]: Trait implemented by serializable objects that are fields in Thrift structs.

use std::cmp::Ordering;
// `write_thrift_field!` is defined in the sibling `parquet_macros` module and
// registered at crate root via `#[macro_export]`, so it resolves unqualified
// in this file. No explicit `use` needed.
use std::io::Error;
use std::io::{Read, Write};
use std::str::Utf8Error;

// Polars integration shim. Arrow exposes `crate::errors::{ParquetError, Result}`
// at crate root; polars keeps them under `parquet::error`. Bound under the same
// names so the rest of this file stays byte-identical to arrow-rs 57.0.
use crate::parquet::error::{ParquetError, ParquetResult as Result};

// Polars doesn't carry arrow's `general_err!` / `eof_err!`. File-local shims
// produce the same `ParquetError::OutOfSpec(format!(...))` value arrow's
// versions construct.
macro_rules! general_err {
    ($($arg:tt)*) => { $crate::parquet::error::ParquetError::oos(format!($($arg)*)) };
}
macro_rules! eof_err {
    ($($arg:tt)*) => { $crate::parquet::error::ParquetError::oos(format!($($arg)*)) };
}

#[derive(Debug)]
pub(crate) enum ThriftProtocolError {
    Eof,
    IO(Error),
    InvalidFieldType(u8),
    InvalidElementType(u8),
    FieldDeltaOverflow { field_delta: u8, last_field_id: i16 },
    InvalidBoolean(u8),
    Utf8Error,
    SkipDepth(FieldType),
    SkipUnsupportedType(FieldType),
}

impl From<ThriftProtocolError> for ParquetError {
    #[inline(never)]
    fn from(e: ThriftProtocolError) -> Self {
        match e {
            ThriftProtocolError::Eof => eof_err!("Unexpected EOF"),
            ThriftProtocolError::IO(e) => e.into(),
            ThriftProtocolError::InvalidFieldType(value) => {
                general_err!("Unexpected struct field type {}", value)
            },
            ThriftProtocolError::InvalidElementType(value) => {
                general_err!("Unexpected list/set element type{}", value)
            },
            ThriftProtocolError::FieldDeltaOverflow {
                field_delta,
                last_field_id,
            } => general_err!("cannot add {} to {}", field_delta, last_field_id),
            ThriftProtocolError::InvalidBoolean(value) => {
                general_err!("cannot convert {} into bool", value)
            },
            ThriftProtocolError::Utf8Error => general_err!("invalid utf8"),
            ThriftProtocolError::SkipDepth(field_type) => {
                general_err!("cannot parse past {:?}", field_type)
            },
            ThriftProtocolError::SkipUnsupportedType(field_type) => {
                general_err!("cannot skip field type {:?}", field_type)
            },
        }
    }
}

impl From<Utf8Error> for ThriftProtocolError {
    fn from(_: Utf8Error) -> Self {
        // ignore error payload to reduce the size of ThriftProtocolError
        Self::Utf8Error
    }
}

impl From<Error> for ThriftProtocolError {
    fn from(e: Error) -> Self {
        Self::IO(e)
    }
}

// Arrow writes `Result<T, ThriftProtocolError>` here, relying on its two-arg
// `std::result::Result` alias. Our local `Result = ParquetResult<T>` takes
// one arg, so spell `std::result::Result` explicitly for this one alias.
pub type ThriftProtocolResult<T> = std::result::Result<T, ThriftProtocolError>;

/// Wrapper for thrift `double` fields. This is used to provide
/// an implementation of `Eq` for floats. This implementation
/// uses IEEE 754 total order.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OrderedF64(f64);

impl From<f64> for OrderedF64 {
    fn from(value: f64) -> Self {
        Self(value)
    }
}

impl From<OrderedF64> for f64 {
    fn from(value: OrderedF64) -> Self {
        value.0
    }
}

impl Eq for OrderedF64 {} // Marker trait, requires PartialEq

impl Ord for OrderedF64 {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.total_cmp(&other.0)
    }
}

impl PartialOrd for OrderedF64 {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

// Thrift compact protocol types for struct fields.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum FieldType {
    Stop = 0,
    BooleanTrue = 1,
    BooleanFalse = 2,
    Byte = 3,
    I16 = 4,
    I32 = 5,
    I64 = 6,
    Double = 7,
    Binary = 8,
    List = 9,
    Set = 10,
    Map = 11,
    Struct = 12,
}

impl TryFrom<u8> for FieldType {
    type Error = ThriftProtocolError;
    fn try_from(value: u8) -> ThriftProtocolResult<Self> {
        match value {
            0 => Ok(Self::Stop),
            1 => Ok(Self::BooleanTrue),
            2 => Ok(Self::BooleanFalse),
            3 => Ok(Self::Byte),
            4 => Ok(Self::I16),
            5 => Ok(Self::I32),
            6 => Ok(Self::I64),
            7 => Ok(Self::Double),
            8 => Ok(Self::Binary),
            9 => Ok(Self::List),
            10 => Ok(Self::Set),
            11 => Ok(Self::Map),
            12 => Ok(Self::Struct),
            _ => Err(ThriftProtocolError::InvalidFieldType(value)),
        }
    }
}

impl TryFrom<ElementType> for FieldType {
    type Error = ThriftProtocolError;
    fn try_from(value: ElementType) -> std::result::Result<Self, Self::Error> {
        match value {
            ElementType::Bool => Ok(Self::BooleanTrue),
            ElementType::Byte => Ok(Self::Byte),
            ElementType::I16 => Ok(Self::I16),
            ElementType::I32 => Ok(Self::I32),
            ElementType::I64 => Ok(Self::I64),
            ElementType::Double => Ok(Self::Double),
            ElementType::Binary => Ok(Self::Binary),
            ElementType::List => Ok(Self::List),
            ElementType::Struct => Ok(Self::Struct),
            _ => Err(ThriftProtocolError::InvalidFieldType(value as u8)),
        }
    }
}

// Thrift compact protocol types for list elements
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ElementType {
    Bool = 2,
    Byte = 3,
    I16 = 4,
    I32 = 5,
    I64 = 6,
    Double = 7,
    Binary = 8,
    List = 9,
    Set = 10,
    Map = 11,
    Struct = 12,
}

impl TryFrom<u8> for ElementType {
    type Error = ThriftProtocolError;
    fn try_from(value: u8) -> ThriftProtocolResult<Self> {
        match value {
            // For historical and compatibility reasons, a reader should be capable to deal with both cases.
            // The only valid value in the original spec was 2, but due to an widespread implementation bug
            // the defacto standard across large parts of the library became 1 instead.
            // As a result, both values are now allowed.
            // https://github.com/apache/thrift/blob/master/doc/specs/thrift-compact-protocol.md#list-and-set
            1 | 2 => Ok(Self::Bool),
            3 => Ok(Self::Byte),
            4 => Ok(Self::I16),
            5 => Ok(Self::I32),
            6 => Ok(Self::I64),
            7 => Ok(Self::Double),
            8 => Ok(Self::Binary),
            9 => Ok(Self::List),
            10 => Ok(Self::Set),
            11 => Ok(Self::Map),
            12 => Ok(Self::Struct),
            _ => Err(ThriftProtocolError::InvalidElementType(value)),
        }
    }
}

/// Struct used to describe a [thrift struct] field during decoding.
///
/// [thrift struct]: https://github.com/apache/thrift/blob/master/doc/specs/thrift-compact-protocol.md#struct-encoding
pub(crate) struct FieldIdentifier {
    /// The type for the field.
    pub(crate) field_type: FieldType,
    /// The field's `id`. May be computed from delta or directly decoded.
    pub(crate) id: i16,
    /// Stores the value for booleans.
    ///
    /// Boolean fields store no data, instead the field type is either boolean true, or
    /// boolean false.
    pub(crate) bool_val: Option<bool>,
}

/// Struct used to describe a [thrift list].
///
/// [thrift list]: https://github.com/apache/thrift/blob/master/doc/specs/thrift-compact-protocol.md#list-and-set
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ListIdentifier {
    /// The type for each element in the list.
    pub(crate) element_type: ElementType,
    /// Number of elements contained in the list.
    pub(crate) size: i32,
}

/// Low-level object used to deserialize structs encoded with the Thrift [compact] protocol.
///
/// Implementation of this trait must provide the low-level functions `read_byte`, `read_bytes`,
/// `skip_bytes`, and `read_double`. These primitives are used by the default functions provided
/// here to perform deserialization.
///
/// [compact]: https://github.com/apache/thrift/blob/master/doc/specs/thrift-compact-protocol.md
pub(crate) trait ThriftCompactInputProtocol<'a> {
    /// Read a single byte from the input.
    fn read_byte(&mut self) -> ThriftProtocolResult<u8>;

    /// Read a Thrift encoded [binary] from the input.
    ///
    /// [binary]: https://github.com/apache/thrift/blob/master/doc/specs/thrift-compact-protocol.md#binary-encoding
    fn read_bytes(&mut self) -> ThriftProtocolResult<&'a [u8]>;

    fn read_bytes_owned(&mut self) -> ThriftProtocolResult<Vec<u8>>;

    /// Skip the next `n` bytes of input.
    fn skip_bytes(&mut self, n: usize) -> ThriftProtocolResult<()>;

    /// Read a ULEB128 encoded unsigned varint from the input.
    ///
    /// Fast path for the 1-byte case (~80-90% of Parquet metadata varints).
    /// Multi-byte fallback is in `read_vlq_slow`, marked `#[cold]` so the
    /// common path stays tight.
    #[inline]
    fn read_vlq(&mut self) -> ThriftProtocolResult<u64> {
        let byte = self.read_byte()?;
        if byte & 0x80 == 0 {
            return Ok(byte as u64);
        }
        self.read_vlq_slow(byte)
    }

    /// Cold-path continuation for [`Self::read_vlq`] when the first byte has
    /// its high bit set. `first` is the already-consumed first byte.
    #[cold]
    #[inline(never)]
    fn read_vlq_slow(&mut self, first: u8) -> ThriftProtocolResult<u64> {
        let mut in_progress = (first & 0x7F) as u64;
        let mut shift = 7;
        loop {
            let byte = self.read_byte()?;
            in_progress |= ((byte & 0x7F) as u64).wrapping_shl(shift);
            if byte & 0x80 == 0 {
                return Ok(in_progress);
            }
            shift += 7;
        }
    }

    /// Read a zig-zag encoded signed varint from the input.
    fn read_zig_zag(&mut self) -> ThriftProtocolResult<i64> {
        let val = self.read_vlq()?;
        Ok((val >> 1) as i64 ^ -((val & 1) as i64))
    }

    /// Read the [`ListIdentifier`] for a Thrift encoded list.
    fn read_list_begin(&mut self) -> ThriftProtocolResult<ListIdentifier> {
        let header = self.read_byte()?;
        let element_type = ElementType::try_from(header & 0x0f)?;

        let possible_element_count = (header & 0xF0) >> 4;
        let element_count = if possible_element_count != 15 {
            // high bits set high if count and type encoded separately
            possible_element_count as i32
        } else {
            self.read_vlq()? as _
        };

        Ok(ListIdentifier {
            element_type,
            size: element_count,
        })
    }

    // Full field ids are uncommon.
    // Not inlining this method reduces the code size of `read_field_begin`, which then ideally gets
    // inlined everywhere.
    #[cold]
    fn read_full_field_id(&mut self) -> ThriftProtocolResult<i16> {
        self.read_i16()
    }

    /// Read the [`FieldIdentifier`] for a field in a Thrift encoded struct.
    fn read_field_begin(&mut self, last_field_id: i16) -> ThriftProtocolResult<FieldIdentifier> {
        // we can read at least one byte, which is:
        // - the type
        // - the field delta and the type
        let field_type = self.read_byte()?;
        let field_delta = (field_type & 0xf0) >> 4;
        let field_type = FieldType::try_from(field_type & 0xf)?;
        let mut bool_val: Option<bool> = None;

        match field_type {
            FieldType::Stop => Ok(FieldIdentifier {
                field_type: FieldType::Stop,
                id: 0,
                bool_val,
            }),
            _ => {
                // special handling for bools
                if field_type == FieldType::BooleanFalse {
                    bool_val = Some(false);
                } else if field_type == FieldType::BooleanTrue {
                    bool_val = Some(true);
                }
                let field_id = if field_delta != 0 {
                    last_field_id.checked_add(field_delta as i16).ok_or(
                        ThriftProtocolError::FieldDeltaOverflow {
                            field_delta,
                            last_field_id,
                        },
                    )?
                } else {
                    self.read_full_field_id()?
                };

                Ok(FieldIdentifier {
                    field_type,
                    id: field_id,
                    bool_val,
                })
            },
        }
    }

    /// This is a specialized version of [`Self::read_field_begin`], solely for use in parsing
    /// simple structs. This function assumes that the delta field will always be less than 0xf,
    /// fields will be in order, and no boolean fields will be read.
    /// This also skips validation of the field type.
    ///
    /// Returns a tuple of `(field_type, field_delta)`.
    fn read_field_header(&mut self) -> ThriftProtocolResult<(u8, u8)> {
        let field_type = self.read_byte()?;
        let field_delta = (field_type & 0xf0) >> 4;
        let field_type = field_type & 0xf;
        Ok((field_type, field_delta))
    }

    /// Read a boolean list element. This should not be used for struct fields. For the latter,
    /// use the [`FieldIdentifier::bool_val`] field.
    fn read_bool(&mut self) -> ThriftProtocolResult<bool> {
        let b = self.read_byte()?;
        // Previous versions of the thrift specification said to use 0 and 1 inside collections,
        // but that differed from existing implementations.
        // The specification was updated in https://github.com/apache/thrift/commit/2c29c5665bc442e703480bb0ee60fe925ffe02e8.
        // At least the go implementation seems to have followed the previously documented values.
        match b {
            0x01 => Ok(true),
            0x00 | 0x02 => Ok(false),
            _ => Err(ThriftProtocolError::InvalidBoolean(b)),
        }
    }

    /// Read a Thrift [binary] as a UTF-8 encoded string.
    ///
    /// [binary]: https://github.com/apache/thrift/blob/master/doc/specs/thrift-compact-protocol.md#binary-encoding
    fn read_string(&mut self) -> ThriftProtocolResult<&'a str> {
        let slice = self.read_bytes()?;
        Ok(std::str::from_utf8(slice)?)
    }

    /// Read an `i8`.
    fn read_i8(&mut self) -> ThriftProtocolResult<i8> {
        Ok(self.read_byte()? as _)
    }

    /// Read an `i16`.
    fn read_i16(&mut self) -> ThriftProtocolResult<i16> {
        Ok(self.read_zig_zag()? as _)
    }

    /// Read an `i32`.
    fn read_i32(&mut self) -> ThriftProtocolResult<i32> {
        Ok(self.read_zig_zag()? as _)
    }

    /// Read an `i64`.
    fn read_i64(&mut self) -> ThriftProtocolResult<i64> {
        self.read_zig_zag()
    }

    /// Read a Thrift `double` as `f64`.
    fn read_double(&mut self) -> ThriftProtocolResult<f64>;

    /// Skip a ULEB128 encoded varint.
    fn skip_vlq(&mut self) -> ThriftProtocolResult<()> {
        loop {
            let byte = self.read_byte()?;
            if byte & 0x80 == 0 {
                return Ok(());
            }
        }
    }

    /// Skip a thrift [binary].
    ///
    /// [binary]: https://github.com/apache/thrift/blob/master/doc/specs/thrift-compact-protocol.md#binary-encoding
    fn skip_binary(&mut self) -> ThriftProtocolResult<()> {
        let len = self.read_vlq()? as usize;
        self.skip_bytes(len)
    }

    /// Skip a thrift list whose element type is a varint (i16/i32/i64). Reads
    /// the list header, then consumes each element as a varint without
    /// dispatching through the generic per-element recursive `skip`. Used in
    /// hot paths where we know the element type up-front (e.g. skipping
    /// `ColumnMetaData.encodings` which is a `list<i32>`).
    fn skip_list_of_varint(&mut self) -> ThriftProtocolResult<()> {
        let list = self.read_list_begin()?;
        for _ in 0..list.size.max(0) {
            self.skip_vlq()?;
        }
        Ok(())
    }

    /// Skip a thrift list whose element type is `binary` (bytes / string).
    /// Used in hot paths where we know the element type up-front (e.g.
    /// skipping `ColumnMetaData.path_in_schema` which is a `list<binary>`).
    fn skip_list_of_binary(&mut self) -> ThriftProtocolResult<()> {
        let list = self.read_list_begin()?;
        for _ in 0..list.size.max(0) {
            self.skip_binary()?;
        }
        Ok(())
    }

    /// Skip a field with type `field_type` recursively until the default
    /// maximum skip depth (currently 64) is reached.
    fn skip(&mut self, field_type: FieldType) -> ThriftProtocolResult<()> {
        const DEFAULT_SKIP_DEPTH: i8 = 64;
        self.skip_till_depth(field_type, DEFAULT_SKIP_DEPTH)
    }

    /// Empty structs in unions consist of a single byte of 0 for the field stop record.
    /// This skips that byte without encuring the cost of processing the [`FieldIdentifier`].
    /// Will return an error if the struct is not actually empty.
    fn skip_empty_struct(&mut self) -> Result<()> {
        let b = self.read_byte()?;
        if b != 0 {
            Err(general_err!("Empty struct has fields"))
        } else {
            Ok(())
        }
    }

    /// Skip a field with type `field_type` recursively up to `depth` levels.
    fn skip_till_depth(&mut self, field_type: FieldType, depth: i8) -> ThriftProtocolResult<()> {
        if depth == 0 {
            return Err(ThriftProtocolError::SkipDepth(field_type));
        }

        match field_type {
            // boolean field has no data
            FieldType::BooleanFalse | FieldType::BooleanTrue => Ok(()),
            FieldType::Byte => self.read_i8().map(|_| ()),
            FieldType::I16 => self.skip_vlq().map(|_| ()),
            FieldType::I32 => self.skip_vlq().map(|_| ()),
            FieldType::I64 => self.skip_vlq().map(|_| ()),
            FieldType::Double => self.skip_bytes(8).map(|_| ()),
            FieldType::Binary => self.skip_binary().map(|_| ()),
            FieldType::Struct => {
                let mut last_field_id = 0i16;
                loop {
                    let field_ident = self.read_field_begin(last_field_id)?;
                    if field_ident.field_type == FieldType::Stop {
                        break;
                    }
                    self.skip_till_depth(field_ident.field_type, depth - 1)?;
                    last_field_id = field_ident.id;
                }
                Ok(())
            },
            FieldType::List => {
                let list_ident = self.read_list_begin()?;
                for _ in 0..list_ident.size {
                    let element_type = FieldType::try_from(list_ident.element_type)?;
                    self.skip_till_depth(element_type, depth - 1)?;
                }
                Ok(())
            },
            // no list or map types in parquet format
            _ => Err(ThriftProtocolError::SkipUnsupportedType(field_type)),
        }
    }
}

/// A high performance Thrift reader that reads from a slice of bytes.
pub(crate) struct ThriftSliceInputProtocol<'a> {
    buf: &'a [u8],
}

impl<'a> ThriftSliceInputProtocol<'a> {
    /// Create a new `ThriftSliceInputProtocol` using the bytes in `buf`.
    pub fn new(buf: &'a [u8]) -> Self {
        Self { buf }
    }

    /// Return the current buffer as a slice.
    pub fn as_slice(&self) -> &'a [u8] {
        self.buf
    }

    /// Byte offset of the cursor from `origin_ptr`. The pointer must point
    /// into the same allocation we were constructed from (typically the
    /// footer start). Used as a value only, never dereferenced, so
    /// caller-side `unsafe` is not needed.
    #[inline]
    pub fn offset_from(&self, origin_ptr: *const u8) -> u32 {
        (self.buf.as_ptr() as usize - origin_ptr as usize) as u32
    }
}

impl<'b, 'a: 'b> ThriftCompactInputProtocol<'b> for ThriftSliceInputProtocol<'a> {
    #[inline]
    fn read_byte(&mut self) -> ThriftProtocolResult<u8> {
        let ret = *self.buf.first().ok_or(ThriftProtocolError::Eof)?;
        self.buf = &self.buf[1..];
        Ok(ret)
    }

    fn read_bytes(&mut self) -> ThriftProtocolResult<&'b [u8]> {
        let len = self.read_vlq()? as usize;
        let ret = self.buf.get(..len).ok_or(ThriftProtocolError::Eof)?;
        self.buf = &self.buf[len..];
        Ok(ret)
    }

    fn read_bytes_owned(&mut self) -> ThriftProtocolResult<Vec<u8>> {
        Ok(self.read_bytes()?.to_vec())
    }

    #[inline]
    fn skip_bytes(&mut self, n: usize) -> ThriftProtocolResult<()> {
        self.buf.get(..n).ok_or(ThriftProtocolError::Eof)?;
        self.buf = &self.buf[n..];
        Ok(())
    }

    fn read_double(&mut self) -> ThriftProtocolResult<f64> {
        let slice = self.buf.get(..8).ok_or(ThriftProtocolError::Eof)?;
        self.buf = &self.buf[8..];
        match slice.try_into() {
            Ok(slice) => Ok(f64::from_le_bytes(slice)),
            Err(_) => unreachable!(),
        }
    }
}

/// A Thrift input protocol that wraps a [`Read`] object.
///
/// Note that this is only intended for use in reading Parquet page headers. This will panic
/// if Thrift `binary` data is encountered because a slice of that data cannot be returned.
pub(crate) struct ThriftReadInputProtocol<R: Read> {
    reader: R,
}

impl<R: Read> ThriftReadInputProtocol<R> {
    pub(crate) fn new(reader: R) -> Self {
        Self { reader }
    }
}

impl<'a, R: Read> ThriftCompactInputProtocol<'a> for ThriftReadInputProtocol<R> {
    #[inline]
    fn read_byte(&mut self) -> ThriftProtocolResult<u8> {
        let mut buf = [0_u8; 1];
        self.reader.read_exact(&mut buf)?;
        Ok(buf[0])
    }

    fn read_bytes(&mut self) -> ThriftProtocolResult<&'a [u8]> {
        unimplemented!()
    }

    fn read_bytes_owned(&mut self) -> ThriftProtocolResult<Vec<u8>> {
        let len = self.read_vlq()? as usize;
        let mut v = Vec::with_capacity(len);
        std::io::copy(&mut self.reader.by_ref().take(len as u64), &mut v)?;
        Ok(v)
    }

    fn skip_bytes(&mut self, n: usize) -> ThriftProtocolResult<()> {
        std::io::copy(
            &mut self.reader.by_ref().take(n as u64),
            &mut std::io::sink(),
        )?;
        Ok(())
    }

    fn read_double(&mut self) -> ThriftProtocolResult<f64> {
        let mut buf = [0_u8; 8];
        self.reader.read_exact(&mut buf)?;
        Ok(f64::from_le_bytes(buf))
    }
}

/// Trait implemented for objects that can be deserialized from a Thrift input stream.
/// Implementations are provided for Thrift primitive types.
pub(crate) trait ReadThrift<'a, R: ThriftCompactInputProtocol<'a>> {
    /// Read an object of type `Self` from the input protocol object.
    fn read_thrift(prot: &mut R) -> Result<Self>
    where
        Self: Sized;
}

impl<'a, R: ThriftCompactInputProtocol<'a>> ReadThrift<'a, R> for bool {
    fn read_thrift(prot: &mut R) -> Result<Self> {
        Ok(prot.read_bool()?)
    }
}

impl<'a, R: ThriftCompactInputProtocol<'a>> ReadThrift<'a, R> for i8 {
    fn read_thrift(prot: &mut R) -> Result<Self> {
        Ok(prot.read_i8()?)
    }
}

impl<'a, R: ThriftCompactInputProtocol<'a>> ReadThrift<'a, R> for i16 {
    fn read_thrift(prot: &mut R) -> Result<Self> {
        Ok(prot.read_i16()?)
    }
}

impl<'a, R: ThriftCompactInputProtocol<'a>> ReadThrift<'a, R> for i32 {
    fn read_thrift(prot: &mut R) -> Result<Self> {
        Ok(prot.read_i32()?)
    }
}

impl<'a, R: ThriftCompactInputProtocol<'a>> ReadThrift<'a, R> for i64 {
    fn read_thrift(prot: &mut R) -> Result<Self> {
        Ok(prot.read_i64()?)
    }
}

impl<'a, R: ThriftCompactInputProtocol<'a>> ReadThrift<'a, R> for OrderedF64 {
    fn read_thrift(prot: &mut R) -> Result<Self> {
        Ok(OrderedF64(prot.read_double()?))
    }
}

impl<'a, R: ThriftCompactInputProtocol<'a>> ReadThrift<'a, R> for &'a str {
    fn read_thrift(prot: &mut R) -> Result<Self> {
        Ok(prot.read_string()?)
    }
}

// Polars-side removal: arrow-rs's `impl ReadThrift for String` would force
// `From<std::string::FromUtf8Error> for ParquetError` into polars's error
// module to support the inner `String::from_utf8(...)?`. Polars doesn't
// invoke that impl (the file-metadata decoder uses `prot.read_string()`
// directly), so the impl and its `From` shim are dropped.

impl<'a, R: ThriftCompactInputProtocol<'a>> ReadThrift<'a, R> for &'a [u8] {
    fn read_thrift(prot: &mut R) -> Result<Self> {
        Ok(prot.read_bytes()?)
    }
}

/// Read a Thrift encoded [list] from the input protocol object.
///
/// [list]: https://github.com/apache/thrift/blob/master/doc/specs/thrift-compact-protocol.md#list-and-set
pub(crate) fn read_thrift_vec<'a, T, R>(prot: &mut R) -> Result<Vec<T>>
where
    R: ThriftCompactInputProtocol<'a>,
    T: ReadThrift<'a, R>,
{
    let list_ident = prot.read_list_begin()?;
    let mut res = Vec::with_capacity(list_ident.size as usize);
    for _ in 0..list_ident.size {
        let val = T::read_thrift(prot)?;
        res.push(val);
    }
    Ok(res)
}

/////////////////////////
// thrift compact output

/// Low-level object used to serialize structs to the Thrift [compact output] protocol.
///
/// This struct serves as a wrapper around a [`Write`] object, to which thrift encoded data
/// will written. The implementation provides functions to write Thrift primitive types, as well
/// as functions used in the encoding of lists and structs. This should rarely be used directly,
/// but is instead intended for use by implementers of [`WriteThrift`] and [`WriteThriftField`].
///
/// [compact output]: https://github.com/apache/thrift/blob/master/doc/specs/thrift-compact-protocol.md
pub(crate) struct ThriftCompactOutputProtocol<W: Write> {
    writer: W,
}

impl<W: Write> ThriftCompactOutputProtocol<W> {
    /// Create a new `ThriftCompactOutputProtocol` wrapping the byte sink `writer`.
    pub(crate) fn new(writer: W) -> Self {
        Self { writer }
    }

    /// Write a single byte to the output stream.
    fn write_byte(&mut self, b: u8) -> Result<()> {
        self.writer.write_all(&[b])?;
        Ok(())
    }

    /// Write the given `u64` as a ULEB128 encoded varint.
    fn write_vlq(&mut self, val: u64) -> Result<()> {
        let mut v = val;
        while v > 0x7f {
            self.write_byte(v as u8 | 0x80)?;
            v >>= 7;
        }
        self.write_byte(v as u8)
    }

    /// Write the given `i64` as a zig-zag encoded varint.
    fn write_zig_zag(&mut self, val: i64) -> Result<()> {
        let s = (val < 0) as i64;
        self.write_vlq((((val ^ -s) << 1) + s) as u64)
    }

    /// Used to mark the start of a Thrift struct field of type `field_type`. `last_field_id`
    /// is used to compute a delta to the given `field_id` per the compact protocol [spec].
    ///
    /// [spec]: https://github.com/apache/thrift/blob/master/doc/specs/thrift-compact-protocol.md#struct-encoding
    pub(crate) fn write_field_begin(
        &mut self,
        field_type: FieldType,
        field_id: i16,
        last_field_id: i16,
    ) -> Result<()> {
        let delta = field_id.wrapping_sub(last_field_id);
        if delta > 0 && delta <= 0xf {
            self.write_byte((delta as u8) << 4 | field_type as u8)
        } else {
            self.write_byte(field_type as u8)?;
            self.write_i16(field_id)
        }
    }

    /// Used to indicate the start of a list of `element_type` elements.
    pub(crate) fn write_list_begin(&mut self, element_type: ElementType, len: usize) -> Result<()> {
        if len < 15 {
            self.write_byte((len as u8) << 4 | element_type as u8)
        } else {
            self.write_byte(0xf0u8 | element_type as u8)?;
            self.write_vlq(len as _)
        }
    }

    /// Used to mark the end of a struct. This must be called after all fields of the struct have
    /// been written.
    pub(crate) fn write_struct_end(&mut self) -> Result<()> {
        self.write_byte(0)
    }

    /// Serialize a slice of `u8`s. This will encode a length, and then write the bytes without
    /// further encoding.
    pub(crate) fn write_bytes(&mut self, val: &[u8]) -> Result<()> {
        self.write_vlq(val.len() as u64)?;
        self.writer.write_all(val)?;
        Ok(())
    }

    /// Short-cut method used to encode structs that have no fields (often used in Thrift unions).
    /// This simply encodes the field id and then immediately writes the end-of-struct marker.
    pub(crate) fn write_empty_struct(&mut self, field_id: i16, last_field_id: i16) -> Result<i16> {
        self.write_field_begin(FieldType::Struct, field_id, last_field_id)?;
        self.write_struct_end()?;
        Ok(last_field_id)
    }

    /// Write a boolean value.
    pub(crate) fn write_bool(&mut self, val: bool) -> Result<()> {
        match val {
            true => self.write_byte(1),
            false => self.write_byte(2),
        }
    }

    /// Write a zig-zag encoded `i8` value.
    pub(crate) fn write_i8(&mut self, val: i8) -> Result<()> {
        self.write_byte(val as u8)
    }

    /// Write a zig-zag encoded `i16` value.
    pub(crate) fn write_i16(&mut self, val: i16) -> Result<()> {
        self.write_zig_zag(val as _)
    }

    /// Write a zig-zag encoded `i32` value.
    pub(crate) fn write_i32(&mut self, val: i32) -> Result<()> {
        self.write_zig_zag(val as _)
    }

    /// Write a zig-zag encoded `i64` value.
    pub(crate) fn write_i64(&mut self, val: i64) -> Result<()> {
        self.write_zig_zag(val as _)
    }

    /// Write a double value.
    pub(crate) fn write_double(&mut self, val: f64) -> Result<()> {
        self.writer.write_all(&val.to_le_bytes())?;
        Ok(())
    }
}

/// Trait implemented by objects that are to be serialized to a Thrift [compact output] protocol
/// stream. Implementations are also provided for primitive Thrift types.
///
/// [compact output]: https://github.com/apache/thrift/blob/master/doc/specs/thrift-compact-protocol.md
pub(crate) trait WriteThrift {
    /// The [`ElementType`] to use when a list of this object is written.
    const ELEMENT_TYPE: ElementType;

    /// Serialize this object to the given `writer`.
    fn write_thrift<W: Write>(&self, writer: &mut ThriftCompactOutputProtocol<W>) -> Result<()>;
}

/// Implementation for a vector of thrift serializable objects that implement [`WriteThrift`].
/// This will write the necessary list header and then serialize the elements one-at-a-time.
impl<T> WriteThrift for Vec<T>
where
    T: WriteThrift,
{
    const ELEMENT_TYPE: ElementType = ElementType::List;

    fn write_thrift<W: Write>(&self, writer: &mut ThriftCompactOutputProtocol<W>) -> Result<()> {
        writer.write_list_begin(T::ELEMENT_TYPE, self.len())?;
        for item in self {
            item.write_thrift(writer)?;
        }
        Ok(())
    }
}

impl WriteThrift for bool {
    const ELEMENT_TYPE: ElementType = ElementType::Bool;

    fn write_thrift<W: Write>(&self, writer: &mut ThriftCompactOutputProtocol<W>) -> Result<()> {
        writer.write_bool(*self)
    }
}

impl WriteThrift for i8 {
    const ELEMENT_TYPE: ElementType = ElementType::Byte;

    fn write_thrift<W: Write>(&self, writer: &mut ThriftCompactOutputProtocol<W>) -> Result<()> {
        writer.write_i8(*self)
    }
}

impl WriteThrift for i16 {
    const ELEMENT_TYPE: ElementType = ElementType::I16;

    fn write_thrift<W: Write>(&self, writer: &mut ThriftCompactOutputProtocol<W>) -> Result<()> {
        writer.write_i16(*self)
    }
}

impl WriteThrift for i32 {
    const ELEMENT_TYPE: ElementType = ElementType::I32;

    fn write_thrift<W: Write>(&self, writer: &mut ThriftCompactOutputProtocol<W>) -> Result<()> {
        writer.write_i32(*self)
    }
}

impl WriteThrift for i64 {
    const ELEMENT_TYPE: ElementType = ElementType::I64;

    fn write_thrift<W: Write>(&self, writer: &mut ThriftCompactOutputProtocol<W>) -> Result<()> {
        writer.write_i64(*self)
    }
}

impl WriteThrift for OrderedF64 {
    const ELEMENT_TYPE: ElementType = ElementType::Double;

    fn write_thrift<W: Write>(&self, writer: &mut ThriftCompactOutputProtocol<W>) -> Result<()> {
        writer.write_double(self.0)
    }
}

impl WriteThrift for f64 {
    const ELEMENT_TYPE: ElementType = ElementType::Double;

    fn write_thrift<W: Write>(&self, writer: &mut ThriftCompactOutputProtocol<W>) -> Result<()> {
        writer.write_double(*self)
    }
}

impl WriteThrift for &[u8] {
    const ELEMENT_TYPE: ElementType = ElementType::Binary;

    fn write_thrift<W: Write>(&self, writer: &mut ThriftCompactOutputProtocol<W>) -> Result<()> {
        writer.write_bytes(self)
    }
}

impl WriteThrift for &str {
    const ELEMENT_TYPE: ElementType = ElementType::Binary;

    fn write_thrift<W: Write>(&self, writer: &mut ThriftCompactOutputProtocol<W>) -> Result<()> {
        writer.write_bytes(self.as_bytes())
    }
}

impl WriteThrift for String {
    const ELEMENT_TYPE: ElementType = ElementType::Binary;

    fn write_thrift<W: Write>(&self, writer: &mut ThriftCompactOutputProtocol<W>) -> Result<()> {
        writer.write_bytes(self.as_bytes())
    }
}

/// Trait implemented by objects that are fields of Thrift structs.
///
/// For example, given the Thrift struct definition
/// ```ignore
/// struct MyStruct {
///   1: required i32 field1
///   2: optional bool field2
///   3: optional OtherStruct field3
/// }
/// ```
///
/// which becomes in Rust
/// ```no_run
/// # struct OtherStruct {}
/// struct MyStruct {
///   field1: i32,
///   field2: Option<bool>,
///   field3: Option<OtherStruct>,
/// }
/// ```
/// the impl of `WriteThrift` for `MyStruct` will use the `WriteThriftField` impls for `i32`,
/// `bool`, and `OtherStruct`.
///
/// ```ignore
/// impl WriteThrift for MyStruct {
///   fn write_thrift<W: Write>(&self, writer: &mut ThriftCompactOutputProtocol<W>) -> Result<()> {
///     let mut last_field_id = 0i16;
///     last_field_id = self.field1.write_thrift_field(writer, 1, last_field_id)?;
///     if self.field2.is_some() {
///       // if field2 is `None` then this assignment won't happen and last_field_id will remain
///       // `1` when writing `field3`
///       last_field_id = self.field2.write_thrift_field(writer, 2, last_field_id)?;
///     }
///     if self.field3.is_some() {
///       // no need to assign last_field_id since this is the final field.
///       self.field3.write_thrift_field(writer, 3, last_field_id)?;
///     }
///     writer.write_struct_end()
///   }
/// }
/// ```
///
pub(crate) trait WriteThriftField {
    /// Used to write struct fields (which may be primitive or IDL defined types). This will
    /// write the field marker for the given `field_id`, using `last_field_id` to compute the
    /// field delta used by the Thrift [compact protocol]. On success this will return `field_id`
    /// to be used in chaining.
    ///
    /// [compact protocol]: https://github.com/apache/thrift/blob/master/doc/specs/thrift-compact-protocol.md#struct-encoding
    fn write_thrift_field<W: Write>(
        &self,
        writer: &mut ThriftCompactOutputProtocol<W>,
        field_id: i16,
        last_field_id: i16,
    ) -> Result<i16>;
}

// bool struct fields are written differently to bool values
impl WriteThriftField for bool {
    fn write_thrift_field<W: Write>(
        &self,
        writer: &mut ThriftCompactOutputProtocol<W>,
        field_id: i16,
        last_field_id: i16,
    ) -> Result<i16> {
        // boolean only writes the field header
        match *self {
            true => writer.write_field_begin(FieldType::BooleanTrue, field_id, last_field_id)?,
            false => writer.write_field_begin(FieldType::BooleanFalse, field_id, last_field_id)?,
        }
        Ok(field_id)
    }
}

write_thrift_field!(i8, FieldType::Byte);
write_thrift_field!(i16, FieldType::I16);
write_thrift_field!(i32, FieldType::I32);
write_thrift_field!(i64, FieldType::I64);
write_thrift_field!(OrderedF64, FieldType::Double);
write_thrift_field!(f64, FieldType::Double);
write_thrift_field!(String, FieldType::Binary);

impl WriteThriftField for &[u8] {
    fn write_thrift_field<W: Write>(
        &self,
        writer: &mut ThriftCompactOutputProtocol<W>,
        field_id: i16,
        last_field_id: i16,
    ) -> Result<i16> {
        writer.write_field_begin(FieldType::Binary, field_id, last_field_id)?;
        writer.write_bytes(self)?;
        Ok(field_id)
    }
}

impl WriteThriftField for &str {
    fn write_thrift_field<W: Write>(
        &self,
        writer: &mut ThriftCompactOutputProtocol<W>,
        field_id: i16,
        last_field_id: i16,
    ) -> Result<i16> {
        writer.write_field_begin(FieldType::Binary, field_id, last_field_id)?;
        writer.write_bytes(self.as_bytes())?;
        Ok(field_id)
    }
}

impl<T> WriteThriftField for Vec<T>
where
    T: WriteThrift,
{
    fn write_thrift_field<W: Write>(
        &self,
        writer: &mut ThriftCompactOutputProtocol<W>,
        field_id: i16,
        last_field_id: i16,
    ) -> Result<i16> {
        writer.write_field_begin(FieldType::List, field_id, last_field_id)?;
        self.write_thrift(writer)?;
        Ok(field_id)
    }
}

#[cfg(test)]
pub(crate) mod tests {
    // Arrow's `test_enum_roundtrip` / `test_union_all_empty_roundtrip` depend on
    // `crate::basic::{TimeUnit, Type}` enums (their `ReadThrift`/`WriteThrift`
    // impls are generated by `thrift_enum!` in arrow). Polars doesn't yet have
    // the equivalent polars-side enum impls. The two tests are dropped.
    // The generic roundtrip helper stays so future polars-native enums can reuse it.
    use std::fmt::Debug;

    use super::*;

    pub(crate) fn test_roundtrip<T>(val: T)
    where
        T: for<'a> ReadThrift<'a, ThriftSliceInputProtocol<'a>> + WriteThrift + PartialEq + Debug,
    {
        let mut buf = Vec::<u8>::new();
        {
            let mut writer = ThriftCompactOutputProtocol::new(&mut buf);
            val.write_thrift(&mut writer).unwrap();
        }

        let mut prot = ThriftSliceInputProtocol::new(&buf);
        let read_val = T::read_thrift(&mut prot).unwrap();
        assert_eq!(val, read_val);
    }
}
