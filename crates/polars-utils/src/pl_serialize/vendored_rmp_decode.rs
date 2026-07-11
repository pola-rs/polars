//! Vendored and modified from `rmp-serde` 1.3.1 (https://crates.io/crates/rmp-serde/1.3.1),
//! specifically `src/decode.rs`, so we replicate its copyright license.
//!
//! Copyright (c) 2017 Evgeny Safronov
//!
//! Permission is hereby granted, free of charge, to any person obtaining a copy
//! of this software and associated documentation files (the "Software"), to deal
//! in the Software without restriction, including without limitation the rights
//! to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
//! copies of the Software, and to permit persons to whom the Software is
//! furnished to do so, subject to the following conditions:
//!
//! The above copyright notice and this permission notice shall be included in all
//! copies or substantial portions of the Software.
//!
//! THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//! IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//! FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//! AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//! LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//! OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
//! SOFTWARE.
//!
//! # Why this exists
//!
//! Upstream `rmp_serde::Deserializer::deserialize_identifier` forwards (via
//! `forward_to_deserialize_any!`) into the fully generic, `#[inline(always)]`
//! `deserialize_any` -> `any_inner`/`any_num`, which is a large match over every
//! MessagePack marker (null/bool/every int width/f32/f64/str/array/map/bin/ext).
//! Because that's force-inlined, the whole body gets monomorphized separately for
//! every distinct `serde::de::Visitor` type it's called with, and
//! `#[derive(Deserialize)]` generates one anonymous `__FieldVisitor` type per
//! struct/enum. Polars' DSL/`Expr` plan tree has ~2000-3000 such types, so
//! identifier decoding -- which only ever needs to read a field/variant name --
//! was getting its entire numeric-coercion machinery duplicated ~3000 times
//! (confirmed via `cargo bloat`: ~10.6 MiB of pure duplication, 6.5% of the
//! compiled `.text` section).
//!
//! `deserialize_identifier` below is narrowed to only accept a MessagePack string
//! marker. This is safe because `rmp_serde::Serializer::with_struct_map()` (what
//! Polars always uses, see `pl_serialize.rs`) *always* writes struct field names
//! and enum variant names via `serialize_str` -- never as integers -- so a
//! non-string identifier marker is never actually produced by our own serializer,
//! and returns the same `Error::TypeMismatch` that upstream's `any_inner` would
//! already return for that case.
//!
//! # Other deliberate deviations from upstream
//!
//! Polars never configures the decoder with `.with_human_readable()` /
//! `.with_binary()`, nor deserializes from a bare zero-copy `&[u8]` source
//! (`rmp_serde::from_slice`/`from_read_ref`) -- only `Deserializer::new`/
//! `from_read` over a `std::io::Read` source is used (confirmed via grep over
//! `crates/`). So, to shrink vendored surface area, this module:
//! - Drops the generic `C: SerializerConfig` parameter entirely (upstream only
//!   uses it to fix the constant `is_human_readable = false`, which we hardcode
//!   directly -- this also sidesteps the fact that the methods backing
//!   `SerializerConfig` live on a `pub(crate)` `sealed` trait upstream, which
//!   external crates like this one cannot name).
//! - Drops `ReadRefReader` / `from_slice` / `from_read_ref` (the zero-copy
//!   slice-native reader). Note this isn't a behavior change: `ReadReader`'s
//!   `read_slice` always copies into an internal buffer regardless of the
//!   underlying reader type, so even today's `&[u8]`-backed deserialization
//!   never actually borrows zero-copy through this path.
//!
//! IMPORTANT: if `rmp-serde`'s pinned version in `Cargo.toml` is ever bumped,
//! re-diff this file against the new `rmp-serde-<version>/src/decode.rs` to pick
//! up any upstream correctness fixes. `crates/polars-plan/src/dsl/serializable_plan.rs`
//! has a permanent cross-check test comparing this decoder's output against real
//! `rmp_serde::Deserializer` on representative `DslPlan`/`Expr` values, which acts
//! as a tripwire for behavioral drift.

use std::convert::TryInto;
use std::error;
use std::fmt::{self, Display, Formatter};
use std::io::{self, Read};
use std::num::TryFromIntError;
use std::str::{self, Utf8Error};

use serde::de::value::SeqDeserializer;
use serde::de::{self, Deserialize, DeserializeOwned, DeserializeSeed, Unexpected, Visitor};
use serde::forward_to_deserialize_any;

use rmp::decode::{self, DecodeStringError, MarkerReadError, NumValueReadError, RmpRead, ValueReadError};
use rmp::Marker;

use rmp_serde::MSGPACK_EXT_STRUCT_NAME;

/// Enum representing errors that can occur while decoding MessagePack data.
#[derive(Debug)]
pub enum Error {
    /// The enclosed I/O error occurred while trying to read a MessagePack
    /// marker.
    InvalidMarkerRead(io::Error),
    /// The enclosed I/O error occurred while trying to read the encoded
    /// MessagePack data.
    InvalidDataRead(io::Error),
    /// A mismatch occurred between the decoded and expected value types.
    TypeMismatch(Marker),
    /// A numeric cast failed due to an out-of-range error.
    OutOfRange,
    /// A decoded array did not have the enclosed expected length.
    LengthMismatch(u32),
    /// An otherwise uncategorized error occurred. See the enclosed `String` for
    /// details.
    Uncategorized(String),
    /// A general error occurred while deserializing the expected type. See the
    /// enclosed `String` for details.
    Syntax(String),
    /// An encoded string could not be parsed as UTF-8.
    Utf8Error(Utf8Error),
    /// The depth limit was exceeded.
    DepthLimitExceeded,
}

macro_rules! depth_count(
    ( $counter:expr, $expr:expr ) => {
        {
            $counter -= 1;
            if $counter == 0 {
                return Err(Error::DepthLimitExceeded)
            }
            let res = $expr;
            $counter += 1;
            res
        }
    }
);

impl error::Error for Error {
    #[cold]
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        match *self {
            Self::TypeMismatch(..) => None,
            Self::InvalidMarkerRead(ref err) => Some(err),
            Self::InvalidDataRead(ref err) => Some(err),
            Self::LengthMismatch(..) => None,
            Self::OutOfRange => None,
            Self::Uncategorized(..) => None,
            Self::Syntax(..) => None,
            Self::Utf8Error(ref err) => Some(err),
            Self::DepthLimitExceeded => None,
        }
    }
}

impl de::Error for Error {
    #[cold]
    fn custom<T: Display>(msg: T) -> Self {
        Self::Syntax(msg.to_string())
    }
}

impl Display for Error {
    #[cold]
    fn fmt(&self, fmt: &mut Formatter<'_>) -> Result<(), fmt::Error> {
        match *self {
            Self::InvalidMarkerRead(ref err) => write!(fmt, "IO error while reading marker: {err}"),
            Self::InvalidDataRead(ref err) => write!(fmt, "IO error while reading data: {err}"),
            Self::TypeMismatch(ref actual_marker) => {
                write!(fmt, "wrong msgpack marker {actual_marker:?}")
            },
            Self::OutOfRange => fmt.write_str("numeric cast found out of range"),
            Self::LengthMismatch(expected_length) => {
                write!(fmt, "array had incorrect length, expected {expected_length}")
            },
            Self::Uncategorized(ref msg) => write!(fmt, "uncategorized error: {msg}"),
            Self::Syntax(ref msg) => fmt.write_str(msg),
            Self::Utf8Error(ref err) => write!(fmt, "string found to be invalid utf8: {err}"),
            Self::DepthLimitExceeded => fmt.write_str("depth limit exceeded"),
        }
    }
}

impl From<MarkerReadError> for Error {
    #[cold]
    fn from(err: MarkerReadError) -> Self {
        match err {
            MarkerReadError(err) => Self::InvalidMarkerRead(err),
        }
    }
}

impl From<Utf8Error> for Error {
    #[cold]
    fn from(err: Utf8Error) -> Self {
        Self::Utf8Error(err)
    }
}

impl From<ValueReadError> for Error {
    #[cold]
    fn from(err: ValueReadError) -> Self {
        match err {
            ValueReadError::TypeMismatch(marker) => Self::TypeMismatch(marker),
            ValueReadError::InvalidMarkerRead(err) => Self::InvalidMarkerRead(err),
            ValueReadError::InvalidDataRead(err) => Self::InvalidDataRead(err),
        }
    }
}

impl From<NumValueReadError> for Error {
    #[cold]
    fn from(err: NumValueReadError) -> Self {
        match err {
            NumValueReadError::TypeMismatch(marker) => Self::TypeMismatch(marker),
            NumValueReadError::InvalidMarkerRead(err) => Self::InvalidMarkerRead(err),
            NumValueReadError::InvalidDataRead(err) => Self::InvalidDataRead(err),
            NumValueReadError::OutOfRange => Self::OutOfRange,
        }
    }
}

impl From<DecodeStringError<'_>> for Error {
    #[cold]
    fn from(err: DecodeStringError<'_>) -> Self {
        match err {
            DecodeStringError::InvalidMarkerRead(err) => Self::InvalidMarkerRead(err),
            DecodeStringError::InvalidDataRead(err) => Self::InvalidDataRead(err),
            DecodeStringError::TypeMismatch(marker) => Self::TypeMismatch(marker),
            DecodeStringError::BufferSizeTooSmall(..) => Self::Uncategorized("BufferSizeTooSmall".to_string()),
            DecodeStringError::InvalidUtf8(..) => Self::Uncategorized("InvalidUtf8".to_string()),
        }
    }
}

impl From<TryFromIntError> for Error {
    #[cold]
    fn from(_: TryFromIntError) -> Self {
        Self::OutOfRange
    }
}

/// A Deserializer that reads bytes from a buffer.
///
/// Unlike upstream `rmp_serde::Deserializer`, this is not generic over a
/// `SerializerConfig` -- Polars only ever uses the default (non-human-readable)
/// configuration on the decode side, so it's hardcoded (see module docs).
#[derive(Debug)]
pub struct Deserializer<R> {
    rd: R,
    marker: Option<Marker>,
    depth: u16,
}

impl<R: Read> Deserializer<ReadReader<R>> {
    /// Constructs a new `Deserializer` by consuming the given reader.
    #[inline]
    pub fn new(rd: R) -> Self {
        Self {
            rd: ReadReader::new(rd),
            // Cached marker in case of deserializing optional values.
            marker: None,
            depth: 1024,
        }
    }
}

impl<R: Read> Deserializer<R> {
    #[inline]
    fn take_or_read_marker(&mut self) -> Result<Marker, MarkerReadError> {
        self.marker
            .take()
            .map_or_else(|| rmp::decode::read_marker(&mut self.rd), Ok)
    }

    #[inline]
    fn peek_or_read_marker(&mut self) -> Result<Marker, MarkerReadError> {
        if let Some(m) = self.marker {
            Ok(m)
        } else {
            let m = rmp::decode::read_marker(&mut self.rd)?;
            Ok(self.marker.insert(m).to_owned())
        }
    }
}

impl<'de, R: ReadSlice<'de>> Deserializer<R> {
    /// Changes the maximum nesting depth that is allowed
    #[inline(always)]
    pub fn set_max_depth(&mut self, depth: usize) {
        self.depth = depth.min(u16::MAX as _) as u16;
    }
}

#[inline(never)]
fn read_i128_marker<'de, R: ReadSlice<'de>>(marker: Marker, rd: &mut R) -> Result<i128, Error> {
    Ok(match marker {
        Marker::FixPos(val) => val.into(),
        Marker::FixNeg(val) => val.into(),
        Marker::U8 => rd.read_data_u8()?.into(),
        Marker::U16 => rd.read_data_u16()?.into(),
        Marker::U32 => rd.read_data_u32()?.into(),
        Marker::U64 => rd.read_data_u64()?.into(),
        Marker::I8 => rd.read_data_i8()?.into(),
        Marker::I16 => rd.read_data_i16()?.into(),
        Marker::I32 => rd.read_data_i32()?.into(),
        Marker::I64 => rd.read_data_i64()?.into(),
        Marker::Bin8 => {
            let len = read_u8(&mut *rd)?;
            read_128_buf(rd, len)?
        },
        Marker::FixArray(len) => read_128_buf(rd, len)?,
        marker => return Err(Error::TypeMismatch(marker)),
    })
}

fn read_128_buf<'de, R: ReadSlice<'de>>(rd: &mut R, len: u8) -> Result<i128, Error> {
    if len != 16 {
        return Err(Error::LengthMismatch(16));
    }
    let buf = match read_bin_data(rd, 16)? {
        Reference::Borrowed(buf) => buf,
        Reference::Copied(buf) => buf,
    };
    Ok(i128::from_be_bytes(buf.try_into().map_err(|_| Error::LengthMismatch(16))?))
}

fn read_str_data<'de, V, R>(rd: &mut R, len: u32, visitor: V) -> Result<V::Value, Error>
    where V: Visitor<'de>, R: ReadSlice<'de>
{
    match read_bin_data(rd, len)? {
        Reference::Borrowed(buf) => {
            match str::from_utf8(buf) {
                Ok(s) => visitor.visit_borrowed_str(s),
                Err(err) => {
                    // Allow to unpack invalid UTF-8 bytes into a byte array.
                    match visitor.visit_borrowed_bytes::<Error>(buf) {
                        Ok(buf) => Ok(buf),
                        Err(..) => Err(Error::Utf8Error(err)),
                    }
                },
            }
        },
        Reference::Copied(buf) => {
            match str::from_utf8(buf) {
                Ok(s) => visitor.visit_str(s),
                Err(err) => {
                    // Allow to unpack invalid UTF-8 bytes into a byte array.
                    match visitor.visit_bytes::<Error>(buf) {
                        Ok(buf) => Ok(buf),
                        Err(..) => Err(Error::Utf8Error(err)),
                    }
                },
            }
        },
    }
}

fn read_bin_data<'a, 'de, R: ReadSlice<'de>>(rd: &'a mut R, len: u32) -> Result<Reference<'de, 'a, [u8]>, Error> {
    rd.read_slice(len as usize).map_err(Error::InvalidDataRead)
}

fn read_u8<R: Read>(rd: &mut R) -> Result<u8, Error> {
    let mut buf = [0; 1];
    rd.read_exact(&mut buf).map_err(Error::InvalidDataRead)?;
    Ok(buf[0])
}

fn read_u16<R: Read>(rd: &mut R) -> Result<u16, Error> {
    let mut buf = [0; 2];
    rd.read_exact(&mut buf).map_err(Error::InvalidDataRead)?;
    Ok(u16::from_be_bytes(buf))
}

fn read_u32<R: Read>(rd: &mut R) -> Result<u32, Error> {
    let mut buf = [0; 4];
    rd.read_exact(&mut buf).map_err(Error::InvalidDataRead)?;
    Ok(u32::from_be_bytes(buf))
}

fn ext_len<R: Read>(rd: &mut R, marker: Marker) -> Result<u32, Error> {
    Ok(match marker {
        Marker::FixExt1 => 1,
        Marker::FixExt2 => 2,
        Marker::FixExt4 => 4,
        Marker::FixExt8 => 8,
        Marker::FixExt16 => 16,
        Marker::Ext8 => u32::from(read_u8(rd)?),
        Marker::Ext16 => u32::from(read_u16(rd)?),
        Marker::Ext32 => read_u32(rd)?,
        _ => return Err(Error::TypeMismatch(marker)),
    })
}

#[derive(Debug)]
enum ExtDeserializerState {
    New,
    ReadTag,
    ReadBinary,
}

#[derive(Debug)]
struct ExtDeserializer<'a, R> {
    rd: &'a mut R,
    len: u32,
    state: ExtDeserializerState,
}

impl<'de, 'a, R: ReadSlice<'de> + 'a> ExtDeserializer<'a, R> {
    const fn new(d: &'a mut Deserializer<R>, len: u32) -> Self {
        ExtDeserializer {
            rd: &mut d.rd,
            len,
            state: ExtDeserializerState::New,
        }
    }
}

impl<'de, 'a, R: ReadSlice<'de> + 'a> de::Deserializer<'de> for ExtDeserializer<'a, R> {
    type Error = Error;

    #[inline(always)]
    fn deserialize_any<V>(self, visitor: V) -> Result<V::Value, Self::Error>
        where V: Visitor<'de>
    {
        visitor.visit_seq(self)
    }

    forward_to_deserialize_any! {
        bool u8 u16 u32 u64 i8 i16 i32 i64 f32 f64 char str string unit option
        seq bytes byte_buf map unit_struct newtype_struct
        struct identifier tuple enum ignored_any tuple_struct
    }
}

impl<'de, 'a, R: ReadSlice<'de> + 'a> de::SeqAccess<'de> for ExtDeserializer<'a, R> {
    type Error = Error;

    #[inline]
    fn next_element_seed<T>(&mut self, seed: T) -> Result<Option<T::Value>, Error>
    where
        T: DeserializeSeed<'de>,
    {
        match self.state {
            ExtDeserializerState::New | ExtDeserializerState::ReadTag => Ok(Some(seed.deserialize(self)?)),
            ExtDeserializerState::ReadBinary => Ok(None),
        }
    }
}

/// Deserializer for Ext `SeqAccess`
impl<'de, 'a, R: ReadSlice<'de> + 'a> de::Deserializer<'de> for &mut ExtDeserializer<'a, R> {
    type Error = Error;

    fn deserialize_any<V>(self, visitor: V) -> Result<V::Value, Self::Error>
        where V: Visitor<'de>
    {
        match self.state {
            ExtDeserializerState::New => {
                let tag = self.rd.read_data_i8()?;
                self.state = ExtDeserializerState::ReadTag;
                visitor.visit_i8(tag)
            },
            ExtDeserializerState::ReadTag => {
                let data = self.rd.read_slice(self.len as usize).map_err(Error::InvalidDataRead)?;
                self.state = ExtDeserializerState::ReadBinary;
                match data {
                    Reference::Borrowed(bytes) => visitor.visit_borrowed_bytes(bytes),
                    Reference::Copied(bytes) => visitor.visit_bytes(bytes),
                }
            },
            ExtDeserializerState::ReadBinary => {
                debug_assert!(false);
                Err(Error::TypeMismatch(Marker::Reserved))
            },
        }
    }

    forward_to_deserialize_any! {
        bool u8 u16 u32 u64 i8 i16 i32 i64 f32 f64 char str string unit option
        seq bytes byte_buf map unit_struct newtype_struct
        tuple_struct struct identifier tuple enum ignored_any
    }
}

#[inline(never)]
fn any_num<'de, R: ReadSlice<'de>, V: Visitor<'de>>(rd: &mut R, visitor: V, marker: Marker) -> Result<V::Value, Error> {
    match marker {
        Marker::Null => visitor.visit_unit(),
        Marker::True |
        Marker::False => visitor.visit_bool(marker == Marker::True),
        Marker::FixPos(val) => visitor.visit_u8(val),
        Marker::FixNeg(val) => visitor.visit_i8(val),
        Marker::U8 => visitor.visit_u8(rd.read_data_u8()?),
        Marker::U16 => visitor.visit_u16(rd.read_data_u16()?),
        Marker::U32 => visitor.visit_u32(rd.read_data_u32()?),
        Marker::U64 => visitor.visit_u64(rd.read_data_u64()?),
        Marker::I8 => visitor.visit_i8(rd.read_data_i8()?),
        Marker::I16 => visitor.visit_i16(rd.read_data_i16()?),
        Marker::I32 => visitor.visit_i32(rd.read_data_i32()?),
        Marker::I64 => visitor.visit_i64(rd.read_data_i64()?),
        Marker::F32 => visitor.visit_f32(rd.read_data_f32()?),
        Marker::F64 => visitor.visit_f64(rd.read_data_f64()?),
        other_marker => Err(Error::TypeMismatch(other_marker)),
    }
}

impl<'de, R: ReadSlice<'de>> Deserializer<R> {
    fn any_inner<V: Visitor<'de>>(&mut self, visitor: V, allow_bytes: bool) -> Result<V::Value, Error> {
        let marker = self.take_or_read_marker()?;
        match marker {
            Marker::Null |
            Marker::True |
            Marker::False |
            Marker::FixPos(_) |
            Marker::FixNeg(_) |
            Marker::U8 |
            Marker::U16 |
            Marker::U32 |
            Marker::U64 |
            Marker::I8 |
            Marker::I16 |
            Marker::I32 |
            Marker::I64 |
            Marker::F32 |
            Marker::F64 => any_num(&mut self.rd, visitor, marker),
            Marker::FixStr(_) | Marker::Str8 | Marker::Str16 | Marker::Str32 => {
                let len = match marker {
                    Marker::FixStr(len) => Ok(len.into()),
                    Marker::Str8 => read_u8(&mut self.rd).map(u32::from),
                    Marker::Str16 => read_u16(&mut self.rd).map(u32::from),
                    Marker::Str32 => read_u32(&mut self.rd),
                    _ => return Err(Error::TypeMismatch(Marker::Reserved)),
                }?;
                read_str_data(&mut self.rd, len, visitor)
            },
            Marker::FixArray(_) | Marker::Array16 | Marker::Array32 => {
                let len = match marker {
                    Marker::FixArray(len) => len.into(),
                    Marker::Array16 => read_u16(&mut self.rd)?.into(),
                    Marker::Array32 => read_u32(&mut self.rd)?,
                    _ => return Err(Error::TypeMismatch(Marker::Reserved)),
                };

                depth_count!(self.depth, {
                    let mut seq = SeqAccess::new(self, len);
                    let res = visitor.visit_seq(&mut seq)?;
                    match seq.left {
                        0 => Ok(res),
                        excess => Err(Error::LengthMismatch(len - excess)),
                    }
                })
            },
            Marker::FixMap(_) | Marker::Map16 | Marker::Map32 => {
                let len = match marker {
                    Marker::FixMap(len) => len.into(),
                    Marker::Map16 => read_u16(&mut self.rd)?.into(),
                    Marker::Map32 => read_u32(&mut self.rd)?,
                    _ => return Err(Error::TypeMismatch(Marker::Reserved)),
                };

                depth_count!(self.depth, {
                    let mut seq = MapAccess::new(self, len);
                    let res = visitor.visit_map(&mut seq)?;
                    match seq.left {
                        0 => Ok(res),
                        excess => Err(Error::LengthMismatch(len - excess)),
                    }
                })
            },
            Marker::Bin8 | Marker::Bin16 | Marker::Bin32 => {
                let len = match marker {
                    Marker::Bin8 => read_u8(&mut self.rd).map(u32::from),
                    Marker::Bin16 => read_u16(&mut self.rd).map(u32::from),
                    Marker::Bin32 => read_u32(&mut self.rd),
                    _ => return Err(Error::TypeMismatch(Marker::Reserved)),
                }?;
                match read_bin_data(&mut self.rd, len)? {
                    Reference::Borrowed(buf) if allow_bytes => visitor.visit_borrowed_bytes(buf),
                    Reference::Copied(buf) if allow_bytes => visitor.visit_bytes(buf),
                    Reference::Borrowed(buf) | Reference::Copied(buf) => {
                        visitor.visit_seq(SeqDeserializer::new(buf.iter().copied()))
                    },
                }
            },
            Marker::FixExt1 |
            Marker::FixExt2 |
            Marker::FixExt4 |
            Marker::FixExt8 |
            Marker::FixExt16 |
            Marker::Ext8 |
            Marker::Ext16 |
            Marker::Ext32 => {
                let len = ext_len(&mut self.rd, marker)?;
                depth_count!(self.depth, visitor.visit_newtype_struct(ExtDeserializer::new(self, len)))
            },
            Marker::Reserved => Err(Error::TypeMismatch(Marker::Reserved)),
        }
    }
}

impl<'de, R: ReadSlice<'de>> serde::Deserializer<'de> for &mut Deserializer<R> {
    type Error = Error;

    #[inline(always)]
    fn is_human_readable(&self) -> bool {
        // Polars never configures the decoder as human-readable (see module docs).
        false
    }

    #[inline(always)]
    fn deserialize_any<V>(self, visitor: V) -> Result<V::Value, Self::Error>
        where V: Visitor<'de>
    {
        self.any_inner(visitor, true)
    }

    fn deserialize_option<V>(self, visitor: V) -> Result<V::Value, Self::Error>
        where V: Visitor<'de>
    {
        // # Important
        //
        // If a nested Option `o ∈ { Option<Opion<t>>, Option<Option<Option<t>>>, ..., Option<Option<...Option<t>...> }`
        // is visited for the first time, the marker (read from the underlying Reader) will determine
        // `o`'s innermost type `t`.
        // For subsequent visits of `o` the marker will not be re-read again but kept until type `t`
        // is visited.
        //
        // # Note
        //
        // Round trips of Options where `Option<t> = None` such as `Some(None)` will fail because
        // they are just seriialized as `nil`. The serialization format has probably to be changed
        // to solve this. But as serde_json behaves the same, I think it's not worth doing this.
        let marker = self.take_or_read_marker()?;

        if marker == Marker::Null {
            visitor.visit_none()
        } else {
            // Keep the marker until `o`'s innermost type `t` is visited.
            self.marker = Some(marker);
            visitor.visit_some(self)
        }
    }

    /// Narrowed replacement for upstream's `forward_to_deserialize_any!`-based
    /// `deserialize_identifier`. See module docs for why this is safe.
    fn deserialize_identifier<V>(self, visitor: V) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        let marker = self.take_or_read_marker()?;
        match marker {
            Marker::FixStr(len) => read_str_data(&mut self.rd, u32::from(len), visitor),
            Marker::Str8 => {
                let len = u32::from(read_u8(&mut self.rd)?);
                read_str_data(&mut self.rd, len, visitor)
            },
            Marker::Str16 => {
                let len = u32::from(read_u16(&mut self.rd)?);
                read_str_data(&mut self.rd, len, visitor)
            },
            Marker::Str32 => {
                let len = read_u32(&mut self.rd)?;
                read_str_data(&mut self.rd, len, visitor)
            },
            other => Err(Error::TypeMismatch(other)),
        }
    }

    fn deserialize_enum<V>(self, _name: &str, _variants: &[&str], visitor: V) -> Result<V::Value, Error>
        where V: Visitor<'de>
    {
        let marker = self.peek_or_read_marker()?;
        match rmp::decode::marker_to_len(&mut self.rd, marker) {
            Ok(len) => match len {
                // Enums are either encoded as maps with a single K/V pair
                // where the K = the variant & V = associated data
                // or as just the variant
                1 => {
                    self.marker = None;
                    visitor.visit_enum(VariantAccess::new(self))
                },
                n => Err(Error::LengthMismatch(n)),
            },
            // TODO: Check this is a string
            Err(_) => visitor.visit_enum(UnitVariantAccess::new(self)),
        }
    }

    fn deserialize_newtype_struct<V>(self, name: &'static str, visitor: V) -> Result<V::Value, Error>
        where V: Visitor<'de>
    {
        if name == MSGPACK_EXT_STRUCT_NAME {
            let marker = self.take_or_read_marker()?;

            let len = ext_len(&mut self.rd, marker)?;
            let ext_de = ExtDeserializer::new(self, len);
            return visitor.visit_newtype_struct(ext_de);
        }

        visitor.visit_newtype_struct(self)
    }

    fn deserialize_unit_struct<V>(self, _name: &'static str, visitor: V) -> Result<V::Value, Self::Error>
        where V: Visitor<'de>
    {
        // We need to special case this so that [] is treated as a unit struct when asked for,
        // but as a sequence otherwise. This is because we serialize unit structs as [] rather
        // than as 'nil'.
        match self.take_or_read_marker()? {
            Marker::Null | Marker::FixArray(0) => visitor.visit_unit(),
            marker => {
                self.marker = Some(marker);
                self.deserialize_any(visitor)
            },
        }
    }

    #[inline]
    fn deserialize_i128<V>(self, visitor: V) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        visitor.visit_i128(read_i128_marker(self.take_or_read_marker()?, &mut self.rd)?)
    }

    #[inline]
    fn deserialize_u128<V>(self, visitor: V) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        visitor.visit_u128(read_i128_marker(self.take_or_read_marker()?, &mut self.rd)? as u128)
    }

    #[inline]
    fn deserialize_seq<V>(self, visitor: V) -> Result<V::Value, Self::Error> where V: Visitor<'de> {
        self.any_inner(visitor, false)
    }

    #[inline]
    fn deserialize_tuple<V>(self, _len: usize, visitor: V) -> Result<V::Value, Self::Error> where V: Visitor<'de> {
        self.any_inner(visitor, false)
    }

    #[inline]
    fn deserialize_struct<V>(self, _: &'static str, _: &'static [&'static str], visitor: V) -> Result<V::Value, Self::Error> where V: Visitor<'de> {
        self.any_inner(visitor, false)
    }

    #[inline]
    fn deserialize_tuple_struct<V>(self, _: &'static str, _: usize, visitor: V) -> Result<V::Value, Self::Error> where V: Visitor<'de> {
        self.any_inner(visitor, false)
    }

    forward_to_deserialize_any! {
        bytes byte_buf unit
        map str string char
        ignored_any
    }

    fn deserialize_bool<V>(self, visitor: V) -> Result<V::Value, Self::Error> where V: Visitor<'de> {
        let marker = self.take_or_read_marker()?;
        any_num(&mut self.rd, visitor, marker)
    }

    fn deserialize_u8<V>(self, visitor: V) -> Result<V::Value, Self::Error> where V: Visitor<'de> {
        let marker = self.take_or_read_marker()?;
        any_num(&mut self.rd, visitor, marker)
    }

    fn deserialize_i8<V>(self, visitor: V) -> Result<V::Value, Self::Error> where V: Visitor<'de> {
        let marker = self.take_or_read_marker()?;
        any_num(&mut self.rd, visitor, marker)
    }

    fn deserialize_i16<V>(self, visitor: V) -> Result<V::Value, Self::Error> where V: Visitor<'de> {
        let marker = self.take_or_read_marker()?;
        any_num(&mut self.rd, visitor, marker)
    }

    fn deserialize_u16<V>(self, visitor: V) -> Result<V::Value, Self::Error> where V: Visitor<'de> {
        let marker = self.take_or_read_marker()?;
        any_num(&mut self.rd, visitor, marker)
    }

    fn deserialize_i32<V>(self, visitor: V) -> Result<V::Value, Self::Error> where V: Visitor<'de> {
        let marker = self.take_or_read_marker()?;
        any_num(&mut self.rd, visitor, marker)
    }

    fn deserialize_u32<V>(self, visitor: V) -> Result<V::Value, Self::Error> where V: Visitor<'de> {
        let marker = self.take_or_read_marker()?;
        any_num(&mut self.rd, visitor, marker)
    }

    fn deserialize_i64<V>(self, visitor: V) -> Result<V::Value, Self::Error> where V: Visitor<'de> {
        let marker = self.take_or_read_marker()?;
        any_num(&mut self.rd, visitor, marker)
    }

    fn deserialize_u64<V>(self, visitor: V) -> Result<V::Value, Self::Error> where V: Visitor<'de> {
        let marker = self.take_or_read_marker()?;
        any_num(&mut self.rd, visitor, marker)
    }

    fn deserialize_f32<V>(self, visitor: V) -> Result<V::Value, Self::Error> where V: Visitor<'de> {
        let marker = self.take_or_read_marker()?;
        any_num(&mut self.rd, visitor, marker)
    }

    fn deserialize_f64<V>(self, visitor: V) -> Result<V::Value, Self::Error> where V: Visitor<'de> {
        let marker = self.take_or_read_marker()?;
        any_num(&mut self.rd, visitor, marker)
    }
}

struct SeqAccess<'a, R> {
    de: &'a mut Deserializer<R>,
    left: u32,
}

impl<'a, R: 'a> SeqAccess<'a, R> {
    #[inline]
    const fn new(de: &'a mut Deserializer<R>, len: u32) -> Self {
        SeqAccess { de, left: len }
    }
}

impl<'de, 'a, R: ReadSlice<'de> + 'a> de::SeqAccess<'de> for SeqAccess<'a, R> {
    type Error = Error;

    #[inline]
    fn next_element_seed<T>(&mut self, seed: T) -> Result<Option<T::Value>, Self::Error>
        where T: DeserializeSeed<'de>
    {
        if self.left > 0 {
            self.left -= 1;
            Ok(Some(seed.deserialize(&mut *self.de)?))
        } else {
            Ok(None)
        }
    }

    #[inline(always)]
    fn size_hint(&self) -> Option<usize> {
        self.left.try_into().ok()
    }
}

struct MapAccess<'a, R> {
    de: &'a mut Deserializer<R>,
    left: u32,
}

impl<'a, R: 'a> MapAccess<'a, R> {
    #[inline]
    const fn new(de: &'a mut Deserializer<R>, len: u32) -> Self {
        MapAccess { de, left: len }
    }
}

impl<'de, 'a, R: ReadSlice<'de> + 'a> de::MapAccess<'de> for MapAccess<'a, R> {
    type Error = Error;

    #[inline]
    fn next_key_seed<K>(&mut self, seed: K) -> Result<Option<K::Value>, Self::Error>
        where K: DeserializeSeed<'de>
    {
        if self.left > 0 {
            self.left -= 1;
            seed.deserialize(&mut *self.de).map(Some)
        } else {
            Ok(None)
        }
    }

    #[inline]
    fn next_value_seed<V>(&mut self, seed: V) -> Result<V::Value, Self::Error>
        where V: DeserializeSeed<'de>
    {
        seed.deserialize(&mut *self.de)
    }

    #[inline(always)]
    fn size_hint(&self) -> Option<usize> {
        self.left.try_into().ok()
    }
}

struct UnitVariantAccess<'a, R: 'a> {
    de: &'a mut Deserializer<R>,
}

impl<'a, R: 'a> UnitVariantAccess<'a, R> {
    pub const fn new(de: &'a mut Deserializer<R>) -> Self {
        UnitVariantAccess { de }
    }
}

impl<'de, R: ReadSlice<'de>> de::EnumAccess<'de>
    for UnitVariantAccess<'_, R>
{
    type Error = Error;
    type Variant = Self;

    #[inline]
    fn variant_seed<V>(self, seed: V) -> Result<(V::Value, Self), Error>
    where
        V: de::DeserializeSeed<'de>,
    {
        let variant = seed.deserialize(&mut *self.de)?;
        Ok((variant, self))
    }
}

impl<'de, 'a, R: ReadSlice<'de> + 'a> de::VariantAccess<'de>
    for UnitVariantAccess<'a, R>
{
    type Error = Error;

    fn unit_variant(self) -> Result<(), Error> {
        Ok(())
    }

    fn newtype_variant_seed<T>(self, _seed: T) -> Result<T::Value, Error>
    where
        T: de::DeserializeSeed<'de>,
    {
        Err(de::Error::invalid_type(
            Unexpected::UnitVariant,
            &"newtype variant",
        ))
    }

    fn tuple_variant<V>(self, _len: usize, _visitor: V) -> Result<V::Value, Error>
    where
        V: de::Visitor<'de>,
    {
        Err(de::Error::invalid_type(
            Unexpected::UnitVariant,
            &"tuple variant",
        ))
    }

    fn struct_variant<V>(
        self,
        _fields: &'static [&'static str],
        _visitor: V,
    ) -> Result<V::Value, Error>
    where
        V: de::Visitor<'de>,
    {
        Err(de::Error::invalid_type(
            Unexpected::UnitVariant,
            &"struct variant",
        ))
    }
}

struct VariantAccess<'a, R> {
    de: &'a mut Deserializer<R>,
}

impl<'a, R: 'a> VariantAccess<'a, R> {
    pub const fn new(de: &'a mut Deserializer<R>) -> Self {
        VariantAccess { de }
    }
}

impl<'de, R: ReadSlice<'de>> de::EnumAccess<'de> for VariantAccess<'_, R> {
    type Error = Error;
    type Variant = Self;

    #[inline]
    fn variant_seed<V>(self, seed: V) -> Result<(V::Value, Self), Error>
        where V: de::DeserializeSeed<'de>,
    {
        Ok((seed.deserialize(&mut *self.de)?, self))
    }
}

impl<'de, R: ReadSlice<'de>> de::VariantAccess<'de> for VariantAccess<'_, R> {
    type Error = Error;

    #[inline]
    fn unit_variant(self) -> Result<(), Error> {
        decode::read_nil(&mut self.de.rd)?;
        Ok(())
    }

    #[inline]
    fn newtype_variant_seed<T>(self, seed: T) -> Result<T::Value, Self::Error>
        where T: DeserializeSeed<'de>
    {
        seed.deserialize(self.de)
    }

    #[inline]
    fn tuple_variant<V>(self, len: usize, visitor: V) -> Result<V::Value, Error>
        where V: Visitor<'de>
    {
        de::Deserializer::deserialize_tuple(self.de, len, visitor)
    }

    #[inline]
    fn struct_variant<V>(self, fields: &'static [&'static str], visitor: V) -> Result<V::Value, Error>
        where V: Visitor<'de>
    {
        de::Deserializer::deserialize_tuple(self.de, fields.len(), visitor)
    }
}

/// Unification of both borrowed and non-borrowed reference types.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Reference<'b, 'c, T: ?Sized + 'static> {
    /// The reference is pointed at data that was borrowed.
    Borrowed(&'b T),
    /// The reference is pointed at data that was copied.
    Copied(&'c T),
}

/// Extends the `Read` trait by allowing to read slices directly by borrowing bytes.
///
/// Used to allow zero-copy reading.
pub trait ReadSlice<'de>: Read {
    /// Reads the exact number of bytes from the underlying byte-array.
    fn read_slice<'a>(&'a mut self, len: usize) -> Result<Reference<'de, 'a, [u8]>, io::Error>;
}

/// Owned reader wrapper.
#[derive(Debug)]
pub struct ReadReader<R: Read> {
    rd: R,
    buf: Vec<u8>,
}

impl<R: Read> ReadReader<R> {
    #[inline]
    fn new(rd: R) -> Self {
        Self {
            rd,
            buf: Vec::with_capacity(128),
        }
    }
}

impl<'de, R: Read> ReadSlice<'de> for ReadReader<R> {
    #[inline]
    fn read_slice<'a>(&'a mut self, len: usize) -> Result<Reference<'de, 'a, [u8]>, io::Error> {
        self.buf.clear();
        let read = self.rd.by_ref().take(len as u64).read_to_end(&mut self.buf)?;
        if read != len {
            return Err(io::ErrorKind::UnexpectedEof.into());
        }

        Ok(Reference::Copied(&self.buf[..]))
    }
}

impl<R: Read> Read for ReadReader<R> {
    #[inline]
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        self.rd.read(buf)
    }

    #[inline]
    fn read_exact(&mut self, buf: &mut [u8]) -> io::Result<()> {
        self.rd.read_exact(buf)
    }
}

/// Deserialize an instance of type `T` from an I/O stream of MessagePack.
///
/// # Errors
///
/// This conversion can fail if the structure of the Value does not match the structure expected
/// by `T`. It can also fail if the structure is correct but `T`'s implementation of `Deserialize`
/// decides that something is wrong with the data, for example required struct fields are missing.
#[inline]
pub fn from_read<R, T>(rd: R) -> Result<T, Error>
where R: Read,
      T: DeserializeOwned
{
    Deserialize::deserialize(&mut Deserializer::new(rd))
}
