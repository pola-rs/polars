use std::convert::{From, TryFrom, TryInto};
use std::io;
use std::io::Read;

use super::super::varint::VarIntReader;

use super::super::{Error, ProtocolError, ProtocolErrorKind, Result};
use super::{
    TFieldIdentifier, TInputProtocol, TListIdentifier, TMapIdentifier, TMessageIdentifier,
    TMessageType,
};
use super::{TSetIdentifier, TStructIdentifier, TType};

pub(super) const COMPACT_PROTOCOL_ID: u8 = 0x82;
pub(super) const COMPACT_VERSION: u8 = 0x01;
pub(super) const COMPACT_VERSION_MASK: u8 = 0x1F;

/// Read messages encoded in the Thrift compact protocol.
#[derive(Debug)]
pub struct TCompactInputProtocol<R>
where
    R: Read,
{
    // Identifier of the last field deserialized for a struct.
    last_read_field_id: i16,
    // Stack of the last read field ids (a new entry is added each time a nested struct is read).
    read_field_id_stack: Vec<i16>,
    // Boolean value for a field.
    // Saved because boolean fields and their value are encoded in a single byte,
    // and reading the field only occurs after the field id is read.
    pending_read_bool_value: Option<bool>,
    // Underlying reader used for byte-level operations.
    reader: R,
    // remaining bytes that can be read before refusing to read more
    remaining: usize,
}

impl<R> TCompactInputProtocol<R>
where
    R: Read,
{
    /// Create a [`TCompactInputProtocol`] that reads bytes from `reader`.

    pub fn new(reader: R, max_bytes: usize) -> Self {
        Self {
            last_read_field_id: 0,
            read_field_id_stack: Vec::new(),
            pending_read_bool_value: None,
            remaining: max_bytes,
            reader,
        }
    }

    fn update_remaining<T>(&mut self, element: usize) -> Result<()> {
        self.remaining = self
            .remaining
            .checked_sub((element).saturating_mul(std::mem::size_of::<T>()))
            .ok_or_else(|| {
                Error::Protocol(ProtocolError {
                    kind: ProtocolErrorKind::SizeLimit,
                    message: "The thrift file would allocate more bytes than allowed".to_string(),
                })
            })?;
        Ok(())
    }

    fn read_list_set_begin(&mut self) -> Result<(TType, u32)> {
        let header = self.read_byte()?;
        let element_type = collection_u8_to_type(header & 0x0F)?;

        let possible_element_count = (header & 0xF0) >> 4;
        let element_count = if possible_element_count != 15 {
            // high bits set high if count and type encoded separately
            possible_element_count.into()
        } else {
            self.reader.read_varint::<u32>()?
        };
        self.update_remaining::<usize>(element_count as usize)?;

        Ok((element_type, element_count))
    }
}

impl<R> TInputProtocol for TCompactInputProtocol<R>
where
    R: Read,
{
    fn read_message_begin(&mut self) -> Result<TMessageIdentifier> {
        let compact_id = self.read_byte()?;
        if compact_id != COMPACT_PROTOCOL_ID {
            Err(Error::Protocol(ProtocolError {
                kind: ProtocolErrorKind::BadVersion,
                message: format!("invalid compact protocol header {:?}", compact_id),
            }))
        } else {
            Ok(())
        }?;

        let type_and_byte = self.read_byte()?;
        let received_version = type_and_byte & COMPACT_VERSION_MASK;
        if received_version != COMPACT_VERSION {
            Err(Error::Protocol(ProtocolError {
                kind: ProtocolErrorKind::BadVersion,
                message: format!(
                    "cannot process compact protocol version {:?}",
                    received_version
                ),
            }))
        } else {
            Ok(())
        }?;

        // NOTE: unsigned right shift will pad with 0s
        let message_type: TMessageType = TMessageType::try_from(type_and_byte >> 5)?;
        let sequence_number = self.reader.read_varint::<u32>()?;
        let service_call_name = self.read_string()?;

        self.last_read_field_id = 0;

        Ok(TMessageIdentifier::new(
            service_call_name,
            message_type,
            sequence_number,
        ))
    }

    fn read_message_end(&mut self) -> Result<()> {
        Ok(())
    }

    fn read_struct_begin(&mut self) -> Result<Option<TStructIdentifier>> {
        self.update_remaining::<i16>(1)?;
        self.read_field_id_stack.push(self.last_read_field_id);
        self.last_read_field_id = 0;
        Ok(None)
    }

    fn read_struct_end(&mut self) -> Result<()> {
        self.last_read_field_id = self
            .read_field_id_stack
            .pop()
            .expect("should have previous field ids");
        Ok(())
    }

    fn read_field_begin(&mut self) -> Result<TFieldIdentifier> {
        // we can read at least one byte, which is:
        // - the type
        // - the field delta and the type
        let field_type = self.read_byte()?;
        let field_delta = (field_type & 0xF0) >> 4;
        let field_type = match field_type & 0x0F {
            0x01 => {
                self.pending_read_bool_value = Some(true);
                Ok(TType::Bool)
            }
            0x02 => {
                self.pending_read_bool_value = Some(false);
                Ok(TType::Bool)
            }
            ttu8 => u8_to_type(ttu8),
        }?;

        match field_type {
            TType::Stop => Ok(
                TFieldIdentifier::new::<Option<String>, String, Option<i16>>(
                    None,
                    TType::Stop,
                    None,
                ),
            ),
            _ => {
                if field_delta != 0 {
                    self.last_read_field_id = self
                        .last_read_field_id
                        .checked_add(field_delta as i16)
                        .ok_or(Error::Protocol(ProtocolError {
                            kind: ProtocolErrorKind::DepthLimit,
                            message: String::new(),
                        }))?;
                } else {
                    self.last_read_field_id = self.read_i16()?;
                };

                Ok(TFieldIdentifier {
                    name: None,
                    field_type,
                    id: Some(self.last_read_field_id),
                })
            }
        }
    }

    fn read_field_end(&mut self) -> Result<()> {
        Ok(())
    }

    fn read_bool(&mut self) -> Result<bool> {
        match self.pending_read_bool_value.take() {
            Some(b) => Ok(b),
            None => {
                let b = self.read_byte()?;
                match b {
                    0x01 => Ok(true),
                    0x02 => Ok(false),
                    unkn => Err(Error::Protocol(ProtocolError {
                        kind: ProtocolErrorKind::InvalidData,
                        message: format!("cannot convert {} into bool", unkn),
                    })),
                }
            }
        }
    }

    fn read_bytes(&mut self) -> Result<Vec<u8>> {
        let len = self.reader.read_varint::<u32>()?;

        self.update_remaining::<u8>(len.try_into()?)?;

        let mut buf = vec![];
        buf.try_reserve(len.try_into()?)?;
        self.reader
            .by_ref()
            .take(len.try_into()?)
            .read_to_end(&mut buf)?;
        Ok(buf)
    }

    fn read_i8(&mut self) -> Result<i8> {
        self.read_byte().map(|i| i as i8)
    }

    fn read_i16(&mut self) -> Result<i16> {
        self.reader.read_varint::<i16>().map_err(From::from)
    }

    fn read_i32(&mut self) -> Result<i32> {
        self.reader.read_varint::<i32>().map_err(From::from)
    }

    fn read_i64(&mut self) -> Result<i64> {
        self.reader.read_varint::<i64>().map_err(From::from)
    }

    fn read_double(&mut self) -> Result<f64> {
        let mut data = [0u8; 8];
        self.reader.read_exact(&mut data)?;
        Ok(f64::from_le_bytes(data))
    }

    fn read_string(&mut self) -> Result<String> {
        let bytes = self.read_bytes()?;
        String::from_utf8(bytes).map_err(From::from)
    }

    fn read_list_begin(&mut self) -> Result<TListIdentifier> {
        let (element_type, element_count) = self.read_list_set_begin()?;
        Ok(TListIdentifier::new(element_type, element_count))
    }

    fn read_list_end(&mut self) -> Result<()> {
        Ok(())
    }

    fn read_set_begin(&mut self) -> Result<TSetIdentifier> {
        let (element_type, element_count) = self.read_list_set_begin()?;
        Ok(TSetIdentifier::new(element_type, element_count))
    }

    fn read_set_end(&mut self) -> Result<()> {
        Ok(())
    }

    fn read_map_begin(&mut self) -> Result<TMapIdentifier> {
        let element_count = self.reader.read_varint::<u32>()?;
        if element_count == 0 {
            Ok(TMapIdentifier::new(None, None, 0))
        } else {
            let type_header = self.read_byte()?;
            let key_type = collection_u8_to_type((type_header & 0xF0) >> 4)?;
            let val_type = collection_u8_to_type(type_header & 0x0F)?;
            self.update_remaining::<usize>(element_count.try_into()?)?;
            Ok(TMapIdentifier::new(key_type, val_type, element_count))
        }
    }

    fn read_map_end(&mut self) -> Result<()> {
        Ok(())
    }

    // utility
    //

    fn read_byte(&mut self) -> Result<u8> {
        let mut buf = [0u8; 1];
        self.reader
            .read_exact(&mut buf)
            .map_err(From::from)
            .map(|_| buf[0])
    }
}

impl<R> io::Seek for TCompactInputProtocol<R>
where
    R: io::Seek + Read,
{
    fn seek(&mut self, pos: io::SeekFrom) -> io::Result<u64> {
        self.reader.seek(pos)
    }
}

pub(super) fn collection_u8_to_type(b: u8) -> Result<TType> {
    match b {
        0x01 => Ok(TType::Bool),
        o => u8_to_type(o),
    }
}

pub(super) fn u8_to_type(b: u8) -> Result<TType> {
    match b {
        0x00 => Ok(TType::Stop),
        0x03 => Ok(TType::I08), // equivalent to TType::Byte
        0x04 => Ok(TType::I16),
        0x05 => Ok(TType::I32),
        0x06 => Ok(TType::I64),
        0x07 => Ok(TType::Double),
        0x08 => Ok(TType::String),
        0x09 => Ok(TType::List),
        0x0A => Ok(TType::Set),
        0x0B => Ok(TType::Map),
        0x0C => Ok(TType::Struct),
        unkn => Err(Error::Protocol(ProtocolError {
            kind: ProtocolErrorKind::InvalidData,
            message: format!("cannot convert {} into TType", unkn),
        })),
    }
}
