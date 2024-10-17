use std::convert::From;
use std::convert::TryInto;
use std::io::Write;

use super::super::varint::VarIntWriter;

use super::super::Result;
use super::{TFieldIdentifier, TListIdentifier, TMapIdentifier, TMessageIdentifier};
use super::{TOutputProtocol, TSetIdentifier, TStructIdentifier, TType};

use super::compact::{COMPACT_PROTOCOL_ID, COMPACT_VERSION};

/// Write messages using the Thrift compact protocol.
#[derive(Debug)]
pub struct TCompactOutputProtocol<T>
where
    T: Write,
{
    // Identifier of the last field serialized for a struct.
    last_write_field_id: i16,
    // Stack of the last written field ids (new entry added each time a nested struct is written).
    write_field_id_stack: Vec<i16>,
    // Field identifier of the boolean field to be written.
    // Saved because boolean fields and their value are encoded in a single byte
    pending_write_bool_field_identifier: Option<TFieldIdentifier>,
    // Underlying transport used for byte-level operations.
    transport: T,
}

impl<T> TCompactOutputProtocol<T>
where
    T: Write,
{
    /// Create a `TCompactOutputProtocol` that writes bytes to `transport`.
    pub fn new(transport: T) -> TCompactOutputProtocol<T> {
        TCompactOutputProtocol {
            last_write_field_id: 0,
            write_field_id_stack: Vec::new(),
            pending_write_bool_field_identifier: None,
            transport,
        }
    }

    // FIXME: field_type as unconstrained u8 is bad
    fn write_field_header(&mut self, field_type: u8, field_id: i16) -> Result<usize> {
        let mut written = 0;

        let field_delta = field_id - self.last_write_field_id;
        if field_delta > 0 && field_delta < 15 {
            written += self.write_byte(((field_delta as u8) << 4) | field_type)?;
        } else {
            written += self.write_byte(field_type)?;
            written += self.write_i16(field_id)?;
        }
        self.last_write_field_id = field_id;
        Ok(written)
    }

    fn write_list_set_begin(&mut self, element_type: TType, element_count: u32) -> Result<usize> {
        let mut written = 0;

        let elem_identifier = collection_type_to_u8(element_type);
        if element_count <= 14 {
            let header = (element_count as u8) << 4 | elem_identifier;
            written += self.write_byte(header)?;
        } else {
            let header = 0xF0 | elem_identifier;
            written += self.write_byte(header)?;
            written += self.transport.write_varint(element_count)?;
        }
        Ok(written)
    }

    fn assert_no_pending_bool_write(&self) {
        if let Some(ref f) = self.pending_write_bool_field_identifier {
            panic!("pending bool field {:?} not written", f)
        }
    }
}

impl<T> TOutputProtocol for TCompactOutputProtocol<T>
where
    T: Write,
{
    fn write_message_begin(&mut self, identifier: &TMessageIdentifier) -> Result<usize> {
        let mut written = 0;
        written += self.write_byte(COMPACT_PROTOCOL_ID)?;
        written += self.write_byte((u8::from(identifier.message_type) << 5) | COMPACT_VERSION)?;
        written += self.transport.write_varint(identifier.sequence_number)?;
        written += self.write_string(&identifier.name)?;
        Ok(written)
    }

    fn write_message_end(&mut self) -> Result<usize> {
        self.assert_no_pending_bool_write();
        Ok(0)
    }

    fn write_struct_begin(&mut self, _: &TStructIdentifier) -> Result<usize> {
        self.write_field_id_stack.push(self.last_write_field_id);
        self.last_write_field_id = 0;
        Ok(0)
    }

    fn write_struct_end(&mut self) -> Result<usize> {
        self.assert_no_pending_bool_write();
        self.last_write_field_id = self
            .write_field_id_stack
            .pop()
            .expect("should have previous field ids");
        Ok(0)
    }

    fn write_field_begin(&mut self, identifier: &TFieldIdentifier) -> Result<usize> {
        match identifier.field_type {
            TType::Bool => {
                if self.pending_write_bool_field_identifier.is_some() {
                    panic!(
                        "should not have a pending bool while writing another bool with id: \
                         {:?}",
                        identifier
                    )
                }
                self.pending_write_bool_field_identifier = Some(identifier.clone());
                Ok(0)
            }
            _ => {
                let field_type = type_to_u8(identifier.field_type);
                let field_id = identifier.id.expect("non-stop field should have field id");
                self.write_field_header(field_type, field_id)
            }
        }
    }

    fn write_field_end(&mut self) -> Result<usize> {
        self.assert_no_pending_bool_write();
        Ok(0)
    }

    fn write_field_stop(&mut self) -> Result<usize> {
        self.assert_no_pending_bool_write();
        self.write_byte(type_to_u8(TType::Stop))
    }

    fn write_bool(&mut self, b: bool) -> Result<usize> {
        match self.pending_write_bool_field_identifier.take() {
            Some(pending) => {
                let field_id = pending.id.expect("bool field should have a field id");
                let field_type_as_u8 = if b { 0x01 } else { 0x02 };
                self.write_field_header(field_type_as_u8, field_id)
            }
            None => {
                if b {
                    self.write_byte(0x01)
                } else {
                    self.write_byte(0x02)
                }
            }
        }
    }

    fn write_bytes(&mut self, b: &[u8]) -> Result<usize> {
        let mut written = 0;
        written += self.transport.write_varint::<u32>(b.len().try_into()?)?;
        self.transport.write_all(b)?;
        written += b.len();
        Ok(written)
    }

    fn write_i8(&mut self, i: i8) -> Result<usize> {
        self.write_byte(i as u8)
    }

    fn write_i16(&mut self, i: i16) -> Result<usize> {
        self.transport.write_varint(i).map_err(From::from)
    }

    fn write_i32(&mut self, i: i32) -> Result<usize> {
        self.transport.write_varint(i).map_err(From::from)
    }

    fn write_i64(&mut self, i: i64) -> Result<usize> {
        self.transport.write_varint(i).map_err(From::from)
    }

    fn write_double(&mut self, d: f64) -> Result<usize> {
        let bytes = d.to_le_bytes();
        self.transport.write_all(&bytes)?;
        Ok(8)
    }

    fn write_string(&mut self, s: &str) -> Result<usize> {
        self.write_bytes(s.as_bytes())
    }

    fn write_list_begin(&mut self, identifier: &TListIdentifier) -> Result<usize> {
        self.write_list_set_begin(identifier.element_type, identifier.size)
    }

    fn write_list_end(&mut self) -> Result<usize> {
        Ok(0)
    }

    fn write_set_begin(&mut self, identifier: &TSetIdentifier) -> Result<usize> {
        self.write_list_set_begin(identifier.element_type, identifier.size)
    }

    fn write_set_end(&mut self) -> Result<usize> {
        Ok(0)
    }

    fn write_map_begin(&mut self, identifier: &TMapIdentifier) -> Result<usize> {
        if identifier.size == 0 {
            self.write_byte(0)
        } else {
            let mut written = 0;
            written += self.transport.write_varint(identifier.size)?;

            let key_type = identifier
                .key_type
                .expect("map identifier to write should contain key type");
            let key_type_byte = collection_type_to_u8(key_type) << 4;

            let val_type = identifier
                .value_type
                .expect("map identifier to write should contain value type");
            let val_type_byte = collection_type_to_u8(val_type);

            let map_type_header = key_type_byte | val_type_byte;
            written += self.write_byte(map_type_header)?;
            Ok(written)
        }
    }

    fn write_map_end(&mut self) -> Result<usize> {
        Ok(0)
    }

    fn flush(&mut self) -> Result<()> {
        self.transport.flush().map_err(From::from)
    }

    // utility
    //

    fn write_byte(&mut self, b: u8) -> Result<usize> {
        self.transport.write(&[b]).map_err(From::from)
    }
}

pub(super) fn collection_type_to_u8(field_type: TType) -> u8 {
    match field_type {
        TType::Bool => 0x01,
        f => type_to_u8(f),
    }
}

pub(super) fn type_to_u8(field_type: TType) -> u8 {
    match field_type {
        TType::Stop => 0x00,
        TType::I08 => 0x03, // equivalent to TType::Byte
        TType::I16 => 0x04,
        TType::I32 => 0x05,
        TType::I64 => 0x06,
        TType::Double => 0x07,
        TType::String => 0x08,
        TType::List => 0x09,
        TType::Set => 0x0A,
        TType::Map => 0x0B,
        TType::Struct => 0x0C,
        _ => panic!("should not have attempted to convert {} to u8", field_type),
    }
}
