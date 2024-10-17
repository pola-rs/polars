use std::convert::{From, TryFrom, TryInto};

use async_trait::async_trait;
#[cfg(feature = "async")]
use futures::io::{AsyncRead, AsyncReadExt};

use super::super::varint::VarIntAsyncReader;

use super::compact::{
    collection_u8_to_type, u8_to_type, COMPACT_PROTOCOL_ID, COMPACT_VERSION, COMPACT_VERSION_MASK,
};
use super::{
    TFieldIdentifier, TInputStreamProtocol, TListIdentifier, TMapIdentifier, TMessageIdentifier,
    TMessageType,
};
use super::{TSetIdentifier, TStructIdentifier, TType};
use crate::thrift::{Error, ProtocolError, ProtocolErrorKind, Result};

#[derive(Debug)]
pub struct TCompactInputStreamProtocol<R: Send> {
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

impl<R: VarIntAsyncReader + AsyncRead + Unpin + Send> TCompactInputStreamProtocol<R> {
    /// Create a `TCompactInputProtocol` that reads bytes from `reader`.
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

    async fn read_list_set_begin(&mut self) -> Result<(TType, u32)> {
        let header = self.read_byte().await?;
        let element_type = collection_u8_to_type(header & 0x0F)?;

        let possible_element_count = (header & 0xF0) >> 4;
        let element_count = if possible_element_count != 15 {
            // high bits set high if count and type encoded separately
            possible_element_count.into()
        } else {
            self.reader.read_varint_async::<u32>().await?
        };

        self.update_remaining::<usize>(element_count.try_into()?)?;

        Ok((element_type, element_count))
    }
}

#[async_trait]
impl<R: VarIntAsyncReader + AsyncRead + Unpin + Send> TInputStreamProtocol
    for TCompactInputStreamProtocol<R>
{
    async fn read_message_begin(&mut self) -> Result<TMessageIdentifier> {
        let compact_id = self.read_byte().await?;
        if compact_id != COMPACT_PROTOCOL_ID {
            Err(Error::Protocol(ProtocolError {
                kind: ProtocolErrorKind::BadVersion,
                message: format!("invalid compact protocol header {:?}", compact_id),
            }))
        } else {
            Ok(())
        }?;

        let type_and_byte = self.read_byte().await?;
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
        let sequence_number = self.reader.read_varint_async::<u32>().await?;
        let service_call_name = self.read_string().await?;

        self.last_read_field_id = 0;

        Ok(TMessageIdentifier::new(
            service_call_name,
            message_type,
            sequence_number,
        ))
    }

    async fn read_message_end(&mut self) -> Result<()> {
        Ok(())
    }

    async fn read_struct_begin(&mut self) -> Result<Option<TStructIdentifier>> {
        self.read_field_id_stack.push(self.last_read_field_id);
        self.last_read_field_id = 0;
        Ok(None)
    }

    async fn read_struct_end(&mut self) -> Result<()> {
        self.last_read_field_id = self
            .read_field_id_stack
            .pop()
            .expect("should have previous field ids");
        Ok(())
    }

    async fn read_field_begin(&mut self) -> Result<TFieldIdentifier> {
        // we can read at least one byte, which is:
        // - the type
        // - the field delta and the type
        let field_type = self.read_byte().await?;
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
                    self.last_read_field_id = self.read_i16().await?;
                };

                Ok(TFieldIdentifier {
                    name: None,
                    field_type,
                    id: Some(self.last_read_field_id),
                })
            }
        }
    }

    async fn read_field_end(&mut self) -> Result<()> {
        Ok(())
    }

    async fn read_bool(&mut self) -> Result<bool> {
        match self.pending_read_bool_value.take() {
            Some(b) => Ok(b),
            None => {
                let b = self.read_byte().await?;
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

    async fn read_bytes(&mut self) -> Result<Vec<u8>> {
        let len = self.reader.read_varint_async::<u32>().await?;

        self.update_remaining::<u8>(len.try_into()?)?;

        let mut buf = vec![];
        buf.try_reserve(len.try_into()?)?;
        (&mut self.reader)
            .take(len.try_into()?)
            .read_to_end(&mut buf)
            .await?;
        Ok(buf)
    }

    async fn read_i8(&mut self) -> Result<i8> {
        self.read_byte().await.map(|i| i as i8)
    }

    async fn read_i16(&mut self) -> Result<i16> {
        self.reader
            .read_varint_async::<i16>()
            .await
            .map_err(From::from)
    }

    async fn read_i32(&mut self) -> Result<i32> {
        self.reader
            .read_varint_async::<i32>()
            .await
            .map_err(From::from)
    }

    async fn read_i64(&mut self) -> Result<i64> {
        self.reader
            .read_varint_async::<i64>()
            .await
            .map_err(From::from)
    }

    async fn read_double(&mut self) -> Result<f64> {
        let mut buf = [0; 8];
        self.reader.read_exact(&mut buf).await?;
        let r = f64::from_le_bytes(buf);
        Ok(r)
    }

    async fn read_string(&mut self) -> Result<String> {
        let bytes = self.read_bytes().await?;
        String::from_utf8(bytes).map_err(From::from)
    }

    async fn read_list_begin(&mut self) -> Result<TListIdentifier> {
        let (element_type, element_count) = self.read_list_set_begin().await?;
        Ok(TListIdentifier::new(element_type, element_count))
    }

    async fn read_list_end(&mut self) -> Result<()> {
        Ok(())
    }

    async fn read_set_begin(&mut self) -> Result<TSetIdentifier> {
        let (element_type, element_count) = self.read_list_set_begin().await?;
        Ok(TSetIdentifier::new(element_type, element_count))
    }

    async fn read_set_end(&mut self) -> Result<()> {
        Ok(())
    }

    async fn read_map_begin(&mut self) -> Result<TMapIdentifier> {
        let element_count = self.reader.read_varint_async::<u32>().await?;
        if element_count == 0 {
            Ok(TMapIdentifier::new(None, None, 0))
        } else {
            let type_header = self.read_byte().await?;
            let key_type = collection_u8_to_type((type_header & 0xF0) >> 4)?;
            let val_type = collection_u8_to_type(type_header & 0x0F)?;
            self.update_remaining::<usize>(element_count.try_into()?)?;
            Ok(TMapIdentifier::new(key_type, val_type, element_count))
        }
    }

    async fn read_map_end(&mut self) -> Result<()> {
        Ok(())
    }

    // utility
    //

    async fn read_byte(&mut self) -> Result<u8> {
        let mut buf = [0u8; 1];
        self.reader
            .read_exact(&mut buf)
            .await
            .map_err(From::from)
            .map(|_| buf[0])
    }
}
