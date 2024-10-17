use std::convert::From;
use std::convert::TryInto;

use async_trait::async_trait;
#[cfg(feature = "async")]
use futures::io::{AsyncWrite, AsyncWriteExt};

use super::super::varint::VarIntAsyncWriter;

use crate::thrift::Result;

use super::compact::{COMPACT_PROTOCOL_ID, COMPACT_VERSION};
use super::compact_write::{collection_type_to_u8, type_to_u8};
use super::{
    TFieldIdentifier, TListIdentifier, TMapIdentifier, TMessageIdentifier, TOutputStreamProtocol,
    TSetIdentifier, TStructIdentifier, TType,
};

/// Write messages asyncronously using the Thrift compact protocol.
#[derive(Debug)]
pub struct TCompactOutputStreamProtocol<T>
where
    T: AsyncWrite + Unpin + Send,
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

impl<T> TCompactOutputStreamProtocol<T>
where
    T: VarIntAsyncWriter + AsyncWrite + Unpin + Send,
{
    /// Create a `TCompactOutputProtocol` that writes bytes to `transport`.
    pub fn new(transport: T) -> Self {
        Self {
            last_write_field_id: 0,
            write_field_id_stack: Vec::new(),
            pending_write_bool_field_identifier: None,
            transport,
        }
    }

    // FIXME: field_type as unconstrained u8 is bad
    async fn write_field_header(&mut self, field_type: u8, field_id: i16) -> Result<usize> {
        let mut written = 0;

        let field_delta = field_id - self.last_write_field_id;
        if field_delta > 0 && field_delta < 15 {
            written += self
                .write_byte(((field_delta as u8) << 4) | field_type)
                .await?;
        } else {
            written += self.write_byte(field_type).await?;
            written += self.write_i16(field_id).await?;
        }
        self.last_write_field_id = field_id;
        Ok(written)
    }

    async fn write_list_set_begin(
        &mut self,
        element_type: TType,
        element_count: u32,
    ) -> Result<usize> {
        let elem_identifier = collection_type_to_u8(element_type);
        if element_count <= 14 {
            let header = (element_count as u8) << 4 | elem_identifier;
            self.write_byte(header).await
        } else {
            let mut written = 0;

            let header = 0xF0 | elem_identifier;
            written += self.write_byte(header).await?;
            written += self.transport.write_varint_async(element_count).await?;
            Ok(written)
        }
    }

    fn assert_no_pending_bool_write(&self) {
        if let Some(ref f) = self.pending_write_bool_field_identifier {
            panic!("pending bool field {:?} not written", f)
        }
    }
}

#[async_trait]
impl<T> TOutputStreamProtocol for TCompactOutputStreamProtocol<T>
where
    T: VarIntAsyncWriter + AsyncWrite + Unpin + Send,
{
    async fn write_message_begin(&mut self, identifier: &TMessageIdentifier) -> Result<usize> {
        let mut written = 0;
        written += self.write_byte(COMPACT_PROTOCOL_ID).await?;
        written += self
            .write_byte((u8::from(identifier.message_type) << 5) | COMPACT_VERSION)
            .await?;
        written += self
            .transport
            .write_varint_async(identifier.sequence_number)
            .await?;
        written += self.write_string(&identifier.name).await?;
        Ok(written)
    }

    async fn write_message_end(&mut self) -> Result<usize> {
        self.assert_no_pending_bool_write();
        Ok(0)
    }

    async fn write_struct_begin(&mut self, _: &TStructIdentifier) -> Result<usize> {
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

    async fn write_field_begin(&mut self, identifier: &TFieldIdentifier) -> Result<usize> {
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
                self.write_field_header(field_type, field_id).await
            }
        }
    }

    fn write_field_end(&mut self) -> Result<usize> {
        self.assert_no_pending_bool_write();
        Ok(0)
    }

    async fn write_field_stop(&mut self) -> Result<usize> {
        self.assert_no_pending_bool_write();
        self.write_byte(type_to_u8(TType::Stop)).await
    }

    async fn write_bool(&mut self, b: bool) -> Result<usize> {
        match self.pending_write_bool_field_identifier.take() {
            Some(pending) => {
                let field_id = pending.id.expect("bool field should have a field id");
                let field_type_as_u8 = if b { 0x01 } else { 0x02 };
                self.write_field_header(field_type_as_u8, field_id).await
            }
            None => {
                if b {
                    self.write_byte(0x01).await
                } else {
                    self.write_byte(0x02).await
                }
            }
        }
    }

    async fn write_bytes(&mut self, b: &[u8]) -> Result<usize> {
        let mut written = self
            .transport
            .write_varint_async::<u32>(b.len().try_into()?)
            .await?;
        self.transport.write_all(b).await?;
        written += b.len();
        Ok(written)
    }

    async fn write_i8(&mut self, i: i8) -> Result<usize> {
        self.write_byte(i as u8).await
    }

    async fn write_i16(&mut self, i: i16) -> Result<usize> {
        self.transport
            .write_varint_async(i)
            .await
            .map_err(From::from)
    }

    async fn write_i32(&mut self, i: i32) -> Result<usize> {
        self.transport
            .write_varint_async(i)
            .await
            .map_err(From::from)
    }

    async fn write_i64(&mut self, i: i64) -> Result<usize> {
        self.transport
            .write_varint_async(i)
            .await
            .map_err(From::from)
    }

    async fn write_double(&mut self, d: f64) -> Result<usize> {
        let r = d.to_le_bytes();
        self.transport.write_all(&r).await?;
        Ok(r.len())
    }

    async fn write_string(&mut self, s: &str) -> Result<usize> {
        self.write_bytes(s.as_bytes()).await
    }

    async fn write_list_begin(&mut self, identifier: &TListIdentifier) -> Result<usize> {
        self.write_list_set_begin(identifier.element_type, identifier.size)
            .await
    }

    async fn write_list_end(&mut self) -> Result<usize> {
        Ok(0)
    }

    async fn write_set_begin(&mut self, identifier: &TSetIdentifier) -> Result<usize> {
        self.write_list_set_begin(identifier.element_type, identifier.size)
            .await
    }

    async fn write_set_end(&mut self) -> Result<usize> {
        Ok(0)
    }

    async fn write_map_begin(&mut self, identifier: &TMapIdentifier) -> Result<usize> {
        if identifier.size == 0 {
            self.write_byte(0).await
        } else {
            self.transport.write_varint_async(identifier.size).await?;

            let key_type = identifier
                .key_type
                .expect("map identifier to write should contain key type");
            let key_type_byte = collection_type_to_u8(key_type) << 4;

            let val_type = identifier
                .value_type
                .expect("map identifier to write should contain value type");
            let val_type_byte = collection_type_to_u8(val_type);

            let map_type_header = key_type_byte | val_type_byte;
            self.write_byte(map_type_header).await
        }
    }

    async fn write_map_end(&mut self) -> Result<usize> {
        Ok(0)
    }

    async fn flush(&mut self) -> Result<()> {
        self.transport.flush().await.map_err(From::from)
    }

    // utility
    //

    async fn write_byte(&mut self, b: u8) -> Result<usize> {
        self.transport.write(&[b]).await.map_err(From::from)
    }
}
