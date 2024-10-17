// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied. See the License for the
// specific language governing permissions and limitations
// under the License.

use async_trait::async_trait;

use crate::thrift::{Error, ProtocolError, ProtocolErrorKind, Result};

use super::*;

#[async_trait]
pub trait AsyncReadThrift: Send + Sized {
    async fn stream_from_in_protocol<T: TInputStreamProtocol>(
        i_prot: &mut T,
    ) -> crate::thrift::Result<Self>;
}

#[async_trait]
impl AsyncReadThrift for String {
    async fn stream_from_in_protocol<T: TInputStreamProtocol>(
        i_prot: &mut T,
    ) -> crate::thrift::Result<Self> {
        i_prot.read_string().await
    }
}

#[async_trait]
impl AsyncReadThrift for bool {
    async fn stream_from_in_protocol<T: TInputStreamProtocol>(
        i_prot: &mut T,
    ) -> crate::thrift::Result<Self> {
        i_prot.read_bool().await
    }
}

#[async_trait]
impl AsyncReadThrift for u8 {
    async fn stream_from_in_protocol<T: TInputStreamProtocol>(
        i_prot: &mut T,
    ) -> crate::thrift::Result<Self> {
        i_prot.read_byte().await
    }
}

#[async_trait]
impl AsyncReadThrift for i64 {
    async fn stream_from_in_protocol<T: TInputStreamProtocol>(
        i_prot: &mut T,
    ) -> crate::thrift::Result<Self> {
        i_prot.read_i64().await
    }
}

#[async_trait]
impl AsyncReadThrift for Vec<u8> {
    async fn stream_from_in_protocol<T: TInputStreamProtocol>(
        i_prot: &mut T,
    ) -> crate::thrift::Result<Self> {
        i_prot.read_bytes().await
    }
}

#[async_trait]
pub trait TInputStreamProtocol: Send + Sized {
    /// Read the beginning of a Thrift message.
    async fn read_message_begin(&mut self) -> Result<TMessageIdentifier>;
    /// Read the end of a Thrift message.
    async fn read_message_end(&mut self) -> Result<()>;
    /// Read the beginning of a Thrift struct.
    async fn read_struct_begin(&mut self) -> Result<Option<TStructIdentifier>>;
    /// Read the end of a Thrift struct.
    async fn read_struct_end(&mut self) -> Result<()>;
    /// Read the beginning of a Thrift struct field.
    async fn read_field_begin(&mut self) -> Result<TFieldIdentifier>;
    /// Read the end of a Thrift struct field.
    async fn read_field_end(&mut self) -> Result<()>;
    /// Read a bool.
    async fn read_bool(&mut self) -> Result<bool>;
    /// Read a fixed-length byte array.
    async fn read_bytes(&mut self) -> Result<Vec<u8>>;
    /// Read a word.
    async fn read_i8(&mut self) -> Result<i8>;
    /// Read a 16-bit signed integer.
    async fn read_i16(&mut self) -> Result<i16>;
    /// Read a 32-bit signed integer.
    async fn read_i32(&mut self) -> Result<i32>;
    /// Read a 64-bit signed integer.
    async fn read_i64(&mut self) -> Result<i64>;
    /// Read a 64-bit float.
    async fn read_double(&mut self) -> Result<f64>;
    /// Read a fixed-length string (not null terminated).
    async fn read_string(&mut self) -> Result<String>;
    /// Read the beginning of a list.
    async fn read_list_begin(&mut self) -> Result<TListIdentifier>;
    /// Read the end of a list.
    async fn read_list_end(&mut self) -> Result<()>;
    /// Read the beginning of a set.
    async fn read_set_begin(&mut self) -> Result<TSetIdentifier>;
    /// Read the end of a set.
    async fn read_set_end(&mut self) -> Result<()>;
    /// Read the beginning of a map.
    async fn read_map_begin(&mut self) -> Result<TMapIdentifier>;
    /// Read the end of a map.
    async fn read_map_end(&mut self) -> Result<()>;

    async fn read_list<P: AsyncReadThrift>(&mut self) -> crate::thrift::Result<Vec<P>> {
        let list_ident = self.read_list_begin().await?;
        let mut val: Vec<P> = Vec::with_capacity(list_ident.size as usize);
        for _ in 0..list_ident.size {
            val.push(P::stream_from_in_protocol(self).await?);
        }
        self.read_list_end().await?;
        Ok(val)
    }

    /// Skip a field with type `field_type` recursively until the default
    /// maximum skip depth is reached.
    async fn skip(&mut self, field_type: TType) -> Result<()> {
        self.skip_till_depth(field_type, MAXIMUM_SKIP_DEPTH).await
    }

    /// Skip a field with type `field_type` recursively up to `depth` levels.
    async fn skip_till_depth(&mut self, field_type: TType, depth: i8) -> Result<()> {
        if depth == 0 {
            return Err(Error::Protocol(ProtocolError {
                kind: ProtocolErrorKind::DepthLimit,
                message: format!("cannot parse past {:?}", field_type),
            }));
        }

        match field_type {
            TType::Bool => self.read_bool().await.map(|_| ()),
            TType::I08 => self.read_i8().await.map(|_| ()),
            TType::I16 => self.read_i16().await.map(|_| ()),
            TType::I32 => self.read_i32().await.map(|_| ()),
            TType::I64 => self.read_i64().await.map(|_| ()),
            TType::Double => self.read_double().await.map(|_| ()),
            TType::String => self.read_string().await.map(|_| ()),
            TType::Struct => {
                self.read_struct_begin().await?;
                loop {
                    let field_ident = self.read_field_begin().await?;
                    if field_ident.field_type == TType::Stop {
                        break;
                    }
                    self.skip_till_depth(field_ident.field_type, depth - 1)
                        .await?;
                }
                self.read_struct_end().await
            }
            TType::List => {
                let list_ident = self.read_list_begin().await?;
                for _ in 0..list_ident.size {
                    self.skip_till_depth(list_ident.element_type, depth - 1)
                        .await?;
                }
                self.read_list_end().await
            }
            TType::Set => {
                let set_ident = self.read_set_begin().await?;
                for _ in 0..set_ident.size {
                    self.skip_till_depth(set_ident.element_type, depth - 1)
                        .await?;
                }
                self.read_set_end().await
            }
            TType::Map => {
                let map_ident = self.read_map_begin().await?;
                for _ in 0..map_ident.size {
                    let key_type = map_ident
                        .key_type
                        .expect("non-zero sized map should contain key type");
                    let val_type = map_ident
                        .value_type
                        .expect("non-zero sized map should contain value type");
                    self.skip_till_depth(key_type, depth - 1).await?;
                    self.skip_till_depth(val_type, depth - 1).await?;
                }
                self.read_map_end().await
            }
            u => Err(Error::Protocol(ProtocolError {
                kind: ProtocolErrorKind::Unknown,
                message: format!("cannot skip field type {:?}", &u),
            })),
        }
    }

    // utility (DO NOT USE IN GENERATED CODE!!!!)
    //

    /// Read an unsigned byte.
    ///
    /// This method should **never** be used in generated code.
    async fn read_byte(&mut self) -> Result<u8>;
}

#[async_trait]
pub trait TOutputStreamProtocol: Send {
    /// Write the beginning of a Thrift message.
    async fn write_message_begin(&mut self, identifier: &TMessageIdentifier) -> Result<usize>;
    /// Write the end of a Thrift message.
    async fn write_message_end(&mut self) -> Result<usize>;
    /// Write the beginning of a Thrift struct.
    async fn write_struct_begin(&mut self, identifier: &TStructIdentifier) -> Result<usize>;
    /// Write the end of a Thrift struct.
    fn write_struct_end(&mut self) -> Result<usize>;
    /// Write the beginning of a Thrift field.
    async fn write_field_begin(&mut self, identifier: &TFieldIdentifier) -> Result<usize>;
    /// Write the end of a Thrift field.
    fn write_field_end(&mut self) -> Result<usize>;
    /// Write a STOP field indicating that all the fields in a struct have been
    /// written.
    async fn write_field_stop(&mut self) -> Result<usize>;
    /// Write a bool.
    async fn write_bool(&mut self, b: bool) -> Result<usize>;
    /// Write a fixed-length byte array.
    async fn write_bytes(&mut self, b: &[u8]) -> Result<usize>;
    /// Write an 8-bit signed integer.
    async fn write_i8(&mut self, i: i8) -> Result<usize>;
    /// Write a 16-bit signed integer.
    async fn write_i16(&mut self, i: i16) -> Result<usize>;
    /// Write a 32-bit signed integer.
    async fn write_i32(&mut self, i: i32) -> Result<usize>;
    /// Write a 64-bit signed integer.
    async fn write_i64(&mut self, i: i64) -> Result<usize>;
    /// Write a 64-bit float.
    async fn write_double(&mut self, d: f64) -> Result<usize>;
    /// Write a fixed-length string.
    async fn write_string(&mut self, s: &str) -> Result<usize>;
    /// Write the beginning of a list.
    async fn write_list_begin(&mut self, identifier: &TListIdentifier) -> Result<usize>;
    /// Write the end of a list.
    async fn write_list_end(&mut self) -> Result<usize>;
    /// Write the beginning of a set.
    async fn write_set_begin(&mut self, identifier: &TSetIdentifier) -> Result<usize>;
    /// Write the end of a set.
    async fn write_set_end(&mut self) -> Result<usize>;
    /// Write the beginning of a map.
    async fn write_map_begin(&mut self, identifier: &TMapIdentifier) -> Result<usize>;
    /// Write the end of a map.
    async fn write_map_end(&mut self) -> Result<usize>;
    /// Flush buffered bytes to the underlying transport.
    async fn flush(&mut self) -> Result<()>;

    // utility (DO NOT USE IN GENERATED CODE!!!!)
    //

    /// Write an unsigned byte.
    ///
    /// This method should **never** be used in generated code.
    async fn write_byte(&mut self, b: u8) -> Result<usize>; // FIXME: REMOVE
}
