use std::io::Result;

use futures::io::{AsyncWrite, AsyncWriteExt};

use super::VarInt;

/// Like VarIntWriter, but asynchronous.
#[async_trait::async_trait]
pub trait VarIntAsyncWriter {
    /// Write a VarInt integer to an asynchronous writer.
    async fn write_varint_async<VI: VarInt + Send>(&mut self, n: VI) -> Result<usize>;
}

#[async_trait::async_trait]
impl<AW: AsyncWrite + Send + Unpin> VarIntAsyncWriter for AW {
    async fn write_varint_async<VI: VarInt + Send>(&mut self, n: VI) -> Result<usize> {
        let mut buf = [0_u8; 10];
        let b = n.encode_var(&mut buf);
        self.write_all(&buf[0..b]).await?;
        Ok(b)
    }
}
