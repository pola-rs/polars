use super::decode::VarIntProcessor;
use super::VarInt;

use futures::io::{AsyncRead, AsyncReadExt};

/// Like a [`VarIntReader`], but returns a future.
#[async_trait::async_trait]
pub trait VarIntAsyncReader {
    async fn read_varint_async<VI: VarInt>(&mut self) -> Result<VI, std::io::Error>;
}

#[async_trait::async_trait]
impl<AR: AsyncRead + Unpin + Send> VarIntAsyncReader for AR {
    async fn read_varint_async<VI: VarInt>(&mut self) -> std::io::Result<VI> {
        let mut buf = [0_u8; 1];
        let mut p = VarIntProcessor::new::<VI>();

        while !p.finished() {
            let read = self.read(&mut buf).await?;

            // EOF
            if read == 0 && p.i == 0 {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    "Reached EOF",
                ));
            }
            if read == 0 {
                break;
            }

            p.push(buf[0])?;
        }

        p.decode()
            .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::UnexpectedEof, "Reached EOF"))
    }
}
