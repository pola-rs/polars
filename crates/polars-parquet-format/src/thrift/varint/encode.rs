use std::io::{Result, Write};

use super::VarInt;

/// A trait for writing integers in VarInt encoding to any `Write` type. This packs encoding and
/// writing into one step.
pub trait VarIntWriter {
    fn write_varint<VI: VarInt>(&mut self, n: VI) -> Result<usize>;
}

impl<Inner: Write> VarIntWriter for Inner {
    fn write_varint<VI: VarInt>(&mut self, n: VI) -> Result<usize> {
        let mut buf = [0_u8; 10];
        let used = n.encode_var(&mut buf[..]);

        self.write_all(&buf[0..used])?;
        Ok(used)
    }
}
