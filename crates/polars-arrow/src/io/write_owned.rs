use polars_buffer::Buffer;

pub trait WriteBytesOwned {
    fn write_all_owned(&mut self, bytes: &Buffer<u8>) -> std::io::Result<()>;
    fn write_all(&mut self, bytes: &[u8]) -> std::io::Result<()>;
    fn len(&self) -> usize;
}

impl<'a> dyn WriteBytesOwned + 'a {
    pub fn as_io_write(&mut self) -> &mut IoWriteWrap {
        unsafe {
            // Safety: IoWriteWrap is repr(transparent) on `dyn WriteBytesOwned`
            std::mem::transmute::<&mut dyn WriteBytesOwned, &mut IoWriteWrap>(self)
        }
    }
}

#[repr(transparent)]
pub struct IoWriteWrap(dyn WriteBytesOwned);

impl std::io::Write for IoWriteWrap {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0.write_all(buf)?;
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}
