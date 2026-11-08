use polars_buffer::Buffer;

pub trait WriteBytesOwned {
    fn write_all_owned(&mut self, bytes: &Buffer<u8>) -> std::io::Result<()>;
    fn write_all(&mut self, bytes: &[u8]) -> std::io::Result<()>;
    fn len(&self) -> usize;
}
