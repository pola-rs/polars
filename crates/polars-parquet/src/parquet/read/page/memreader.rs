use std::io;
use std::ops::Deref;
use std::sync::Arc;

/// A cursor over a segment of heap allocated memory. This is used for the Parquet reader to avoid
/// sequential allocations.
#[derive(Debug, Clone)]
pub struct MemReader {
    data: Arc<[u8]>,
    position: usize,
}

/// A reference to a slice of a memory reader.
///
/// This should not outlast the original the original [`MemReader`] because it still owns all the
/// memory.
#[derive(Debug, Clone)]
pub struct MemReaderSlice {
    data: Arc<[u8]>,
    start: usize,
    end: usize,
}

impl Default for MemReaderSlice {
    fn default() -> Self {
        let slice: &[u8] = &[];
        Self {
            data: Arc::from(slice),
            start: 0,
            end: 0,
        }
    }
}

impl Deref for MemReaderSlice {
    type Target = [u8];

    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        &self.data[self.start..self.end]
    }
}

#[derive(Debug, Clone)]
pub enum CowBuffer {
    Borrowed(MemReaderSlice),
    Owned(Vec<u8>),
}

impl Deref for CowBuffer {
    type Target = [u8];

    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        match self {
            CowBuffer::Borrowed(v) => v.deref(),
            CowBuffer::Owned(v) => v.deref(),
        }
    }
}

impl MemReader {
    #[inline(always)]
    pub fn new(data: Arc<[u8]>) -> Self {
        Self { data, position: 0 }
    }

    #[inline(always)]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    #[inline(always)]
    pub fn remaining_len(&self) -> usize {
        self.data.len() - self.position
    }

    #[inline(always)]
    pub fn position(&self) -> usize {
        self.position
    }

    #[inline(always)]
    pub fn from_slice(data: &[u8]) -> Self {
        let data = data.into();
        Self { data, position: 0 }
    }

    #[inline(always)]
    pub fn from_vec(data: Vec<u8>) -> Self {
        let data = data.into_boxed_slice().into();
        Self { data, position: 0 }
    }

    #[inline(always)]
    pub fn from_reader<R: io::Read>(mut reader: R) -> io::Result<Self> {
        let mut vec = Vec::new();
        reader.read_to_end(&mut vec)?;
        Ok(Self::from_vec(vec))
    }

    #[inline(always)]
    pub fn read_slice(&mut self, n: usize) -> MemReaderSlice {
        let start = self.position;
        let end = usize::min(self.position + n, self.data.len());

        self.position = end;

        MemReaderSlice {
            data: self.data.clone(),
            start,
            end,
        }
    }
}

impl io::Read for MemReader {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let n = usize::min(buf.len(), self.remaining_len());
        buf[..n].copy_from_slice(&self.data[self.position..self.position + n]);
        self.position += n;
        Ok(n)
    }
}

impl io::Seek for MemReader {
    fn seek(&mut self, pos: io::SeekFrom) -> io::Result<u64> {
        let position = match pos {
            io::SeekFrom::Start(position) => usize::min(position as usize, self.len()),
            io::SeekFrom::End(offset) => {
                let Some(position) = self.len().checked_add_signed(offset as isize) else {
                    return Err(io::Error::new(
                        io::ErrorKind::Other,
                        "Seek before to before buffer",
                    ));
                };

                position
            },
            io::SeekFrom::Current(offset) => {
                let Some(position) = self.len().checked_add_signed(offset as isize) else {
                    return Err(io::Error::new(
                        io::ErrorKind::Other,
                        "Seek before to before buffer",
                    ));
                };

                position
            },
        };

        eprintln!(
            "pos = {}, new_pos = {}, seek = {:?}",
            self.position, position, pos
        );

        self.position = position;

        Ok(position as u64)
    }
}

impl MemReaderSlice {
    #[inline(always)]
    pub fn to_vec(self) -> Vec<u8> {
        <[u8]>::to_vec(&self)
    }

    #[inline]
    pub fn from_vec(v: Vec<u8>) -> Self {
        let end = v.len();

        Self {
            data: v.into(),
            start: 0,
            end,
        }
    }
}

impl CowBuffer {
    pub fn to_mut(&mut self) -> &mut Vec<u8> {
        match self {
            CowBuffer::Borrowed(v) => {
                *self = Self::Owned(v.clone().to_vec());
                self.to_mut()
            },
            CowBuffer::Owned(v) => v,
        }
    }
}
