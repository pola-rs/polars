use std::fs::File;
use std::io::{BufReader, Cursor, Read, Seek};

/// Trait used to get a hold to file handler or to the underlying bytes
/// without performing a Read.
pub trait MmapBytesReader: Read + Seek + Send + Sync {
    fn to_file(&self) -> Option<&File> {
        None
    }

    fn to_bytes(&self) -> Option<&[u8]> {
        None
    }
}

impl MmapBytesReader for File {
    fn to_file(&self) -> Option<&File> {
        Some(self)
    }
}

impl MmapBytesReader for BufReader<File> {
    fn to_file(&self) -> Option<&File> {
        Some(self.get_ref())
    }
}

impl<T> MmapBytesReader for Cursor<T>
where
    T: AsRef<[u8]> + Send + Sync,
{
    fn to_bytes(&self) -> Option<&[u8]> {
        Some(self.get_ref().as_ref())
    }
}

impl<T: MmapBytesReader + ?Sized> MmapBytesReader for Box<T> {
    fn to_file(&self) -> Option<&File> {
        T::to_file(self)
    }

    fn to_bytes(&self) -> Option<&[u8]> {
        T::to_bytes(self)
    }
}

impl<T: MmapBytesReader> MmapBytesReader for &mut T {
    fn to_file(&self) -> Option<&File> {
        T::to_file(self)
    }

    fn to_bytes(&self) -> Option<&[u8]> {
        T::to_bytes(self)
    }
}

// Handle various forms of input bytes
pub enum ReaderBytes<'a> {
    Borrowed(&'a [u8]),
    Owned(Vec<u8>),
    Mapped(memmap::Mmap),
}

impl std::ops::Deref for ReaderBytes<'_> {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        match self {
            Self::Borrowed(ref_bytes) => ref_bytes,
            Self::Owned(vec) => vec,
            Self::Mapped(mmap) => mmap,
        }
    }
}

impl<'a, T: 'a + MmapBytesReader> From<&'a T> for ReaderBytes<'a> {
    fn from(m: &'a T) -> Self {
        match m.to_bytes() {
            Some(s) => ReaderBytes::Borrowed(s),
            None => {
                let f = m.to_file().unwrap();
                let mmap = unsafe { memmap::Mmap::map(f).unwrap() };
                ReaderBytes::Mapped(mmap)
            }
        }
    }
}
