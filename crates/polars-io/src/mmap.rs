use std::fs::File;
use std::io::{BufReader, Cursor, Read, Seek};

use polars_buffer::Buffer;
use polars_core::config::verbose;
use polars_utils::mmap::MMapSemaphore;

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

impl MmapBytesReader for BufReader<&File> {
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

// Handle various forms of input bytes.
// TODO: we should just remove this and use Buffer.
pub enum ReaderBytes<'a> {
    Borrowed(&'a [u8]),
    Owned(Buffer<u8>),
}

impl std::ops::Deref for ReaderBytes<'_> {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        match self {
            Self::Borrowed(ref_bytes) => ref_bytes,
            Self::Owned(vec) => vec,
        }
    }
}

impl<'a, T: 'a + MmapBytesReader> From<&'a mut T> for ReaderBytes<'a> {
    fn from(m: &'a mut T) -> Self {
        match m.to_bytes() {
            // , but somehow bchk doesn't see that lifetime is 'a.
            Some(s) => {
                let s = unsafe { std::mem::transmute::<&[u8], &'a [u8]>(s) };
                ReaderBytes::Borrowed(s)
            },
            None => {
                if let Some(f) = m.to_file() {
                    let mmap = MMapSemaphore::new_from_file(f).unwrap();
                    ReaderBytes::Owned(Buffer::from_owner(mmap))
                } else {
                    if verbose() {
                        eprintln!("could not memory map file; read to buffer.")
                    }
                    let mut buf = vec![];
                    m.read_to_end(&mut buf).expect("could not read");
                    ReaderBytes::Owned(Buffer::from_vec(buf))
                }
            },
        }
    }
}
