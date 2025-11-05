use std::fs::File;
use std::io::{BufReader, Cursor, Read, Seek};

use polars_core::config::verbose;
use polars_utils::file::ClosableFile;
use polars_utils::mmap::MemSlice;

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

impl MmapBytesReader for ClosableFile {
    fn to_file(&self) -> Option<&File> {
        Some(self.as_ref())
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

// Handle various forms of input bytes
pub enum ReaderBytes<'a> {
    Borrowed(&'a [u8]),
    Owned(MemSlice),
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

/// There are some places that perform manual lifetime management after transmuting `ReaderBytes`
/// to have a `'static` inner lifetime. The advantage to doing this is that it lets you construct a
/// `MemSlice` from the `ReaderBytes` in a zero-copy manner regardless of the underlying enum
/// variant.
impl ReaderBytes<'static> {
    /// Construct a `MemSlice` in a zero-copy manner from the underlying bytes, with the assumption
    /// that the underlying bytes have a `'static` lifetime.
    pub fn to_memslice(&self) -> MemSlice {
        match self {
            ReaderBytes::Borrowed(v) => MemSlice::from_static(v),
            ReaderBytes::Owned(v) => v.clone(),
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
                    ReaderBytes::Owned(MemSlice::from_file(f).unwrap())
                } else {
                    if verbose() {
                        eprintln!("could not memory map file; read to buffer.")
                    }
                    let mut buf = vec![];
                    m.read_to_end(&mut buf).expect("could not read");
                    ReaderBytes::Owned(MemSlice::from_vec(buf))
                }
            },
        }
    }
}
