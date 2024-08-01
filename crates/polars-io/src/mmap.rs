use std::collections::btree_map::Entry;
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufReader, Cursor, Read, Seek};
use std::sync::{Arc, Mutex};

use memmap::Mmap;
use once_cell::sync::Lazy;
use polars_core::config::verbose;
use polars_error::{polars_bail, PolarsResult};
use polars_utils::mmap::MemSlice;

// Keep track of memory mapped files so we don't write to them while reading
// Use a btree as it uses less memory than a hashmap and this thing never shrinks.
// Write handle in Windows is exclusive, so this is only necessary in Unix.
#[cfg(target_family = "unix")]
static MEMORY_MAPPED_FILES: Lazy<Mutex<BTreeMap<(u64, u64), u32>>> =
    Lazy::new(|| Mutex::new(Default::default()));

pub(crate) struct MMapSemaphore {
    #[cfg(target_family = "unix")]
    key: (u64, u64),
    mmap: Mmap,
}

impl MMapSemaphore {
    #[cfg(target_family = "unix")]
    pub(super) fn new(dev: u64, ino: u64, mmap: Mmap) -> Self {
        let mut guard = MEMORY_MAPPED_FILES.lock().unwrap();
        let key = (dev, ino);
        guard.insert(key, 1);
        Self { key, mmap }
    }

    #[cfg(not(target_family = "unix"))]
    pub(super) fn new(mmap: Mmap) -> Self {
        Self { mmap }
    }
}

impl AsRef<[u8]> for MMapSemaphore {
    #[inline]
    fn as_ref(&self) -> &[u8] {
        self.mmap.as_ref()
    }
}

#[cfg(target_family = "unix")]
impl Drop for MMapSemaphore {
    fn drop(&mut self) {
        let mut guard = MEMORY_MAPPED_FILES.lock().unwrap();
        if let Entry::Occupied(mut e) = guard.entry(self.key) {
            let v = e.get_mut();
            *v -= 1;

            if *v == 0 {
                e.remove_entry();
            }
        }
    }
}

pub fn ensure_not_mapped(file: &File) -> PolarsResult<()> {
    #[cfg(target_family = "unix")]
    {
        use std::os::unix::fs::MetadataExt;
        let guard = MEMORY_MAPPED_FILES.lock().unwrap();
        let metadata = file.metadata()?;
        if guard.contains_key(&(metadata.dev(), metadata.ino())) {
            polars_bail!(ComputeError: "cannot write to file: already memory mapped");
        }
    }
    Ok(())
}

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
    Mapped(memmap::Mmap, &'a File),
}

impl std::ops::Deref for ReaderBytes<'_> {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        match self {
            Self::Borrowed(ref_bytes) => ref_bytes,
            Self::Owned(vec) => vec,
            Self::Mapped(mmap, _) => mmap,
        }
    }
}

/// Require 'static to force the caller to do any transmute as it's usually much
/// clearer to see there whether it's sound.
impl ReaderBytes<'static> {
    pub fn into_mem_slice(self) -> MemSlice {
        match self {
            ReaderBytes::Borrowed(v) => MemSlice::from_slice(v),
            ReaderBytes::Owned(v) => MemSlice::from_vec(v),
            ReaderBytes::Mapped(v, _) => MemSlice::from_mmap(Arc::new(v)),
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
                    let f = unsafe { std::mem::transmute::<&File, &'a File>(f) };
                    let mmap = unsafe { memmap::Mmap::map(f).unwrap() };
                    ReaderBytes::Mapped(mmap, f)
                } else {
                    if verbose() {
                        eprintln!("could not memory map file; read to buffer.")
                    }
                    let mut buf = vec![];
                    m.read_to_end(&mut buf).expect("could not read");
                    ReaderBytes::Owned(buf)
                }
            },
        }
    }
}
