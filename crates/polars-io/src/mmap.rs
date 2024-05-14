use std::collections::btree_map::Entry;
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufReader, Cursor, Read, Seek};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use memmap::Mmap;
use once_cell::sync::Lazy;
use polars_error::{polars_bail, PolarsResult};
use polars_utils::create_file;

// Keep track of memory mapped files so we don't write to them while reading
// Use a btree as it uses less memory than a hashmap and this thing never shrinks.
static MEMORY_MAPPED_FILES: Lazy<Mutex<BTreeMap<PathBuf, u32>>> =
    Lazy::new(|| Mutex::new(Default::default()));

pub(crate) struct MMapSemaphore {
    path: PathBuf,
    mmap: Mmap,
}

impl MMapSemaphore {
    pub(super) fn new(path: PathBuf, mmap: Mmap) -> Self {
        let mut guard = MEMORY_MAPPED_FILES.lock().unwrap();
        guard.insert(path.clone(), 1);
        Self { path, mmap }
    }
}

impl AsRef<[u8]> for MMapSemaphore {
    #[inline]
    fn as_ref(&self) -> &[u8] {
        self.mmap.as_ref()
    }
}

impl Drop for MMapSemaphore {
    fn drop(&mut self) {
        let mut guard = MEMORY_MAPPED_FILES.lock().unwrap();
        if let Entry::Occupied(mut e) = guard.entry(std::mem::take(&mut self.path)) {
            let v = e.get_mut();
            *v -= 1;

            if *v == 0 {
                e.remove_entry();
            }
        }
    }
}

/// Open a file to get write access. This will check if the file is currently registered as memory mapped.
pub fn try_create_file(path: &Path) -> PolarsResult<File> {
    let guard = MEMORY_MAPPED_FILES.lock().unwrap();
    if guard.contains_key(path) {
        polars_bail!(ComputeError: "cannot write to file: already memory mapped")
    }
    drop(guard);
    create_file(path)
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

impl<'a, T: 'a + MmapBytesReader> From<&'a T> for ReaderBytes<'a> {
    fn from(m: &'a T) -> Self {
        match m.to_bytes() {
            Some(s) => ReaderBytes::Borrowed(s),
            None => {
                let f = m.to_file().unwrap();
                let mmap = unsafe { memmap::Mmap::map(f).unwrap() };
                ReaderBytes::Mapped(mmap, f)
            },
        }
    }
}
