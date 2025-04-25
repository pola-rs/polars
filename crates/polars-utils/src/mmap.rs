use std::ffi::c_void;
use std::fs::File;
use std::io;
use std::mem::ManuallyDrop;
use std::sync::LazyLock;

pub use memmap::Mmap;

mod private {
    use std::fs::File;
    use std::ops::Deref;
    use std::sync::Arc;

    use polars_error::PolarsResult;

    use super::MMapSemaphore;
    use crate::mem::prefetch::prefetch_l2;

    /// A read-only reference to a slice of memory that can potentially be memory-mapped.
    ///
    /// A reference count is kept to the underlying buffer to ensure the memory is kept alive.
    /// [`MemSlice::slice`] can be used to slice the memory in a zero-copy manner.
    ///
    /// This still owns the all the original memory and therefore should probably not be a long-lasting
    /// structure.
    #[derive(Clone, Debug)]
    pub struct MemSlice {
        // Store the `&[u8]` to make the `Deref` free.
        // `slice` is not 'static - it is backed by `inner`. This is safe as long as `slice` is not
        // directly accessed, and we are in a private module to guarantee that. Access should only
        // be done through `Deref<Target = [u8]>`, which automatically gives the correct lifetime.
        slice: &'static [u8],
        #[allow(unused)]
        inner: MemSliceInner,
    }

    /// Keeps the underlying buffer alive. This should be cheaply cloneable.
    #[derive(Clone, Debug)]
    #[allow(unused)]
    enum MemSliceInner {
        Bytes(bytes::Bytes), // Separate because it does atomic refcounting internally
        Arc(Arc<dyn std::fmt::Debug + Send + Sync>),
    }

    impl Deref for MemSlice {
        type Target = [u8];

        #[inline(always)]
        fn deref(&self) -> &Self::Target {
            self.slice
        }
    }

    impl AsRef<[u8]> for MemSlice {
        #[inline(always)]
        fn as_ref(&self) -> &[u8] {
            self.slice
        }
    }

    impl Default for MemSlice {
        fn default() -> Self {
            Self::from_bytes(bytes::Bytes::new())
        }
    }

    impl From<Vec<u8>> for MemSlice {
        fn from(value: Vec<u8>) -> Self {
            Self::from_vec(value)
        }
    }

    impl MemSlice {
        pub const EMPTY: Self = Self::from_static(&[]);

        /// Copy the contents into a new owned `Vec`
        #[inline(always)]
        pub fn to_vec(self) -> Vec<u8> {
            <[u8]>::to_vec(self.deref())
        }

        /// Construct a `MemSlice` from an existing `Vec<u8>`. This is zero-copy.
        #[inline]
        pub fn from_vec(v: Vec<u8>) -> Self {
            Self::from_bytes(bytes::Bytes::from(v))
        }

        /// Construct a `MemSlice` from [`bytes::Bytes`]. This is zero-copy.
        #[inline]
        pub fn from_bytes(bytes: bytes::Bytes) -> Self {
            Self {
                slice: unsafe { std::mem::transmute::<&[u8], &'static [u8]>(bytes.as_ref()) },
                inner: MemSliceInner::Bytes(bytes),
            }
        }

        #[inline]
        pub fn from_mmap(mmap: Arc<MMapSemaphore>) -> Self {
            Self {
                slice: unsafe {
                    std::mem::transmute::<&[u8], &'static [u8]>(mmap.as_ref().as_ref())
                },
                inner: MemSliceInner::Arc(mmap),
            }
        }

        #[inline]
        pub fn from_arc<T>(slice: &[u8], arc: Arc<T>) -> Self
        where
            T: std::fmt::Debug + Send + Sync + 'static,
        {
            Self {
                slice: unsafe { std::mem::transmute::<&[u8], &'static [u8]>(slice) },
                inner: MemSliceInner::Arc(arc),
            }
        }

        #[inline]
        pub fn from_file(file: &File) -> PolarsResult<Self> {
            let mmap = MMapSemaphore::new_from_file(file)?;
            Ok(Self::from_mmap(Arc::new(mmap)))
        }

        /// Construct a `MemSlice` that simply wraps around a `&[u8]`.
        #[inline]
        pub const fn from_static(slice: &'static [u8]) -> Self {
            let inner = MemSliceInner::Bytes(bytes::Bytes::from_static(slice));
            Self { slice, inner }
        }

        /// Attempt to prefetch the memory belonging to to this [`MemSlice`]
        #[inline]
        pub fn prefetch(&self) {
            prefetch_l2(self.as_ref());
        }

        /// # Panics
        /// Panics if range is not in bounds.
        #[inline]
        #[track_caller]
        pub fn slice(&self, range: std::ops::Range<usize>) -> Self {
            let mut out = self.clone();
            out.slice = &out.slice[range];
            out
        }
    }

    impl From<bytes::Bytes> for MemSlice {
        fn from(value: bytes::Bytes) -> Self {
            Self::from_bytes(value)
        }
    }
}

use memmap::MmapOptions;
use polars_error::PolarsResult;
#[cfg(target_family = "unix")]
use polars_error::polars_bail;
pub use private::MemSlice;
use rayon::{ThreadPool, ThreadPoolBuilder};

use crate::mem::PAGE_SIZE;

/// A cursor over a [`MemSlice`].
#[derive(Debug, Clone)]
pub struct MemReader {
    data: MemSlice,
    position: usize,
}

impl MemReader {
    pub fn new(data: MemSlice) -> Self {
        Self { data, position: 0 }
    }

    #[inline(always)]
    pub fn remaining_len(&self) -> usize {
        self.data.len() - self.position
    }

    #[inline(always)]
    pub fn total_len(&self) -> usize {
        self.data.len()
    }

    #[inline(always)]
    pub fn position(&self) -> usize {
        self.position
    }

    /// Construct a `MemSlice` from an existing `Vec<u8>`. This is zero-copy.
    #[inline(always)]
    pub fn from_vec(v: Vec<u8>) -> Self {
        Self::new(MemSlice::from_vec(v))
    }

    /// Construct a `MemSlice` from [`bytes::Bytes`]. This is zero-copy.
    #[inline(always)]
    pub fn from_bytes(bytes: bytes::Bytes) -> Self {
        Self::new(MemSlice::from_bytes(bytes))
    }

    // Construct a `MemSlice` that simply wraps around a `&[u8]`. The caller must ensure the
    /// slice outlives the returned `MemSlice`.
    #[inline]
    pub fn from_slice(slice: &'static [u8]) -> Self {
        Self::new(MemSlice::from_static(slice))
    }

    #[inline(always)]
    pub fn from_reader<R: io::Read>(mut reader: R) -> io::Result<Self> {
        let mut vec = Vec::new();
        reader.read_to_end(&mut vec)?;
        Ok(Self::from_vec(vec))
    }

    #[inline(always)]
    pub fn read_slice(&mut self, n: usize) -> MemSlice {
        let start = self.position;
        let end = usize::min(self.position + n, self.data.len());
        self.position = end;
        self.data.slice(start..end)
    }
}

impl From<MemSlice> for MemReader {
    fn from(data: MemSlice) -> Self {
        Self { data, position: 0 }
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
            io::SeekFrom::Start(position) => usize::min(position as usize, self.total_len()),
            io::SeekFrom::End(offset) => {
                let Some(position) = self.total_len().checked_add_signed(offset as isize) else {
                    return Err(io::Error::other("Seek before to before buffer"));
                };

                position
            },
            io::SeekFrom::Current(offset) => {
                let Some(position) = self.position.checked_add_signed(offset as isize) else {
                    return Err(io::Error::other("Seek before to before buffer"));
                };

                position
            },
        };

        self.position = position;

        Ok(position as u64)
    }
}

pub static UNMAP_POOL: LazyLock<ThreadPool> = LazyLock::new(|| {
    let thread_name = std::env::var("POLARS_THREAD_NAME").unwrap_or_else(|_| "polars".to_string());
    ThreadPoolBuilder::new()
        .num_threads(1)
        .thread_name(move |i| format!("{}-unmap-{}", thread_name, i))
        .build()
        .expect("could not spawn threads")
});

// Keep track of memory mapped files so we don't write to them while reading
// Use a btree as it uses less memory than a hashmap and this thing never shrinks.
// Write handle in Windows is exclusive, so this is only necessary in Unix.
#[cfg(target_family = "unix")]
static MEMORY_MAPPED_FILES: std::sync::LazyLock<
    std::sync::Mutex<std::collections::BTreeMap<(u64, u64), u32>>,
> = std::sync::LazyLock::new(|| std::sync::Mutex::new(Default::default()));

#[derive(Debug)]
pub struct MMapSemaphore {
    #[cfg(target_family = "unix")]
    key: (u64, u64),
    mmap: ManuallyDrop<Mmap>,
}

impl Drop for MMapSemaphore {
    fn drop(&mut self) {
        #[cfg(target_family = "unix")]
        {
            let mut guard = MEMORY_MAPPED_FILES.lock().unwrap();
            if let std::collections::btree_map::Entry::Occupied(mut e) = guard.entry(self.key) {
                let v = e.get_mut();
                *v -= 1;

                if *v == 0 {
                    e.remove_entry();
                }
            }
        }

        unsafe {
            let mmap = ManuallyDrop::take(&mut self.mmap);
            // If the unmap is 1 MiB or bigger, we do it in a background thread.
            let len = self.mmap.len();
            if len >= 1024 * 1024 {
                UNMAP_POOL.spawn(move || {
                    #[cfg(target_family = "unix")]
                    {
                        // If the unmap is bigger than our chunk size (32 MiB), we do it in chunks.
                        // This is because munmap holds a lock on the unmap file, which we don't
                        // want to hold for extended periods of time.
                        let chunk_size = (32_usize * 1024 * 1024).next_multiple_of(*PAGE_SIZE);
                        if len > chunk_size {
                            let mmap = ManuallyDrop::new(mmap);
                            let ptr: *const u8 = mmap.as_ptr();
                            let mut offset = 0;
                            while offset < len {
                                let remaining = len - offset;
                                libc::munmap(
                                    ptr.add(offset) as *mut c_void,
                                    remaining.min(chunk_size),
                                );
                                offset += chunk_size;
                            }
                            return;
                        }
                    }
                    drop(mmap)
                });
            } else {
                drop(mmap);
            }
        }
    }
}

impl MMapSemaphore {
    pub fn new_from_file_with_options(
        file: &File,
        options: MmapOptions,
    ) -> PolarsResult<MMapSemaphore> {
        let mmap = unsafe { options.map(file) }?;

        #[cfg(target_family = "unix")]
        {
            // FIXME: We aren't handling the case where the file is already open in write-mode here.

            use std::os::unix::fs::MetadataExt;
            let metadata = file.metadata()?;

            let mut guard = MEMORY_MAPPED_FILES.lock().unwrap();
            let key = (metadata.dev(), metadata.ino());
            match guard.entry(key) {
                std::collections::btree_map::Entry::Occupied(mut e) => *e.get_mut() += 1,
                std::collections::btree_map::Entry::Vacant(e) => _ = e.insert(1),
            }
            Ok(Self {
                key,
                mmap: ManuallyDrop::new(mmap),
            })
        }

        #[cfg(not(target_family = "unix"))]
        Ok(Self {
            mmap: ManuallyDrop::new(mmap),
        })
    }

    pub fn new_from_file(file: &File) -> PolarsResult<MMapSemaphore> {
        Self::new_from_file_with_options(file, MmapOptions::default())
    }

    pub fn as_ptr(&self) -> *const u8 {
        self.mmap.as_ptr()
    }
}

impl AsRef<[u8]> for MMapSemaphore {
    #[inline]
    fn as_ref(&self) -> &[u8] {
        self.mmap.as_ref()
    }
}

pub fn ensure_not_mapped(
    #[cfg_attr(not(target_family = "unix"), allow(unused))] file_md: &std::fs::Metadata,
) -> PolarsResult<()> {
    // TODO: We need to actually register that this file has been write-opened and prevent
    // read-opening this file based on that.
    #[cfg(target_family = "unix")]
    {
        use std::os::unix::fs::MetadataExt;
        let guard = MEMORY_MAPPED_FILES.lock().unwrap();
        if guard.contains_key(&(file_md.dev(), file_md.ino())) {
            polars_bail!(ComputeError: "cannot write to file: already memory mapped");
        }
    }
    Ok(())
}

mod tests {
    #[test]
    fn test_mem_slice_zero_copy() {
        use std::sync::Arc;

        use super::MemSlice;

        {
            let vec = vec![1u8, 2, 3, 4, 5];
            let ptr = vec.as_ptr();

            let mem_slice = MemSlice::from_vec(vec);
            let ptr_out = mem_slice.as_ptr();

            assert_eq!(ptr_out, ptr);
        }

        {
            let mut vec = vec![1u8, 2, 3, 4, 5];
            vec.truncate(2);
            let ptr = vec.as_ptr();

            let mem_slice = MemSlice::from_vec(vec);
            let ptr_out = mem_slice.as_ptr();

            assert_eq!(ptr_out, ptr);
        }

        {
            let bytes = bytes::Bytes::from(vec![1u8, 2, 3, 4, 5]);
            let ptr = bytes.as_ptr();

            let mem_slice = MemSlice::from_bytes(bytes);
            let ptr_out = mem_slice.as_ptr();

            assert_eq!(ptr_out, ptr);
        }

        {
            use crate::mmap::MMapSemaphore;

            let path = "../../examples/datasets/foods1.csv";
            let file = std::fs::File::open(path).unwrap();
            let mmap = MMapSemaphore::new_from_file(&file).unwrap();
            let ptr = mmap.as_ptr();

            let mem_slice = MemSlice::from_mmap(Arc::new(mmap));
            let ptr_out = mem_slice.as_ptr();

            assert_eq!(ptr_out, ptr);
        }

        {
            let vec = vec![1u8, 2, 3, 4, 5];
            let slice = vec.as_slice();
            let ptr = slice.as_ptr();

            let mem_slice = MemSlice::from_static(unsafe {
                std::mem::transmute::<&[u8], &'static [u8]>(slice)
            });
            let ptr_out = mem_slice.as_ptr();

            assert_eq!(ptr_out, ptr);
        }
    }

    #[test]
    fn test_mem_slice_slicing() {
        use super::MemSlice;

        {
            let vec = vec![1u8, 2, 3, 4, 5];
            let slice = vec.as_slice();

            let mem_slice = MemSlice::from_static(unsafe {
                std::mem::transmute::<&[u8], &'static [u8]>(slice)
            });

            let out = &*mem_slice.slice(3..5);
            assert_eq!(out, &slice[3..5]);
            assert_eq!(out.as_ptr(), slice[3..5].as_ptr());
        }
    }
}
