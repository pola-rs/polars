use std::ffi::c_void;
use std::fs::File;
use std::mem::ManuallyDrop;
use std::sync::LazyLock;

pub use memmap::Mmap;
use memmap::MmapOptions;
use polars_error::PolarsResult;
#[cfg(target_family = "unix")]
use polars_error::polars_bail;
use rayon::{ThreadPool, ThreadPoolBuilder};

use crate::mem::PAGE_SIZE;

pub static UNMAP_POOL: LazyLock<ThreadPool> = LazyLock::new(|| {
    let thread_name = std::env::var("POLARS_THREAD_NAME").unwrap_or_else(|_| "polars".to_string());
    ThreadPoolBuilder::new()
        .num_threads(1)
        .thread_name(move |i| format!("{thread_name}-unmap-{i}"))
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
        let mmap = match unsafe { options.map(file) } {
            Ok(m) => m,

            // Mmap can fail with ENODEV on filesystems which don't support
            // MAP_SHARED, try MAP_PRIVATE instead, see #24343.
            #[cfg(target_family = "unix")]
            Err(e) if e.raw_os_error() == Some(libc::ENODEV) => unsafe {
                options.map_copy_read_only(file)?
            },

            Err(e) => return Err(e.into()),
        };

        #[cfg(target_family = "unix")]
        {
            // TODO: We aren't handling the case where the file is already open in write-mode here.

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
