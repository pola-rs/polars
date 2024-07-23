use std::ops::Deref;
use std::sync::Arc;
use std::{fmt, io};

pub use memmap::Mmap;

use crate::mem::prefetch_l2;

/// A read-only slice over an [`Mmap`]
pub struct MmapSlice {
    // We keep the Mmap around to ensure it is still valid.
    mmap: Arc<Mmap>,
    ptr: *const u8,
    len: usize,
}

/// A cursor over a [`MemSlice`].
#[derive(Debug, Clone)]
pub struct MemReader {
    data: MemSlice,
    position: usize,
}

/// A read-only reference to a slice of memory.
///
/// This memory can either be heap-allocated or be mmap-ed into memory.
///
/// This still owns the all the original memory and therefore should probably not be a long-lasting
/// structure.
#[derive(Clone)]
pub struct MemSlice(MemSliceInner);

#[derive(Clone)]
enum MemSliceInner {
    Mmap(MmapSlice),
    Allocated(AllocatedSlice),
}

#[derive(Clone)]
struct AllocatedSlice {
    data: Arc<[u8]>,
    start: usize,
    end: usize,
}

impl Deref for MmapSlice {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        // SAFETY: Invariant of MmapSlice
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}

impl Deref for MemSlice {
    type Target = [u8];

    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        match &self.0 {
            MemSliceInner::Mmap(v) => v.deref(),
            MemSliceInner::Allocated(v) => &v.data[v.start..v.end],
        }
    }
}

impl Default for AllocatedSlice {
    fn default() -> Self {
        let slice: &[u8] = &[];
        Self {
            data: Arc::from(slice),
            start: 0,
            end: 0,
        }
    }
}

impl fmt::Debug for MemSlice {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt.debug_tuple("MemSlice").field(&self.deref()).finish()
    }
}

impl Default for MemSlice {
    fn default() -> Self {
        Self(MemSliceInner::Allocated(AllocatedSlice::default()))
    }
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

    #[inline(always)]
    pub fn from_slice(data: &[u8]) -> Self {
        Self::new(MemSlice::from_slice(data))
    }

    #[inline(always)]
    pub fn from_vec(data: Vec<u8>) -> Self {
        Self::new(MemSlice::from_vec(data))
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
        self.data.slice(start, end)
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
                    return Err(io::Error::new(
                        io::ErrorKind::Other,
                        "Seek before to before buffer",
                    ));
                };

                position
            },
            io::SeekFrom::Current(offset) => {
                let Some(position) = self.position.checked_add_signed(offset as isize) else {
                    return Err(io::Error::new(
                        io::ErrorKind::Other,
                        "Seek before to before buffer",
                    ));
                };

                position
            },
        };

        self.position = position;

        Ok(position as u64)
    }
}

impl MemSlice {
    #[inline(always)]
    pub fn to_vec(self) -> Vec<u8> {
        <[u8]>::to_vec(self.deref())
    }

    #[inline]
    pub fn from_vec(v: Vec<u8>) -> Self {
        let end = v.len();

        Self(MemSliceInner::Allocated(AllocatedSlice {
            data: v.into_boxed_slice().into(),
            start: 0,
            end,
        }))
    }

    #[inline]
    pub fn from_slice(slice: &[u8]) -> Self {
        let end = slice.len();
        Self(MemSliceInner::Allocated(AllocatedSlice {
            data: slice.into(),
            start: 0,
            end,
        }))
    }

    /// Attempt to prefetch the memory belonging to to this [`MemSlice`]
    #[inline]
    pub fn prefetch(&self) {
        if self.len() == 0 {
            return;
        }

        // @TODO: We can play a bit more with this prefetching. Maybe introduce a maximum number of
        // prefetches as to not overwhelm the processor. The linear prefetcher should pick it up
        // at a certain point.

        const PAGE_SIZE: usize = 4096;
        for i in 0..self.len() / PAGE_SIZE {
            unsafe { prefetch_l2(self[i * PAGE_SIZE..].as_ptr()) };
        }
        unsafe { prefetch_l2(self[self.len() - 1..].as_ptr()) }
    }

    #[inline]
    pub fn from_mmap(mmap: MmapSlice) -> Self {
        Self(MemSliceInner::Mmap(mmap))
    }

    #[inline]
    #[track_caller]
    pub fn slice(&self, start: usize, end: usize) -> Self {
        Self(match &self.0 {
            MemSliceInner::Mmap(v) => MemSliceInner::Mmap(v.slice(start, end)),
            MemSliceInner::Allocated(v) => MemSliceInner::Allocated({
                let len = v.end - v.start;

                assert!(start <= end);
                assert!(start <= len);
                assert!(end <= len);

                AllocatedSlice {
                    data: v.data.clone(),
                    start: v.start + start,
                    end: v.start + end,
                }
            }),
        })
    }
}

// SAFETY: This structure is read-only and does not contain any non-sync or non-send data.
unsafe impl Sync for MmapSlice {}
unsafe impl Send for MmapSlice {}

impl fmt::Debug for MmapSlice {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt.debug_tuple("MmapSlice").field(&self.deref()).finish()
    }
}

impl Clone for MmapSlice {
    fn clone(&self) -> Self {
        Self {
            mmap: self.mmap.clone(),
            ptr: self.ptr,
            len: self.len,
        }
    }
}

impl MmapSlice {
    #[inline]
    pub fn new(mmap: Mmap) -> Self {
        let slice: &[u8] = &mmap;

        let ptr = slice as *const [u8] as *const u8;
        let len = slice.len();

        let mmap = Arc::new(mmap);

        Self { mmap, ptr, len }
    }

    /// Take a slice of the current [`MmapSlice`]
    #[inline]
    #[track_caller]
    pub fn slice(&self, start: usize, end: usize) -> Self {
        assert!(start <= end);
        assert!(start <= self.len());
        assert!(end <= self.len());

        // SAFETY: Start and end are within the slice
        let ptr = unsafe { self.ptr.add(start) };
        let len = end - start;

        Self {
            mmap: self.mmap.clone(),
            ptr,
            len,
        }
    }
}
