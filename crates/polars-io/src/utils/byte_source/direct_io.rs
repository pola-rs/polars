use std::alloc::{Layout, alloc, dealloc};
use std::fs::{File, OpenOptions};
use std::os::fd::AsRawFd;
use std::os::unix::fs::FileExt;
use std::path::Path;

use polars_buffer::Buffer;
use polars_error::PolarsResult;

use super::dio_align::DioAlign;

/// Aligned buffer, required for O_DIRECT
struct AlignedBuf {
    ptr: *mut u8,
    len: usize,
    layout: Layout,
}

// Safety: exclusive ownership of a heap allocation; no interior mutability.
unsafe impl Send for AlignedBuf {}
unsafe impl Sync for AlignedBuf {}

impl AsRef<[u8]> for AlignedBuf {
    fn as_ref(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}

impl Drop for AlignedBuf {
    fn drop(&mut self) {
        unsafe { dealloc(self.ptr, self.layout) }
    }
}

pub(crate) fn open_o_direct(path: &Path) -> std::io::Result<std::fs::File> {
    use std::os::unix::fs::OpenOptionsExt;

    OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_DIRECT)
        .open(path)
}

pub(crate) fn probe_fd(file: &std::fs::File) -> Option<DioAlign> {
    let flags = unsafe { libc::fcntl(file.as_raw_fd(), libc::F_GETFL) };
    (flags >= 0 && (flags & libc::O_DIRECT) != 0)
        .then(|| DioAlign::probe(file))
        .flatten()
}

pub(crate) fn read_aligned(
    file: &File,
    align: DioAlign,
    offset: u64,
    len: usize,
    size: u64,
) -> PolarsResult<Buffer<u8>> {
    let (lo, hi, pad) = align.span(offset, len);
    let span = (hi - lo) as usize;

    // Memory alignment may be smaller than the offset alignment (4 is common).
    let align_memory = align.memory.max(512);

    let layout = Layout::from_size_align(span, align_memory).unwrap();
    let ptr = unsafe { alloc(layout) };
    if ptr.is_null() {
        std::alloc::handle_alloc_error(layout)
    }
    let owner = AlignedBuf {
        ptr,
        len: span,
        layout,
    };
    let raw = unsafe { std::slice::from_raw_parts_mut(ptr, span) };

    if hi <= size {
        file.read_exact_at(raw, lo)?;
    } else {
        // Last block: O_DIRECT rejects an unaligned length, and the aligned
        // length runs past EOF. Read what exists, leave the tail untouched.
        let mut filled = 0;
        while (lo + filled as u64) < size {
            let n = file.read_at(&mut raw[filled..], lo + filled as u64)?;
            if n == 0 {
                break;
            }
            filled += n;
        }
    }

    Ok(Buffer::from_owner(owner).sliced(pad..pad + len))
}
