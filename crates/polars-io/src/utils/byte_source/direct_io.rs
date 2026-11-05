use std::alloc::{Layout, alloc, dealloc};
use std::fs::{File, OpenOptions};
use std::os::fd::AsRawFd;
use std::os::unix::fs::FileExt;
use std::path::Path;

use polars_buffer::Buffer;
use polars_error::{PolarsResult, polars_bail};

use super::dio_align::DioAlign;

/// Aligned buffer, required for O_DIRECT.
///
/// Invariant: the first `len` bytes are initialized, and `as_ref` exposes only
/// those. `len` may be less than the allocation when a read stops short at EOF.
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
        // Safety: `ptr` owns `len` initialized bytes (see the type's docs).
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

/// Returns the alignment `O_DIRECT` requires for `file`, or `None` if `file`
/// must be read through the page cache instead.
///
/// Clears `O_DIRECT` when the fd has it set but no workable alignment exists,
/// so that unaligned reads work. The flag lives on the open file description,
/// which `dup(2)` - and so `File::try_clone` - shares, making that visible
/// through every descriptor sharing it.
pub(crate) fn probe_or_disable(file: &File) -> std::io::Result<Option<DioAlign>> {
    let fd = file.as_raw_fd();

    let flags = unsafe { libc::fcntl(fd, libc::F_GETFL) };
    if flags < 0 {
        return Err(std::io::Error::last_os_error());
    }

    if flags & libc::O_DIRECT == 0 {
        // Already buffered: nothing to probe, and nothing to undo.
        return Ok(None);
    }

    if let Some(align) = DioAlign::probe(file) {
        return Ok(Some(align));
    }

    clear_o_direct(fd, flags)?;

    Ok(None)
}

/// Turn `O_DIRECT` off on an already-open fd.
fn clear_o_direct(fd: std::os::fd::RawFd, flags: libc::c_int) -> std::io::Result<()> {
    if unsafe { libc::fcntl(fd, libc::F_SETFL, flags & !libc::O_DIRECT) } < 0 {
        return Err(std::io::Error::last_os_error());
    }

    Ok(())
}

pub(crate) fn read_aligned(
    file: &File,
    align: DioAlign,
    offset: u64,
    len: usize,
    size: u64,
) -> PolarsResult<Buffer<u8>> {
    if len == 0 {
        return Ok(Buffer::new());
    }

    let (lo, hi, pad) = align.span(offset, len);
    let span = (hi - lo) as usize;

    let layout = Layout::from_size_align(span, align.memory).unwrap();
    let ptr = unsafe { alloc(layout) };
    if ptr.is_null() {
        std::alloc::handle_alloc_error(layout)
    }

    let mut owner = AlignedBuf {
        ptr,
        len: 0,
        layout,
    };
    let raw = unsafe { std::slice::from_raw_parts_mut(ptr, span) };

    let filled = if hi <= size {
        file.read_exact_at(raw, lo)?;
        span
    } else {
        // Last block: O_DIRECT rejects an unaligned length, and the aligned
        // length runs past EOF. Read what exists and leave the tail unread.
        let mut filled = 0;
        while (lo + filled as u64) < size {
            let n = file.read_at(&mut raw[filled..], lo + filled as u64)?;
            if n == 0 {
                break;
            }
            filled += n;
        }
        filled
    };
    owner.len = filled;

    if pad + len > filled {
        polars_bail!(
            ComputeError:
            "direct IO read at offset {lo} returned {filled} bytes, expected at least {}",
            pad + len
        );
    }

    Ok(Buffer::from_owner(owner).sliced(pad..pad + len))
}
