//! O_DIRECT alignment discovery.
//!
//! `O_DIRECT` requires the file offset, the transfer length, and the buffer
//! address to be aligned.
//!
//! Sources, in order of accuracy:
//!  1. `statx(STATX_DIOALIGN)` (Linux 6.1+) - authoritative, per-file, and
//!     distinguishes memory alignment from offset/length alignment. It can
//!     also say *definitively* that a file does not support direct I/O.
//!     glibc only for now (see comments below for musl)
//!  2. `/sys/dev/block/<major>:<minor>/queue/logical_block_size` - the device
//!     sector size. Right for most filesystems, but misses cases where the
//!     filesystem imposes something stricter.
//!
//! When neither can answer, the alignment stays unknown and the caller reads
//! through the page cache.

#[cfg(all(target_os = "linux", target_env = "gnu"))]
use std::os::fd::AsRawFd;

/// Substituted for a reported value that cannot be an alignment.
#[cfg(target_os = "linux")]
const FALLBACK_ALIGN: usize = 4096;

/// Floor for the buffer alignment, applied even when the kernel reports less.
///
/// `statx` can report a memory alignment below the sector size (ext4 reports 4),
/// and the other two probes infer the alignment rather than being told it.
#[cfg(target_os = "linux")]
const MIN_MEM_ALIGN: usize = 512;

#[derive(Debug, Clone, Copy)]
pub struct DioAlign {
    /// Alignment required for the file offset and the transfer length.
    pub offset: usize,
    /// Alignment required for the buffer address.
    /// Invariant: no less than `MIN_MEM_ALIGN`.
    pub memory: usize,
}

/// What `statx` was able to tell us.
#[cfg(all(target_os = "linux", target_env = "gnu"))]
#[derive(Debug, Clone)]
enum StatxAlign {
    /// The kernel reported concrete alignments.
    Known(DioAlign),
    /// The kernel reported zero: this file does not support direct I/O.
    /// Distinct from `Unknown` - here we must not guess.
    Unsupported,
    /// The syscall failed or the mask is unavailable (kernel < 6.1).
    Unknown,
}

impl DioAlign {
    /// Round `offset` down and `end` up to the offset alignment.
    ///
    /// Returns `(lo, hi, pad)` where `pad` is how far into the aligned span the
    /// caller's data begins. Note `hi` is deliberately *not* clamped to the
    /// file size: the length must stay aligned, so a read at EOF is expected to
    /// come up short and the caller must handle the tail.
    pub fn span(&self, offset: u64, len: usize) -> (u64, u64, usize) {
        debug_assert!(self.offset > 0);

        let a = self.offset as u64;
        let lo = offset & !(a - 1);
        let hi = (offset + len as u64).next_multiple_of(a);
        (lo, hi, (offset - lo) as usize)
    }

    /// Query the alignment `O_DIRECT` requires for this file.
    ///
    /// `None` means direct I/O is not supported here and the caller should use
    /// buffered reads.
    #[cfg(target_os = "linux")]
    pub fn probe(file: &std::fs::File) -> Option<Self> {
        #[cfg(target_env = "gnu")]
        match statx_dioalign(file) {
            StatxAlign::Unsupported => return None,
            StatxAlign::Known(a) => return Some(Self::new(a.offset, a.memory)),
            StatxAlign::Unknown => {},
        }

        sysfs_logical_block_size(file).map(|a| Self::new(a.offset, a.memory))
    }

    #[cfg(not(target_os = "linux"))]
    pub fn probe(_file: &std::fs::File) -> Option<Self> {
        None
    }

    /// Satisfy type invariants.
    #[cfg(target_os = "linux")]
    fn new(offset: usize, memory: usize) -> Self {
        Self {
            offset: normalize(offset),
            memory: normalize(memory).max(MIN_MEM_ALIGN),
        }
    }
}

#[cfg(target_os = "linux")]
fn normalize(v: usize) -> usize {
    if v == 0 || !v.is_power_of_two() {
        FALLBACK_ALIGN
    } else {
        v
    }
}

/// `statx(2)` on an open file, as a raw syscall: glibc's wrapper needs glibc 2.28,
/// manylinux2014 is 2.17.
// TODO: also probe on musl (needs our own `struct statx`, #29306); musl uses sysfs meanwhile.
#[cfg(all(target_os = "linux", target_env = "gnu"))]
fn statx(file: &std::fs::File, mask: u32) -> Option<libc::statx> {
    let mut stx: libc::statx = unsafe { std::mem::zeroed() };
    // Safety: `stx` matches the kernel's layout and outlives the call. `syscall` reads every
    // integer argument as a `c_long`.
    let rc = unsafe {
        libc::syscall(
            libc::SYS_statx,
            file.as_raw_fd() as libc::c_long,
            c"".as_ptr(),
            libc::AT_EMPTY_PATH as libc::c_long,
            mask as libc::c_long,
            &mut stx as *mut libc::statx,
        )
    };
    (rc == 0).then_some(stx)
}

#[cfg(all(target_os = "linux", target_env = "gnu"))]
fn statx_dioalign(file: &std::fs::File) -> StatxAlign {
    let Some(stx) =
        statx(file, libc::STATX_DIOALIGN).filter(|stx| stx.stx_mask & libc::STATX_DIOALIGN != 0)
    else {
        return StatxAlign::Unknown;
    };

    let offset = stx.stx_dio_offset_align as usize;
    let memory = stx.stx_dio_mem_align as usize;

    if offset == 0 || memory == 0 {
        StatxAlign::Unsupported
    } else {
        StatxAlign::Known(DioAlign { offset, memory })
    }
}

/// `/sys/dev/block/<major>:<minor>/queue/logical_block_size`.
///
/// Keyed by device number, so this needs no device-name lookup. Used when
/// `statx` cannot answer: kernels before 6.1, filesystems that do not report
/// `STATX_DIOALIGN`, and musl.
#[cfg(target_os = "linux")]
fn sysfs_logical_block_size(file: &std::fs::File) -> Option<DioAlign> {
    use std::os::unix::fs::MetadataExt;

    let dev = file.metadata().ok()?.dev();
    let (major, minor) = (libc::major(dev), libc::minor(dev));
    let path = format!("/sys/dev/block/{major}:{minor}/queue/logical_block_size");
    let v: usize = std::fs::read_to_string(path).ok()?.trim().parse().ok()?;

    // The device sector size constrains offset and length. Buffer alignment is
    // not reported here, so assume the same per historical behavior.
    Some(DioAlign {
        offset: v,
        memory: v,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn span_invariants() {
        for align in [512usize, 4096] {
            let a = DioAlign {
                offset: align,
                memory: align,
            };
            for (off, len) in [(206959u64, 206955usize), (4095, 2), (0, 1), (0, align)] {
                let (lo, hi, pad) = a.span(off, len);
                assert_eq!(lo as usize % align, 0, "lo unaligned");
                assert_eq!((hi - lo) as usize % align, 0, "span unaligned");
                assert!(hi >= off + len as u64, "span does not cover the range");
                assert!(pad + len <= (hi - lo) as usize, "pad + len exceeds span");
            }
        }
    }

    /// A wrong argument order fails the call; a wrong struct layout misreads the size.
    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    #[test]
    fn statx_reads_the_file_29800() {
        use std::io::Write;

        let mut file = tempfile::tempfile().unwrap();
        file.write_all(&[0; 1234]).unwrap();
        let stx = statx(&file, libc::STATX_SIZE).unwrap();
        assert_ne!(stx.stx_mask & libc::STATX_SIZE, 0);
        assert_eq!(stx.stx_size, 1234);
    }
}
