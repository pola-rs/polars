//! O_DIRECT alignment discovery.
//!
//! `O_DIRECT` requires the file offset, the transfer length, and the buffer
//! address to be aligned.
//!
//! Sources, in order of accuracy:
//!  1. `statx(STATX_DIOALIGN)` (Linux 6.1+) - authoritative, per-file, and
//!     distinguishes memory alignment from offset/length alignment. It can
//!     also say *definitively* that a file does not support direct I/O.
//!  2. `/sys/dev/block/<major>:<minor>/queue/logical_block_size` - the device
//!     sector size. Right for most filesystems, but misses cases where the
//!     filesystem imposes something stricter.
//!  3. `FALLBACK_ALIGN` - a safe over-alignment when nothing else is known.

use std::os::fd::AsRawFd;

/// Used only when the alignment is *unknown*.
const FALLBACK_ALIGN: usize = 4096;

#[derive(Debug, Clone, Copy)]
pub struct DioAlign {
    /// Alignment required for the file offset and the transfer length.
    pub offset: usize,
    /// Alignment required for the buffer address.
    pub memory: usize,
}

/// What `statx` was able to tell us.
#[cfg(target_os = "linux")]
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
        match statx_dioalign(file) {
            StatxAlign::Unsupported => None,
            StatxAlign::Known(a) => Some(Self {
                offset: normalize(a.offset),
                memory: normalize(a.memory),
            }),
            // statx could not tell us; fall back to the device sector size,
            // then to a safe over-alignment.
            StatxAlign::Unknown => Some(match sysfs_logical_block_size(file) {
                Some(a) => Self {
                    offset: normalize(a.offset),
                    memory: normalize(a.memory),
                },
                None => Self {
                    offset: FALLBACK_ALIGN,
                    memory: FALLBACK_ALIGN,
                },
            }),
        }
    }

    #[cfg(not(target_os = "linux"))]
    pub fn probe(_file: &std::fs::File) -> Option<Self> {
        None
    }
}

fn normalize(v: usize) -> usize {
    if v == 0 || !v.is_power_of_two() {
        FALLBACK_ALIGN
    } else {
        v
    }
}

#[cfg(target_os = "linux")]
fn statx_dioalign(file: &std::fs::File) -> StatxAlign {
    let mut stx: libc::statx = unsafe { std::mem::zeroed() };
    let rc = unsafe {
        libc::statx(
            file.as_raw_fd(),
            c"".as_ptr(),
            libc::AT_EMPTY_PATH,
            libc::STATX_DIOALIGN,
            &mut stx,
        )
    };
    if rc != 0 || stx.stx_mask & libc::STATX_DIOALIGN == 0 {
        return StatxAlign::Unknown;
    }

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
/// Keyed by device number, so this needs no device-name lookup. Used when the
/// kernel predates `STATX_DIOALIGN`.
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
}
