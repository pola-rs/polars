#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Hash, strum_macros::IntoStaticStr)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
pub enum SyncOnCloseType {
    /// Don't call sync on close.
    #[default]
    None,

    /// Sync only the file contents.
    Data,
    /// Synce the file contents and the metadata.
    All,
}

/// Queue writeback for the range written since `from`, without waiting for it.
#[cfg(target_os = "linux")]
pub fn start_writeback(fd: std::os::fd::RawFd, from: &mut u64) {
    let mut stat: libc::stat = unsafe { std::mem::zeroed() };
    if unsafe { libc::fstat(fd, &mut stat) } != 0 {
        return;
    }
    let size = stat.st_size as u64;
    if size <= *from {
        return;
    }
    let _ = unsafe {
        libc::sync_file_range(
            fd,
            *from as libc::off64_t,
            (size - *from) as libc::off64_t,
            libc::SYNC_FILE_RANGE_WRITE,
        )
    };
    *from = size;
}
