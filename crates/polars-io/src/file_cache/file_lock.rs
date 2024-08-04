use std::fs::{File, OpenOptions};
use std::path::Path;

use fs4::fs_std::FileExt;

/// Note: this creates the file if it does not exist when acquiring locks.
pub(super) struct FileLock<T: AsRef<Path>>(T);
pub(super) struct FileLockSharedGuard(File);
pub(super) struct FileLockExclusiveGuard(File);

/// Trait to specify a file is lock-guarded without needing a particular type of
/// guard (i.e. shared/exclusive).
pub(super) trait FileLockAnyGuard:
    std::ops::Deref<Target = File> + std::ops::DerefMut<Target = File>
{
    const IS_EXCLUSIVE: bool;
}
impl FileLockAnyGuard for FileLockSharedGuard {
    const IS_EXCLUSIVE: bool = false;
}
impl FileLockAnyGuard for FileLockExclusiveGuard {
    const IS_EXCLUSIVE: bool = true;
}

impl<T: AsRef<Path>> From<T> for FileLock<T> {
    fn from(path: T) -> Self {
        Self(path)
    }
}

impl<T: AsRef<Path>> FileLock<T> {
    pub(super) fn acquire_shared(&self) -> Result<FileLockSharedGuard, std::io::Error> {
        let file = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(self.0.as_ref())?;
        file.lock_shared().map(|_| FileLockSharedGuard(file))
    }

    pub(super) fn acquire_exclusive(&self) -> Result<FileLockExclusiveGuard, std::io::Error> {
        let file = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(self.0.as_ref())?;
        file.lock_exclusive().map(|_| FileLockExclusiveGuard(file))
    }
}

impl std::ops::Deref for FileLockSharedGuard {
    type Target = File;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl std::ops::DerefMut for FileLockSharedGuard {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl Drop for FileLockSharedGuard {
    fn drop(&mut self) {
        self.0.unlock().unwrap();
    }
}

impl std::ops::Deref for FileLockExclusiveGuard {
    type Target = File;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl std::ops::DerefMut for FileLockExclusiveGuard {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl Drop for FileLockExclusiveGuard {
    fn drop(&mut self) {
        self.0.unlock().unwrap();
    }
}
