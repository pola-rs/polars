use std::fs::File;
use std::io::{Read, Seek, Write};

impl From<File> for ClosableFile {
    fn from(value: File) -> Self {
        ClosableFile { inner: value }
    }
}

impl From<ClosableFile> for File {
    fn from(value: ClosableFile) -> Self {
        value.inner
    }
}

pub struct ClosableFile {
    inner: File,
}

impl ClosableFile {
    #[cfg(unix)]
    pub fn close(self) -> std::io::Result<()> {
        use std::os::fd::IntoRawFd;
        let fd = self.inner.into_raw_fd();

        match unsafe { libc::close(fd) } {
            0 => Ok(()),
            _ => Err(std::io::Error::last_os_error()),
        }
    }

    #[cfg(not(unix))]
    pub fn close(self) -> std::io::Result<()> {
        Ok(())
    }
}

impl AsMut<File> for ClosableFile {
    fn as_mut(&mut self) -> &mut File {
        &mut self.inner
    }
}

impl AsRef<File> for ClosableFile {
    fn as_ref(&self) -> &File {
        &self.inner
    }
}

impl Seek for ClosableFile {
    fn seek(&mut self, pos: std::io::SeekFrom) -> std::io::Result<u64> {
        self.inner.seek(pos)
    }
}

impl Read for ClosableFile {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        self.inner.read(buf)
    }
}

impl Write for ClosableFile {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.inner.write(buf)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.inner.flush()
    }
}

pub trait WriteClose: Write {
    fn close(self: Box<Self>) -> std::io::Result<()> {
        Ok(())
    }
}

impl WriteClose for ClosableFile {
    fn close(self: Box<Self>) -> std::io::Result<()> {
        let f = *self;
        f.close()
    }
}
