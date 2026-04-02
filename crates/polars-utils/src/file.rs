pub fn close_file(file: std::fs::File) -> std::io::Result<()> {
    #[cfg(unix)]
    {
        use std::os::fd::IntoRawFd;
        let fd = file.into_raw_fd();

        if unsafe { libc::close(fd) } != 0 {
            return Err(std::io::Error::last_os_error());
        }
    }

    Ok(())
}
