use std::fs::File;
use std::io;
use std::path::Path;

use polars_error::*;

use crate::config::verbose;

pub fn _limit_path_len_io_err(path: &Path, err: io::Error) -> PolarsError {
    let path = path.to_string_lossy();
    let msg = if path.len() > 88 && !verbose() {
        let truncated_path: String = path.chars().skip(path.len() - 88).collect();
        format!("{err}: ...{truncated_path} (set POLARS_VERBOSE=1 to see full path)")
    } else {
        format!("{err}: {path}")
    };
    io::Error::new(err.kind(), msg).into()
}

pub fn open_file(path: &Path) -> PolarsResult<File> {
    File::open(path).map_err(|err| _limit_path_len_io_err(path, err))
}

pub fn open_file_write(path: &Path) -> PolarsResult<File> {
    std::fs::OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(path)
        .map_err(|err| _limit_path_len_io_err(path, err))
}

pub fn create_file(path: &Path) -> PolarsResult<File> {
    File::create(path).map_err(|err| _limit_path_len_io_err(path, err))
}
