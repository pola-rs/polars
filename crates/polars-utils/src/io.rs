use std::fs::File;
use std::io;
use std::path::Path;

use polars_error::*;

fn verbose() -> bool {
    std::env::var("POLARS_VERBOSE").as_deref().unwrap_or("") == "1"
}

pub fn _limit_path_len_io_err(path: &Path, err: io::Error) -> PolarsError {
    let path = path.to_string_lossy();
    let msg = if path.len() > 88 && !verbose() {
        let truncated_path: String = path.chars().skip(path.len() - 88).collect();
        format!("{err}: ...{truncated_path}")
    } else {
        format!("{err}: {path}")
    };
    io::Error::new(err.kind(), msg).into()
}

pub fn open_file(path: &Path) -> PolarsResult<File> {
    File::open(path).map_err(|err| _limit_path_len_io_err(path, err))
}

pub fn create_file(path: &Path) -> PolarsResult<File> {
    File::create(path).map_err(|err| _limit_path_len_io_err(path, err))
}
