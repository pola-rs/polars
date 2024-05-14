use std::fs::File;
use std::io;
use std::path::Path;

use polars_error::*;

fn map_err(path: &Path, err: io::Error) -> PolarsError {
    let path = path.to_string_lossy();
    let msg = if path.len() > 88 {
        let truncated_path: String = path.chars().skip(path.len() - 88).collect();
        format!("{err}: ...{truncated_path}")
    } else {
        format!("{err}: {path}")
    };
    io::Error::new(err.kind(), msg).into()
}

pub fn open_file<P>(path: P) -> PolarsResult<File>
where
    P: AsRef<Path>,
{
    File::open(&path).map_err(|err| map_err(path.as_ref(), err))
}

pub fn create_file<P>(path: P) -> PolarsResult<File>
where
    P: AsRef<Path>,
{
    File::create(&path).map_err(|err| map_err(path.as_ref(), err))
}
