use std::fs::File;
use std::io::Error;
use std::path::Path;

use polars_error::*;

pub fn open_file<P>(path: P) -> PolarsResult<File>
where
    P: AsRef<Path>,
{
    std::fs::File::open(&path).map_err(|err| {
        let path = path.as_ref().to_string_lossy();
        let msg = if path.len() > 88 {
            let truncated_path: String = path.chars().skip(path.len() - 88).collect();
            format!("{err}: ...{truncated_path}")
        } else {
            format!("{err}: {path}")
        };
        PolarsError::Io(Error::new(err.kind(), msg))
    })
}
