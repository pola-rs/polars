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
        PolarsError::Io(Error::new(err.kind(), format!("{err}: {path}")))
    })
}
