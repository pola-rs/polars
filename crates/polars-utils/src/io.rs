use std::fs::File;
use std::path::Path;

use polars_error::*;

pub fn open_file<P>(path: P) -> PolarsResult<File>
where
    P: AsRef<Path>,
{
    std::fs::File::open(&path).map_err(|e| {
        let path = path.as_ref().to_string_lossy();
        if path.len() > 88 {
            let path: String = path.chars().skip(path.len() - 88).collect();
            polars_err!(ComputeError: "error open file: ...{}, {}", path, e)
        } else {
            polars_err!(ComputeError: "error open file: {}, {}", path, e)
        }
    })
}
