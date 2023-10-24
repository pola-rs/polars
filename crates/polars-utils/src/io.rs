use std::fs::File;
use std::path::Path;
use std::io::ErrorKind;
use polars_error::*;

pub fn open_file<P>(path: P) -> PolarsResult<File>
where
    P: AsRef<Path>,
{
    std::fs::File::open(&path).map_err(|e| {
        let mut path = path.as_ref().to_string_lossy();
        if path.len() > 88 {
            path = path.chars().skip(path.len() - 88).collect();
        }

        match e.kind() {
            ErrorKind::NotFound => {
                polars_err!(FileNotFound: "No such file or directory: {}", path)
            },        
            _ => {
                polars_err!(ComputeError: "error open file: {}, {}", path, e)
                
            }
        }
    })
}
