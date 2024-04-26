use std::io::BufReader;
use std::path::PathBuf;

use polars::io::ipc::{IpcReader, IpcWriterOption};
use polars::io::prelude::SerReader;
use polars::io::PartitionedWriter;
use polars_error::PolarsResult;
use tempfile;

#[test]
#[cfg(feature = "ipc")]
fn test_ipc_partition() -> PolarsResult<()> {
    let tmp_dir = tempfile::tempdir()?;

    let df = df!("a" => [1, 1, 2, 3], "b" => [2, 2, 3, 4], "c" => [2, 3, 4, 5]).unwrap();
    let by = ["a", "b"];
    let rootdir = tmp_dir.path().join("ipc-partition");

    let option = IpcWriterOption::new();

    PartitionedWriter::new(option, rootdir.clone(), by).finish(&df)?;

    let expected_dfs = [
        df!("a" => [1, 1], "b" => [2, 2], "c" => [2, 3])?,
        df!("a" => [2], "b" => [3], "c" => [4])?,
        df!("a" => [3], "b" => [4], "c" => [5])?,
    ];

    let expected: Vec<(PathBuf, DataFrame)> = ["a=1/b=2", "a=2/b=3", "a=3/b=4"]
        .into_iter()
        .zip(expected_dfs)
        .map(|(p, df)| (rootdir.join(p), df))
        .collect();

    for (expected_dir, expected_df) in expected.iter() {
        assert!(expected_dir.exists());

        let ipc_paths = std::fs::read_dir(expected_dir)?
            .map(|e| {
                let entry = e?;
                Ok(entry.path())
            })
            .collect::<PolarsResult<Vec<_>>>()?;

        assert_eq!(ipc_paths.len(), 1);
        let reader = BufReader::new(polars_utils::open_file(&ipc_paths[0])?);
        let df = IpcReader::new(reader).finish()?;
        assert!(expected_df.equals(&df));
    }

    Ok(())
}
