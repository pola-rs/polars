use crate::{utils::resolve_homedir, WriterFactory};
use polars_core::prelude::*;
use rayon::prelude::*;
use std::{
    fs::File,
    io::BufWriter,
    path::{Path, PathBuf},
};

/// partition_df must be created by the same way of partition_by
fn resolve_partition_dir<I, S>(rootdir: &Path, by: I, partition_df: &DataFrame) -> PathBuf
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    let mut path = PathBuf::new();
    path.push(resolve_homedir(rootdir));

    for key in by.into_iter() {
        let value = partition_df[key.as_ref()].get(0).to_string();
        path.push(format!("{}={}", key.as_ref(), value))
    }
    path
}

/// Write a DataFrame with disk partitioning
///
/// # Example
/// ```
/// use polars_core::prelude::*;
/// use polars_io::ipc::IpcWriterOption;
/// use polars_io::partition::PartitionedWriter;
///
/// fn example(df: &mut DataFrame) -> Result<()> {
///     let option = IpcWriterOption::default();
///     PartitionedWriter::new(option, "./rootdir", ["a", "b"])
///         .finish(df)
/// }
/// ```
///
pub struct PartitionedWriter<F> {
    option: F,
    rootdir: PathBuf,
    by: Vec<String>,
    parallel: bool,
}

impl<F> PartitionedWriter<F>
where
    F: WriterFactory + Send + Sync,
{
    pub fn new<P, I, S>(option: F, rootdir: P, by: I) -> Self
    where
        P: Into<PathBuf>,
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        Self {
            option,
            rootdir: rootdir.into(),
            by: by.into_iter().map(|s| s.as_ref().to_string()).collect(),
            parallel: true,
        }
    }

    /// Write the parquet file in parallel (default).
    pub fn with_parallel(mut self, pararell: bool) -> Self {
        self.parallel = pararell;
        self
    }

    fn write_partition_df(&self, partition_df: &mut DataFrame, i: usize) -> Result<()> {
        let mut path = resolve_partition_dir(&self.rootdir, &self.by, partition_df);
        std::fs::create_dir_all(&path)?;

        path.push(format!(
            "data-{:04}.{}",
            i,
            self.option.extension().display()
        ));

        let file = std::fs::File::create(path)?;
        let writer = BufWriter::new(file);

        self.option
            .create_writer::<BufWriter<File>>(writer)
            .finish(partition_df)
    }

    pub fn finish(self, df: &DataFrame) -> Result<()> {
        df._partition_by_impl(&self.by, false)?
            .enumerate()
            .map(|(i, mut part_df)| self.write_partition_df(&mut part_df, i))
            .collect::<Result<Vec<_>>>()?;

        Ok(())
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    #[cfg(feature = "ipc")]
    fn test_ipc_partition() -> Result<()> {
        use crate::ipc::IpcReader;
        use crate::SerReader;
        use std::{io::BufReader, path::PathBuf};

        use tempdir::TempDir;

        use crate::prelude::IpcWriterOption;

        let tempdir = TempDir::new("ipc-partition")?;

        let df = df!("a" => [1, 1, 2, 3], "b" => [2, 2, 3, 4], "c" => [2, 3, 4, 5]).unwrap();
        let by = ["a", "b"];
        let rootdir = tempdir.path();

        let option = IpcWriterOption::new();

        PartitionedWriter::new(option, &rootdir, by).finish(&df)?;

        let expected_dfs = [
            df!("a" => [1, 1], "b" => [2, 2], "c" => [2, 3])?,
            df!("a" => [2], "b" => [3], "c" => [4])?,
            df!("a" => [3], "b" => [4], "c" => [5])?,
        ];

        let expected: Vec<(PathBuf, DataFrame)> = ["a=1/b=2", "a=2/b=3", "a=3/b=4"]
            .into_iter()
            .zip(expected_dfs.into_iter())
            .map(|(p, df)| (PathBuf::from(rootdir.join(p)), df))
            .collect();

        for (expected_dir, expected_df) in expected.iter() {
            assert!(expected_dir.exists());

            let ipc_paths = std::fs::read_dir(&expected_dir)?
                .map(|e| {
                    let entry = e?;
                    Ok(entry.path())
                })
                .collect::<Result<Vec<_>>>()?;

            assert_eq!(ipc_paths.len(), 1);
            let reader = BufReader::new(std::fs::File::open(&ipc_paths[0])?);
            let df = IpcReader::new(reader).finish()?;
            assert!(expected_df.frame_equal(&df));
        }

        Ok(())
    }
}
