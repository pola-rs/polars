use std::fs::File;
use std::io::BufWriter;
use std::path::{Path, PathBuf};

use polars_core::prelude::*;
use polars_core::POOL;
use rayon::prelude::*;

use crate::utils::resolve_homedir;
use crate::WriterFactory;

/// partition_df must be created by the same way of partition_by
fn resolve_partition_dir<I, S>(rootdir: &Path, by: I, partition_df: &DataFrame) -> PathBuf
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    let mut path = PathBuf::new();
    path.push(resolve_homedir(rootdir));

    for key in by.into_iter() {
        let value = partition_df[key.as_ref()].get(0).unwrap().to_string();
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
/// fn example(df: &mut DataFrame) -> PolarsResult<()> {
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
    pub fn with_parallel(mut self, parallel: bool) -> Self {
        self.parallel = parallel;
        self
    }

    fn write_partition_df(&self, partition_df: &mut DataFrame, i: usize) -> PolarsResult<()> {
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

    pub fn finish(self, df: &DataFrame) -> PolarsResult<()> {
        let groups = df.groupby(self.by.clone())?;
        let groups = groups.get_groups();

        // don't parallelize this
        // there is a lot of parallelization in take and this may easily SO
        POOL.install(|| {
            match groups {
                GroupsProxy::Idx(idx) => {
                    idx.par_iter()
                        .enumerate()
                        .map(|(i, (_, group))| {
                            // groups are in bounds
                            let mut part_df = unsafe { df._take_unchecked_slice(group, false) };
                            self.write_partition_df(&mut part_df, i)
                        })
                        .collect::<PolarsResult<Vec<_>>>()
                }
                GroupsProxy::Slice { groups, .. } => groups
                    .par_iter()
                    .enumerate()
                    .map(|(i, [first, len])| {
                        let mut part_df = df.slice(*first as i64, *len as usize);
                        self.write_partition_df(&mut part_df, i)
                    })
                    .collect::<PolarsResult<Vec<_>>>(),
            }
        })?;

        Ok(())
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    #[cfg(feature = "ipc")]
    fn test_ipc_partition() -> PolarsResult<()> {
        use std::io::BufReader;
        use std::path::PathBuf;

        use tempdir::TempDir;

        use crate::ipc::IpcReader;
        use crate::prelude::IpcWriterOption;
        use crate::SerReader;

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
                .collect::<PolarsResult<Vec<_>>>()?;

            assert_eq!(ipc_paths.len(), 1);
            let reader = BufReader::new(std::fs::File::open(&ipc_paths[0])?);
            let df = IpcReader::new(reader).finish()?;
            assert!(expected_df.frame_equal(&df));
        }

        Ok(())
    }
}
