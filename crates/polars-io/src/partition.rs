//! Functionality for writing a DataFrame partitioned into multiple files.

use std::fs::File;
use std::io::BufWriter;
use std::path::{Path, PathBuf};

use polars_core::prelude::*;
use polars_core::series::IsSorted;
use polars_core::POOL;
use rayon::prelude::*;

use crate::utils::resolve_homedir;
use crate::WriterFactory;

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
        let groups = df.group_by(self.by.clone())?;
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
                            // and sorted
                            let mut part_df = unsafe {
                                df._take_unchecked_slice_sorted(group, false, IsSorted::Ascending)
                            };
                            self.write_partition_df(&mut part_df, i)
                        })
                        .collect::<PolarsResult<Vec<_>>>()
                },
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

/// `partition_df` must be created in the same way as `partition_by`.
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

/// Creates an iterator of (hive partition path, DataFrame) pairs, e.g.:
/// ("a=1/b=1", DataFrame)
pub fn get_hive_partitions_iter<'a, S>(
    df: &'a DataFrame,
    partition_by: &'a [S],
) -> PolarsResult<Box<dyn Iterator<Item = (String, DataFrame)> + 'a>>
where
    S: AsRef<str>,
{
    for x in partition_by.iter() {
        if df.get_column_index(x.as_ref()).is_none() {
            polars_bail!(ColumnNotFound: "{}", x.as_ref());
        }
    }

    let get_hive_path_part = |df: &DataFrame| {
        const CHAR_SET: &percent_encoding::AsciiSet =
            &percent_encoding::CONTROLS.add(b'/').add(b'=').add(b':');

        partition_by
            .iter()
            .map(|x| {
                let s = df.column(x.as_ref()).unwrap().slice(0, 1);

                format!(
                    "{}={}",
                    s.name(),
                    percent_encoding::percent_encode(
                        s.cast(&DataType::String)
                            .unwrap()
                            .str()
                            .unwrap()
                            .get(0)
                            .unwrap_or("__HIVE_DEFAULT_PARTITION__")
                            .as_bytes(),
                        CHAR_SET
                    )
                )
            })
            .collect::<Vec<_>>()
            .join("/")
    };

    let groups = df.group_by(partition_by)?;
    let groups = groups.take_groups();

    let out: Box<dyn Iterator<Item = (String, DataFrame)>> = match groups {
        GroupsProxy::Idx(idx) => Box::new(idx.into_iter().map(move |(_, group)| {
            let part_df =
                unsafe { df._take_unchecked_slice_sorted(&group, false, IsSorted::Ascending) };
            (get_hive_path_part(&part_df), part_df)
        })),
        GroupsProxy::Slice { groups, .. } => {
            Box::new(groups.into_iter().map(move |[offset, len]| {
                let part_df = df.slice(offset as i64, len as usize);
                (get_hive_path_part(&part_df), part_df)
            }))
        },
    };

    Ok(out)
}
