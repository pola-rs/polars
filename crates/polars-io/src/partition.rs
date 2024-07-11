//! Functionality for writing a DataFrame partitioned into multiple files.

use std::fs::File;
use std::io::BufWriter;
use std::path::{Path, PathBuf};

use polars_core::prelude::*;
use polars_core::series::IsSorted;
use polars_core::POOL;
use rayon::prelude::*;

use crate::parquet::write::ParquetWriteOptions;
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

/// Write a partitioned parquet dataset. This functionality is unstable.
pub fn write_partitioned_dataset<S>(
    df: &DataFrame,
    path: &Path,
    partition_by: &[S],
    file_write_options: &ParquetWriteOptions,
    chunk_size: usize,
) -> PolarsResult<()>
where
    S: AsRef<str>,
{
    // Note: When adding support for formats other than Parquet, avoid writing the partitioned
    // columns into the file. We write them for parquet because they are encoded efficiently with
    // RLE and also gives us a way to get the hive schema from the parquet file for free.
    let get_hive_path_part = {
        let schema = &df.schema();

        let partition_by_col_idx = partition_by
            .iter()
            .map(|x| {
                let Some(i) = schema.index_of(x.as_ref()) else {
                    polars_bail!(ColumnNotFound: "{}", x.as_ref())
                };
                Ok(i)
            })
            .collect::<PolarsResult<Vec<_>>>()?;

        const CHAR_SET: &percent_encoding::AsciiSet = &percent_encoding::CONTROLS
            .add(b'/')
            .add(b'=')
            .add(b':')
            .add(b' ');

        move |df: &DataFrame| {
            let cols = df.get_columns();

            partition_by_col_idx
                .iter()
                .map(|&i| {
                    let s = &cols[i].slice(0, 1).cast(&DataType::String).unwrap();

                    format!(
                        "{}={}",
                        s.name(),
                        percent_encoding::percent_encode(
                            s.str()
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
        }
    };

    let base_path = path;
    let groups = df.group_by(partition_by)?.take_groups();

    let init_part_base_dir = |part_df: &DataFrame| {
        let path_part = get_hive_path_part(part_df);
        let dir = base_path.join(path_part);
        std::fs::create_dir_all(&dir)?;

        PolarsResult::Ok(dir)
    };

    fn get_path_for_index(i: usize) -> String {
        // Use a fixed-width file name so that it sorts properly.
        format!("{:08x}.parquet", i)
    }

    let get_n_files_and_rows_per_file = |part_df: &DataFrame| {
        let n_files = (part_df.estimated_size() / chunk_size).clamp(1, 0xffff_ffff);
        let rows_per_file = (df.height() / n_files).saturating_add(1);
        (n_files, rows_per_file)
    };

    let write_part = |mut df: DataFrame, path: &Path| {
        let f = std::fs::File::create(path)?;
        file_write_options.to_writer(f).finish(&mut df)?;
        PolarsResult::Ok(())
    };

    // This is sqrt(N) of the actual limit - we chunk the input both at the groups
    // proxy level and within every group.
    const MAX_OPEN_FILES: usize = 8;

    let finish_part_df = |df: DataFrame| {
        let dir_path = init_part_base_dir(&df)?;
        let (n_files, rows_per_file) = get_n_files_and_rows_per_file(&df);

        if n_files == 1 {
            write_part(df.clone(), &dir_path.join(get_path_for_index(0)))
        } else {
            (0..df.height())
                .step_by(rows_per_file)
                .enumerate()
                .collect::<Vec<_>>()
                .chunks(MAX_OPEN_FILES)
                .map(|chunk| {
                    chunk
                        .into_par_iter()
                        .map(|&(idx, slice_start)| {
                            let df = df.slice(slice_start as i64, rows_per_file);
                            write_part(df.clone(), &dir_path.join(get_path_for_index(idx)))
                        })
                        .reduce(
                            || PolarsResult::Ok(()),
                            |a, b| if a.is_err() { a } else { b },
                        )
                })
                .collect::<PolarsResult<Vec<()>>>()?;
            Ok(())
        }
    };

    POOL.install(|| match groups {
        GroupsProxy::Idx(idx) => idx
            .all()
            .chunks(MAX_OPEN_FILES)
            .map(|chunk| {
                chunk
                    .par_iter()
                    .map(|group| {
                        let df = unsafe {
                            df._take_unchecked_slice_sorted(group, false, IsSorted::Ascending)
                        };
                        finish_part_df(df)
                    })
                    .reduce(
                        || PolarsResult::Ok(()),
                        |a, b| if a.is_err() { a } else { b },
                    )
            })
            .collect::<PolarsResult<Vec<()>>>(),
        GroupsProxy::Slice { groups, .. } => groups
            .chunks(MAX_OPEN_FILES)
            .map(|chunk| {
                chunk
                    .into_par_iter()
                    .map(|&[offset, len]| {
                        let df = df.slice(offset as i64, len as usize);
                        finish_part_df(df)
                    })
                    .reduce(
                        || PolarsResult::Ok(()),
                        |a, b| if a.is_err() { a } else { b },
                    )
            })
            .collect::<PolarsResult<Vec<()>>>(),
    })?;

    Ok(())
}
