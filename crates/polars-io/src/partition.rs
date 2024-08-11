//! Functionality for writing a DataFrame partitioned into multiple files.

use std::path::Path;

use polars_core::prelude::*;
use polars_core::series::IsSorted;
use polars_core::POOL;
use rayon::prelude::*;

use crate::parquet::write::ParquetWriteOptions;
#[cfg(feature = "ipc")]
use crate::prelude::IpcWriterOptions;
use crate::prelude::URL_ENCODE_CHAR_SET;
use crate::{SerWriter, WriteDataFrameToFile};

impl WriteDataFrameToFile for ParquetWriteOptions {
    fn write_df_to_file<W: std::io::Write>(&self, mut df: DataFrame, file: W) -> PolarsResult<()> {
        self.to_writer(file).finish(&mut df)?;
        Ok(())
    }
}

#[cfg(feature = "ipc")]
impl WriteDataFrameToFile for IpcWriterOptions {
    fn write_df_to_file<W: std::io::Write>(&self, mut df: DataFrame, file: W) -> PolarsResult<()> {
        self.to_writer(file).finish(&mut df)?;
        Ok(())
    }
}

/// Write a partitioned parquet dataset. This functionality is unstable.
pub fn write_partitioned_dataset<S, O>(
    df: &mut DataFrame,
    path: &Path,
    partition_by: &[S],
    file_write_options: &O,
    chunk_size: usize,
) -> PolarsResult<()>
where
    S: AsRef<str>,
    O: WriteDataFrameToFile + Send + Sync,
{
    // Ensure we have a single chunk as the gather will otherwise rechunk per group.
    df.as_single_chunk_par();

    // Note: When adding support for formats other than Parquet, avoid writing the partitioned
    // columns into the file. We write them for parquet because they are encoded efficiently with
    // RLE and also gives us a way to get the hive schema from the parquet file for free.
    let get_hive_path_part = {
        let schema = &df.schema();

        let partition_by_col_idx = partition_by
            .iter()
            .map(|x| {
                let Some(i) = schema.index_of(x.as_ref()) else {
                    polars_bail!(col_not_found = x.as_ref())
                };
                Ok(i)
            })
            .collect::<PolarsResult<Vec<_>>>()?;

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
                            URL_ENCODE_CHAR_SET
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

    let write_part = |df: DataFrame, path: &Path| {
        let f = std::fs::File::create(path)?;
        file_write_options.write_df_to_file(df, f)?;
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
                            df._take_unchecked_slice_sorted(group, true, IsSorted::Ascending)
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
