use std::path::{Path, PathBuf};

use polars_core::utils::accumulate_dataframes_vertical;
use polars_core::utils::arrow::io::parquet::read::FileMetaData;
use polars_io::cloud::CloudOptions;
use polars_io::{is_cloud_url, RowCount};

use super::*;

pub struct ParquetExec {
    paths: Arc<[PathBuf]>,
    file_info: FileInfo,
    predicate: Option<Arc<dyn PhysicalExpr>>,
    options: ParquetOptions,
    #[allow(dead_code)]
    cloud_options: Option<CloudOptions>,
    file_options: FileScanOptions,
    metadata: Option<Arc<FileMetaData>>,
}

impl ParquetExec {
    pub(crate) fn new(
        paths: Arc<[PathBuf]>,
        file_info: FileInfo,
        predicate: Option<Arc<dyn PhysicalExpr>>,
        options: ParquetOptions,
        cloud_options: Option<CloudOptions>,
        file_options: FileScanOptions,
        metadata: Option<Arc<FileMetaData>>,
    ) -> Self {
        ParquetExec {
            paths,
            file_info,
            predicate,
            options,
            cloud_options,
            file_options,
            metadata,
        }
    }

    fn read_par(&mut self) -> PolarsResult<Vec<DataFrame>> {
        let parallel = match self.options.parallel {
            ParallelStrategy::Auto if self.paths.len() > POOL.current_num_threads() => {
                ParallelStrategy::RowGroups
            },
            identity => identity,
        };
        let n_rows = self.file_options.n_rows;

        // self.paths.iter().map(|path| {
        //     let (file, projection, predicate) = prepare_scan_args(
        //         path,
        //         &self.predicate,
        //         &mut self.file_options.with_columns.clone(),
        //         &mut self.file_info.schema.clone(),
        //         self.file_options.row_count.is_some(),
        //         hive_partitions.as_deref(),
        //     );
        //
        //     let reader = if let Some(file) = file {
        //         ParquetReader::new(file)
        //             .with_schema(Some(self.file_info.reader_schema.clone()))
        //             .read_parallel(parallel)
        //             .set_low_memory(self.options.low_memory)
        //             .use_statistics(self.options.use_statistics)
        //             .with_hive_partition_columns(hive_partitions);
        //             )
        //     } else {
        //         polars_bail!(ComputeError: "could not read {}", path.display())
        //     }?;
        //
        //
        //
        // })

        POOL.install(|| {
            self.paths
                .par_iter()
                .map(|path| {
                    let mut file_info = self.file_info.clone();
                    file_info.update_hive_partitions(path);

                    let hive_partitions = file_info
                        .hive_parts
                        .as_ref()
                        .map(|hive| hive.materialize_partition_columns());

                    let (file, projection, predicate) = prepare_scan_args(
                        path,
                        &self.predicate,
                        &mut self.file_options.with_columns.clone(),
                        &mut self.file_info.schema.clone(),
                        self.file_options.row_count.is_some(),
                        hive_partitions.as_deref(),
                    );

                    let df = if let Some(file) = file {
                        ParquetReader::new(file)
                            .with_schema(Some(self.file_info.reader_schema.clone()))
                            .read_parallel(parallel)
                            .with_n_rows(n_rows)
                            .set_low_memory(self.options.low_memory)
                            .use_statistics(self.options.use_statistics)
                            .with_hive_partition_columns(hive_partitions)
                            ._finish_with_scan_ops(
                                predicate,
                                projection.as_ref().map(|v| v.as_ref()),
                            )
                    } else {
                        polars_bail!(ComputeError: "could not read {}", path.display())
                    }?;
                    Ok(df)
                })
                .collect()
        })
    }

    fn read_seq(&mut self) -> PolarsResult<Vec<DataFrame>> {
        let mut base_offset = 0 as IdxSize;
        let mut n_rows_total = self.file_options.n_rows;

        self.paths
            .iter()
            .map(|path| {
                let row_count = self.file_options.row_count.as_ref().map(|rc| RowCount {
                    name: rc.name.clone(),
                    offset: rc.offset + base_offset,
                });

                self.file_info.update_hive_partitions(path);

                let hive_partitions = self
                    .file_info
                    .hive_parts
                    .as_ref()
                    .map(|hive| hive.materialize_partition_columns());

                let (file, projection, predicate) = prepare_scan_args(
                    path,
                    &self.predicate,
                    &mut self.file_options.with_columns.clone(),
                    &mut self.file_info.schema.clone(),
                    self.file_options.row_count.is_some(),
                    hive_partitions.as_deref(),
                );

                let df = if let Some(file) = file {
                    ParquetReader::new(file)
                        .with_schema(Some(self.file_info.reader_schema.clone()))
                        .with_n_rows(n_rows_total)
                        .read_parallel(self.options.parallel)
                        .with_row_count(row_count)
                        .set_low_memory(self.options.low_memory)
                        .use_statistics(self.options.use_statistics)
                        .with_hive_partition_columns(hive_partitions)
                        ._finish_with_scan_ops(predicate, projection.as_ref().map(|v| v.as_ref()))
                } else {
                    polars_bail!(ComputeError: "could not read {}", path.display())
                }?;
                let read = df.height();
                base_offset += read as IdxSize;
                if let Some(total) = n_rows_total.as_mut() {
                    *total -= read
                }
                Ok(df)
            })
            .collect()
    }

    #[cfg(feature = "cloud")]
    async fn read_async(&mut self) -> PolarsResult<Vec<DataFrame>> {
        let first_schema = &self.file_info.reader_schema;
        let first_metadata = &self.metadata;
        let cloud_options = self.cloud_options.as_ref();
        // First initialize the readers and get the metadata concurrently.
        let iter = self.paths.iter().enumerate().map(|(i, path)| async move {
            // use the cached one as this saves a cloud call
            let metadata = if i == 0 { first_metadata.clone() } else { None };
            let mut reader = ParquetAsyncReader::from_uri(
                &path.to_string_lossy(),
                cloud_options,
                // Schema must be the same for all files. The hive partitions are included in this schema.
                Some(first_schema.clone()),
                metadata,
            )
            .await?;
            let num_rows = reader.num_rows().await?;
            PolarsResult::Ok((num_rows, reader))
        });
        let readers_and_metadata = futures::future::try_join_all(iter).await?;

        // Then compute `n_rows` to be taken per file up front, so we can actually read concurrently
        // after this.
        let n_rows_to_read = self.file_options.n_rows.unwrap_or(usize::MAX);
        let iter = readers_and_metadata
            .iter()
            .map(|(num_rows, _)| num_rows)
            .copied();

        let rows_statistics = get_sequential_row_statistics(iter, n_rows_to_read);

        // Now read the actual data.
        let base_row_count = &self.file_options.row_count;
        let file_info = &self.file_info;
        let file_options = &self.file_options;
        let paths = &self.paths;
        let use_statistics = self.options.use_statistics;
        let predicate = &self.predicate;

        let iter = readers_and_metadata
            .into_iter()
            .zip(rows_statistics.iter())
            .zip(paths.as_ref().iter())
            .map(
                |(
                    ((num_rows_this_file, reader), (remaining_rows_to_read, cumulative_read)),
                    path,
                )| async move {
                    let mut file_info = file_info.clone();
                    let remaining_rows_to_read = *remaining_rows_to_read;
                    let remaining_rows_to_read = if num_rows_this_file < remaining_rows_to_read {
                        None
                    } else {
                        Some(remaining_rows_to_read)
                    };
                    let row_count = base_row_count.as_ref().map(|rc| RowCount {
                        name: rc.name.clone(),
                        offset: rc.offset + *cumulative_read as IdxSize,
                    });

                    file_info.update_hive_partitions(path);

                    let hive_partitions = file_info
                        .hive_parts
                        .as_ref()
                        .map(|hive| hive.materialize_partition_columns());

                    let (_, projection, predicate) = prepare_scan_args(
                        Path::new(""),
                        predicate,
                        &mut file_options.with_columns.clone(),
                        &mut file_info.schema.clone(),
                        row_count.is_some(),
                        hive_partitions.as_deref(),
                    );

                    reader
                        .with_n_rows(remaining_rows_to_read)
                        .with_row_count(row_count)
                        .with_projection(projection)
                        .use_statistics(use_statistics)
                        .with_predicate(predicate)
                        .with_hive_partition_columns(hive_partitions)
                        .finish()
                        .await
                        .map(Some)
                },
            );

        let dfs = futures::future::try_join_all(iter).await?;
        Ok(dfs.into_iter().flatten().collect())
    }

    fn read(&mut self) -> PolarsResult<DataFrame> {
        let is_cloud = is_cloud_url(self.paths[0].as_path());
        let force_async = std::env::var("POLARS_FORCE_ASYNC").as_deref().unwrap_or("") == "1";

        let out = if self.file_options.n_rows.is_some()
            || self.file_options.row_count.is_some()
            || is_cloud
            || force_async
        {
            if is_cloud || force_async {
                #[cfg(not(feature = "cloud"))]
                {
                    panic!("activate cloud feature")
                }

                #[cfg(feature = "cloud")]
                {
                    polars_io::pl_async::get_runtime()
                        .block_on_potential_spawn(self.read_async())?
                }
            } else {
                self.read_seq()?
            }
        } else {
            self.read_par()?
        };

        accumulate_dataframes_vertical(out)
    }
}

impl Executor for ParquetExec {
    fn execute(&mut self, state: &mut ExecutionState) -> PolarsResult<DataFrame> {
        let finger_print = FileFingerPrint {
            paths: self.paths.clone(),
            predicate: self
                .predicate
                .as_ref()
                .map(|ae| ae.as_expression().unwrap().clone()),
            slice: (0, self.file_options.n_rows),
        };

        let profile_name = if state.has_node_timer() {
            let mut ids = vec![self.paths[0].to_string_lossy().into()];
            if self.predicate.is_some() {
                ids.push("predicate".into())
            }
            let name = comma_delimited("parquet".to_string(), &ids);
            Cow::Owned(name)
        } else {
            Cow::Borrowed("")
        };

        state.record(
            || {
                state
                    .file_cache
                    .read(finger_print, self.file_options.file_counter, &mut || {
                        self.read()
                    })
            },
            profile_name,
        )
    }
}
