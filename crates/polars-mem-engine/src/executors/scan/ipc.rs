use hive::HivePartitions;
use polars_core::config;
use polars_core::utils::accumulate_dataframes_vertical;
use polars_error::feature_gated;
use polars_io::cloud::CloudOptions;
use polars_io::path_utils::is_cloud_url;
use polars_io::predicates::{apply_predicate, SkipBatchPredicate};
use polars_utils::mmap::MemSlice;
use polars_utils::open_file;
use rayon::prelude::*;

use super::*;
use crate::ScanPredicate;

pub struct IpcExec {
    pub(crate) sources: ScanSources,
    pub(crate) file_info: FileInfo,
    pub(crate) predicate: Option<ScanPredicate>,
    #[allow(dead_code)]
    pub(crate) options: IpcScanOptions,
    pub(crate) file_options: FileScanOptions,
    pub(crate) hive_parts: Option<Arc<Vec<HivePartitions>>>,
    pub(crate) cloud_options: Option<CloudOptions>,
    pub(crate) metadata: Option<Arc<arrow::io::ipc::read::FileMetadata>>,
}

impl IpcExec {
    fn read(&mut self) -> PolarsResult<DataFrame> {
        let is_cloud = match &self.sources {
            ScanSources::Paths(paths) => paths.iter().any(is_cloud_url),
            ScanSources::Files(_) | ScanSources::Buffers(_) => false,
        };
        let force_async = config::force_async();

        let mut out = if is_cloud || (self.sources.is_paths() && force_async) {
            feature_gated!("cloud", {
                if force_async && config::verbose() {
                    eprintln!("ASYNC READING FORCED");
                }

                polars_io::pl_async::get_runtime().block_on_potential_spawn(self.read_async())?
            })
        } else {
            self.read_sync()?
        };

        if self.file_options.rechunk {
            out.as_single_chunk_par();
        }

        Ok(out)
    }

    fn read_impl(
        &mut self,
        idx_to_cached_file: impl Fn(usize) -> Option<PolarsResult<std::fs::File>> + Send + Sync,
    ) -> PolarsResult<DataFrame> {
        if config::verbose() {
            eprintln!("executing ipc read sync with row_index = {:?}, n_rows = {:?}, predicate = {:?} for paths {:?}",
                self.file_options.row_index.as_ref(),
                self.file_options.slice.map(|x| {
                    assert_eq!(x.0, 0);
                    x.1
                }).as_ref(),
                self.predicate.is_some(),
                self.sources,
            );
        }

        let projection = materialize_projection(
            self.file_options.with_columns.as_deref(),
            &self.file_info.schema,
            None,
            self.file_options.row_index.is_some(),
        );

        let read_path = |index: usize, n_rows: Option<usize>| {
            let source = self.sources.at(index);

            let memslice = match source {
                ScanSourceRef::Path(path) => {
                    let file = match idx_to_cached_file(index) {
                        None => open_file(path)?,
                        Some(f) => f?,
                    };

                    MemSlice::from_file(&file)?
                },
                ScanSourceRef::File(file) => MemSlice::from_file(file)?,
                ScanSourceRef::Buffer(buff) => buff.clone(),
            };

            IpcReader::new(std::io::Cursor::new(memslice))
                .with_n_rows(n_rows)
                .with_row_index(self.file_options.row_index.clone())
                .with_projection(projection.clone())
                .with_hive_partition_columns(
                    self.hive_parts
                        .as_ref()
                        .map(|x| x[index].materialize_partition_columns()),
                )
                .with_include_file_path(
                    self.file_options
                        .include_file_paths
                        .as_ref()
                        .map(|x| (x.clone(), Arc::from(source.to_include_path_name()))),
                )
                .finish()
        };

        let mut dfs = if let Some(mut n_rows) = self.file_options.slice.map(|x| {
            assert_eq!(x.0, 0);
            x.1
        }) {
            let mut out = Vec::with_capacity(self.sources.len());

            for i in 0..self.sources.len() {
                let df = read_path(i, Some(n_rows))?;
                let df_height = df.height();
                out.push(df);

                assert!(
                    df_height <= n_rows,
                    "impl error: got more rows than expected"
                );
                if df_height == n_rows {
                    break;
                }
                n_rows -= df_height;
            }

            out
        } else {
            POOL.install(|| {
                (0..self.sources.len())
                    .into_par_iter()
                    .map(|i| read_path(i, None))
                    .collect::<PolarsResult<Vec<_>>>()
            })?
        };

        if let Some(ref row_index) = self.file_options.row_index {
            let mut offset = 0;
            for df in &mut dfs {
                df.apply(&row_index.name, |series| series.idx().unwrap() + offset)
                    .unwrap();
                offset += df.height();
            }
        };

        let dfs = if let Some(predicate) = self.predicate.clone() {
            let predicate = phys_expr_to_io_expr(predicate.predicate);
            let predicate = Some(predicate.as_ref());

            POOL.install(|| {
                dfs.into_par_iter()
                    .map(|mut df| {
                        apply_predicate(&mut df, predicate, true)?;
                        Ok(df)
                    })
                    .collect::<PolarsResult<Vec<_>>>()
            })?
        } else {
            dfs
        };

        accumulate_dataframes_vertical(dfs)
    }

    fn read_sync(&mut self) -> PolarsResult<DataFrame> {
        self.read_impl(|_| None)
    }

    #[cfg(feature = "cloud")]
    async fn read_async(&mut self) -> PolarsResult<DataFrame> {
        // TODO: Better async impl that can download only the parts of the file it needs, and do it
        // concurrently.
        use polars_io::file_cache::init_entries_from_uri_list;

        let paths = self.sources.into_paths().unwrap();

        tokio::task::block_in_place(|| {
            let cache_entries = init_entries_from_uri_list(
                paths
                    .iter()
                    .map(|x| Arc::from(x.to_str().unwrap()))
                    .collect::<Vec<_>>()
                    .as_slice(),
                self.cloud_options.as_ref(),
            )?;

            self.read_impl(|i| Some(cache_entries[i].try_open_check_latest()))
        })
    }
}

impl ScanExec for IpcExec {
    fn read(
        &mut self,
        with_columns: Option<Arc<[PlSmallStr]>>,
        slice: Option<(usize, usize)>,
        predicate: Option<ScanPredicate>,
        _skip_batch_predicate: Option<Arc<dyn SkipBatchPredicate>>,
        row_index: Option<polars_io::RowIndex>,
    ) -> PolarsResult<DataFrame> {
        self.file_options.with_columns = with_columns;
        self.file_options.slice = slice.map(|(s, l)| (s as i64, l));
        self.predicate = predicate;
        self.file_options.row_index = row_index;

        if self.file_info.reader_schema.is_none() {
            self.schema()?;
        }
        self.read()
    }

    fn schema(&mut self) -> PolarsResult<&SchemaRef> {
        if self.file_info.reader_schema.is_some() {
            return Ok(&self.file_info.schema);
        }

        let arrow_schema = match &self.metadata {
            None => {
                // @TODO!: Cache the memslice here.
                let memslice = self
                    .sources
                    .at(0)
                    .to_memslice_async_assume_latest(self.sources.is_cloud_url())?;
                IpcReader::new(std::io::Cursor::new(memslice)).schema()?
            },
            Some(md) => md.schema.clone(),
        };
        self.file_info.schema =
            Arc::new(Schema::from_iter(arrow_schema.iter().map(
                |(name, field)| (name.clone(), DataType::from_arrow_field(field)),
            )));
        self.file_info.reader_schema = Some(arrow::Either::Left(arrow_schema));

        Ok(&self.file_info.schema)
    }

    fn num_unfiltered_rows(&mut self) -> PolarsResult<IdxSize> {
        let (lb, ub) = self.file_info.row_estimation;
        if lb.is_some_and(|lb| lb == ub) {
            return Ok(ub as IdxSize);
        }

        // @TODO!: Cache the memslice here.
        let memslice = self
            .sources
            .at(0)
            .to_memslice_async_assume_latest(self.sources.is_cloud_url())?;
        let mut reader = std::io::Cursor::new(memslice);

        let num_unfiltered_rows = match &self.metadata {
            None => arrow::io::ipc::read::get_row_count(&mut reader)?,
            Some(md) => arrow::io::ipc::read::get_row_count_from_blocks(&mut reader, &md.blocks)?,
        } as usize;

        self.file_info.row_estimation = (Some(num_unfiltered_rows), num_unfiltered_rows);

        Ok(num_unfiltered_rows as IdxSize)
    }
}

impl Executor for IpcExec {
    fn execute(&mut self, state: &mut ExecutionState) -> PolarsResult<DataFrame> {
        let profile_name = if state.has_node_timer() {
            let mut ids = vec![self.sources.id()];
            if self.predicate.is_some() {
                ids.push("predicate".into())
            }
            let name = comma_delimited("ipc".to_string(), &ids);
            Cow::Owned(name)
        } else {
            Cow::Borrowed("")
        };

        state.record(|| self.read(), profile_name)
    }
}
