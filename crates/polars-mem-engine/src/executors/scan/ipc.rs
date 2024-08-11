use std::path::PathBuf;

use hive::HivePartitions;
use polars_core::config;
use polars_core::utils::accumulate_dataframes_vertical;
use polars_io::cloud::CloudOptions;
use polars_io::path_utils::is_cloud_url;
use polars_io::predicates::apply_predicate;
use rayon::prelude::*;

use super::*;

pub struct IpcExec {
    pub(crate) paths: Arc<Vec<PathBuf>>,
    pub(crate) file_info: FileInfo,
    pub(crate) predicate: Option<Arc<dyn PhysicalExpr>>,
    pub(crate) options: IpcScanOptions,
    pub(crate) file_options: FileScanOptions,
    pub(crate) hive_parts: Option<Arc<Vec<HivePartitions>>>,
    pub(crate) cloud_options: Option<CloudOptions>,
}

impl IpcExec {
    fn read(&mut self) -> PolarsResult<DataFrame> {
        let is_cloud = self.paths.iter().any(is_cloud_url);
        let force_async = config::force_async();

        let mut out = if is_cloud || force_async {
            #[cfg(not(feature = "cloud"))]
            {
                panic!("activate cloud feature")
            }

            #[cfg(feature = "cloud")]
            {
                if force_async && config::verbose() {
                    eprintln!("ASYNC READING FORCED");
                }

                polars_io::pl_async::get_runtime().block_on_potential_spawn(self.read_async())?
            }
        } else {
            self.read_sync()?
        };

        if self.file_options.rechunk {
            out.as_single_chunk_par();
        }

        Ok(out)
    }

    fn read_impl<F: Fn(usize) -> PolarsResult<std::fs::File> + Send + Sync>(
        &mut self,
        path_idx_to_file: F,
    ) -> PolarsResult<DataFrame> {
        if config::verbose() {
            eprintln!("executing ipc read sync with row_index = {:?}, n_rows = {:?}, predicate = {:?} for paths {:?}",
                self.file_options.row_index.as_ref(),
                self.file_options.slice.map(|x| {
                    assert_eq!(x.0, 0);
                    x.1
                }).as_ref(),
                self.predicate.is_some(),
                self.paths
            );
        }

        let projection = materialize_projection(
            self.file_options.with_columns.as_deref(),
            &self.file_info.schema,
            None,
            self.file_options.row_index.is_some(),
        );

        let read_path = |path_index: usize, n_rows: Option<usize>| {
            IpcReader::new(path_idx_to_file(path_index)?)
                .with_n_rows(n_rows)
                .with_row_index(self.file_options.row_index.clone())
                .with_projection(projection.clone())
                .with_hive_partition_columns(
                    self.hive_parts
                        .as_ref()
                        .map(|x| x[path_index].materialize_partition_columns()),
                )
                .with_include_file_path(self.file_options.include_file_paths.as_ref().map(|x| {
                    (
                        x.clone(),
                        Arc::from(self.paths[path_index].to_str().unwrap().to_string()),
                    )
                }))
                .memory_mapped(
                    self.options
                        .memory_map
                        .then(|| self.paths[path_index].clone()),
                )
                .finish()
        };

        let mut dfs = if let Some(mut n_rows) = self.file_options.slice.map(|x| {
            assert_eq!(x.0, 0);
            x.1
        }) {
            let mut out = Vec::with_capacity(self.paths.len());

            for i in 0..self.paths.len() {
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
                (0..self.paths.len())
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
            let predicate = phys_expr_to_io_expr(predicate);
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
        let paths = self.paths.clone();
        self.read_impl(move |i| std::fs::File::open(&paths[i]).map_err(Into::into))
    }

    #[cfg(feature = "cloud")]
    async fn read_async(&mut self) -> PolarsResult<DataFrame> {
        // TODO: Better async impl that can download only the parts of the file it needs, and do it
        // concurrently.
        use polars_io::file_cache::init_entries_from_uri_list;

        tokio::task::block_in_place(|| {
            let cache_entries = init_entries_from_uri_list(
                self.paths
                    .iter()
                    .map(|x| Arc::from(x.to_str().unwrap()))
                    .collect::<Vec<_>>()
                    .as_slice(),
                self.cloud_options.as_ref(),
            )?;

            self.read_impl(move |i| cache_entries[i].try_open_check_latest())
        })
    }
}

impl Executor for IpcExec {
    fn execute(&mut self, state: &mut ExecutionState) -> PolarsResult<DataFrame> {
        let profile_name = if state.has_node_timer() {
            let mut ids = vec![self.paths[0].to_string_lossy().into()];
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
