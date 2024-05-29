use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::RwLock;

use polars_core::config;
use polars_core::utils::accumulate_dataframes_vertical;
use polars_io::cloud::CloudOptions;
use polars_io::predicates::apply_predicate;
use polars_io::utils::is_cloud_url;
use polars_io::RowIndex;
use rayon::prelude::*;

use super::*;

pub struct IpcExec {
    pub(crate) paths: Arc<[PathBuf]>,
    pub(crate) schema: SchemaRef,
    pub(crate) predicate: Option<Arc<dyn PhysicalExpr>>,
    pub(crate) options: IpcScanOptions,
    pub(crate) file_options: FileScanOptions,
    pub(crate) cloud_options: Option<CloudOptions>,
    pub(crate) metadata: Option<arrow::io::ipc::read::FileMetadata>,
}

impl IpcExec {
    fn read(&mut self, verbose: bool) -> PolarsResult<DataFrame> {
        let is_cloud = self.paths.iter().any(is_cloud_url);
        let mut out = if is_cloud || config::force_async() {
            #[cfg(not(feature = "cloud"))]
            {
                panic!("activate cloud feature")
            }

            #[cfg(feature = "cloud")]
            {
                if !is_cloud && verbose {
                    eprintln!("ASYNC READING FORCED");
                }

                polars_io::pl_async::get_runtime()
                    .block_on_potential_spawn(self.read_async(verbose))?
            }
        } else {
            self.read_sync()?
        };

        if self.file_options.rechunk {
            out.as_single_chunk_par();
        }

        Ok(out)
    }

    fn read_sync(&mut self) -> PolarsResult<DataFrame> {
        if config::verbose() {
            eprintln!("executing ipc read sync with row_index = {:?}, n_rows = {:?}, predicate = {:?} for paths {:?}",
                self.file_options.row_index.as_ref(),
                self.file_options.n_rows.as_ref(),
                self.predicate.is_some(),
                self.paths
            );
        }

        let projection = materialize_projection(
            self.file_options
                .with_columns
                .as_deref()
                .map(|cols| cols.deref()),
            &self.schema,
            None,
            self.file_options.row_index.is_some(),
        );

        let n_rows = self
            .file_options
            .n_rows
            .map(|n| IdxSize::try_from(n).unwrap());

        let row_limit = n_rows.unwrap_or(IdxSize::MAX);

        // Used to determine the next file to open. This guarantees the order.
        let path_index = AtomicUsize::new(0);
        let row_counter = RwLock::new(ConsecutiveCountState::new(self.paths.len()));

        let index_and_dfs = (0..self.paths.len())
            .into_par_iter()
            .map(|_| -> PolarsResult<(usize, DataFrame)> {
                let index = path_index.fetch_add(1, Ordering::Relaxed);
                let path = &self.paths[index];

                let already_read_in_sequence = row_counter.read().unwrap().sum();
                if already_read_in_sequence >= row_limit {
                    return Ok((index, Default::default()));
                }

                let file = std::fs::File::open(path)?;

                let memory_mapped = if self.options.memory_map {
                    Some(path.clone())
                } else {
                    None
                };

                let df = IpcReader::new(file)
                    .with_n_rows(
                        // NOTE: If there is any file that by itself exceeds the
                        // row limit, passing the total row limit to each
                        // individual reader helps.
                        n_rows.map(|n| {
                            n.saturating_sub(already_read_in_sequence)
                                .try_into()
                                .unwrap()
                        }),
                    )
                    .with_row_index(self.file_options.row_index.clone())
                    .with_projection(projection.clone())
                    .memory_mapped(memory_mapped)
                    .finish()?;

                row_counter
                    .write()
                    .unwrap()
                    .write(index, df.height().try_into().unwrap());

                Ok((index, df))
            })
            .collect::<PolarsResult<Vec<_>>>()?;

        finish_index_and_dfs(
            index_and_dfs,
            row_counter.into_inner().unwrap(),
            self.file_options.row_index.as_ref(),
            row_limit,
            self.predicate.as_ref(),
        )
    }

    #[cfg(feature = "cloud")]
    async fn read_async(&mut self, verbose: bool) -> PolarsResult<DataFrame> {
        use futures::stream::{self, StreamExt};
        use futures::TryStreamExt;

        /// See https://users.rust-lang.org/t/implementation-of-fnonce-is-not-general-enough-with-async-block/83427/3.
        trait AssertSend {
            fn assert_send<R>(self) -> impl Send + stream::Stream<Item = R>
            where
                Self: Send + stream::Stream<Item = R> + Sized,
            {
                self
            }
        }

        impl<T: Send + stream::Stream + Sized> AssertSend for T {}

        let n_rows = self
            .file_options
            .n_rows
            .map(|limit| limit.try_into().unwrap());

        let row_limit = n_rows.unwrap_or(IdxSize::MAX);

        let row_counter = RwLock::new(ConsecutiveCountState::new(self.paths.len()));

        let index_and_dfs = stream::iter(&*self.paths)
            .enumerate()
            .map(|(index, path)| {
                let this = &*self;
                let row_counter = &row_counter;
                async move {
                    let already_read_in_sequence = row_counter.read().unwrap().sum();
                    if already_read_in_sequence >= row_limit {
                        return Ok((index, Default::default()));
                    }

                    let reader = IpcReaderAsync::from_uri(
                        path.to_str().unwrap(),
                        this.cloud_options.as_ref(),
                    )
                    .await?;
                    let df = reader
                        .data(
                            this.metadata.as_ref(),
                            IpcReadOptions::default()
                                .with_row_limit(
                                    // NOTE: If there is any file that by itself
                                    // exceeds the row limit, passing the total
                                    // row limit to each individual reader
                                    // helps.
                                    n_rows.map(|n| {
                                        n.saturating_sub(already_read_in_sequence)
                                            .try_into()
                                            .unwrap()
                                    }),
                                )
                                .with_row_index(this.file_options.row_index.clone())
                                .with_projection(
                                    this.file_options.with_columns.as_deref().cloned(),
                                ),
                            verbose,
                        )
                        .await?;

                    row_counter
                        .write()
                        .unwrap()
                        .write(index, df.height().try_into().unwrap());

                    PolarsResult::Ok((index, df))
                }
            })
            .assert_send()
            .buffer_unordered(config::get_file_prefetch_size())
            .try_collect::<Vec<_>>()
            .await?;

        finish_index_and_dfs(
            index_and_dfs,
            row_counter.into_inner().unwrap(),
            self.file_options.row_index.as_ref(),
            row_limit,
            self.predicate.as_ref(),
        )
    }
}

fn finish_index_and_dfs(
    mut index_and_dfs: Vec<(usize, DataFrame)>,
    row_counter: ConsecutiveCountState,
    row_index: Option<&RowIndex>,
    row_limit: IdxSize,
    predicate: Option<&Arc<dyn PhysicalExpr>>,
) -> PolarsResult<DataFrame> {
    index_and_dfs.sort_unstable_by(|(a, _), (b, _)| a.cmp(b));

    debug_assert!(
        index_and_dfs.iter().enumerate().all(|(a, &(b, _))| a == b),
        "expected dataframe indices in order from 0 to len"
    );

    debug_assert_eq!(index_and_dfs.len(), row_counter.len());
    let mut offset = 0;
    let mut df = accumulate_dataframes_vertical(
        index_and_dfs
            .into_iter()
            .zip(row_counter.counts())
            .filter_map(|((_, mut df), count)| {
                let count = count?;

                let remaining = row_limit.checked_sub(offset)?;

                // If necessary, correct having read too much from a single file.
                if remaining < count {
                    df = df.slice(0, remaining.try_into().unwrap());
                }

                // If necessary, correct row indices now that we know the offset.
                if let Some(row_index) = row_index {
                    df.apply(&row_index.name, |series| {
                        series.idx().expect("index column should be of index type") + offset
                    })
                    .expect("index column should exist");
                }

                offset += count;

                Some(df)
            }),
    )?;

    let predicate = predicate.cloned().map(phys_expr_to_io_expr);
    apply_predicate(&mut df, predicate.as_deref(), true)?;

    Ok(df)
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

        state.record(|| self.read(state.verbose()), profile_name)
    }
}
