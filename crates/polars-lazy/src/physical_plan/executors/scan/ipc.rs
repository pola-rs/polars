use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::RwLock;

use polars_core::config;
use polars_core::utils::accumulate_dataframes_vertical;
#[cfg(feature = "cloud")]
use polars_io::cloud::CloudOptions;
use polars_io::is_cloud_url;
use polars_io::predicates::apply_predicate;
use rayon::prelude::*;

use super::*;

pub struct IpcExec {
    pub(crate) paths: Arc<[PathBuf]>,
    pub(crate) schema: SchemaRef,
    pub(crate) predicate: Option<Arc<dyn PhysicalExpr>>,
    pub(crate) options: IpcScanOptions,
    pub(crate) file_options: FileScanOptions,
    #[cfg(feature = "cloud")]
    pub(crate) cloud_options: Option<CloudOptions>,
    pub(crate) metadata: Option<arrow::io::ipc::read::FileMetadata>,
}

fn prefix_sum_in_place<'a, I: IntoIterator<Item = &'a mut IdxSize>>(values: I) {
    let mut values = values.into_iter();
    let Some(first) = values.next() else {
        return;
    };
    let mut sum = *first;
    *first = 0;
    for val in values {
        let new_sum = sum + *val;
        *val = sum;
        sum = new_sum;
    }
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
            println!("executing ipc read sync with row_index = {:?}, n_rows = {:?}, predicate = {:?} for paths {:?}",
                self.file_options.row_index.as_ref(),
                self.file_options.n_rows.as_ref(),
                self.predicate.is_some(),
                self.paths
            );
        }

        let (projection, predicate) = prepare_scan_args(
            self.predicate.clone(),
            &mut self.file_options.with_columns,
            &mut self.schema,
            self.file_options.row_index.is_some(),
            None,
        );

        // TODO: Make `n_rows: IdxSize`.
        let n_rows = self
            .file_options
            .n_rows
            .map(|n| IdxSize::try_from(n).unwrap());

        let row_limit = n_rows.unwrap_or(IdxSize::MAX);

        // Used to determine the next file to open. This guarantees the order.
        let path_index = AtomicUsize::new(0);
        let row_counter = RwLock::new(ConsecutiveCountState::new(self.paths.len()));

        let mut index_and_dfs = (0..self.paths.len())
            .into_par_iter()
            .map(|_| -> PolarsResult<(usize, DataFrame)> {
                let index = path_index.fetch_add(1, Ordering::SeqCst);
                let path = &self.paths[index];

                let already_read_in_sequence = row_counter.read().unwrap().sum();
                if already_read_in_sequence >= row_limit {
                    return Ok(Default::default());
                }

                let file = std::fs::File::open(path)?;

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
                    .memory_mapped(self.options.memmap)
                    .finish()?;
                // TODO: We can not supply a filter until the readers return
                // how many total rows have been read before applying the
                // filter. Without that, we can not correctly compute the
                // pre-filter row count.
                // .finish_with_scan_ops(
                //     predicate.clone(),
                //     verbose,
                // )?;

                row_counter
                    .write()
                    .unwrap()
                    .write(index, df.height().try_into().unwrap());

                Ok((index, df))
            })
            .collect::<PolarsResult<Vec<_>>>()?;

        index_and_dfs.sort_unstable_by(|(a, _), (b, _)| a.cmp(b));

        let offsets = {
            let mut counts = row_counter.into_inner().unwrap().into_counts();
            prefix_sum_in_place(&mut counts[..]);
            counts
        };

        // Update the row indices now that we know how many rows are in each
        // file.
        if let Some(row_index) = self.file_options.row_index.as_ref() {
            for &mut (index, ref mut df) in index_and_dfs.iter_mut() {
                let offset = offsets[index];
                df.apply(&row_index.name, |series| {
                    series.idx().expect("index column should be of index type") + offset
                })
                .expect("index column should exist");
            }
        }

        // Accumulate dataframes while adjusting for the possibility of having
        // read more rows than the limit.
        let mut df =
            accumulate_dataframes_vertical(index_and_dfs.into_iter().filter_map(|(index, df)| {
                let count: IdxSize = df.height().try_into().unwrap();
                if count == 0 {
                    return None;
                }
                let remaining = row_limit.checked_sub(offsets[index])?;
                Some(if remaining < count {
                    df.slice(0, remaining.try_into().unwrap())
                } else {
                    df
                })
            }))?;

        apply_predicate(&mut df, predicate.as_deref(), true)?;

        Ok(df)
    }

    #[cfg(feature = "cloud")]
    async fn read_async(&mut self, verbose: bool) -> PolarsResult<DataFrame> {
        let predicate = self.predicate.clone().map(phys_expr_to_io_expr);

        let mut dfs = vec![];

        for path in self.paths.iter() {
            let reader =
                IpcReaderAsync::from_uri(path.to_str().unwrap(), self.cloud_options.as_ref())
                    .await?;
            dfs.push(
                reader
                    .data(
                        self.metadata.as_ref(),
                        IpcReadOptions::default()
                            .with_row_limit(self.file_options.n_rows)
                            .with_row_index(self.file_options.row_index.clone())
                            .with_projection(self.file_options.with_columns.as_deref().cloned())
                            .with_predicate(predicate.clone()),
                        verbose,
                    )
                    .await?,
            );
        }

        accumulate_dataframes_vertical(dfs)

        // TODO: WIP
        // let paths = self.paths.clone();
        // let cloud_options = self.cloud_options.clone();
        // let metadata
        // use futures::{stream::{self, StreamExt}, TryStreamExt};
        // let dfs = stream::iter(&*paths).map(
        //     move |path| async move {
        //         let reader =
        //             IpcReaderAsync::from_uri(path.to_str().unwrap(), cloud_options.as_ref())
        //                 .await?;
        //         reader
        //                 .data(
        //                     self.metadata.as_ref(),
        //                     IpcReadOptions::default()
        //                         .with_row_limit(self.file_options.n_rows)
        //                         .with_row_index(self.file_options.row_index.clone())
        //                         .with_projection(self.file_options.with_columns.as_deref().cloned())
        //                         .with_predicate(predicate.clone()),
        //                     verbose,
        //                 )
        //                 .await
        //     }
        // ).buffer_unordered(100).try_collect::<Vec<_>>().await?;
        // accumulate_dataframes_vertical(dfs)
    }
}

impl Executor for IpcExec {
    fn execute(&mut self, state: &mut ExecutionState) -> PolarsResult<DataFrame> {
        let finger_print = FileFingerPrint {
            paths: Arc::clone(&self.paths),
            #[allow(clippy::useless_asref)]
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
            let name = comma_delimited("ipc".to_string(), &ids);
            Cow::Owned(name)
        } else {
            Cow::Borrowed("")
        };

        state.record(
            || {
                state
                    .file_cache
                    .read(finger_print, self.file_options.file_counter, &mut || {
                        self.read(state.verbose())
                    })
            },
            profile_name,
        )
    }
}

// Tracks the sum of consecutive values in a dynamically sized array where the values can be written
// in any order.
struct ConsecutiveCountState {
    counts: Box<[IdxSize]>,
    next_index: usize,
    sum: IdxSize,
}

impl ConsecutiveCountState {
    fn new(len: usize) -> Self {
        Self {
            counts: vec![IdxSize::MAX; len].into_boxed_slice(),
            next_index: 0,
            sum: 0,
        }
    }

    /// Sum of all consecutive counts.
    fn sum(&self) -> IdxSize {
        self.sum
    }

    /// Write count at index.
    fn write(&mut self, index: usize, count: IdxSize) {
        debug_assert!(
            self.counts[index] == IdxSize::MAX,
            "second write to same index"
        );
        debug_assert!(count != IdxSize::MAX, "count can not be IdxSize::MAX");

        self.counts[index] = count;

        // Update sum and next index.
        while self.next_index < self.counts.len() {
            let count = self.counts[self.next_index];
            if count == IdxSize::MAX {
                break;
            }
            self.sum += count;
            self.next_index += 1;
        }
    }

    fn into_counts(mut self) -> Box<[IdxSize]> {
        // Set the counts to which no value has been written to zero.
        for i in self.next_index..self.counts.len() {
            if self.counts[i] == IdxSize::MAX {
                self.counts[i] = 0;
            }
        }
        self.counts
    }
}
