use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::RwLock;

use polars_core::config;
use polars_core::utils::accumulate_dataframes_vertical;
use polars_io::predicates::apply_predicate;
use polars_io::{is_cloud_url, RowIndex};

use super::*;

pub struct CsvExec {
    pub paths: Arc<[PathBuf]>,
    pub schema: SchemaRef,
    pub options: CsvParserOptions,
    pub file_options: FileScanOptions,
    pub predicate: Option<Arc<dyn PhysicalExpr>>,
}

impl CsvExec {
    fn read(&mut self) -> PolarsResult<DataFrame> {
        let is_cloud = self.paths.iter().any(is_cloud_url);
        let mut out = if is_cloud || config::force_async() {
            #[cfg(not(feature = "cloud"))]
            {
                panic!("activate cloud feature")
            }

            #[cfg(feature = "cloud")]
            {
                if !is_cloud && config::verbose() {
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

    fn read_sync(&mut self) -> PolarsResult<DataFrame> {
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
                let index = path_index.fetch_add(1, Ordering::SeqCst);
                let path = &self.paths[index];

                let already_read_in_sequence = row_counter.read().unwrap().sum();
                if already_read_in_sequence >= row_limit {
                    return Ok((index, Default::default()));
                }

                let df = CsvReader::from_path(path)?
                    .has_header(self.options.has_header)
                    .with_dtypes(Some(self.schema.clone()))
                    .with_separator(self.options.separator)
                    .with_ignore_errors(self.options.ignore_errors)
                    .with_skip_rows(self.options.skip_rows)
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
                    .with_projection(projection.clone())
                    .low_memory(self.options.low_memory)
                    .with_null_values(self.options.null_values.clone())
                    .with_encoding(CsvEncoding::LossyUtf8)
                    ._with_comment_prefix(self.options.comment_prefix.clone())
                    .with_quote_char(self.options.quote_char)
                    .with_end_of_line_char(self.options.eol_char)
                    .with_encoding(self.options.encoding)
                    .with_rechunk(self.file_options.rechunk)
                    .with_row_index(self.file_options.row_index.clone())
                    .with_try_parse_dates(self.options.try_parse_dates)
                    .with_n_threads(self.options.n_threads)
                    .truncate_ragged_lines(self.options.truncate_ragged_lines)
                    .raise_if_empty(self.options.raise_if_empty)
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
    async fn read_async(&mut self) -> PolarsResult<DataFrame> {
        todo!();

        // use futures::stream::{self, StreamExt};
        // use futures::TryStreamExt;

        // /// See https://users.rust-lang.org/t/implementation-of-fnonce-is-not-general-enough-with-async-block/83427/3.
        // trait AssertSend {
        //     fn assert_send<R>(self) -> impl Send + stream::Stream<Item = R>
        //     where
        //         Self: Send + stream::Stream<Item = R> + Sized,
        //     {
        //         self
        //     }
        // }

        // impl<T: Send + stream::Stream + Sized> AssertSend for T {}

        // let n_rows = self
        //     .file_options
        //     .n_rows
        //     .map(|limit| limit.try_into().unwrap());

        // let row_limit = n_rows.unwrap_or(IdxSize::MAX);

        // let row_counter = RwLock::new(ConsecutiveCountState::new(self.paths.len()));

        // let index_and_dfs = stream::iter(&*self.paths)
        //     .enumerate()
        //     .map(|(index, path)| {
        //         let this = &*self;
        //         let row_counter = &row_counter;
        //         async move {
        //             let already_read_in_sequence = row_counter.read().unwrap().sum();
        //             if already_read_in_sequence >= row_limit {
        //                 return Ok((index, Default::default()));
        //             }

        //             let reader = IpcReaderAsync::from_uri(
        //                 path.to_str().unwrap(),
        //                 this.cloud_options.as_ref(),
        //             )
        //             .await?;
        //             let df = reader
        //                 .data(
        //                     this.metadata.as_ref(),
        //                     IpcReadOptions::default()
        //                         .with_row_limit(
        //                             // NOTE: If there is any file that by itself
        //                             // exceeds the row limit, passing the total
        //                             // row limit to each individual reader
        //                             // helps.
        //                             n_rows.map(|n| {
        //                                 n.saturating_sub(already_read_in_sequence)
        //                                     .try_into()
        //                                     .unwrap()
        //                             }),
        //                         )
        //                         .with_row_index(this.file_options.row_index.clone())
        //                         .with_projection(
        //                             this.file_options.with_columns.as_deref().cloned(),
        //                         ),
        //                     verbose,
        //                 )
        //                 .await?;

        //             row_counter
        //                 .write()
        //                 .unwrap()
        //                 .write(index, df.height().try_into().unwrap());

        //             PolarsResult::Ok((index, df))
        //         }
        //     })
        //     .assert_send()
        //     .buffer_unordered(100)
        //     .try_collect::<Vec<_>>()
        //     .await?;

        // finish_index_and_dfs(
        //     index_and_dfs,
        //     row_counter.into_inner().unwrap(),
        //     self.file_options.row_index.as_ref(),
        //     row_limit,
        //     self.predicate.as_ref(),
        // )
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

    #[cfg(debug_assertions)]
    {
        assert!(
            index_and_dfs.iter().enumerate().all(|(a, &(b, _))| a == b),
            "expected dataframe indices in order from 0 to len"
        );
    }

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

impl Executor for CsvExec {
    fn execute(&mut self, state: &mut ExecutionState) -> PolarsResult<DataFrame> {
        #[allow(clippy::useless_asref)]
        let finger_print = FileFingerPrint {
            paths: Arc::clone(&self.paths),
            predicate: self
                .predicate
                .as_ref()
                .map(|ae| ae.as_expression().unwrap().clone()),
            slice: (self.options.skip_rows, self.file_options.n_rows),
        };

        let profile_name = if state.has_node_timer() {
            let mut items = self
                .paths
                .iter()
                .map(|path| path.to_string_lossy().into())
                .collect::<Vec<_>>();
            if self.predicate.is_some() {
                items.push("predicate".into())
            }
            let name = comma_delimited("csv".to_string(), &items);
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
