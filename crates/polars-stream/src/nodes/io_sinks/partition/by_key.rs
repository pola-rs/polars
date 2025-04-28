use std::cmp::Reverse;
use std::path::PathBuf;
use std::sync::Arc;

use futures::StreamExt;
use futures::stream::FuturesUnordered;
use polars_core::config;
use polars_core::frame::DataFrame;
use polars_core::prelude::{Column, PlHashSet, PlIndexMap, row_encode};
use polars_core::schema::SchemaRef;
use polars_core::utils::arrow::buffer::Buffer;
use polars_error::PolarsResult;
use polars_plan::dsl::{PartitionTargetCallback, SinkOptions};
use polars_utils::pl_str::PlSmallStr;
use polars_utils::priority::Priority;

use super::CreateNewSinkFn;
use crate::async_executor::{AbortOnDropHandle, spawn};
use crate::execute::StreamingExecutionState;
use crate::morsel::SourceToken;
use crate::nodes::io_sinks::partition::{SinkSender, open_new_sink};
use crate::nodes::io_sinks::phase::PhaseOutcome;
use crate::nodes::io_sinks::{SinkInputPort, SinkNode, parallelize_receive_task};
use crate::nodes::{JoinHandle, Morsel, MorselSeq, TaskPriority};

type Linearized =
    Priority<Reverse<MorselSeq>, (SourceToken, Vec<(Buffer<u8>, Vec<Column>, DataFrame)>)>;
pub struct PartitionByKeySinkNode {
    // This is not be the same as the input_schema, e.g. when include_key=false then this will not
    // include the keys columns.
    sink_input_schema: SchemaRef,

    key_cols: Arc<[PlSmallStr]>,

    max_open_partitions: usize,
    include_key: bool,

    base_path: Arc<PathBuf>,
    file_path_cb: Option<PartitionTargetCallback>,
    create_new: CreateNewSinkFn,
    ext: PlSmallStr,

    sink_options: SinkOptions,
}

impl PartitionByKeySinkNode {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        input_schema: SchemaRef,
        key_cols: Arc<[PlSmallStr]>,
        base_path: Arc<PathBuf>,
        file_path_cb: Option<PartitionTargetCallback>,
        create_new: CreateNewSinkFn,
        ext: PlSmallStr,
        sink_options: SinkOptions,
        include_key: bool,
    ) -> Self {
        assert!(!key_cols.is_empty());

        let mut sink_input_schema = input_schema.clone();
        if !include_key {
            let keys_col_hm = PlHashSet::from_iter(key_cols.iter().map(|s| s.as_str()));
            sink_input_schema = Arc::new(
                sink_input_schema
                    .try_project(
                        input_schema
                            .iter_names()
                            .filter(|n| !keys_col_hm.contains(n.as_str()))
                            .cloned(),
                    )
                    .unwrap(),
            );
        }

        const DEFAULT_MAX_OPEN_PARTITIONS: usize = 128;
        let max_open_partitions =
            std::env::var("POLARS_MAX_OPEN_PARTITIONS").map_or(DEFAULT_MAX_OPEN_PARTITIONS, |v| {
                v.parse::<usize>()
                    .expect("unable to parse POLARS_MAX_OPEN_PARTITIONS")
            });

        Self {
            sink_input_schema,
            key_cols,
            max_open_partitions,
            include_key,
            base_path,
            file_path_cb,
            create_new,
            ext,
            sink_options,
        }
    }
}

impl SinkNode for PartitionByKeySinkNode {
    fn name(&self) -> &str {
        "partition-by-key-sink"
    }

    fn is_sink_input_parallel(&self) -> bool {
        true
    }
    fn do_maintain_order(&self) -> bool {
        self.sink_options.maintain_order
    }

    fn spawn_sink(
        &mut self,
        recv_port_rx: crate::async_primitives::connector::Receiver<(PhaseOutcome, SinkInputPort)>,
        state: &StreamingExecutionState,
        join_handles: &mut Vec<JoinHandle<polars_error::PolarsResult<()>>>,
    ) {
        let (pass_rxs, mut io_rx) = parallelize_receive_task::<Linearized>(
            join_handles,
            recv_port_rx,
            state.num_pipelines,
            self.sink_options.maintain_order,
        );

        join_handles.extend(pass_rxs.into_iter().map(|mut pass_rx| {
            let key_cols = self.key_cols.clone();
            let stable = self.sink_options.maintain_order;
            let include_key = self.include_key;

            spawn(TaskPriority::High, async move {
                while let Ok((mut rx, mut lin_tx)) = pass_rx.recv().await {
                    while let Ok(morsel) = rx.recv().await {
                        let (df, seq, source_token, consume_token) = morsel.into_inner();

                        let partition_include_key = true; // We need the keys to send to the
                        // appropriate sink.
                        let parallel = false; // We handle parallel processing in the streaming
                        // engine.
                        let partitions = df._partition_by_impl(
                            &key_cols,
                            stable,
                            partition_include_key,
                            parallel,
                        )?;

                        let partitions = partitions
                            .into_iter()
                            .map(|mut df| {
                                let keys = df.select_columns(key_cols.iter().cloned())?;
                                let keys = keys
                                    .into_iter()
                                    .map(|c| c.head(Some(1)))
                                    .collect::<Vec<_>>();

                                let row_encoded = row_encode::encode_rows_unordered(&keys)?
                                    .downcast_into_iter()
                                    .next()
                                    .unwrap();
                                let row_encoded = row_encoded.into_inner().2;

                                if !include_key {
                                    df = df.drop_many(key_cols.iter().cloned());
                                }

                                PolarsResult::Ok((row_encoded, keys, df))
                            })
                            .collect::<PolarsResult<Vec<(Buffer<u8>, Vec<Column>, DataFrame)>>>()?;

                        if lin_tx
                            .insert(Priority(Reverse(seq), (source_token, partitions)))
                            .await
                            .is_err()
                        {
                            return Ok(());
                        }
                        // It is important that we don't pass the consume
                        // token to the sinks, because that leads to
                        // deadlocks.
                        drop(consume_token);
                    }
                }

                Ok(())
            })
        }));

        let state = state.clone();
        let sink_input_schema = self.sink_input_schema.clone();
        let max_open_partitions = self.max_open_partitions;
        let base_path = self.base_path.clone();
        let file_path_cb = self.file_path_cb.clone();
        let create_new_sink = self.create_new.clone();
        let ext = self.ext.clone();
        join_handles.push(spawn(TaskPriority::High, async move {
            enum OpenPartition {
                Sink(
                    SinkSender,
                    FuturesUnordered<AbortOnDropHandle<PolarsResult<()>>>,
                ),
                Buffer(Vec<Column>, Vec<DataFrame>),
            }

            let verbose = config::verbose();
            let mut file_idx = 0;
            let mut open_partitions: PlIndexMap<Buffer<u8>, OpenPartition> = PlIndexMap::default();

            // Wrap this in a closure so that a failure to send (which signifies a failure) can be
            // caught while waiting for tasks.
            let mut receive_and_pass = async || {
                while let Ok(mut lin_rx) = io_rx.recv().await {
                    while let Some(Priority(Reverse(seq), (source_token, partitions))) =
                        lin_rx.get().await
                    {
                        for (row_encoded, keys, partition) in partitions {
                            let num_open_partitions = open_partitions.len();
                            let open_partition = match open_partitions.get_mut(&row_encoded) {
                                None if num_open_partitions >= max_open_partitions => {
                                    if num_open_partitions == max_open_partitions && verbose {
                                        eprintln!(
                                            "[partition[by-key]]: Reached maximum open partitions. Buffering the rest to memory before writing.",
                                        );
                                    }

                                    let (idx, previous) = open_partitions.insert_full(
                                        row_encoded,
                                        OpenPartition::Buffer(keys, Vec::new()),
                                    );
                                    debug_assert!(previous.is_none());
                                    open_partitions.get_index_mut(idx).unwrap().1
                                },
                                None => {
                                    let result = open_new_sink(
                                        base_path.as_path(),
                                        file_path_cb.as_ref(),
                                        super::default_by_key_file_path_cb,
                                        file_idx,
                                        file_idx,
                                        0,
                                        Some(keys.as_slice()),
                                        &create_new_sink,
                                        sink_input_schema.clone(),
                                        "by-key",
                                        ext.as_str(),
                                        verbose,
                                        &state,
                                    ).await?;
                                    file_idx += 1;

                                    let Some((join_handles, sender)) = result else {
                                        return Ok(());
                                    };

                                    let (idx, previous) = open_partitions.insert_full(
                                        row_encoded,
                                        OpenPartition::Sink(sender, join_handles),
                                    );
                                    debug_assert!(previous.is_none());
                                    open_partitions.get_index_mut(idx).unwrap().1
                                },
                                Some(open_partition) => open_partition,
                            };

                            match open_partition {
                                OpenPartition::Sink(input, _) => {
                                    let morsel = Morsel::new(partition, seq, source_token.clone());
                                    if input.send(morsel).await.is_err() {
                                        return Ok(());
                                    }
                                },
                                OpenPartition::Buffer(_keys, buffer) => buffer.push(partition),
                            }
                        }
                    }
                }

                PolarsResult::Ok(())
            };
            receive_and_pass().await?;

            // At this point, we need to wait for all sinks to finish writing and close them. Also,
            // sinks that ended up buffering need to output their data.
            for open_partition in open_partitions.into_values() {
                match open_partition {
                    OpenPartition::Sink(sink_sender, mut join_handles) => {
                        drop(sink_sender); // Signal to the sink that nothing more is coming.
                        while let Some(res) = join_handles.next().await {
                            res?;
                        }
                    },
                    OpenPartition::Buffer(keys, buffered) => {
                        let result = open_new_sink(
                            base_path.as_path(),
                            file_path_cb.as_ref(),
                            super::default_by_key_file_path_cb,
                            file_idx,
                            file_idx,
                            0,
                            Some(keys.as_slice()),
                            &create_new_sink,
                            sink_input_schema.clone(),
                            "by-key",
                            ext.as_str(),
                            verbose,
                            &state
                        ).await?;
                        file_idx += 1;
                        let Some((mut join_handles, mut sender)) = result else {
                            return Ok(());
                        };

                        let source_token = SourceToken::new();
                        let mut seq = MorselSeq::default();
                        for df in buffered {
                            let morsel = Morsel::new(df, seq, source_token.clone());
                            if sender.send(morsel).await.is_err() {
                                return Ok(());
                            }
                            seq = seq.successor();
                        }

                        drop(sender); // Signal to the sink that nothing more is coming.
                        while let Some(res) = join_handles.next().await {
                            res?;
                        }
                    },
                }
            }

            Ok(())
        }));
    }
}
