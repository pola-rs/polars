use std::sync::Arc;

use futures::StreamExt;
use futures::stream::FuturesUnordered;
use polars_core::prelude::{Column, DataType, SortMultipleOptions};
use polars_core::scalar::Scalar;
use polars_core::schema::SchemaRef;
use polars_error::PolarsResult;
use polars_io::cloud::CloudOptions;
use polars_plan::dsl::{
    FileType, PartitionTargetCallback, PartitionTargetCallbackResult, PartitionTargetContext,
    SinkOptions, SinkTarget,
};
use polars_utils::format_pl_smallstr;
use polars_utils::plpath::PlPathRef;

use super::{DEFAULT_SINK_DISTRIBUTOR_BUFFER_SIZE, SinkInputPort, SinkNode};
use crate::async_executor::{AbortOnDropHandle, spawn};
use crate::async_primitives::wait_group::WaitGroup;
use crate::async_primitives::{connector, distributor_channel};
use crate::execute::StreamingExecutionState;
use crate::expression::StreamExpr;
use crate::morsel::{MorselSeq, SourceToken};
use crate::nodes::io_sinks::phase::PhaseOutcome;
use crate::nodes::{Morsel, TaskPriority};

pub mod by_key;
pub mod max_size;
pub mod parted;

#[derive(Clone)]
pub struct PerPartitionSortBy {
    // Invariant: all vecs have the same length.
    pub selectors: Vec<StreamExpr>,
    pub descending: Vec<bool>,
    pub nulls_last: Vec<bool>,
    pub maintain_order: bool,
}

pub type CreateNewSinkFn =
    Arc<dyn Send + Sync + Fn(SchemaRef, SinkTarget) -> PolarsResult<Box<dyn SinkNode + Send>>>;

pub fn get_create_new_fn(
    file_type: FileType,
    sink_options: SinkOptions,
    cloud_options: Option<CloudOptions>,
    collect_metrics: bool,
) -> CreateNewSinkFn {
    match file_type {
        #[cfg(feature = "ipc")]
        FileType::Ipc(ipc_writer_options) => Arc::new(move |input_schema, target| {
            let sink = Box::new(super::ipc::IpcSinkNode::new(
                input_schema,
                target,
                sink_options.clone(),
                ipc_writer_options,
                cloud_options.clone(),
            )) as Box<dyn SinkNode + Send>;
            Ok(sink)
        }) as _,
        #[cfg(feature = "json")]
        FileType::Json(_ndjson_writer_options) => Arc::new(move |_input_schema, target| {
            let sink = Box::new(super::json::NDJsonSinkNode::new(
                target,
                sink_options.clone(),
                cloud_options.clone(),
            )) as Box<dyn SinkNode + Send>;
            Ok(sink)
        }) as _,
        #[cfg(feature = "parquet")]
        FileType::Parquet(parquet_writer_options) => {
            Arc::new(move |input_schema, target: SinkTarget| {
                let sink = Box::new(super::parquet::ParquetSinkNode::new(
                    input_schema,
                    target,
                    sink_options.clone(),
                    &parquet_writer_options,
                    cloud_options.clone(),
                    collect_metrics,
                )?) as Box<dyn SinkNode + Send>;
                Ok(sink)
            }) as _
        },
        #[cfg(feature = "csv")]
        FileType::Csv(csv_writer_options) => Arc::new(move |input_schema, target| {
            let sink = Box::new(super::csv::CsvSinkNode::new(
                target,
                input_schema,
                sink_options.clone(),
                csv_writer_options.clone(),
                cloud_options.clone(),
            )) as Box<dyn SinkNode + Send>;
            Ok(sink)
        }) as _,
        #[cfg(not(any(
            feature = "csv",
            feature = "parquet",
            feature = "json",
            feature = "ipc"
        )))]
        _ => {
            panic!("activate source feature")
        },
    }
}

enum SinkSender {
    Connector(connector::Sender<Morsel>),
    Distributor(distributor_channel::Sender<Morsel>),
}

impl SinkSender {
    pub async fn send(&mut self, morsel: Morsel) -> Result<(), Morsel> {
        match self {
            SinkSender::Connector(sender) => sender.send(morsel).await,
            SinkSender::Distributor(sender) => sender.send(morsel).await,
        }
    }
}

fn default_by_key_file_path_cb(
    ext: &str,
    _file_idx: usize,
    _part_idx: usize,
    in_part_idx: usize,
    columns: Option<&[Column]>,
    separator: char,
) -> PolarsResult<String> {
    use std::fmt::Write;

    let columns = columns.unwrap();
    assert!(!columns.is_empty());

    let mut file_path = String::new();
    for c in columns {
        let name = c.name();
        let value = c.head(Some(1)).strict_cast(&DataType::String)?;
        let value = value.str().unwrap();
        let value = value
            .get(0)
            .unwrap_or("__HIVE_DEFAULT_PARTITION__")
            .as_bytes();
        let value = percent_encoding::percent_encode(value, polars_io::utils::URL_ENCODE_CHAR_SET);
        write!(&mut file_path, "{name}={value}").unwrap();
        file_path.push(separator);
    }
    write!(&mut file_path, "{in_part_idx}.{ext}").unwrap();

    Ok(file_path)
}

type FilePathCallback =
    fn(&str, usize, usize, usize, Option<&[Column]>, char) -> PolarsResult<String>;

#[allow(clippy::too_many_arguments)]
async fn open_new_sink(
    base_path: PlPathRef<'_>,
    file_path_cb: Option<&PartitionTargetCallback>,
    default_file_path_cb: FilePathCallback,
    file_idx: usize,
    part_idx: usize,
    in_part_idx: usize,
    keys: Option<&[Column]>,
    create_new_sink: &CreateNewSinkFn,
    sink_input_schema: SchemaRef,
    partition_name: &'static str,
    ext: &str,
    verbose: bool,
    state: &StreamingExecutionState,
    per_partition_sort_by: Option<&PerPartitionSortBy>,
) -> PolarsResult<
    Option<(
        FuturesUnordered<AbortOnDropHandle<PolarsResult<()>>>,
        SinkSender,
        Box<dyn SinkNode + Send>,
    )>,
> {
    let separator = '/'; // note: accepted by both Windows and Linux
    let file_path = default_file_path_cb(ext, file_idx, part_idx, in_part_idx, keys, separator)?;
    let path = base_path.join(file_path.as_str());

    // If the user provided their own callback, modify the path to that.
    let target = if let Some(file_path_cb) = file_path_cb {
        let keys = keys.map_or(Vec::new(), |keys| {
            keys.iter()
                .map(|k| polars_plan::dsl::PartitionTargetContextKey {
                    name: k.name().clone(),
                    raw_value: Scalar::new(k.dtype().clone(), k.get(0).unwrap().into_static()),
                })
                .collect()
        });

        let target = file_path_cb.call(PartitionTargetContext {
            file_idx,
            part_idx,
            in_part_idx,
            keys,
            file_path,
            full_path: path,
        })?;
        match target {
            // Offset the given path by the base_path.
            PartitionTargetCallbackResult::Str(p) => SinkTarget::Path(base_path.join(p)),
            PartitionTargetCallbackResult::Dyn(t) => SinkTarget::Dyn(t),
        }
    } else {
        SinkTarget::Path(path)
    };

    if verbose {
        match &target {
            SinkTarget::Path(p) => eprintln!(
                "[partition[{partition_name}]]: Start on new file '{}'",
                p.display(),
            ),
            SinkTarget::Dyn(_) => eprintln!("[partition[{partition_name}]]: Start on new file",),
        }
    }

    let mut node = (create_new_sink)(sink_input_schema.clone(), target)?;
    let mut join_handles = Vec::new();
    let (sink_input, mut sender) = if node.is_sink_input_parallel() {
        let (tx, dist_rxs) = distributor_channel::distributor_channel(
            state.num_pipelines,
            *DEFAULT_SINK_DISTRIBUTOR_BUFFER_SIZE,
        );
        let (txs, rxs) = (0..state.num_pipelines)
            .map(|_| connector::connector())
            .collect::<(Vec<_>, Vec<_>)>();
        join_handles.extend(dist_rxs.into_iter().zip(txs).map(|(mut dist_rx, mut tx)| {
            spawn(TaskPriority::High, async move {
                while let Ok(morsel) = dist_rx.recv().await {
                    if tx.send(morsel).await.is_err() {
                        break;
                    }
                }
                Ok(())
            })
        }));

        (SinkInputPort::Parallel(rxs), SinkSender::Distributor(tx))
    } else {
        let (tx, rx) = connector::connector();
        (SinkInputPort::Serial(rx), SinkSender::Connector(tx))
    };

    // Handle sorting per partition.
    if let Some(per_partition_sort_by) = per_partition_sort_by {
        let num_selectors = per_partition_sort_by.selectors.len();
        let (tx, mut rx) = connector::connector();

        let state = state.in_memory_exec_state.split();
        let selectors = per_partition_sort_by.selectors.clone();
        let descending = per_partition_sort_by.descending.clone();
        let nulls_last = per_partition_sort_by.nulls_last.clone();
        let maintain_order = per_partition_sort_by.maintain_order;

        // Tell the partitioning sink to send stuff here instead.
        let mut old_sender = std::mem::replace(&mut sender, SinkSender::Connector(tx));

        // This all happens in a single thread per partition. Acceptable for now as the main
        // usecase here is writing many partitions, not the best idea for the future.
        join_handles.push(spawn(TaskPriority::High, async move {
            // Gather all morsels for this partition. We expect at least one morsel per partition.
            let Ok(morsel) = rx.recv().await else {
                return Ok(());
            };
            let mut df = morsel.into_df();
            while let Ok(next_morsel) = rx.recv().await {
                df.vstack_mut_owned(next_morsel.into_df())?;
            }

            let mut names = Vec::with_capacity(num_selectors);
            for (i, s) in selectors.into_iter().enumerate() {
                // @NOTE: This evaluation cannot be done as chunks come in since it might contain
                // non-elementwise expressions.
                let c = s.evaluate(&df, &state).await?;
                let name = format_pl_smallstr!("__POLARS_PART_SORT_COL{i}");
                names.push(name.clone());
                df.with_column(c.with_name(name))?;
            }
            df.sort_in_place(
                names,
                SortMultipleOptions {
                    descending,
                    nulls_last,
                    multithreaded: false,
                    maintain_order,
                    limit: None,
                },
            )?;
            df = df.select_by_range(0..df.width() - num_selectors)?;

            _ = old_sender
                .send(Morsel::new(df, MorselSeq::default(), SourceToken::new()))
                .await;
            Ok(())
        }));
    }

    let (mut sink_input_tx, sink_input_rx) = connector::connector();
    node.initialize(state)?;
    node.spawn_sink(sink_input_rx, state, &mut join_handles);
    let mut join_handles =
        FuturesUnordered::from_iter(join_handles.into_iter().map(AbortOnDropHandle::new));

    let (_, outcome) = PhaseOutcome::new_shared_wait(WaitGroup::default().token());
    if sink_input_tx.send((outcome, sink_input)).await.is_err() {
        // If this sending failed, probably some error occurred.
        drop(sender);
        while let Some(res) = join_handles.next().await {
            res?;
        }

        return Ok(None);
    }

    Ok(Some((join_handles, sender, node)))
}
