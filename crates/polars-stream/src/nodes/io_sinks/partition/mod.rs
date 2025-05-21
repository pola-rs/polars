use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow::array::builder::ShareStrategy;
use futures::StreamExt;
use futures::stream::FuturesUnordered;
use polars_core::frame::DataFrame;
use polars_core::prelude::{
    ChunkedBuilder, Column, DataType, IntoColumn, PrimitiveChunkedBuilder, SortMultipleOptions,
    StringChunkedBuilder, StructChunked, UInt64Type,
};
use polars_core::scalar::Scalar;
use polars_core::schema::{Schema, SchemaRef};
use polars_core::series::IntoSeries;
use polars_core::series::builder::SeriesBuilder;
use polars_error::PolarsResult;
use polars_expr::reduce::{GroupedReduction, new_max_reduction, new_min_reduction};
use polars_io::cloud::CloudOptions;
use polars_plan::dsl::{
    FileType, PartitionTargetCallback, PartitionTargetContext, SinkOptions, SinkTarget,
};
use polars_utils::format_pl_smallstr;
use polars_utils::pl_str::PlSmallStr;

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

pub struct WrittenPartitionColumn {
    pub null_count: u64,
    pub nan_count: u64,
    pub lower_bound: Box<dyn GroupedReduction>,
    pub upper_bound: Box<dyn GroupedReduction>,
}

pub struct WrittenPartition {
    pub path: String,
    pub height: u64,
    pub columns: Vec<WrittenPartitionColumn>,
}

impl WrittenPartition {
    pub fn new(path: String, schema: &Schema) -> Self {
        Self {
            path,
            height: 0,
            columns: schema
                .iter_values()
                .map(|dtype| {
                    let mut lower_bound = new_min_reduction(dtype.clone(), false);
                    let mut upper_bound = new_max_reduction(dtype.clone(), false);

                    lower_bound.resize(1);
                    upper_bound.resize(1);
                    WrittenPartitionColumn {
                        null_count: 0,
                        nan_count: 0,
                        lower_bound,
                        upper_bound,
                    }
                })
                .collect(),
        }
    }

    pub fn append(&mut self, df: &DataFrame) -> PolarsResult<()> {
        assert_eq!(self.columns.len(), df.width());
        self.height += df.height() as u64;
        for (w, c) in self.columns.iter_mut().zip(df.get_columns()) {
            let null_count = c.null_count();
            w.null_count += c.null_count() as u64;

            let mut has_non_null_non_nan_values = df.height() != null_count;
            if c.dtype().is_float() {
                let nan_count = c.is_nan()?.sum().unwrap_or_default() as u64;
                has_non_null_non_nan_values = nan_count as usize + null_count < df.height();
                w.nan_count += nan_count;
            }

            if has_non_null_non_nan_values {
                w.lower_bound.update_group(&c, 0, 0)?;
                w.upper_bound.update_group(&c, 0, 0)?;
            }
        }
        Ok(())
    }
}

pub fn written_partitions_to_df(wp: Vec<WrittenPartition>, input_schema: &Schema) -> DataFrame {
    let num_partitions = wp.len();

    let mut path = StringChunkedBuilder::new(PlSmallStr::from_static("path"), wp.len());
    let mut height =
        PrimitiveChunkedBuilder::<UInt64Type>::new(PlSmallStr::from_static("height"), wp.len());
    let mut columns = input_schema
        .iter_values()
        .map(|dtype| {
            let null_count = PrimitiveChunkedBuilder::<UInt64Type>::new(
                PlSmallStr::from_static("null_count"),
                wp.len(),
            );
            let nan_count = PrimitiveChunkedBuilder::<UInt64Type>::new(
                PlSmallStr::from_static("nan_count"),
                wp.len(),
            );
            let mut lower_bound = SeriesBuilder::new(dtype.clone());
            let mut upper_bound = SeriesBuilder::new(dtype.clone());
            lower_bound.reserve(wp.len());
            upper_bound.reserve(wp.len());

            (null_count, nan_count, lower_bound, upper_bound)
        })
        .collect::<Vec<_>>();

    for p in wp {
        path.append_value(p.path);
        height.append_value(p.height);

        for (mut w, c) in p.columns.into_iter().zip(columns.iter_mut()) {
            c.0.append_value(w.null_count);
            c.1.append_value(w.nan_count);
            c.2.extend(&w.lower_bound.finalize().unwrap(), ShareStrategy::Always);
            c.3.extend(&w.upper_bound.finalize().unwrap(), ShareStrategy::Always);
        }
    }

    let mut df_columns = Vec::with_capacity(input_schema.len() + 2);
    df_columns.push(path.finish().into_column());
    df_columns.push(height.finish().into_column());
    for (name, column) in input_schema.iter_names().zip(columns) {
        let struct_ca = StructChunked::from_series(
            format_pl_smallstr!("{name}_stats"),
            num_partitions,
            [
                column.0.finish().into_series(),
                column.1.finish().into_series(),
                column.2.freeze(PlSmallStr::from_static("lower_bound")),
                column.3.freeze(PlSmallStr::from_static("upper_bound")),
            ]
            .iter(),
        )
        .unwrap();
        df_columns.push(struct_ca.into_column());
    }

    DataFrame::new_with_height(num_partitions, df_columns).unwrap()
}

#[derive(Clone)]
pub struct PerPartitionSortBy {
    // Invariant: all vecs have the same length.
    pub selectors: Vec<StreamExpr>,
    pub descending: Vec<bool>,
    pub nulls_last: Vec<bool>,
    pub maintain_order: bool,
}

pub type CreateNewSinkFn = Arc<
    dyn Send + Sync + Fn(SchemaRef, SinkTarget) -> PolarsResult<Box<dyn SinkNode + Send + Sync>>,
>;

pub fn get_create_new_fn(
    file_type: FileType,
    sink_options: SinkOptions,
    cloud_options: Option<CloudOptions>,
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
            )) as Box<dyn SinkNode + Send + Sync>;
            Ok(sink)
        }) as _,
        #[cfg(feature = "json")]
        FileType::Json(_ndjson_writer_options) => Arc::new(move |_input_schema, target| {
            let sink = Box::new(super::json::NDJsonSinkNode::new(
                target,
                sink_options.clone(),
                cloud_options.clone(),
            )) as Box<dyn SinkNode + Send + Sync>;
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
                )?) as Box<dyn SinkNode + Send + Sync>;
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
            )) as Box<dyn SinkNode + Send + Sync>;
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
) -> PolarsResult<PathBuf> {
    let columns = columns.unwrap();
    assert!(!columns.is_empty());

    let mut file_path = PathBuf::new();
    for c in columns {
        let name = c.name();
        let value = c.head(Some(1)).strict_cast(&DataType::String)?;
        let value = value.str().unwrap();
        let value = value
            .get(0)
            .unwrap_or("__HIVE_DEFAULT_PARTITION__")
            .as_bytes();
        let value = percent_encoding::percent_encode(value, polars_io::utils::URL_ENCODE_CHAR_SET);
        file_path = file_path.join(format!("{name}={value}"));
    }
    file_path = file_path.join(format!("{in_part_idx}.{ext}"));

    Ok(file_path)
}

type FilePathCallback = fn(&str, usize, usize, usize, Option<&[Column]>) -> PolarsResult<PathBuf>;

#[allow(clippy::too_many_arguments)]
async fn open_new_sink(
    base_path: &Path,
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
        String,
        FuturesUnordered<AbortOnDropHandle<PolarsResult<()>>>,
        SinkSender,
    )>,
> {
    let file_path = default_file_path_cb(ext, file_idx, part_idx, in_part_idx, keys)?;
    let path = base_path.join(file_path.as_path());

    let mut output_path = path.to_string_lossy().into_owned();
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
        // Offset the given path by the base_path.
        match target {
            SinkTarget::Path(p) => {
                let path = base_path.join(p.as_path());
                output_path = path.to_string_lossy().into_owned();
                SinkTarget::Path(Arc::new(path))
            },
            target => target,
        }
    } else {
        SinkTarget::Path(Arc::new(path))
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

    Ok(Some((output_path, join_handles, sender)))
}
