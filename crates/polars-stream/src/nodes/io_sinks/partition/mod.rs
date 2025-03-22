use std::path::{Path, PathBuf};
use std::sync::Arc;

use futures::StreamExt;
use futures::stream::FuturesUnordered;
use polars_core::prelude::{Column, PlHashMap};
use polars_core::schema::SchemaRef;
use polars_error::PolarsResult;
use polars_io::cloud::CloudOptions;
use polars_io::utils::URL_ENCODE_CHAR_SET;
use polars_plan::dsl::{FileType, SinkOptions};
use polars_utils::format_pl_smallstr;
use polars_utils::pl_str::PlSmallStr;

use super::{DEFAULT_SINK_DISTRIBUTOR_BUFFER_SIZE, SinkInputPort, SinkNode};
use crate::async_executor::{AbortOnDropHandle, spawn};
use crate::async_primitives::wait_group::WaitGroup;
use crate::async_primitives::{connector, distributor_channel};
use crate::execute::StreamingExecutionState;
use crate::nodes::{Morsel, PhaseOutcome, TaskPriority};

pub mod by_key;
pub mod max_size;
pub mod parted;

pub type CreateNewSinkFn =
    Arc<dyn Send + Sync + Fn(SchemaRef, PathBuf) -> PolarsResult<Box<dyn SinkNode + Send + Sync>>>;

pub fn format_path(path: &Path, format_args: &PlHashMap<PlSmallStr, PlSmallStr>) -> PathBuf {
    // @Optimize: This can use aho-corasick.
    let mut path = path.display().to_string();
    for (name, value) in format_args {
        let needle = &format!("{{{name}}}");
        path = path.replace(needle, value);
    }
    std::path::PathBuf::from(path)
}

pub fn get_create_new_fn(
    file_type: FileType,
    sink_options: SinkOptions,
    cloud_options: Option<CloudOptions>,
) -> CreateNewSinkFn {
    match file_type {
        #[cfg(feature = "ipc")]
        FileType::Ipc(ipc_writer_options) => Arc::new(move |input_schema, path| {
            let sink = Box::new(super::ipc::IpcSinkNode::new(
                input_schema,
                path,
                sink_options.clone(),
                ipc_writer_options,
                cloud_options.clone(),
            )) as Box<dyn SinkNode + Send + Sync>;
            Ok(sink)
        }) as _,
        #[cfg(feature = "json")]
        FileType::Json(_ndjson_writer_options) => Arc::new(move |_input_schema, path| {
            let sink = Box::new(super::json::NDJsonSinkNode::new(
                path,
                sink_options.clone(),
                cloud_options.clone(),
            )) as Box<dyn SinkNode + Send + Sync>;
            Ok(sink)
        }) as _,
        #[cfg(feature = "parquet")]
        FileType::Parquet(parquet_writer_options) => Arc::new(move |input_schema, path: PathBuf| {
            let sink = Box::new(super::parquet::ParquetSinkNode::new(
                input_schema,
                path.as_path(),
                sink_options.clone(),
                &parquet_writer_options,
                cloud_options.clone(),
            )?) as Box<dyn SinkNode + Send + Sync>;
            Ok(sink)
        }) as _,
        #[cfg(feature = "csv")]
        FileType::Csv(csv_writer_options) => Arc::new(move |input_schema, path| {
            let sink = Box::new(super::csv::CsvSinkNode::new(
                path,
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

async fn open_new_sink(
    path_f_string: &Path,
    create_new_sink: &CreateNewSinkFn,
    format_args: &PlHashMap<PlSmallStr, PlSmallStr>,
    sink_input_schema: SchemaRef,
    partition_name: &'static str,
    verbose: bool,
    state: &StreamingExecutionState,
) -> PolarsResult<
    Option<(
        FuturesUnordered<AbortOnDropHandle<PolarsResult<()>>>,
        SinkSender,
    )>,
> {
    let path = format_path(path_f_string, format_args);

    if verbose {
        eprintln!(
            "[partition[{partition_name}]]: Start on new file '{}'",
            path.display()
        );
    }

    let mut node = (create_new_sink)(sink_input_schema.clone(), path)?;
    let mut join_handles = Vec::new();
    let (sink_input, sender) = if node.is_sink_input_parallel() {
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

    Ok(Some((join_handles, sender)))
}

fn insert_key_value_into_format_args(
    args: &mut PlHashMap<PlSmallStr, PlSmallStr>,
    keys: &[Column],
) {
    for (i, key) in keys.iter().enumerate() {
        *args
            .get_mut(&format_pl_smallstr!("key[{i}].value"))
            .unwrap() = percent_encoding::percent_encode(
            key.get(0).unwrap().to_string().as_bytes(),
            URL_ENCODE_CHAR_SET,
        )
        .to_string()
        .into();
    }
}
