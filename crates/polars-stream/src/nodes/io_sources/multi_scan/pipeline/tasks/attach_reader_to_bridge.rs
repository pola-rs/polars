use std::sync::Arc;

use futures::StreamExt;
use futures::stream::FuturesUnordered;
use polars_async::executor::{self, AbortOnDropHandle, TaskMetricAggregator, TaskPriority};
use polars_async::primitives::connector;
use polars_error::PolarsResult;

use crate::nodes::io_sources::multi_scan::components::bridge::BridgeRecvPort;
use crate::nodes::io_sources::multi_scan::pipeline::models::{
    StartedReader, StartedReaderState, UnorderedFiles,
};

pub struct AttachReaderToBridge {
    /// Its capacity bounds concurrent readers when they carry no slot.
    pub started_reader_rx: tokio::sync::mpsc::Receiver<StartedReader>,
    /// Awaited once the channel closes, so a failed initialization stops the active readers.
    pub reader_starter_handle: AbortOnDropHandle<PolarsResult<()>>,
    pub bridge_recv_port_tx: connector::Sender<BridgeRecvPort>,
    pub unordered_files: Option<UnorderedFiles>,
    pub verbose: bool,
    pub task_metrics: Option<Arc<TaskMetricAggregator>>,
}

impl AttachReaderToBridge {
    pub async fn run(self) -> PolarsResult<()> {
        let AttachReaderToBridge {
            mut started_reader_rx,
            reader_starter_handle,
            mut bridge_recv_port_tx,
            unordered_files,
            verbose,
            task_metrics,
        } = self;

        if let Some(unordered) = unordered_files {
            return run_unordered(
                started_reader_rx,
                reader_starter_handle,
                bridge_recv_port_tx,
                unordered.merge_capacity,
                verbose,
                task_metrics.as_deref(),
            )
            .await;
        }

        let mut n_readers_received: usize = 0;

        while let Some(StartedReader {
            handle,
            wait_token,
            slot: _,
        }) = started_reader_rx.recv().await
        {
            n_readers_received = n_readers_received.saturating_add(1);

            if verbose {
                eprintln!(
                    "[AttachReaderToBridge]: received reader (n_readers_received: {n_readers_received})"
                );
            }

            let StartedReaderState {
                bridge_recv_port,
                post_apply_pipeline_handle,
                reader_handle,
            } = handle.await?;

            if bridge_recv_port_tx.send(bridge_recv_port).await.is_err() {
                break;
            }

            drop(wait_token);
            join_reader(reader_handle, post_apply_pipeline_handle).await?;
        }

        // Unblock the starter's send before waiting for it.
        drop(started_reader_rx);
        reader_starter_handle.await
    }
}

/// Attaches readers as they start; each forwards its morsels into one merged port.
async fn run_unordered(
    mut started_reader_rx: tokio::sync::mpsc::Receiver<StartedReader>,
    mut reader_starter_handle: AbortOnDropHandle<PolarsResult<()>>,
    mut bridge_recv_port_tx: connector::Sender<BridgeRecvPort>,
    capacity: usize,
    verbose: bool,
    task_metrics: Option<&TaskMetricAggregator>,
) -> PolarsResult<()> {
    let (merged_tx, merged_rx) = tokio::sync::mpsc::channel(capacity);
    if bridge_recv_port_tx
        .send(BridgeRecvPort::Merged { rx: merged_rx })
        .await
        .is_err()
    {
        return Ok(());
    }

    let mut active = FuturesUnordered::new();
    let mut n_readers_received: usize = 0;
    let mut starter_done = false;
    let mut bridge_disconnected = false;

    loop {
        tokio::select! {
            biased;

            Some(result) = active.next() => result?,

            // Only the bridge exiting early can drop the receiver while we hold a sender.
            _ = merged_tx.closed() => {
                bridge_disconnected = true;
                break;
            },

            // Fail fast on initialization errors instead of starting the buffered readers first.
            result = &mut reader_starter_handle, if !starter_done => {
                result?;
                starter_done = true;
            },

            v = started_reader_rx.recv() => {
                // The attach token only matters with a single concurrent scan, which is never unordered.
                let Some(StartedReader { handle, wait_token: _, slot }) = v else {
                    break;
                };

                n_readers_received = n_readers_received.saturating_add(1);

                if verbose {
                    eprintln!(
                        "[AttachReaderToBridge]: received reader \
                        (n_readers_received: {n_readers_received}, active: {})",
                        active.len() + 1,
                    );
                }

                let tx = merged_tx.clone();

                // Forwarding runs in its own task, so readers don't share one poller.
                let forward_handle = executor::spawn(TaskPriority::High, task_metrics, async move {
                    let StartedReaderState {
                        mut bridge_recv_port,
                        post_apply_pipeline_handle,
                        reader_handle,
                    } = handle.await?;

                    while let Ok(morsel) = bridge_recv_port.recv().await {
                        if tx.send(morsel).await.is_err() {
                            break;
                        }
                    }

                    // Disconnect so the reader stops if the bridge is gone.
                    drop(bridge_recv_port);
                    drop(tx);

                    join_reader(reader_handle, post_apply_pipeline_handle).await?;
                    drop(slot);
                    PolarsResult::Ok(())
                });
                active.push(AbortOnDropHandle::new(forward_handle));
            },
        }
    }

    // Unblock the starter's send before waiting for it.
    drop(started_reader_rx);

    if bridge_disconnected {
        // Nothing can be delivered anymore; abort the readers.
        active.clear();
    } else {
        // The merged port closes once the last forwarder is done.
        drop(merged_tx);
        drop(bridge_recv_port_tx);
    }

    if !starter_done {
        reader_starter_handle.await?;
    }

    while let Some(result) = active.next().await {
        result?;
    }

    Ok(())
}

/// Waits for a reader and its post-apply pipeline to finish.
async fn join_reader(
    reader_handle: AbortOnDropHandle<PolarsResult<()>>,
    post_apply_pipeline_handle: Option<AbortOnDropHandle<PolarsResult<()>>>,
) -> PolarsResult<()> {
    reader_handle.await?;
    if let Some(handle) = post_apply_pipeline_handle {
        handle.await?;
    }
    Ok(())
}
