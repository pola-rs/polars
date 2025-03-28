use std::sync::Arc;

use polars_error::PolarsResult;
use polars_utils::IdxSize;

use crate::async_executor::{self, AbortOnDropHandle, TaskPriority};
use crate::async_primitives::distributor_channel::distributor_channel;
use crate::async_primitives::morsel_linearizer::{MorselInserter, MorselLinearizer};
use crate::async_primitives::{connector, distributor_channel};
use crate::morsel::Morsel;
use crate::nodes::io_sources::multi_file_reader::bridge::BridgeRecvPort;
use crate::nodes::io_sources::multi_file_reader::extra_ops::apply::ApplyExtraOps;
use crate::nodes::io_sources::multi_file_reader::reader_interface::output::FileReaderOutputRecv;

/// Pool of workers to apply operations on morsels originating from a reader.
/// Kept alive for the duration of the multiscan.
pub struct PostApplyPool {
    workers: Vec<WorkerData>,
    reader_in_progress: bool,
}

struct WorkerData {
    task_data_tx: connector::Sender<TaskData>,
    notify_success_rx: connector::Receiver<()>,
    handle: AbortOnDropHandle<PolarsResult<()>>,
}

impl PostApplyPool {
    pub fn new(num_pipelines: usize) -> Self {
        let workers = (0..num_pipelines)
            .map(|_| {
                let (task_data_tx, task_data_rx) = connector::connector();
                let (notify_success_tx, notify_success_rx) = connector::connector();

                let handle = AbortOnDropHandle::new(async_executor::spawn(
                    TaskPriority::Low,
                    PostApplyWorker {
                        rx: task_data_rx,
                        notify_success: notify_success_tx,
                    }
                    .run(),
                ));

                WorkerData {
                    task_data_tx,
                    notify_success_rx,
                    handle,
                }
            })
            .collect();

        Self {
            workers,
            reader_in_progress: false,
        }
    }

    /// `wait_current_reader` should be awaited on afterwards to catch errors from this file.
    ///
    /// # Panics
    /// Panics if called twice without `wait_current_reader` in between.
    pub async fn run_with_reader(
        &mut self,
        mut reader_port: FileReaderOutputRecv,
        ops_applier: Arc<ApplyExtraOps>,
        // We have this because initialization of `ops_applier` causes `reader_port` to have the
        // first morsel consumed.
        first_morsel: Morsel,
    ) -> PolarsResult<BridgeRecvPort> {
        assert!(
            !self.reader_in_progress,
            "previous reader still in progress"
        );

        let (mut distr_tx, distr_receivers) = distributor_channel(self.workers.len(), 1);

        // Distributor
        {
            let ops_applier = ops_applier.clone();
            async_executor::spawn(TaskPriority::Low, async move {
                // Number of rows received from this reader.
                let mut n_rows_received: IdxSize = 0;

                let mut morsel = first_morsel;

                // Should only run the pipeline if we have an operation we need to apply.
                let ApplyExtraOps::Initialized { pre_slice, .. } = ops_applier.as_ref() else {
                    unreachable!();
                };

                loop {
                    let h = morsel.df().height();
                    let h = IdxSize::try_from(h).unwrap_or(IdxSize::MAX);

                    // We hit this if a reader does not support PRE_SLICE.
                    if pre_slice.clone().is_some_and(|x| {
                        x.offsetted(usize::try_from(n_rows_received).unwrap()).len() == 0
                    }) {
                        // Note: We do not return any flag indicating that we have reached end of slice
                        // from this context. The read should be stopped on a higher level by using
                        // the `row_position_on_end_tx` callback from the reader.
                        break;
                    }

                    if distr_tx.send((morsel, n_rows_received)).await.is_err() {
                        break;
                    }

                    n_rows_received = n_rows_received.saturating_add(h);

                    let Ok(v) = reader_port.recv().await else {
                        break;
                    };

                    morsel = v;
                }
            })
        };

        let (rx, senders) = MorselLinearizer::new(self.workers.len(), 1);

        for (
            WorkerData {
                task_data_tx,
                notify_success_rx: _,
                handle,
            },
            (reader_morsel_rx, morsel_tx),
        ) in self
            .workers
            .iter_mut()
            .zip(distr_receivers.into_iter().zip(senders))
        {
            let ops_applier = ops_applier.clone();
            use crate::async_primitives::connector::SendError;

            match task_data_tx.try_send(TaskData {
                reader_morsel_rx,
                ops_applier,
                morsel_tx,
            }) {
                Err(SendError::Full(_)) => panic!("impl error: worker rx port full"),
                Err(SendError::Closed(_)) => return Err(handle.await.unwrap_err()),
                Ok(_) => {},
            }
        }

        self.reader_in_progress = true;

        Ok(BridgeRecvPort::Linearized { rx })
    }

    pub async fn wait_current_reader(&mut self) -> PolarsResult<()> {
        if !self.reader_in_progress {
            return Ok(());
        }

        for WorkerData {
            task_data_tx: _,
            notify_success_rx,
            handle,
        } in self.workers.iter_mut()
        {
            if notify_success_rx.recv().await.is_err() {
                return Err(handle.await.unwrap_err());
            }
        }

        self.reader_in_progress = false;

        Ok(())
    }

    /// Note: This should not be called if any other function previously returned an error.
    pub async fn shutdown(self) -> PolarsResult<()> {
        for WorkerData {
            task_data_tx,
            notify_success_rx,
            handle,
        } in self.workers.into_iter()
        {
            drop(task_data_tx);
            drop(notify_success_rx);
            handle.await?;
        }

        Ok(())
    }
}

struct PostApplyWorker {
    rx: connector::Receiver<TaskData>,
    notify_success: connector::Sender<()>,
}

struct TaskData {
    reader_morsel_rx: distributor_channel::Receiver<(Morsel, IdxSize)>,
    ops_applier: Arc<ApplyExtraOps>,
    morsel_tx: MorselInserter,
}

impl PostApplyWorker {
    async fn run(self) -> PolarsResult<()> {
        let PostApplyWorker {
            mut rx,
            mut notify_success,
        } = self;

        while let Ok(TaskData {
            mut reader_morsel_rx,
            ops_applier,
            mut morsel_tx,
        }) = rx.recv().await
        {
            while let Ok((mut morsel, row_offset)) = reader_morsel_rx.recv().await {
                ops_applier.apply_to_df(morsel.df_mut(), row_offset)?;

                if morsel_tx.insert(morsel).await.is_err() {
                    break;
                }
            }

            if notify_success.send(()).await.is_err() {
                break;
            }
        }

        Ok(())
    }
}
