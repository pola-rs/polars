use std::cmp::Reverse;
use std::sync::Arc;

use polars_error::PolarsResult;
use polars_io::ndjson;
use polars_utils::mmap::MemSlice;
use polars_utils::priority::Priority;

use super::chunk_reader::ChunkReader;
use crate::async_primitives::connector::Receiver;
use crate::async_primitives::distributor_channel;
use crate::async_primitives::linearizer::Inserter;
use crate::async_primitives::wait_group::WaitGroup;
use crate::morsel::SourceToken;
use crate::nodes::MorselSeq;
use crate::nodes::compute_node_prelude::*;
use crate::nodes::io_sources::MorselOutput;

/// Parses chunks into DataFrames (or counts rows depending on state).
pub(super) struct LineBatchProcessor {
    /// Mainly for logging
    pub(super) worker_idx: usize,

    /// We need to hold a ref to this as `LineBatch` we receive contains `&[u8]`
    /// references to it.
    pub(super) global_bytes: MemSlice,
    pub(super) chunk_reader: Arc<ChunkReader>,

    // Input
    pub(super) line_batch_rx: distributor_channel::Receiver<LineBatch>,
    // Output
    pub(super) output_port: LineBatchProcessorOutputPort,

    /// When this is true, it means both a negative slice and row index is requested.
    /// This will cause the worker to fully consume all chunks even after the output port
    /// is closed.
    pub(super) needs_total_row_count: bool,
    pub(super) verbose: bool,
}

impl LineBatchProcessor {
    /// Returns the number of rows processed.
    pub(super) async fn run(self) -> PolarsResult<usize> {
        let LineBatchProcessor {
            worker_idx,
            global_bytes: _global_bytes,
            chunk_reader,
            mut line_batch_rx,
            mut output_port,
            needs_total_row_count,
            verbose,
        } = self;

        if verbose {
            eprintln!(
                "[NDJSON LineBatchProcessor {}]: begin run(): port_type: {}",
                worker_idx,
                output_port.port_type()
            );
        }

        if output_port.init().await.is_err() {
            if verbose {
                eprintln!(
                    "[NDJSON LineBatchProcessor {}]: phase receiver ended at init, returning",
                    worker_idx
                );
            }

            return Ok(0);
        };

        let mut n_rows_processed: usize = 0;

        while let Ok(LineBatch { bytes, chunk_idx }) = line_batch_rx.recv().await {
            let df = chunk_reader.read_chunk(bytes)?;

            n_rows_processed = n_rows_processed.saturating_add(df.height());

            let morsel_seq = MorselSeq::new(chunk_idx as u64);

            if output_port.send(morsel_seq, df).await.is_err() {
                break;
            }
        }

        if needs_total_row_count {
            if verbose {
                eprintln!(
                    "[NDJSON LineBatchProcessor {}]: entering row count mode",
                    worker_idx
                );
            }

            while let Ok(LineBatch {
                bytes,
                chunk_idx: _,
            }) = line_batch_rx.recv().await
            {
                n_rows_processed = n_rows_processed.saturating_add(ndjson::count_rows(bytes));
            }
        }

        if verbose {
            eprintln!("[NDJSON LineBatchProcessor {}]: returning", worker_idx);
        }

        Ok(n_rows_processed)
    }
}

/// Represents a complete chunk of NDJSON data (i.e. no partial lines).
pub(super) struct LineBatch {
    /// Safety: This is sent between 2 places that both hold a reference to the underlying MemSlice.
    pub(super) bytes: &'static [u8],
    pub(super) chunk_idx: usize,
}

/// We are connected to different outputs depending on query.
pub(super) enum LineBatchProcessorOutputPort {
    /// Connected directly to source node output.
    Direct {
        phase_tx: Option<MorselOutput>,
        phase_tx_receiver: Receiver<MorselOutput>,
        source_token: SourceToken,
        wait_group: WaitGroup,
    },
    /// Connected to:
    /// * Morsel reverser (negative slice)
    /// * Row index / limit applier
    Linearize {
        tx: Inserter<Priority<Reverse<MorselSeq>, DataFrame>>,
    },
    Closed,
}

impl LineBatchProcessorOutputPort {
    fn port_type(&self) -> &'static str {
        use LineBatchProcessorOutputPort::*;
        match self {
            Direct { .. } => "direct",
            Linearize { .. } => "linearize",
            Closed => "closed",
        }
    }

    async fn init(&mut self) -> Result<(), ()> {
        if let Self::Direct {
            phase_tx,
            phase_tx_receiver,
            ..
        } = self
        {
            assert!(phase_tx.is_none());
            *phase_tx = Some(phase_tx_receiver.recv().await?);
        }

        Ok(())
    }

    async fn send(&mut self, morsel_seq: MorselSeq, df: DataFrame) -> Result<(), ()> {
        use LineBatchProcessorOutputPort::*;

        let result = async {
            match self {
                Direct {
                    phase_tx,
                    phase_tx_receiver,
                    source_token,
                    wait_group,
                } => {
                    let mut morsel = Morsel::new(df, morsel_seq, source_token.clone());
                    morsel.set_consume_token(wait_group.token());

                    if phase_tx.as_mut().unwrap().port.send(morsel).await.is_err() {
                        return Err(());
                    };

                    wait_group.wait().await;

                    if source_token.stop_requested() {
                        let v = phase_tx.take().unwrap();
                        v.outcome.stop();
                        drop(v);
                        *phase_tx = Some(phase_tx_receiver.recv().await?);
                    }

                    Ok(())
                },
                Linearize { tx } => tx
                    .insert(Priority(Reverse(morsel_seq), df))
                    .await
                    .map_err(|_| ()),
                Closed => unreachable!(),
            }
        }
        .await;

        if result.is_err() {
            *self = Self::Closed;
        }

        result
    }
}
