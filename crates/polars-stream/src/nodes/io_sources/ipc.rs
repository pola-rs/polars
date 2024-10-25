use std::io::Cursor;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;

use futures::stream::StreamExt;
use polars_core::frame::DataFrame;
use polars_core::prelude::{ArrowField, ArrowSchema, Column, IntoColumn};
use polars_core::series::Series;
use polars_core::utils::arrow::io::ipc::read::{read_file_metadata, FileMetadata, FileReader};
use polars_error::PolarsResult;
use polars_expr::prelude::{phys_expr_to_io_expr, PhysicalExpr};
use polars_expr::state::ExecutionState;
use polars_io::cloud::CloudOptions;
use polars_io::ipc::IpcScanOptions;
use polars_io::predicates::PhysicalIoExpr;
use polars_io::utils::{apply_projection, columns_to_projection};
use polars_io::RowIndex;
use polars_plan::plans::hive::HivePartitions;
use polars_plan::plans::{FileInfo, ScanSources};
use polars_plan::prelude::FileScanOptions;
use polars_utils::mmap::MemSlice;

use crate::async_primitives::connector::{connector, Receiver, Sender};
use crate::async_primitives::wait_group::WaitGroup;
use crate::morsel::SourceToken;
use crate::nodes::{
    ComputeNode, JoinHandle, Morsel, MorselSeq, PortState, TaskPriority, TaskScope,
};
use crate::pipe::{RecvPort, SendPort};

enum Predicate {
    None,
    Slice { offset: i64, length: usize },
    Expr(Arc<dyn PhysicalIoExpr>),
}

pub struct IpcSourceNode {
    sources: ScanSources,

    row_index: Option<RowIndex>,
    projection: Option<Vec<usize>>,
    predicate: Predicate,

    projected_schema: Arc<ArrowSchema>,

    seq: AtomicU64,

    is_finished: AtomicBool,

    // @TODO: This should be some sort of synchronization primitive.
    opened_files: (FileMetadata, MemSlice),
}

impl IpcSourceNode {
    pub fn new(
        sources: ScanSources,
        file_info: FileInfo,
        hive_parts: Option<Arc<Vec<HivePartitions>>>,
        predicate: Option<Arc<dyn PhysicalExpr>>,
        options: IpcScanOptions,
        cloud_options: Option<CloudOptions>,
        file_options: FileScanOptions,
        first_metadata: Option<FileMetadata>,
    ) -> PolarsResult<Self> {
        if sources.len() != 1 {
            todo!();
        }

        let source = match &sources {
            ScanSources::Paths(paths) => {
                let file = std::fs::File::open(paths[0].as_path()).unwrap();
                MemSlice::from_file(&file).unwrap()
            },
            ScanSources::Files(_) => todo!(),
            ScanSources::Buffers(_) => todo!(),
        };

        let metadata = read_file_metadata(&mut std::io::Cursor::new(&*source))?;

        let FileScanOptions {
            slice,
            with_columns,
            cache: _, // @TODO
            row_index,
            rechunk: _, // @TODO: What to do with this?
            file_counter,
            hive_options,
            glob,
            include_file_paths,
            allow_missing_columns,
        } = file_options;

        let projection = with_columns
            .as_ref()
            .map(|cols| columns_to_projection(&cols, &metadata.schema))
            .transpose()?;

        if predicate.is_some() && row_index.is_some() {
            todo!();
        }

        let predicate = match (predicate, slice) {
            (None, None) => Predicate::None,
            (None, Some((offset, _))) if offset != 0 => todo!(),
            (None, Some((offset, length))) => Predicate::Slice { offset, length },
            (Some(expr), None) => Predicate::Expr(phys_expr_to_io_expr(expr)),
            (Some(_), Some(_)) => unreachable!(),
        };

        let projected_schema = projection.as_ref().map_or_else(
            || metadata.schema.clone(),
            |prj| Arc::new(apply_projection(&metadata.schema, prj)),
        );

        Ok(IpcSourceNode {
            sources,

            row_index,
            projection,
            predicate,

            projected_schema,

            seq: AtomicU64::new(0),

            is_finished: AtomicBool::new(false),

            opened_files: (metadata, source),
        })
    }
}

impl ComputeNode for IpcSourceNode {
    fn name(&self) -> &str {
        "ipc_source"
    }

    fn update_state(&mut self, recv: &mut [PortState], send: &mut [PortState]) -> PolarsResult<()> {
        assert!(recv.is_empty());
        assert_eq!(send.len(), 1);

        let (metadata, _) = &self.opened_files;

        let seq = self.seq.load(Ordering::Relaxed);
        let is_finished = self.is_finished.load(Ordering::Relaxed);
        if is_finished || seq as usize >= metadata.blocks.len() {
            send[0] = PortState::Done;
        }

        if send[0] != PortState::Done {
            send[0] = PortState::Ready;
        }

        Ok(())
    }

    fn spawn<'env, 's>(
        &'env mut self,
        scope: &'s TaskScope<'s, 'env>,
        recv: &mut [Option<RecvPort<'_>>],
        send: &mut [Option<SendPort<'_>>],
        _state: &'s ExecutionState,
        join_handles: &mut Vec<JoinHandle<PolarsResult<()>>>,
    ) {
        assert!(recv.is_empty());
        assert_eq!(send.len(), 1);

        let senders = send[0].take().unwrap().parallel();

        let mut rxs = Vec::with_capacity(senders.len());

        let slf = &*self;
        for _ in &senders {
            let (mut tx, rx) = connector();
            rxs.push(rx);

            join_handles.push(scope.spawn_task(TaskPriority::Low, async move {
                loop {
                    if slf.is_finished.load(Ordering::Relaxed) {
                        break;
                    }

                    let (metadata, source) = &slf.opened_files;

                    let mut reader = FileReader::new(
                        Cursor::new(source),
                        metadata.clone(),
                        slf.projection.clone(),
                        None,
                    );

                    loop {
                        if slf.is_finished.load(Ordering::Relaxed) {
                            break;
                        }

                        let seq = slf.seq.fetch_add(1, Ordering::Relaxed);

                        if seq as usize >= metadata.blocks.len() {
                            break;
                        }

                        reader.set_current_block(seq as usize);
                        let record_batch = reader.next().unwrap()?;

                        let schema = reader.schema();
                        assert_eq!(record_batch.arrays().len(), schema.len());

                        let arrays = record_batch.into_arrays();

                        let columns = arrays
                            .into_iter()
                            .zip(slf.projected_schema.iter())
                            .map(|(array, (name, field))| {
                                let field =
                                    ArrowField::new(name.clone(), field.dtype.clone(), true);
                                Ok(Series::try_from((&field, vec![array]))?.into_column())
                            })
                            .collect::<PolarsResult<Vec<Column>>>()?;

                        let mut df = DataFrame::new(columns)?;

                        if let Predicate::Expr(predicate) = &slf.predicate {
                            let s = predicate.evaluate_io(&df)?;
                            let mask = s.bool().expect("filter predicates was not of type boolean");

                            df = df.filter(mask)?;
                        }

                        if tx.send((seq, df)).await.is_err() {
                            break;
                        };
                    }

                    break;
                }

                PolarsResult::Ok(())
            }));
        }

        join_handles.push(scope.spawn_task(TaskPriority::High, async move {
            let mut senders = senders;
            let num_senders = senders.len();

            let source_token = SourceToken::new();

            let mut next = 0;
            let mut buffered = Vec::new();

            let mut rxs = rxs;

            let needs_linearization = matches!(slf.predicate, Predicate::Slice { .. }) || slf.row_index.is_some();

            let mut num_collected = 0;
            loop {
                if slf.is_finished.load(Ordering::Relaxed) {
                    break;
                }

                let (Ok((idx, mut df)), _, _) =
                    futures::future::select_all(rxs.iter_mut().map(|rx| rx.recv())).await
                else {
                    break;
                };

                if needs_linearization {
                    if idx != next {
                        buffered.push((idx, df));
                        let Some(next_idx) = buffered.iter().position(|(idx, _)| *idx == next)
                        else {
                            continue;
                        };

                        (_, df) = buffered.remove(next_idx);
                    }

                    next += 1;

                    if let Some(ri) = &slf.row_index {
                        df = df.with_row_index(ri.name.clone(), Some(ri.offset))?;
                    }

                    if let Predicate::Slice { offset: _, length } = &slf.predicate {
                        if num_collected + df.height() > *length {
                            df = df.slice(0, length - num_collected);
                            slf.is_finished.store(true, Ordering::Relaxed);
                        }

                        num_collected += df.height();
                    }
                }

                let morsel = Morsel::new(df, MorselSeq::new(idx), source_token.clone());
                if senders[(idx as usize) % num_senders]
                    .send(morsel)
                    .await
                    .is_err()
                {
                    break;
                }
            }

            Ok(())
        }));
    }
}
