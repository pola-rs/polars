use std::cmp::Reverse;
use std::future::Future;
use std::marker::PhantomData;
use std::ops::Range;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use futures::stream::FuturesUnordered;
use futures::StreamExt;
use polars_core::config;
use polars_core::frame::column::ScalarColumn;
use polars_core::frame::DataFrame;
use polars_core::prelude::{Column, IntoColumn};
use polars_core::scalar::Scalar;
use polars_core::schema::{Schema, SchemaRef};
use polars_core::utils::arrow::bitmap::{Bitmap, MutableBitmap};
use polars_error::{polars_bail, PolarsResult};
use polars_expr::state::ExecutionState;
use polars_io::cloud::CloudOptions;
use polars_io::RowIndex;
use polars_mem_engine::ScanPredicate;
use polars_plan::plans::hive::HivePartitions;
use polars_plan::plans::{ScanSource, ScanSourceRef, ScanSources};
use polars_utils::index::AtomicIdxSize;
use polars_utils::pl_str::PlSmallStr;
use polars_utils::priority::Priority;
use polars_utils::{format_pl_smallstr, IdxSize};

use super::{SourceNode, SourceOutput};
use crate::async_executor::{spawn, AbortOnDropHandle};
use crate::async_primitives::connector::{connector, Receiver};
use crate::async_primitives::linearizer::{self, Linearizer};
use crate::async_primitives::wait_group::{WaitGroup, WaitToken};
use crate::morsel::SourceToken;
use crate::nodes::io_sources::SourceOutputPort;
use crate::nodes::{JoinHandle, Morsel, MorselSeq, TaskPriority};
use crate::DEFAULT_LINEARIZER_BUFFER_SIZE;

fn source_name(scan_source: ScanSourceRef<'_>, index: usize) -> PlSmallStr {
    match scan_source {
        ScanSourceRef::Path(path) => PlSmallStr::from(path.to_string_lossy().as_ref()),
        ScanSourceRef::File(_) => {
            format_pl_smallstr!("file descriptor #{}", index + 1)
        },
        ScanSourceRef::Buffer(_) => {
            format_pl_smallstr!("in-memory buffer #{}", index + 1)
        },
    }
}

#[derive(Clone, Debug)]
pub enum RowRestrication {
    Slice(Range<usize>),
    #[expect(dead_code)]
    Predicate(ScanPredicate),
}

pub struct MultiScanNode<T: MultiScanable> {
    name: String,
    sources: ScanSources,

    hive_parts: Option<Arc<Vec<HivePartitions>>>,
    allow_missing_columns: bool,
    include_file_paths: Option<PlSmallStr>,

    file_schema: SchemaRef,
    projection: Option<Bitmap>,
    row_index: Option<RowIndex>,
    row_restriction: Option<RowRestrication>,

    read_options: Arc<T::ReadOptions>,
    cloud_options: Arc<Option<CloudOptions>>,

    _pd: PhantomData<T>,
}

impl<T: MultiScanable> MultiScanNode<T> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        sources: ScanSources,

        hive_parts: Option<Arc<Vec<HivePartitions>>>,
        allow_missing_columns: bool,
        include_file_paths: Option<PlSmallStr>,

        file_schema: SchemaRef,
        projection: Option<Bitmap>,
        row_index: Option<RowIndex>,
        row_restriction: Option<RowRestrication>,

        read_options: T::ReadOptions,
        cloud_options: Option<CloudOptions>,
    ) -> Self {
        Self {
            name: format!("multi-scan[{}]", T::BASE_NAME),
            sources,

            hive_parts,
            allow_missing_columns,
            include_file_paths,

            file_schema,
            projection,
            row_index,
            row_restriction,

            read_options: Arc::new(read_options),
            cloud_options: Arc::new(cloud_options),

            _pd: PhantomData,
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn process_dataframe(
    mut df: DataFrame,
    source_name: &PlSmallStr,
    hive_part: Option<&HivePartitions>,
    missing_columns: Option<&Bitmap>,
    include_file_paths: Option<&PlSmallStr>,

    file_schema: &Schema,
    projection: Option<&Bitmap>,
    row_index: Option<&RowIndex>,
) -> PolarsResult<DataFrame> {
    _ = df.schema();

    if let Some(ri) = row_index {
        let ri_column = df
            .get_column_index(ri.name.as_str())
            .expect("should have row index column here");

        let columns = unsafe { df.get_columns_mut() };
        columns[ri_column] = (std::mem::take(&mut columns[ri_column]).take_materialized_series()
            + Scalar::from(ri.offset).into_series(PlSmallStr::EMPTY))?
        .into_column();
    }

    if let Some(hive_part) = hive_part {
        let height = df.height();

        if cfg!(debug_assertions) {
            let schema = df.schema();
            // We should have projected the hive column out when we read the file.
            for column in hive_part.get_statistics().column_stats().iter() {
                assert!(!schema.contains(column.field_name()));
            }
        }
        let mut columns = df.take_columns();

        columns.extend(hive_part.get_statistics().column_stats().iter().filter_map(
            |column_stat| {
                let value = column_stat.get_min_state().unwrap().clone();
                let column_idx = file_schema
                    .index_of(value.name())
                    .expect("hive column not in schema");

                // If the hive column is not included in the projection, skip it.
                if projection.is_some_and(|p| !p.get_bit(column_idx)) {
                    return None;
                }

                Some(ScalarColumn::from_single_value_series(value, height).into_column())
            },
        ));

        df = DataFrame::new_with_height(height, columns)?;
    }

    if let Some(missing_columns) = missing_columns {
        assert_eq!(missing_columns.len(), file_schema.len());

        for column_idx in missing_columns.true_idx_iter() {
            let (name, dtype) = file_schema.get_at_index(column_idx).unwrap();

            // If the hive column is not included in the projection, skip it.
            if projection.is_none_or(|p| p.get_bit(column_idx)) {
                df.with_column(Column::new_scalar(
                    name.clone(),
                    Scalar::null(dtype.clone()),
                    df.height(),
                ))
                .unwrap();
            }
        }
    }

    if let Some(col_name) = include_file_paths {
        df.with_column(Column::new_scalar(
            col_name.clone(),
            source_name.clone().into(),
            df.height(),
        ))
        .unwrap();
    }

    // Project into the right column order.
    df = df.select(
        file_schema
            .iter_names()
            .enumerate()
            .filter(|(i, _)| projection.is_none_or(|p| p.get_bit(*i)))
            .map(|(_, column)| column.clone()),
    )?;

    Ok(df)
}

/// Resolve a projection and missing columns bitmap for a specific source schema from the global
/// schema.
fn resolve_source_projection(
    file_schema: &Schema,
    source_schema: &Schema,

    // Which columns from the file_schema would be physically present in the file. Hive columns,
    // include_file_paths and row_index are for example not physically present in the file.
    physical_columns: &Bitmap,
    allow_missing_columns: bool,
    file_projection: Option<&Bitmap>,
    source: ScanSourceRef<'_>,
    source_idx: usize,
) -> PolarsResult<(Bitmap, Option<Bitmap>)> {
    let mut source_extra = Vec::new();
    let mut base_extra = Vec::new();

    // Get the difference between the two schemas.
    source_schema.field_compare(file_schema, &mut source_extra, &mut base_extra);

    if !source_extra.is_empty() {
        let source_name = source_name(source, source_idx);
        let columns = source_extra
            .iter()
            .map(|(_, (name, _))| format!("'{}'", name))
            .collect::<Vec<_>>()
            .join(", ");
        polars_bail!(
            SchemaMismatch:
            "'{source_name}' contains column(s) {columns}, which are not present in the first scanned file"
        );
    }

    // Filter out the non-physical-columns as those may or may not be present in files, we don't
    // really care.
    base_extra.retain(|(i, _)| physical_columns.get_bit(*i));
    if !allow_missing_columns && !base_extra.is_empty() {
        let source_name = source_name(source, source_idx);
        let columns = base_extra
            .iter()
            .map(|(_, (name, _))| format!("'{}'", name))
            .collect::<Vec<_>>()
            .join(", ");
        polars_bail!(
            SchemaMismatch:
            "'{source_name}' does not contains column(s) {columns}, which are present in the first scanned file. Consider enabling `allow_missing_columns`."
        );
    }

    let missing_columns = allow_missing_columns.then(|| {
        let mut bm = MutableBitmap::from_len_zeroed(file_schema.len());
        for (_, (c, _)) in base_extra {
            bm.set(file_schema.index_of(c).unwrap(), true);
        }
        bm.freeze()
    });

    let mut source_projection = MutableBitmap::from_len_zeroed(source_schema.len());

    let mut j = 0;
    for (i, (source_col_name, source_dtype)) in source_schema.iter().enumerate() {
        while let Some((file_col_name, _)) = file_schema.get_at_index(j) {
            if source_col_name == file_col_name {
                break;
            }
            j += 1;
        }

        let Some((_, file_dtype)) = file_schema.get_at_index(j) else {
            let source_name = source_name(source, source_idx);
            polars_bail!(
                SchemaMismatch:
                "the column order of '{source_name}' does not match the column order of the first scanned file"
            );
        };

        if file_dtype != source_dtype {
            polars_bail!(
                mismatch,
                col = source_col_name,
                expected = file_dtype,
                found = source_dtype
            );
        }

        // Don't load logical columns even if they are in the file. Looking at you hive!
        if physical_columns.get_bit(j) {
            source_projection.set(i, file_projection.is_none_or(|p| p.get_bit(j)));
        }
        j += 1;
    }

    Ok((source_projection.freeze(), missing_columns))
}

pub trait MultiScanable: SourceNode + Sized + Send + Sync {
    type ReadOptions: Send + Sync + 'static;

    const BASE_NAME: &'static str;

    const DOES_PRED_PD: bool;
    const DOES_SLICE_PD: bool;

    fn new(
        source: ScanSource,
        options: &Self::ReadOptions,
        cloud_options: Option<&CloudOptions>,
        row_index: Option<PlSmallStr>,
    ) -> impl Future<Output = PolarsResult<Self>> + Send;

    /// Provide a selection of physical columns to be loaded.
    ///
    /// The provided bitmap should have the same length as a schema given by
    /// [`MultiScanable::physical_schema`].
    fn with_projection(&mut self, projection: Option<&Bitmap>);
    fn with_row_restriction(&mut self, row_restriction: Option<RowRestrication>);

    /// Get the number of physical rows in this source.
    fn unrestricted_row_count(&mut self) -> impl Future<Output = PolarsResult<IdxSize>> + Send;

    /// Schema inferred from of the source.
    ///
    /// This should **NOT** include any logical columns (e.g. file path, row index, hive columns).
    fn physical_schema(&mut self) -> impl Future<Output = PolarsResult<SchemaRef>> + Send;
}

enum SourceInput {
    Serial(Receiver<Morsel>),
    Parallel(Vec<Receiver<Morsel>>),
}

const DEFAULT_MAX_CONCURRENT_SCANS: usize = 8;
fn num_concurrent_scans(num_pipelines: usize) -> usize {
    let max_num_concurrent_scans =
        std::env::var("POLARS_MAX_CONCURRENT_SCANS").map_or(DEFAULT_MAX_CONCURRENT_SCANS, |v| {
            v.parse::<usize>()
                .expect("unable to parse POLARS_MAX_CONCURRENT_SCANS")
                .max(1)
        });
    num_pipelines.min(max_num_concurrent_scans)
}

enum SourcePhaseContent {
    /// 1+ columns, 0+ rows
    Channels(SourceInput),
    /// 0 columns / 0 rows
    OneShot(DataFrame),
}
struct SourcePhase {
    content: SourcePhaseContent,
    unrestricted_row_count: Option<Arc<AtomicIdxSize>>,
    missing_columns: Option<Bitmap>,
}

impl<T: MultiScanable> SourceNode for MultiScanNode<T> {
    fn name(&self) -> &str {
        &self.name
    }

    fn is_source_output_parallel(&self, _is_receiver_serial: bool) -> bool {
        false
    }

    fn spawn_source(
        &mut self,
        num_pipelines: usize,
        mut send_port_recv: Receiver<SourceOutput>,
        _state: &ExecutionState,
        join_handles: &mut Vec<JoinHandle<PolarsResult<()>>>,
        unrestricted_row_count: Option<Arc<AtomicIdxSize>>,
    ) {
        assert!(unrestricted_row_count.is_none());

        let num_concurrent_scans = num_concurrent_scans(num_pipelines);
        let sources = &self.sources;
        let read_options = &self.read_options;
        let cloud_options = &self.cloud_options;
        let file_schema = &self.file_schema;
        let projection = &self.projection;
        let row_index_name = self.row_index.as_ref().map(|ri| &ri.name);
        let allow_missing_columns = self.allow_missing_columns;
        let hive_schema = self
            .hive_parts
            .as_ref()
            .and_then(|p| Some(p.first()?.get_statistics().schema().clone()))
            .unwrap_or_else(|| Arc::new(Schema::default()));
        let physical_columns: Bitmap = file_schema
            .iter_names()
            .map(|n| {
                !hive_schema.contains(n)
                    && self.include_file_paths.as_ref().is_none_or(|c| c != n)
                    && self.row_index.as_ref().is_none_or(|c| c.name != n)
            })
            .collect();

        let (si_send, mut si_recv) = (0..num_concurrent_scans)
            .map(|_| connector::<SourcePhase>())
            .collect::<(Vec<_>, Vec<_>)>();

        let slice_tx = if let Some(RowRestrication::Slice(slice)) = &self.row_restriction {
            let (slice_tx, mut slice_rx) = (0..num_concurrent_scans)
                .map(|_| {
                    let (tx, rx) =
                        connector::<(IdxSize, Arc<(AtomicUsize, AtomicUsize)>, WaitToken)>();
                    (Some(tx), rx)
                })
                .collect::<(Vec<_>, Vec<_>)>();

            // Task that handles the slicing.
            //
            // Since we need someone to keep track of the slice centrally, this thread does the
            // minimal amount of work to keep track of this and makes it is so that most of th work
            // can still happen in parallel.
            let sources = self.sources.clone();
            let mut slice = slice.clone();
            join_handles.push(spawn(TaskPriority::High, async move {
                let verbose = config::verbose();

                for i in 0..sources.len() {
                    if slice.is_empty() {
                        if verbose {
                            let source_name = source_name(sources.at(i), i);
                            eprintln!("[MultiScan]: Slice is at '{source_name}' but no more data is needed. Stopping.");
                        }

                        // Flush all remaining workers waiting for their slice.
                        for mut rx in slice_rx {
                            let Ok((_, slice_range, wait_token)) = rx.recv().await else {
                                continue;
                            };

                            // The order here is necessary to avoid race-conditions.
                            drop(rx);
                            slice_range.0.store(0, Ordering::Relaxed);
                            slice_range.1.store(0, Ordering::Relaxed);
                            drop(wait_token);
                        }

                        break;
                    }

                    let handler = i % num_concurrent_scans;
                    let Ok((num_rows, slice_range, wait_token)) = slice_rx[handler].recv().await
                    else {
                        break;
                    };

                    let num_rows = num_rows as usize;
                    if slice.start >= num_rows {
                        if verbose {
                            let source_name = source_name(sources.at(i), i);
                            eprintln!("[MultiScan]: Skipping '{source_name}' using the slice.");
                        }

                        slice_range.0.store(0, Ordering::Relaxed);
                        slice_range.1.store(0, Ordering::Relaxed);
                    } else {
                        let offset = slice.start;
                        let length = (num_rows - slice.start).min(slice.len());

                        // Only print something if it is actually interesting.
                        if verbose && length < num_rows {
                            let source_name = source_name(sources.at(i), i);
                            eprintln!(
                                "[MultiScan]: Slice for '{source_name}' is ({offset}, {length})."
                            );
                        }

                        slice_range.0.store(offset, Ordering::Relaxed);
                        slice_range.1.store(length, Ordering::Relaxed);
                    }

                    slice.start = slice.start.saturating_sub(num_rows);
                    slice.end = slice.end.saturating_sub(num_rows);

                    drop(wait_token);
                }

                Ok(())
            }));

            slice_tx
        } else {
            (0..num_concurrent_scans).map(|_| None).collect()
        };

        join_handles.extend(si_send.into_iter().zip(slice_tx).enumerate().map(
            |(mut i, (mut si_send, mut slice_tx))| {
                let sources = sources.clone();
                let read_options = read_options.clone();
                let cloud_options = cloud_options.clone();
                let file_schema = file_schema.clone();
                let projection = projection.clone();
                let row_index_name = row_index_name.cloned();
                let physical_columns = physical_columns.clone();

                spawn(TaskPriority::High, async move {
                    let state = ExecutionState::new();
                    let mut join_handles = Vec::new();

                    // Handling of slices
                    let slice_range = Arc::new((AtomicUsize::default(), AtomicUsize::default()));
                    let slice_wg = WaitGroup::default();

                    let mut stop = false;
                    let verbose = config::verbose();

                    while i < sources.len() && !stop {
                        join_handles.clear();

                        let source = sources.at(i).into_owned()?;
                        let (mut output_send, output_recv) = connector();
                        let mut source = T::new(
                            source,
                            read_options.as_ref(),
                            cloud_options.as_ref().as_ref(),
                            row_index_name.clone(),
                        )
                        .await?;

                        let source_schema = source.physical_schema().await?;
                        let (source_projection, missing_columns) = resolve_source_projection(
                            file_schema.as_ref(),
                            source_schema.as_ref(),
                            &physical_columns,
                            allow_missing_columns,
                            projection.as_ref(),
                            sources.at(i),
                            i,
                        )?;

                        if let Some(slice_tx) = &mut slice_tx {
                            let row_count = source.unrestricted_row_count().await?;
                            if slice_tx
                                .send((row_count, slice_range.clone(), slice_wg.token()))
                                .await
                                .is_err()
                            {
                                break;
                            };
                            slice_wg.wait().await;

                            let (start, length) = slice_range.as_ref();
                            let start = start.load(Ordering::Relaxed);
                            let length = length.load(Ordering::Relaxed);

                            // If nothing needs to be loaded from this source, continue to the next
                            // file. This is also the case when we overshoot the slices.
                            if length == 0 {
                                let mut df = DataFrame::empty_with_schema(&source_schema);
                                if let Some(name) = &row_index_name {
                                    df.with_row_index_mut(name.clone(), None);
                                }
                                let phase = SourcePhase {
                                    content: SourcePhaseContent::OneShot(df),
                                    missing_columns: missing_columns.clone(),
                                    unrestricted_row_count: Some(Arc::new(AtomicIdxSize::new(
                                        row_count,
                                    ))),
                                };

                                // Wait for the orchestrator task to actually be interested in the output
                                // of this file.
                                if si_send.send(phase).await.is_err() {
                                    break;
                                };
                                i += num_concurrent_scans;
                                continue;
                            }

                            // If we are stopping before the end, this means that we don't have to
                            // go any further. This saves one count rows.
                            stop |= start + length < row_count as usize;

                            // A slice might cause the source to need to linearize. So if we have a
                            // slice that scans everything. Don't do anything.
                            if length < row_count as usize {
                                source.with_row_restriction(Some(RowRestrication::Slice(
                                    start..start + length,
                                )));
                            }
                        }

                        let unrestricted_row_count = row_index_name
                            .is_some()
                            .then(|| Arc::new(AtomicIdxSize::new(0)));

                        source.with_projection(Some(&source_projection));

                        source.spawn_source(
                            num_pipelines,
                            output_recv,
                            &state,
                            &mut join_handles,
                            unrestricted_row_count.clone(),
                        );

                        // Loop until a phase result indicated that the source is empty.
                        loop {
                            let (tx, rx) = if source.is_source_output_parallel(true) {
                                let (tx, rx) =
                                    (0..num_pipelines)
                                        .map(|_| connector())
                                        .collect::<(Vec<_>, Vec<_>)>();
                                (SourceOutputPort::Parallel(tx), SourceInput::Parallel(rx))
                            } else {
                                let (tx, rx) = connector();
                                (SourceOutputPort::Serial(tx), SourceInput::Serial(rx))
                            };

                            let phase = SourcePhase {
                                content: SourcePhaseContent::Channels(rx),
                                missing_columns: missing_columns.clone(),
                                unrestricted_row_count: unrestricted_row_count.clone(),
                            };

                            // Wait for the orchestrator task to actually be interested in the output
                            // of this file.
                            if si_send.send(phase).await.is_err() {
                                break;
                            };

                            let (outcome, wait_group, tx) = SourceOutput::from_port(tx);

                            // Start draining the source into the created channels.
                            if output_send.send(tx).await.is_err() {
                                break;
                            };

                            // Wait for the phase to end.
                            wait_group.wait().await;
                            if outcome.did_finish() {
                                break;
                            }
                        }

                        if verbose {
                            let source_name = source_name(sources.at(i), i);
                            eprintln!("[MultiScan]: Last data received from '{source_name}'.",);
                        }

                        // One of the tasks might throw an error. In which case, we need to cancel all
                        // handles and find the error.
                        let mut join_handles: FuturesUnordered<_> =
                            join_handles.drain(..).map(AbortOnDropHandle::new).collect();
                        while let Some(ret) = join_handles.next().await {
                            ret?;
                        }

                        i += num_concurrent_scans;
                    }

                    PolarsResult::Ok(())
                })
            },
        ));

        let (mut pass_task_send, pass_task_recv) = (0..num_pipelines)
            .map(|_| {
                connector::<(
                    Receiver<Morsel>,
                    linearizer::Inserter<Priority<Reverse<MorselSeq>, Morsel>>,
                )>()
            })
            .collect::<(Vec<_>, Vec<_>)>();
        join_handles.extend(pass_task_recv.into_iter().map(|mut pass_task_recv| {
            spawn(TaskPriority::High, async move {
                while let Ok((mut recv, mut send)) = pass_task_recv.recv().await {
                    while let Ok(v) = recv.recv().await {
                        if send.insert(Priority(Reverse(v.seq()), v)).await.is_err() {
                            break;
                        }
                    }
                }

                Ok(())
            })
        }));

        let hive_parts = self.hive_parts.clone();
        let include_file_paths = self.include_file_paths.clone();
        let file_schema = self.file_schema.clone();
        let projection = self.projection.clone();
        let mut row_index = self.row_index.clone();
        let sources = sources.clone();
        join_handles.push(spawn(TaskPriority::High, async move {
            let mut seq = MorselSeq::default();
            let mut current_scan = 0;

            // Every phase we are given a new send channel.
            'phase_loop: while let Ok(phase_output) = send_port_recv.recv().await {
                // @TODO: Make this parallel compatible if there is no row count or slice.
                let mut send = phase_output.port.serial();

                let source_token = SourceToken::new();
                let wait_group = WaitGroup::default();

                while current_scan < sources.len() {
                    let source_name = source_name(sources.at(current_scan), current_scan);
                    let hive_part = hive_parts.as_deref().map(|parts| &parts[current_scan]);
                    let si_recv = &mut si_recv[current_scan % num_concurrent_scans];
                    let Ok(phase) = si_recv.recv().await else {
                        return Ok(());
                    };

                    match phase.content {
                        // In certain cases, we don't actually need to read physical data from the
                        // file so we get back a row count.
                        SourcePhaseContent::OneShot(df) => {
                            let df = process_dataframe(
                                df,
                                &source_name,
                                hive_part,
                                phase.missing_columns.as_ref(),
                                include_file_paths.as_ref(),
                                file_schema.as_ref(),
                                projection.as_ref(),
                                row_index.as_ref(),
                            );
                            let df = match df {
                                Ok(df) => df,
                                Err(err) => {
                                    return Err(err);
                                },
                            };

                            let mut morsel = Morsel::new(df, seq, source_token.clone());
                            morsel.set_consume_token(wait_group.token());
                            seq = seq.successor();

                            if send.send(morsel).await.is_err() {
                                break 'phase_loop;
                            }

                            wait_group.wait().await;
                            if source_token.stop_requested() {
                                phase_output.outcome.stop();
                                continue 'phase_loop;
                            }
                        },
                        SourcePhaseContent::Channels(rx) => match rx {
                            SourceInput::Serial(mut rx) => {
                                while let Ok(rg) = rx.recv().await {
                                    let original_source_token = rg.source_token().clone();

                                    let df = rg.into_df();
                                    let df = process_dataframe(
                                        df,
                                        &source_name,
                                        hive_part,
                                        phase.missing_columns.as_ref(),
                                        include_file_paths.as_ref(),
                                        file_schema.as_ref(),
                                        projection.as_ref(),
                                        row_index.as_ref(),
                                    );
                                    let df = match df {
                                        Ok(df) => df,
                                        Err(err) => {
                                            return Err(err);
                                        },
                                    };

                                    let mut morsel = Morsel::new(df, seq, source_token.clone());
                                    morsel.set_consume_token(wait_group.token());
                                    seq = seq.successor();

                                    if send.send(morsel).await.is_err() {
                                        break 'phase_loop;
                                    }

                                    wait_group.wait().await;
                                    if source_token.stop_requested() {
                                        original_source_token.stop();
                                        phase_output.outcome.stop();
                                        continue 'phase_loop;
                                    }
                                }
                            },
                            SourceInput::Parallel(rxs) => {
                                let (mut linearizer, inserters) =
                                    Linearizer::new(num_pipelines, DEFAULT_LINEARIZER_BUFFER_SIZE);
                                for ((rx, pass_task_send), inserter) in rxs
                                    .into_iter()
                                    .zip(pass_task_send.iter_mut())
                                    .zip(inserters)
                                {
                                    if pass_task_send.send((rx, inserter)).await.is_err() {
                                        return Ok(());
                                    };
                                }

                                while let Some(rg) = linearizer.get().await {
                                    let rg = rg.1;

                                    let original_source_token = rg.source_token().clone();

                                    let df = rg.into_df();
                                    let df = process_dataframe(
                                        df,
                                        &source_name,
                                        hive_part,
                                        phase.missing_columns.as_ref(),
                                        include_file_paths.as_ref(),
                                        file_schema.as_ref(),
                                        projection.as_ref(),
                                        row_index.as_ref(),
                                    );
                                    let df = match df {
                                        Ok(df) => df,
                                        Err(err) => {
                                            return Err(err);
                                        },
                                    };

                                    let mut morsel = Morsel::new(df, seq, source_token.clone());
                                    morsel.set_consume_token(wait_group.token());
                                    seq = seq.successor();

                                    if send.send(morsel).await.is_err() {
                                        break 'phase_loop;
                                    }

                                    wait_group.wait().await;
                                    if source_token.stop_requested() {
                                        original_source_token.stop();
                                        phase_output.outcome.stop();
                                        continue 'phase_loop;
                                    }
                                }
                            },
                        },
                    }

                    if let Some(ri) = row_index.as_mut() {
                        let source_num_rows = phase
                            .unrestricted_row_count
                            .unwrap()
                            .load(Ordering::Relaxed);
                        ri.offset += source_num_rows;
                    }
                    current_scan += 1;
                }
                break;
            }

            Ok(())
        }));
    }
}
