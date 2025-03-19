use std::cmp::Reverse;
use std::future::Future;
use std::marker::PhantomData;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use futures::StreamExt;
use futures::stream::FuturesUnordered;
use polars_core::config;
use polars_core::frame::DataFrame;
use polars_core::frame::column::ScalarColumn;
use polars_core::prelude::{Column, IDX_DTYPE, IntoColumn};
use polars_core::scalar::Scalar;
use polars_core::schema::{Schema, SchemaRef};
use polars_core::utils::arrow::bitmap::{Bitmap, MutableBitmap};
use polars_error::{PolarsResult, polars_bail};
use polars_io::RowIndex;
use polars_io::cloud::CloudOptions;
use polars_io::predicates::ScanIOPredicate;
use polars_plan::dsl::{ScanSource, ScanSourceRef, ScanSources};
use polars_plan::plans::hive::HivePartitionsDf;
use polars_utils::pl_str::PlSmallStr;
use polars_utils::priority::Priority;
use polars_utils::{IdxSize, format_pl_smallstr};
use tokio::sync::oneshot;

use super::{RowRestriction, SourceNode, SourceOutput};
use crate::DEFAULT_LINEARIZER_BUFFER_SIZE;
use crate::async_executor::{AbortOnDropHandle, spawn};
use crate::async_primitives::connector::{Receiver, connector};
use crate::async_primitives::linearizer::Linearizer;
use crate::async_primitives::wait_group::{WaitGroup, WaitToken};
use crate::execute::StreamingExecutionState;
use crate::morsel::SourceToken;
use crate::nodes::io_sources::SourceOutputPort;
use crate::nodes::{JoinHandle, Morsel, MorselSeq, TaskPriority};

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
pub enum MultiscanRowRestriction {
    /// A negative slice needs to be resolved before scanning can start.
    NegativeSlice(usize, usize),
    Source(RowRestriction),
}

pub struct MultiScanNode<T: MultiScanable> {
    name: String,
    sources: ScanSources,

    hive_parts: Option<Arc<HivePartitionsDf>>,
    allow_missing_columns: bool,
    include_file_paths: Option<PlSmallStr>,

    file_schema: SchemaRef,
    projection: Option<Bitmap>,
    row_index: Option<RowIndex>,
    row_restriction: Option<MultiscanRowRestriction>,

    read_options: Arc<T::ReadOptions>,
    cloud_options: Arc<Option<CloudOptions>>,

    _pd: PhantomData<T>,
}

impl<T: MultiScanable> MultiScanNode<T> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        sources: ScanSources,

        hive_parts: Option<Arc<HivePartitionsDf>>,
        allow_missing_columns: bool,
        include_file_paths: Option<PlSmallStr>,

        file_schema: SchemaRef,
        projection: Option<Bitmap>,
        row_index: Option<RowIndex>,
        mut row_restriction: Option<MultiscanRowRestriction>,
        predicate: Option<ScanIOPredicate>,

        read_options: T::ReadOptions,
        cloud_options: Option<CloudOptions>,
    ) -> Self {
        if let Some(predicate) = predicate {
            assert!(row_restriction.is_none());
            row_restriction = Some(MultiscanRowRestriction::Source(RowRestriction::Predicate(
                predicate,
            )));
        }

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
    current_scan: usize,
    hive_parts: Option<&HivePartitionsDf>,
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

    if let Some(hive_parts) = hive_parts {
        let height = df.height();

        if cfg!(debug_assertions) {
            let schema = df.schema();
            // We should have projected the hive column out when we read the file.
            for column in hive_parts.df().get_columns() {
                assert!(!schema.contains(column.name()));
            }
        }
        let mut columns = df.take_columns();

        columns.extend(hive_parts.df().get_columns().iter().filter_map(|column| {
            let column_idx = file_schema
                .index_of(column.name())
                .expect("hive column not in schema");

            // If the hive column is not included in the projection, skip it.
            if projection.is_some_and(|p| !p.get_bit(column_idx)) {
                return None;
            }

            // @TODO: Do without cloning the series several times here.
            let value = column
                .slice(current_scan as i64, 1)
                .take_materialized_series();
            Some(ScalarColumn::from_single_value_series(value, height).into_column())
        }));

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
            "'{source_name}' does not contain column(s) {columns}, which are present in the first scanned file. Consider enabling `allow_missing_columns`."
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

pub fn scan_predicate_to_mask(
    scan_predicate: &ScanIOPredicate,
    file_schema: &Schema,
    hive_schema: &Schema,
    hive_parts: &HivePartitionsDf,
) -> PolarsResult<(Option<Bitmap>, bool)> {
    let Some(sbp) = scan_predicate.skip_batch_predicate.as_ref() else {
        return Ok((None, true));
    };

    let non_hive_live_columns = scan_predicate
        .live_columns
        .iter()
        .filter(|lc| !hive_schema.contains(lc))
        .collect::<Vec<_>>();

    if non_hive_live_columns.len() == scan_predicate.live_columns.len() {
        return Ok((None, true));
    }

    let mut statistics_columns =
        Vec::with_capacity(1 + 3 * hive_schema.len() + 3 * non_hive_live_columns.len());

    // We don't know the sizes of the files here yet.
    statistics_columns.push(Column::new_scalar(
        "len".into(),
        Scalar::null(IDX_DTYPE),
        hive_parts.df().height(),
    ));
    for column in hive_parts.df().get_columns() {
        let c = column.name();

        // If the hive value is not null, we know we have 0 nulls for the hive column in the file
        // otherwise we don't know. Same reasoning as with the `len`.
        let mut nc = Column::new_scalar(
            format_pl_smallstr!("{c}_nc"),
            (0 as IdxSize).into(),
            hive_parts.df().height(),
        );
        if column.has_nulls() {
            nc = nc.zip_with_same_type(
                &column.is_null(),
                &Column::new_scalar(PlSmallStr::EMPTY, Scalar::null(IDX_DTYPE), 1),
            )?;
        }

        statistics_columns.extend([
            column.clone().with_name(format_pl_smallstr!("{c}_min")),
            column.clone().with_name(format_pl_smallstr!("{c}_max")),
            nc,
        ]);
    }
    for c in &non_hive_live_columns {
        let dtype = file_schema.try_get(c)?;
        statistics_columns.extend([
            Column::full_null(
                format_pl_smallstr!("{c}_min"),
                hive_parts.df().height(),
                dtype,
            ),
            Column::full_null(
                format_pl_smallstr!("{c}_max"),
                hive_parts.df().height(),
                dtype,
            ),
            Column::full_null(
                format_pl_smallstr!("{c}_nc"),
                hive_parts.df().height(),
                &IDX_DTYPE,
            ),
        ]);
    }

    let statistics_df = DataFrame::new(statistics_columns)?;
    let mask = sbp.evaluate_with_stat_df(&statistics_df)?;

    if config::verbose() {
        eprintln!(
            "[MultiScan]: Predicate pushdown allows skipping {} / {} sources",
            mask.set_bits(),
            mask.len()
        );
    }

    Ok((Some(mask), !non_hive_live_columns.is_empty()))
}

pub trait MultiScanable: SourceNode + Sized + Send + Sync {
    type ReadOptions: Send + Sync + 'static;

    const BASE_NAME: &'static str;

    /// Does the SourceNode have a specialized implementation for filtering, i.e. is it better to
    /// filter within the ScanSource or just place a filter after the scan source.
    const SPECIALIZED_PRED_PD: bool;

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
    fn with_row_restriction(&mut self, row_restriction: Option<RowRestriction>);

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
pub fn max_concurrent_scans(num_pipelines: usize) -> usize {
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
    Channels(SourceInput, oneshot::Receiver<bool>),
    /// 0 columns / 0 rows
    OneShot(DataFrame),
}
struct SourcePhase {
    /// We don't use this to communicate anything, it is just a sanity check.
    source_idx: usize,
    content: SourcePhaseContent,
    unrestricted_row_count: Option<tokio::sync::oneshot::Receiver<IdxSize>>,
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
        mut send_port_recv: Receiver<SourceOutput>,
        state: &StreamingExecutionState,
        join_handles: &mut Vec<JoinHandle<PolarsResult<()>>>,
        unrestricted_row_count: Option<tokio::sync::oneshot::Sender<IdxSize>>,
    ) {
        assert!(unrestricted_row_count.is_none());

        let state = state.clone();
        let max_concurrent_scans = max_concurrent_scans(state.num_pipelines);
        let sources = self.sources.clone();
        let num_concurrent_scans = max_concurrent_scans.min(sources.len());
        let read_options = self.read_options.clone();
        let cloud_options = self.cloud_options.clone();
        let file_schema = self.file_schema.clone();
        let projection = self.projection.clone();
        let row_index_name = self.row_index.as_ref().map(|ri| ri.name.clone());
        let allow_missing_columns = self.allow_missing_columns;
        let hive_parts = self.hive_parts.clone();
        let include_file_paths = self.include_file_paths.clone();
        let mut row_index = self.row_index.clone();
        let hive_schema = self
            .hive_parts
            .as_ref()
            .map_or_else(|| Arc::new(Schema::default()), |p| p.schema().clone());
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

        let row_restriction = self.row_restriction.clone();

        join_handles.push(spawn(TaskPriority::Low, async move {
            let verbose = config::verbose();
            let mut first_data_source = 0;
            let mut skipable_file_mask = None;

            let row_restriction = match row_restriction {
                Some(MultiscanRowRestriction::NegativeSlice(mut offset, length)) => {
                    let start_index;
                    let positive_slice;
                    let mut source_length_sum = 0;

                    if verbose {
                        eprintln!("[MultiScan]: Converting negative slice (-{offset}, {length}) to positive slice");
                    }

                    let mut i = sources.len();
                    loop {
                        if i == 0 {
                            start_index = 0;
                            positive_slice = 0..(length.saturating_sub(offset)).min(source_length_sum);
                            break;
                        }

                        i -= 1;
                        let source = sources.at(i).into_owned()?;
                        let mut source = T::new(
                            source,
                            read_options.as_ref(),
                            cloud_options.as_ref().as_ref(),
                            None,
                        )
                        .await?;

                        let num_rows = source.unrestricted_row_count().await?;
                        let num_rows = num_rows as usize;
                        source_length_sum += offset.min(num_rows);

                        if offset < num_rows {
                            start_index = i;
                            let start_row = num_rows - offset;
                            let end_row = start_row + length.min(source_length_sum);

                            positive_slice = start_row..end_row;
                            break;
                        }

                        offset -= num_rows;
                    }

                    if verbose {
                        eprintln!("[MultiScan]: Resulting positive slice ({}, {}) and allows skipping {start_index} sources.", positive_slice.start, positive_slice.len());
                    }

                    first_data_source = start_index;
                    Some(RowRestriction::Slice(positive_slice))
                },
                Some(MultiscanRowRestriction::Source(RowRestriction::Predicate(scan_predicate))) => {
                    if let Some(hive_parts) = hive_parts.as_ref() {
                        let (mask, need_pred_for_inner_readers) = scan_predicate_to_mask(
                            &scan_predicate,
                            file_schema.as_ref(),
                            hive_schema.as_ref(),
                            hive_parts
                        )?;

                        skipable_file_mask = mask;
                        if need_pred_for_inner_readers {
                            Some(RowRestriction::Predicate(scan_predicate))
                        } else {
                            None
                        }
                    } else {
                        Some(RowRestriction::Predicate(scan_predicate))
                    }
                },
                Some(MultiscanRowRestriction::Source(r)) => Some(r),
                None => None,
            };

            let first_scan_source = if row_index_name.is_some() {
                0
            } else {
                first_data_source
            };

            let mut join_handles = Vec::new();
            let slice_tx = if let Some(RowRestriction::Slice(slice)) = &row_restriction {
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
                let mut slice = slice.clone();
                let sources = sources.clone();
                join_handles.push(spawn(TaskPriority::High, async move {
                    let verbose = config::verbose();

                    for i in first_scan_source..sources.len() {
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

                        let handler = i % max_concurrent_scans;
                        let Ok((num_rows, slice_range, wait_token)) = slice_rx[handler].recv().await
                        else {
                            break;
                        };

                        let num_rows = num_rows as usize;

                        if i < first_data_source {
                            slice_range.0.store(0, Ordering::Relaxed);
                            slice_range.1.store(0, Ordering::Relaxed);
                        } else {
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
                        }

                        drop(wait_token);
                    }

                    Ok(())
                }));

                slice_tx
            } else {
                (0..max_concurrent_scans).map(|_| None).collect()
            };

            join_handles.extend(si_send.into_iter().zip(slice_tx).enumerate().map(
                |(mut i, (mut si_send, mut slice_tx))| {
                    let state = state.clone();
                    let sources = sources.clone();
                    let hive_parts = hive_parts.clone();
                    let include_file_paths = include_file_paths.clone();
                    let row_restriction = row_restriction.clone();
                    let read_options = read_options.clone();
                    let cloud_options = cloud_options.clone();
                    let file_schema = file_schema.clone();
                    let projection = projection.clone();
                    let row_index_name = row_index_name.clone();
                    let physical_columns = physical_columns.clone();
                    let skipable_file_mask = skipable_file_mask.clone();

                    spawn(TaskPriority::High, async move {
                        let mut join_handles = Vec::new();

                        // Handling of slices
                        let slice_range = Arc::new((AtomicUsize::default(), AtomicUsize::default()));
                        let slice_wg = WaitGroup::default();

                        let mut stop = false;
                        let verbose = config::verbose();

                        if first_scan_source > 0 {
                            let n1 = first_scan_source.next_multiple_of(max_concurrent_scans) + i;
                            let n0 = n1 - max_concurrent_scans;

                            i = if n0 >= first_scan_source {
                                n0
                            } else {
                                n1
                            };
                        }

                        while i < sources.len() && !stop {
                            let is_selected = skipable_file_mask.as_ref().is_some_and(|s| s.get_bit(i));

                            // If the predicate allowed skipping this file, do.
                            if row_index_name.is_none() && is_selected {
                                i += max_concurrent_scans;
                                continue;
                            }

                            let source = sources.at(i).into_owned()?;
                            let (mut output_send, output_recv) = connector();
                            let mut source = T::new(
                                source,
                                read_options.as_ref(),
                                cloud_options.as_ref().as_ref(),
                                row_index_name.clone(),
                            )
                            .await?;

                            if is_selected {
                                let row_count = source.unrestricted_row_count().await?;
                                let unrestricted_row_count_rx = {
                                    let (tx, rx) = tokio::sync::oneshot::channel();
                                    tx.send(row_count).unwrap();
                                    rx
                                };
                                let phase = SourcePhase {
                                    source_idx: i,
                                    content: SourcePhaseContent::OneShot(DataFrame::empty()),
                                    missing_columns: None,
                                    unrestricted_row_count: Some(unrestricted_row_count_rx),
                                };
                                // Wait for the orchestrator task to actually be interested in the output
                                // of this file.
                                if si_send.send(phase).await.is_err() {
                                    break;
                                };
                                i += max_concurrent_scans;
                                continue;
                            }

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
                                        unsafe            {        df.with_row_index_mut(name.clone(), None)};
                                    }

                                    let unrestricted_row_count_rx = {
                                        let (tx, rx) = tokio::sync::oneshot::channel();
                                      tx.send(row_count).unwrap();
                                        rx
                                    };

                                    let phase = SourcePhase {
                                        source_idx: i,
                                        content: SourcePhaseContent::OneShot(df),
                                        missing_columns: missing_columns.clone(),
                                        unrestricted_row_count: Some(unrestricted_row_count_rx),
                                    };

                                    // Wait for the orchestrator task to actually be interested in the output
                                    // of this file.
                                    if si_send.send(phase).await.is_err() {
                                        break;
                                    };
                                    i += max_concurrent_scans;
                                    continue;
                                }

                                // If we are stopping before the end, this means that we don't have to
                                // go any further. This saves one count rows.
                                stop |= start + length < row_count as usize;

                                // A slice might cause the source to need to linearize. So if we have a
                                // slice that scans everything. Don't do anything.
                                if length < row_count as usize {
                                    source.with_row_restriction(Some(RowRestriction::Slice(
                                        start..start + length,
                                    )));
                                }
                            }

                            let predicate = match &row_restriction {
                                Some(RowRestriction::Predicate(predicate)) => Some(predicate),
                                _ => None,
                            };
                            if let Some(predicate) = predicate.filter(|_| T::SPECIALIZED_PRED_PD) {
                                let mut num_live_logical_columns = 0;
                                num_live_logical_columns += usize::from(
                                        include_file_paths
                                        .as_ref()
                                        .is_some_and(|ifp| predicate.live_columns.contains(ifp)),
                                );
                                if let Some(hive_df) = hive_parts.as_deref() {
                                    for c in hive_df.df().get_columns() {
                                        num_live_logical_columns += usize::from(predicate.live_columns.contains(c.name()));
                                    }
                                }

                                if num_live_logical_columns < predicate.live_columns.len() {
                                    let mut predicate = predicate.clone();
                                    let mut constant_columns = Vec::new();

                                    if let Some(ifp) = include_file_paths.as_ref().filter(|ifp| predicate.live_columns.contains(*ifp)) {
                                        constant_columns.push((ifp.clone(), Scalar::from(source_name(sources.at(i), i))));
                                    }
                                    // @NOTE: No row index as that is generated by the source.
                                    if let Some(hive_df) = hive_parts.as_deref() {
                                        for c in hive_df.df().get_columns() {
                                            if predicate.live_columns.contains(c.name()) {
                                                constant_columns.push((c.name().clone(), Scalar::new(c.dtype().clone(), c.get(i).unwrap().into_static())));
                                            }
                                        }
                                    }
                                    if let Some(missing_columns) = missing_columns.as_ref() {
                                        for idx in missing_columns.true_idx_iter() {
                                            let (name, dtype) = file_schema.get_at_index(idx).unwrap();
                                            if predicate.live_columns.contains(name) {
                                                constant_columns.push((name.clone(), Scalar::null(dtype.clone())));
                                            }
                                        }
                                    }

                                    if !constant_columns.is_empty() {
                                        predicate.set_external_constant_columns(constant_columns);
                                    }
                                    source.with_row_restriction(Some(RowRestriction::Predicate(predicate)));
                                }
                            }

                            let (unrestricted_row_count_tx, mut unrestricted_row_count_rx) =
                                if row_index_name.is_some() {
                                    let (tx, rx) = tokio::sync::oneshot::channel() ;
                                    (Some(tx), Some(rx))
                                } else {
                                    (None,None)
                                };

                            source.with_projection(Some(&source_projection));
                            source.spawn_source(
                                output_recv,
                                &state,
                                &mut join_handles,
                                unrestricted_row_count_tx,
                            );
                            let mut join_handles: FuturesUnordered<_> =
                                join_handles.drain(..).map(AbortOnDropHandle::new).collect();

                            // Loop until a phase result indicated that the source is empty.
                            loop {
                                let (tx, rx) = if source.is_source_output_parallel(true) {
                                    let (tx, rx) =
                                        (0..state.num_pipelines)
                                            .map(|_| connector())
                                            .collect::<(Vec<_>, Vec<_>)>();
                                    (SourceOutputPort::Parallel(tx), SourceInput::Parallel(rx))
                                } else {
                                    let (tx, rx) = connector();
                                    (SourceOutputPort::Serial(tx), SourceInput::Serial(rx))
                                };

                                let (source_finished_tx, source_finished_rx) = oneshot::channel();
                                let (outcome, wait_group, tx) = SourceOutput::from_port(tx);
                                let phase = SourcePhase {
                                    source_idx: i,
                                    content: SourcePhaseContent::Channels(rx, source_finished_rx),
                                    missing_columns: missing_columns.clone(),
                                    unrestricted_row_count: unrestricted_row_count_rx.take(),
                                };

                                // Wait for the orchestrator task to actually be interested in the output
                                // of this file.
                                if si_send.send(phase).await.is_err() {
                                    break;
                                };

                                // Start draining the source into the created channels.
                                if output_send.send(tx).await.is_err() {
                                    break;
                                };

                                // Wait for the phase to end.
                                wait_group.wait().await;
                                let did_finish = outcome.did_finish();
                                if source_finished_tx.send(did_finish).is_err() {
                                    return Ok(());
                                }
                                if did_finish {
                                    break;
                                }
                            }

                            if verbose {
                                let source_name = source_name(sources.at(i), i);
                                eprintln!("[MultiScan]: Last data received from '{source_name}'.",);
                            }

                            // One of the tasks might throw an error. In which case, we need to cancel all
                            // handles and find the error.
                            while let Some(ret) = join_handles.next().await {
                                ret?;
                            }

                            i += max_concurrent_scans;
                        }

                        PolarsResult::Ok(())
                    })
                }
            ));

            let sources = sources.clone();
            let mut skipable_file_mask = skipable_file_mask.clone();
            join_handles.push(spawn(TaskPriority::High, async move {
                let mut seq = MorselSeq::default();
                let mut current_scan = first_scan_source;
                let mut unrestricted_row_count_rx = None;

                // Every phase we are given a new send channel.
                'phase_loop: while let Ok(phase_output) = send_port_recv.recv().await {
                    let source_token = SourceToken::new();
                    let wait_group = WaitGroup::default();

                    let mut send = phase_output.port.serial();
                    while current_scan < sources.len() {
                        if let Some(skipable_file_mask) = skipable_file_mask.as_mut() {
                            assert_eq!(sources.len() - current_scan, skipable_file_mask.len());

                            if row_index.is_none() {
                                // Skip over all skipable files. The workers must do the same
                                // otherwise they get out of sync.
                                current_scan += skipable_file_mask.take_leading_ones();
                            }
                            if skipable_file_mask.unset_bits() == 0 {
                                break 'phase_loop;
                            }
                        }

                        let source_name = source_name(sources.at(current_scan), current_scan);
                        let si_recv = &mut si_recv[current_scan % max_concurrent_scans];
                        let Ok(phase) = si_recv.recv().await else {
                            return Ok(());
                        };

                        // This is sent with the first phase.
                        unrestricted_row_count_rx = unrestricted_row_count_rx.or(phase.unrestricted_row_count);

                        // Sanity Check: Is the worker currently on the same source as we are?
                        assert_eq!(phase.source_idx, current_scan);

                        let is_selected = skipable_file_mask.as_ref().is_none_or(|s| !s.get_bit(0));
                        if let Some(skipable_file_mask) = skipable_file_mask.as_mut() {
                            skipable_file_mask.slice(1, skipable_file_mask.len() - 1);
                        }

                        let mut stopped = false;
                        match phase.content {
                            // In certain cases, we don't actually need to read physical data from the
                            // file so we get back a row count.
                            SourcePhaseContent::OneShot(df) => {
                                if is_selected {
                                    let df = process_dataframe(
                                        df,
                                        &source_name,
                                        current_scan,
                                        hive_parts.as_deref(),
                                        phase.missing_columns.as_ref(),
                                        include_file_paths.as_ref(),
                                        file_schema.as_ref(),
                                        projection.as_ref(),
                                        row_index.as_ref(),
                                    )?;

                                    let mut morsel = Morsel::new(df, seq, source_token.clone());
                                    seq = seq.successor();
                                    morsel.set_consume_token(wait_group.token());

                                    if send.send(morsel).await.is_err() {
                                        return Ok(());
                                    }

                                    wait_group.wait().await;
                                    if source_token.stop_requested() {
                                        phase_output.outcome.stop();
                                        continue 'phase_loop;
                                    }
                                }
                            },
                            SourcePhaseContent::Channels(rx, source_finished_rx) => {
                                assert!(is_selected);

                                let mut linearizer_tasks: FuturesUnordered<AbortOnDropHandle<PolarsResult<()>>> = FuturesUnordered::new();
                                let mut rx = match rx {
                                    SourceInput::Serial(rx) => rx,
                                    SourceInput::Parallel(rxs) => {
                                        let (mut tx, rx) = connector();
                                        let (mut lin_rx, lin_txs) = Linearizer::new(state.num_pipelines, *DEFAULT_LINEARIZER_BUFFER_SIZE);

                                        linearizer_tasks.extend(rxs.into_iter().zip(lin_txs).map(|(mut rx, mut lin_tx)|
                                            AbortOnDropHandle::new(spawn(TaskPriority::High, async move {
                                                while let Ok(mut m) = rx.recv().await {
                                                    let consume_token = m.take_consume_token();
                                                    if lin_tx.insert(Priority(Reverse(m.seq()), m)).await.is_err() {
                                                        return Ok(());
                                                    }
                                                    drop(consume_token);
                                                }
                                                PolarsResult::Ok(())
                                            }
                                        ))));
                                        linearizer_tasks.push(AbortOnDropHandle::new(spawn(TaskPriority::High, async move {
                                            while let Some(Priority(_, m)) = lin_rx.get().await {
                                                if tx.send(m).await.is_err() {
                                                    return Ok(());
                                                }
                                            }
                                            PolarsResult::Ok(())
                                        })));

                                        rx
                                    },
                                };

                                while let Ok(morsel) = rx.recv().await {
                                    let (df, _, original_source_token, consume_token) = morsel.into_inner();
                                    drop(consume_token);
                                    let df = process_dataframe(
                                        df,
                                        &source_name,
                                        current_scan,
                                        hive_parts.as_deref(),
                                        phase.missing_columns.as_ref(),
                                        include_file_paths.as_ref(),
                                        file_schema.as_ref(),
                                        projection.as_ref(),
                                        row_index.as_ref(),
                                    )?;

                                    let mut morsel = Morsel::new(df, seq, source_token.clone());
                                    seq = seq.successor();
                                    morsel.set_consume_token(wait_group.token());

                                    if send.send(morsel).await.is_err() {
                                        return Ok(());
                                    }

                                    wait_group.wait().await;
                                    if source_token.stop_requested() {
                                        original_source_token.stop();
                                        stopped = true;
                                    }
                                }

                                drop(rx);
                                let Ok(is_finished) = source_finished_rx.await else {
                                    return Ok(());
                                };
                                while let Some(res) = linearizer_tasks.next().await {
                                    res?
                                }

                                if !is_finished {
                                    phase_output.outcome.stop();
                                    continue 'phase_loop;
                                }
                            },
                        }


                        if let Some(ri) = row_index.as_mut() {
                            let source_num_rows = unrestricted_row_count_rx
                                .take()
                                .unwrap()
                                .await
                                .unwrap();
                            ri.offset += source_num_rows;
                        }
                        current_scan += 1;

                        if stopped {
                            phase_output.outcome.stop();
                            continue 'phase_loop;
                        }
                    }
                    break;
                }

                Ok(())
            }));

            let mut join_handles: FuturesUnordered<_> = join_handles
                .drain(..)
                .map(AbortOnDropHandle::new)
                .collect();
            while let Some(ret) = join_handles.next().await {
                ret?;
            }

            Ok(())
        }));
    }
}
