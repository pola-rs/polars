use std::sync::Arc;

use polars_core::frame::DataFrame;
use polars_core::prelude::{
    AnyValue, ArrowSchema, Column, DataType, PlHashMap, PlIndexSet, IDX_DTYPE,
};
use polars_core::scalar::Scalar;
use polars_core::schema::Schema;
use polars_core::series::IsSorted;
use polars_core::utils::arrow::bitmap::Bitmap;
use polars_core::utils::operation_exceeded_idxsize_msg;
use polars_error::{polars_err, PolarsError, PolarsResult};
use polars_io::predicates::{ScanIOPredicate, SkipBatchPredicate};
use polars_io::prelude::{collect_statistics, create_sorting_map, FileMetadata};
use polars_io::utils::byte_source::{ByteSource, DynByteSource};
use polars_io::utils::slice::SplitSlicePosition;
use polars_parquet::read::RowGroupMetadata;
use polars_utils::mmap::MemSlice;
use polars_utils::pl_str::PlSmallStr;
use polars_utils::{format_pl_smallstr, IdxSize};

use super::mem_prefetch_funcs;
use super::row_group_decode::SharedFileState;
use crate::utils::task_handles_ext;

/// Represents byte-data that can be transformed into a DataFrame after some computation.
pub(super) struct RowGroupData {
    pub(super) fetched_bytes: FetchedBytes,
    pub(super) path_index: usize,
    pub(super) row_offset: usize,
    pub(super) slice: Option<(usize, usize)>,
    pub(super) file_max_row_group_height: usize,
    pub(super) row_group_metadata: RowGroupMetadata,
    pub(super) sorting_map: PlHashMap<usize, IsSorted>,
    pub(super) shared_file_state: Arc<tokio::sync::OnceCell<SharedFileState>>,
}

pub(super) struct RowGroupDataFetcher {
    pub(super) metadata_rx: crate::async_primitives::connector::Receiver<(
        usize,
        usize,
        Arc<DynByteSource>,
        FileMetadata,
    )>,
    pub(super) use_statistics: bool,
    pub(super) verbose: bool,
    pub(super) reader_schema: Arc<ArrowSchema>,
    pub(super) projection: Option<Arc<[PlSmallStr]>>,
    #[allow(unused)] // TODO: Fix!
    pub(super) predicate: Option<ScanIOPredicate>,
    pub(super) slice_range: Option<std::ops::Range<usize>>,
    pub(super) memory_prefetch_func: fn(&[u8]) -> (),
    pub(super) current_path_index: usize,
    pub(super) current_byte_source: Arc<DynByteSource>,
    pub(super) current_rg_selection: Option<Bitmap>,
    pub(super) current_row_groups: std::vec::IntoIter<RowGroupMetadata>,
    pub(super) current_row_group_idx: usize,
    pub(super) current_max_row_group_height: usize,
    pub(super) current_row_offset: usize,
    pub(super) current_shared_file_state: Arc<tokio::sync::OnceCell<SharedFileState>>,
}

fn create_select_mask(
    sbp: &dyn SkipBatchPredicate,
    live_columns: &PlIndexSet<PlSmallStr>,
    schema: &ArrowSchema,
    rgs: &[RowGroupMetadata],
) -> PolarsResult<Bitmap> {
    let idxs = live_columns
        .iter()
        .map(|l| schema.index_of(l.as_str()).unwrap())
        .collect::<Vec<_>>();
    let stat_schema: Schema = std::iter::once((PlSmallStr::from_static("len"), IDX_DTYPE))
        .chain(
            live_columns
                .iter()
                .map(|l| {
                    let dtype = DataType::from_arrow_field(schema.get(l).unwrap());
                    [
                        (format_pl_smallstr!("{}_min", l), dtype.clone()),
                        (format_pl_smallstr!("{}_max", l), dtype),
                        (format_pl_smallstr!("{}_nc", l), IDX_DTYPE),
                    ]
                    .into_iter()
                })
                .flatten(),
        )
        .collect();
    let mut df = DataFrame::empty_with_schema(&stat_schema);

    for md in rgs {
        let mut columns = Vec::with_capacity(stat_schema.len());

        columns.push(Column::new_scalar(
            PlSmallStr::from_static("len"),
            (md.num_rows() as IdxSize).into(),
            1,
        ));

        if let Some(stats) = collect_statistics(md, schema)? {
            for col_idx in &idxs {
                let col = &stats.column_stats()[*col_idx];
                let dtype = col.dtype();

                let min = Scalar::new(
                    dtype.clone(),
                    col.to_min()
                        .map_or(AnyValue::Null, |s| s.get(0).unwrap().into_static()),
                );
                let max = Scalar::new(
                    dtype.clone(),
                    col.to_max()
                        .map_or(AnyValue::Null, |s| s.get(0).unwrap().into_static()),
                );
                let null_count = col
                    .null_count()
                    .map_or(Scalar::null(IDX_DTYPE), |nc| Scalar::from(nc as IdxSize));

                columns.extend([
                    Column::new_scalar(format_pl_smallstr!("{}_min", col.field_name()), min, 1),
                    Column::new_scalar(format_pl_smallstr!("{}_max", col.field_name()), max, 1),
                    Column::new_scalar(
                        format_pl_smallstr!("{}_nc", col.field_name()),
                        null_count,
                        1,
                    ),
                ]);
            }
        } else {
            for (i, l) in live_columns.iter().enumerate() {
                let (_, dtype) = stat_schema.get_at_index(1 + i * 3).unwrap();
                columns.extend([
                    Column::full_null(format_pl_smallstr!("{l}_min"), 1, dtype),
                    Column::full_null(format_pl_smallstr!("{l}_max"), 1, dtype),
                    Column::full_null(format_pl_smallstr!("{l}_nc"), 1, &IDX_DTYPE),
                ]);
            }
        }

        df.vstack_mut_owned_unchecked(unsafe { DataFrame::new_no_checks(1, columns) });
    }

    df.rechunk_mut();
    let pred_result = sbp.evaluate_with_stat_df(&df);
    // a parquet file may not have statistics of all columns
    match pred_result {
        Err(PolarsError::ColumnNotFound(errstr)) => {
            return Err(PolarsError::ColumnNotFound(errstr))
        },
        Ok(bm) => Ok(bm),
        _ => Ok(Bitmap::new_with_value(false, rgs.len())),
    }
}

impl RowGroupDataFetcher {
    pub(super) async fn init_next_file_state(&mut self) -> PolarsResult<bool> {
        let Ok((path_index, row_offset, byte_source, metadata)) = self.metadata_rx.recv().await
        else {
            return Ok(false);
        };

        self.current_rg_selection = None;
        if self.use_statistics && !std::env::var("POLARS_NO_PARQUET_STATISTICS").is_ok() {
            if let Some(pred) = self.predicate.as_ref() {
                if let Some(sbp) = &pred.skip_batch_predicate {
                    let mask = create_select_mask(
                        sbp.as_ref(),
                        pred.live_columns.as_ref(),
                        &self.reader_schema,
                        &metadata.row_groups,
                    )?;

                    if self.verbose {
                        eprintln!(
                            "[ParquetSource]: Predicate pushdown: \
                            Skipping {} / {} row groups in file {}",
                            mask.set_bits(),
                            metadata.row_groups.len(),
                            self.current_path_index
                        );
                    }

                    self.current_rg_selection = Some(mask);
                }
            }
        }

        self.current_path_index = path_index;
        self.current_byte_source = byte_source;
        self.current_max_row_group_height = metadata.max_row_group_height;
        // The metadata task also sends a row offset to start counting from as it may skip files
        // during slice pushdown.
        self.current_row_offset = row_offset;
        self.current_row_group_idx = 0;
        self.current_row_groups = metadata.row_groups.into_iter();
        self.current_shared_file_state = Default::default();

        Ok(true)
    }

    pub(super) async fn next(
        &mut self,
    ) -> Option<PolarsResult<task_handles_ext::AbortOnDropHandle<PolarsResult<RowGroupData>>>> {
        'main: loop {
            for (rg_idx, row_group_metadata) in self.current_row_groups.by_ref().enumerate() {
                let current_row_offset = self.current_row_offset;
                let current_row_group_idx = self.current_row_group_idx;

                let num_rows = row_group_metadata.num_rows();
                let sorting_map = create_sorting_map(&row_group_metadata);

                self.current_row_offset = current_row_offset.saturating_add(num_rows);
                self.current_row_group_idx += 1;

                if self.use_statistics
                    && self
                        .current_rg_selection
                        .as_ref()
                        .is_some_and(|s| s.get_bit(rg_idx))
                {
                    continue;
                }

                if num_rows > IdxSize::MAX as usize {
                    let msg = operation_exceeded_idxsize_msg(
                        format!("number of rows in row group ({})", num_rows).as_str(),
                    );
                    return Some(Err(polars_err!(ComputeError: msg)));
                }

                let slice = if let Some(slice_range) = self.slice_range.clone() {
                    let (offset, len) = match SplitSlicePosition::split_slice_at_file(
                        current_row_offset,
                        num_rows,
                        slice_range,
                    ) {
                        SplitSlicePosition::Before => {
                            if self.verbose {
                                eprintln!(
                                    "[ParquetSource]: Slice pushdown: \
                                    Skipped row group {} in file {} ({} rows)",
                                    current_row_group_idx, self.current_path_index, num_rows
                                );
                            }
                            continue;
                        },
                        SplitSlicePosition::After => {
                            if self.verbose {
                                eprintln!(
                                    "[ParquetSource]: Slice pushdown: \
                                    Stop at row group {} in file {} \
                                    (remaining {} row groups will not be read)",
                                    current_row_group_idx,
                                    self.current_path_index,
                                    self.current_row_groups.len(),
                                );
                            };
                            break 'main;
                        },
                        SplitSlicePosition::Overlapping(offset, len) => (offset, len),
                    };

                    Some((offset, len))
                } else {
                    None
                };

                let current_byte_source = self.current_byte_source.clone();
                let projection = self.projection.clone();
                let current_shared_file_state = self.current_shared_file_state.clone();
                let memory_prefetch_func = self.memory_prefetch_func;
                let io_runtime = polars_io::pl_async::get_runtime();
                let current_path_index = self.current_path_index;
                let current_max_row_group_height = self.current_max_row_group_height;

                let handle = io_runtime.spawn(async move {
                    let fetched_bytes = if let DynByteSource::MemSlice(mem_slice) =
                        current_byte_source.as_ref()
                    {
                        // Skip byte range calculation for `no_prefetch`.
                        if memory_prefetch_func as usize != mem_prefetch_funcs::no_prefetch as usize
                        {
                            let slice = mem_slice.0.as_ref();

                            if let Some(columns) = projection.as_ref() {
                                for range in get_row_group_byte_ranges_for_projection(
                                    &row_group_metadata,
                                    columns.as_ref(),
                                ) {
                                    memory_prefetch_func(unsafe { slice.get_unchecked(range) })
                                }
                            } else {
                                let range = row_group_metadata.full_byte_range();
                                let range = range.start as usize..range.end as usize;

                                memory_prefetch_func(unsafe { slice.get_unchecked(range) })
                            };
                        }

                        // We have a mmapped or in-memory slice representing the entire
                        // file that can be sliced directly, so we can skip the byte-range
                        // calculations and HashMap allocation.
                        let mem_slice = mem_slice.0.clone();
                        FetchedBytes::MemSlice {
                            offset: 0,
                            mem_slice,
                        }
                    } else if let Some(columns) = projection.as_ref() {
                        let mut ranges = get_row_group_byte_ranges_for_projection(
                            &row_group_metadata,
                            columns.as_ref(),
                        )
                        .collect::<Vec<_>>();

                        let n_ranges = ranges.len();

                        let bytes_map = current_byte_source.get_ranges(&mut ranges).await?;

                        assert_eq!(bytes_map.len(), n_ranges);

                        FetchedBytes::BytesMap(bytes_map)
                    } else {
                        // We still prefer `get_ranges()` over a single `get_range()` for downloading
                        // the entire row group, as it can have less memory-copying. A single `get_range()`
                        // would naively concatenate the memory blocks of the entire row group, while
                        // `get_ranges()` can skip concatenation since the downloaded blocks are
                        // aligned to the columns.
                        let mut ranges = row_group_metadata
                            .byte_ranges_iter()
                            .map(|x| x.start as usize..x.end as usize)
                            .collect::<Vec<_>>();

                        let n_ranges = ranges.len();

                        let bytes_map = current_byte_source.get_ranges(&mut ranges).await?;

                        assert_eq!(bytes_map.len(), n_ranges);

                        FetchedBytes::BytesMap(bytes_map)
                    };

                    PolarsResult::Ok(RowGroupData {
                        fetched_bytes,
                        path_index: current_path_index,
                        row_offset: current_row_offset,
                        slice,
                        file_max_row_group_height: current_max_row_group_height,
                        row_group_metadata,
                        sorting_map,
                        shared_file_state: current_shared_file_state.clone(),
                    })
                });

                let handle = task_handles_ext::AbortOnDropHandle(handle);
                return Some(Ok(handle));
            }

            // Initialize state to the next file.
            match self.init_next_file_state().await {
                Ok(true) => {},
                Ok(false) => break,
                Err(err) => return Some(Err(err)),
            }
        }

        None
    }
}

pub(super) enum FetchedBytes {
    MemSlice { mem_slice: MemSlice, offset: usize },
    BytesMap(PlHashMap<usize, MemSlice>),
}

impl FetchedBytes {
    pub(super) fn get_range(&self, range: std::ops::Range<usize>) -> MemSlice {
        match self {
            Self::MemSlice { mem_slice, offset } => {
                let offset = *offset;
                debug_assert!(range.start >= offset);
                mem_slice.slice(range.start - offset..range.end - offset)
            },
            Self::BytesMap(v) => {
                let v = v.get(&range.start).unwrap();
                debug_assert_eq!(v.len(), range.len());
                v.clone()
            },
        }
    }
}

fn get_row_group_byte_ranges_for_projection<'a>(
    row_group_metadata: &'a RowGroupMetadata,
    columns: &'a [PlSmallStr],
) -> impl Iterator<Item = std::ops::Range<usize>> + 'a {
    columns.iter().flat_map(|col_name| {
        row_group_metadata
            .columns_under_root_iter(col_name)
            // `Option::into_iter` so that we return an empty iterator for the
            // `allow_missing_columns` case
            .into_iter()
            .flatten()
            .map(|col| {
                let byte_range = col.byte_range();
                byte_range.start as usize..byte_range.end as usize
            })
    })
}
