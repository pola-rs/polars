use std::borrow::Cow;
use std::ops::Range;
use std::sync::Arc;

use arrow::datatypes::ArrowDataType;
use polars_core::frame::DataFrame;
use polars_core::prelude::{Column, DataType, IDX_DTYPE, IntoColumn};
use polars_core::series::Series;
use polars_core::utils::arrow::bitmap::Bitmap;
use polars_core::utils::arrow::datatypes::ArrowSchemaRef;
use polars_error::{PolarsResult, polars_ensure};
use polars_io::RowIndex;
use polars_io::predicates::ScanIOPredicate;
use polars_io::prelude::_internal::{PrefilterMaskSetting, collect_statistics_with_live_columns};
use polars_io::prelude::{FileMetadata, ParallelStrategy};
use polars_utils::{IdxSize, format_pl_smallstr};

use super::row_group_data_fetch::RowGroupDataFetcher;
use super::row_group_decode::RowGroupDecoder;
use super::{AsyncTaskData, ParquetReadImpl};
use crate::async_executor;
use crate::morsel::{Morsel, SourceToken, get_ideal_morsel_size};
use crate::nodes::io_sources::multi_file_reader::reader_interface::output::FileReaderOutputSend;
use crate::nodes::{MorselSeq, TaskPriority};
use crate::utils::task_handles_ext::{self, AbortOnDropHandle};

async fn calculate_row_group_pred_pushdown_skip_mask(
    row_group_slice: Range<usize>,
    use_statistics: bool,
    predicate: Option<&ScanIOPredicate>,
    metadata: &Arc<FileMetadata>,
    reader_schema: &ArrowSchemaRef,
    mut row_index: Option<RowIndex>,
    verbose: bool,
) -> PolarsResult<Option<Bitmap>> {
    if !use_statistics {
        return Ok(None);
    }

    let Some(predicate) = predicate else {
        return Ok(None);
    };
    let Some(sbp) = predicate.skip_batch_predicate.as_ref() else {
        return Ok(None);
    };
    let sbp = sbp.clone();

    let num_row_groups = row_group_slice.len();
    let metadata = metadata.clone();
    let live_columns = predicate.live_columns.clone();
    let reader_schema = reader_schema.clone();
    let skip_row_group_mask = async_executor::spawn(TaskPriority::High, async move {
        if let Some(ri) = &mut row_index {
            for md in metadata.row_groups[0..row_group_slice.start].iter() {
                ri.offset = ri
                    .offset
                    .saturating_add(IdxSize::try_from(md.num_rows()).unwrap_or(IdxSize::MAX));
            }
        }

        let stats = collect_statistics_with_live_columns(
            &metadata.row_groups[row_group_slice.clone()],
            reader_schema.as_ref(),
            &live_columns,
            row_index.as_ref().map(|ri| (&ri.name, ri.offset)),
        )?;

        let mut columns = Vec::with_capacity(1 + live_columns.len() * 3);

        let lengths: Vec<IdxSize> = metadata.row_groups[row_group_slice.clone()]
            .iter()
            .map(|rg| rg.num_rows() as IdxSize)
            .collect();
        columns.push(Column::new("len".into(), lengths));
        for (c, stat) in live_columns.iter().zip(stats) {
            let field = reader_schema.get(c).map(Cow::Borrowed).unwrap_or_else(|| {
                let row_index = row_index.clone().unwrap();
                assert_eq!(c, &row_index.name);

                Cow::Owned(arrow::datatypes::Field {
                    name: row_index.name,
                    dtype: ArrowDataType::IDX_DTYPE,
                    is_nullable: false,
                    metadata: None,
                })
            });

            let min_name = format_pl_smallstr!("{c}_min");
            let max_name = format_pl_smallstr!("{c}_max");
            let nc_name = format_pl_smallstr!("{c}_nc");

            let (min, max, nc) = match stat {
                None => {
                    let dtype = DataType::from_arrow_field(field.as_ref());

                    (
                        Column::full_null(min_name, num_row_groups, &dtype),
                        Column::full_null(max_name, num_row_groups, &dtype),
                        Column::full_null(nc_name, num_row_groups, &IDX_DTYPE),
                    )
                },
                Some(stat) => {
                    let md = field.metadata.as_deref();

                    (
                        unsafe {
                            Series::_try_from_arrow_unchecked_with_md(
                                min_name,
                                vec![stat.min_value],
                                field.dtype(),
                                md,
                            )
                        }?
                        .into_column(),
                        unsafe {
                            Series::_try_from_arrow_unchecked_with_md(
                                max_name,
                                vec![stat.max_value],
                                field.dtype(),
                                md,
                            )
                        }?
                        .into_column(),
                        Series::from_arrow(nc_name, stat.null_count.boxed())?.into_column(),
                    )
                },
            };

            columns.extend([min, max, nc]);
        }

        let statistics_df = DataFrame::new_with_height(num_row_groups, columns)?;
        sbp.evaluate_with_stat_df(&statistics_df)
    })
    .await?;

    if verbose {
        eprintln!(
            "[ParquetFileReader]: Predicate pushdown: \
                                reading {} / {} row groups",
            skip_row_group_mask.unset_bits(),
            num_row_groups,
        );
    }

    Ok(Some(skip_row_group_mask))
}

impl ParquetReadImpl {
    /// Constructs the task that distributes morsels across the engine pipelines.
    #[allow(clippy::type_complexity)]
    pub(super) fn init_morsel_distributor(&mut self) -> AsyncTaskData {
        let verbose = self.verbose;
        let io_runtime = polars_io::pl_async::get_runtime();

        let use_statistics = self.options.use_statistics;

        let (mut morsel_sender, morsel_rx) = FileReaderOutputSend::new_serial();

        if let Some((_, 0)) = self.normalized_pre_slice {
            return (
                morsel_rx,
                task_handles_ext::AbortOnDropHandle(io_runtime.spawn(std::future::ready(Ok(())))),
            );
        }

        let reader_schema = self.schema.clone();

        let row_group_prefetch_size = self.config.row_group_prefetch_size;
        // For row group fetching, only set this if we have a projection, as it will cause individual
        // byte range requests for every column in the row group.
        let projection = (self.projected_arrow_schema.len() < self.schema.len())
            .then_some(self.projected_arrow_schema.clone());
        let predicate = self.predicate.clone();
        let memory_prefetch_func = self.memory_prefetch_func;

        let row_group_decoder = self.init_row_group_decoder();
        let row_group_decoder = Arc::new(row_group_decoder);

        let ideal_morsel_size = get_ideal_morsel_size();

        if verbose {
            eprintln!(
                "[ParquetFileReader]: ideal_morsel_size: {}",
                ideal_morsel_size
            );
        }

        let metadata = self.metadata.clone();
        let normalized_pre_slice = self.normalized_pre_slice;
        let byte_source = self.byte_source.clone();

        // Prefetch loop (spawns prefetches on the tokio scheduler).
        let (prefetch_send, mut prefetch_recv) =
            tokio::sync::mpsc::channel(row_group_prefetch_size);

        let row_index = self.row_index.clone();

        let prefetch_task = AbortOnDropHandle(io_runtime.spawn(async move {
            polars_ensure!(
                metadata.num_rows < IdxSize::MAX as usize,
                bigidx,
                ctx = "parquet file",
                size = metadata.num_rows
            );

            // Calculate the row groups that need to be read and the slice range relative to those
            // row groups.
            let mut row_offset = 0;
            let mut slice_range =
                normalized_pre_slice.map(|(offset, length)| offset..offset + length);
            let mut row_group_slice = 0..metadata.row_groups.len();
            if let Some(pre_slice) = normalized_pre_slice {
                let mut start = 0;
                let mut start_offset = 0;

                let mut num_offset_remaining = pre_slice.0;
                let mut num_length_remaining = pre_slice.1;

                for rg in &metadata.row_groups {
                    if rg.num_rows() > num_offset_remaining {
                        start_offset = num_offset_remaining;
                        num_length_remaining = num_length_remaining
                            .saturating_sub(rg.num_rows() - num_offset_remaining);
                        break;
                    }

                    row_offset += rg.num_rows();
                    num_offset_remaining -= rg.num_rows();
                    start += 1;
                }

                let mut end = start + 1;

                while num_length_remaining > 0 {
                    num_length_remaining =
                        num_length_remaining.saturating_sub(metadata.row_groups[end].num_rows());
                    end += 1;
                }

                slice_range = Some(start_offset..start_offset + pre_slice.1);
                row_group_slice = start..end;

                if verbose {
                    eprintln!(
                        "[ParquetFileReader]: Slice pushdown: \
                            reading {} / {} row groups",
                        row_group_slice.len(),
                        metadata.row_groups.len()
                    );
                }
            }

            let row_group_mask = calculate_row_group_pred_pushdown_skip_mask(
                row_group_slice.clone(),
                use_statistics,
                predicate.as_ref(),
                &metadata,
                &reader_schema,
                row_index,
                verbose,
            )
            .await?;

            let mut row_group_data_fetcher = RowGroupDataFetcher {
                projection,
                predicate,
                slice_range,
                memory_prefetch_func,
                metadata,
                byte_source,
                row_group_slice,
                row_group_mask,
                row_offset,
            };

            while let Some(prefetch) = row_group_data_fetcher.next().await {
                if prefetch_send.send(prefetch?).await.is_err() {
                    break;
                }
            }
            PolarsResult::Ok(())
        }));

        // Decode loop (spawns decodes on the computational executor).
        let (decode_send, mut decode_recv) = tokio::sync::mpsc::channel(self.config.num_pipelines);
        let decode_task = AbortOnDropHandle(io_runtime.spawn(async move {
            while let Some(prefetch) = prefetch_recv.recv().await {
                let row_group_data = prefetch.await.unwrap()?;
                let row_group_decoder = row_group_decoder.clone();
                let decode_fut = async_executor::spawn(TaskPriority::High, async move {
                    row_group_decoder.row_group_data_to_df(row_group_data).await
                });
                if decode_send.send(decode_fut).await.is_err() {
                    break;
                }
            }
            PolarsResult::Ok(())
        }));

        // Distributes morsels across pipelines. This does not perform any CPU or I/O bound work -
        // it is purely a dispatch loop. Run on the computational executor to reduce context switches.
        let last_morsel_min_split = self.config.num_pipelines;
        let distribute_task = async_executor::spawn(TaskPriority::High, async move {
            let mut morsel_seq = MorselSeq::default();
            // Note: We don't use this (it is handled by the bridge). But morsels require a source token.
            let source_token = SourceToken::new();

            // Decode first non-empty morsel.
            let mut next = None;
            loop {
                let Some(decode_fut) = decode_recv.recv().await else {
                    break;
                };
                let df = decode_fut.await?;
                if df.height() == 0 {
                    continue;
                }
                next = Some(df);
                break;
            }

            while let Some(df) = next.take() {
                // Try to decode the next non-empty morsel first, so we know
                // whether the df is the last morsel.
                loop {
                    let Some(decode_fut) = decode_recv.recv().await else {
                        break;
                    };
                    let next_df = decode_fut.await?;
                    if next_df.height() == 0 {
                        continue;
                    }
                    next = Some(next_df);
                    break;
                }

                for df in split_to_morsels(
                    &df,
                    ideal_morsel_size,
                    next.is_none(),
                    last_morsel_min_split,
                ) {
                    if morsel_sender
                        .send_morsel(Morsel::new(df, morsel_seq, source_token.clone()))
                        .await
                        .is_err()
                    {
                        return Ok(());
                    }
                    morsel_seq = morsel_seq.successor();
                }
            }
            PolarsResult::Ok(())
        });

        let join_task = io_runtime.spawn(async move {
            prefetch_task.await.unwrap()?;
            decode_task.await.unwrap()?;
            distribute_task.await?;
            Ok(())
        });

        (morsel_rx, AbortOnDropHandle(join_task))
    }

    /// Creates a `RowGroupDecoder` that turns `RowGroupData` into DataFrames.
    /// This must be called AFTER the following have been initialized:
    /// * `self.projected_arrow_schema`
    /// * `self.physical_predicate`
    pub(super) fn init_row_group_decoder(&self) -> RowGroupDecoder {
        let projected_arrow_schema = self.projected_arrow_schema.clone();
        let row_index = self.row_index.clone();
        let min_values_per_thread = self.config.min_values_per_thread;

        let mut use_prefiltered = matches!(self.options.parallel, ParallelStrategy::Prefiltered);
        use_prefiltered |=
            self.predicate.is_some() && matches!(self.options.parallel, ParallelStrategy::Auto);

        let mut predicate_arrow_field_indices = vec![];
        if use_prefiltered {
            if let Some(predicate) = self.predicate.as_ref() {
                predicate_arrow_field_indices = predicate
                    .live_columns
                    .iter()
                    // Can be `None` - if the column is e.g. a hive column, or the row index column.
                    .filter_map(|x| projected_arrow_schema.index_of(x))
                    .collect::<Vec<_>>();
                predicate_arrow_field_indices.sort_unstable();
            }
        }

        let use_prefiltered = use_prefiltered.then(PrefilterMaskSetting::init_from_env);

        let non_predicate_arrow_field_indices = if use_prefiltered.is_some() {
            filtered_range(
                predicate_arrow_field_indices.as_slice(),
                projected_arrow_schema.len(),
            )
        } else {
            vec![]
        };

        if use_prefiltered.is_some() && self.verbose {
            eprintln!(
                "[ParquetFileReader]: Pre-filtered decode enabled ({} live, {} non-live)",
                predicate_arrow_field_indices.len(),
                non_predicate_arrow_field_indices.len()
            )
        }

        let predicate_arrow_field_indices = Arc::new(predicate_arrow_field_indices);
        let non_predicate_arrow_field_indices = Arc::new(non_predicate_arrow_field_indices);

        RowGroupDecoder {
            num_pipelines: self.config.num_pipelines,
            projected_arrow_schema,
            row_index,
            predicate: self.predicate.clone(),
            use_prefiltered,
            predicate_arrow_field_indices,
            non_predicate_arrow_field_indices,
            min_values_per_thread,
        }
    }
}

/// Returns 0..len in a Vec, excluding indices in `exclude`.
/// `exclude` needs to be a sorted list of unique values.
fn filtered_range(exclude: &[usize], len: usize) -> Vec<usize> {
    if cfg!(debug_assertions) {
        assert!(exclude.windows(2).all(|x| x[1] > x[0]));
    }

    let mut j = 0;

    (0..len)
        .filter(|&i| {
            if j == exclude.len() || i != exclude[j] {
                true
            } else {
                j += 1;
                false
            }
        })
        .collect()
}

fn split_to_morsels(
    df: &DataFrame,
    ideal_morsel_size: usize,
    last_morsel: bool,
    last_morsel_min_split: usize,
) -> impl Iterator<Item = DataFrame> + '_ {
    let mut n_morsels = if df.height() > 3 * ideal_morsel_size / 2 {
        // num_rows > (1.5 * ideal_morsel_size)
        (df.height() / ideal_morsel_size).max(2)
    } else {
        1
    };

    if last_morsel {
        n_morsels = n_morsels.max(last_morsel_min_split);
    }

    let rows_per_morsel = df.height().div_ceil(n_morsels).max(1);

    (0..i64::try_from(df.height()).unwrap())
        .step_by(rows_per_morsel)
        .map(move |offset| df.slice(offset, rows_per_morsel))
        .filter(|df| df.height() > 0)
}

mod tests {

    #[test]
    fn test_filtered_range() {
        use super::filtered_range;
        assert_eq!(filtered_range(&[1, 3], 7).as_slice(), &[0, 2, 4, 5, 6]);
        assert_eq!(filtered_range(&[1, 6], 7).as_slice(), &[0, 2, 3, 4, 5]);
    }
}
