use std::sync::{Arc, Mutex};

use polars_arrow::datatypes::ArrowDataType;
use polars_async::executor;
use polars_core::frame::DataFrame;
use polars_core::prelude::DataType;
use polars_core::runtime::ASYNC;
use polars_error::{PolarsResult, polars_ensure};
use polars_io::predicates::{ColumnPredicateExpr, SpecializedColumnPredicate};
use polars_io::prelude::ParallelStrategy;
use polars_parquet::read::PredicateFilter;
use polars_utils::IdxSize;

use super::row_group_data_fetch::RowGroupDataFetcher;
use super::row_group_decode::{DynamicConjunct, PredicateColumn, RowGroupDecoder, Source};
use super::{AsyncTaskData, ParquetReadImpl};
use crate::morsel::{Morsel, SourceToken, get_ideal_morsel_size};
use crate::nodes::io_sources::multi_scan::reader_interface::output::FileReaderOutputSend;
use crate::nodes::io_sources::parquet::projection::ArrowFieldProjection;
use crate::nodes::io_sources::parquet::statistics::calculate_row_group_pred_pushdown_skip_mask;
use crate::nodes::{MorselSeq, TaskPriority};
use crate::utils::tokio_handle_ext::{self, AbortOnDropHandle};

/// Whether the decoder may evaluate a predicate on this column as it decodes the values.
fn filter_while_decoding(projection: &ArrowFieldProjection) -> bool {
    use ArrowDataType as A;
    let ArrowFieldProjection::Plain(arrow_field) = projection else {
        return false;
    };
    match arrow_field.dtype() {
        A::Dictionary(..)
        | A::Decimal(..)
        | A::Decimal32(..)
        | A::Decimal64(..)
        | A::Decimal256(..)
        | A::Float16
        | A::Float32
        | A::Float64
        | A::Int128
        | A::UInt128
        | A::FixedSizeBinary(_) => false,
        dtype => !dtype.is_nested(),
    }
}

impl ParquetReadImpl {
    /// Constructs the task that distributes morsels across the engine pipelines.
    #[allow(clippy::type_complexity)]
    pub(super) fn init_morsel_distributor(&mut self) -> AsyncTaskData {
        let verbose = self.verbose;
        let use_statistics = self.options.use_statistics;

        let (mut morsel_sender, morsel_rx) = FileReaderOutputSend::new_serial();

        if let Some((_, 0)) = self.normalized_pre_slice {
            return (
                morsel_rx,
                tokio_handle_ext::AbortOnDropHandle(ASYNC.spawn(std::future::ready(Ok(())))),
            );
        }

        let projected_arrow_fields = self.projected_arrow_fields.clone();
        let is_full_projection = self.is_full_projection;

        let row_group_prefetch_size = self.config.row_group_prefetch_size;
        let predicate = self.predicate.clone();
        let memory_prefetch_func = self.memory_prefetch_func;

        let row_group_decoder = self.init_row_group_decoder();
        let row_group_decoder = Arc::new(row_group_decoder);

        let ideal_morsel_size = get_ideal_morsel_size();

        if verbose {
            eprintln!(
                "[ParquetFileReader]: ideal_morsel_size: {ideal_morsel_size}, \
                use_async_prefetch: {}, \
                concurrency: {:?}, \
                chunk_size: {:?}",
                self.byte_source.is_cloud(),
                self.byte_source.concurrency_strategy(),
                self.byte_source.chunk_size(),
            );
        }

        let metadata = self.metadata.clone();
        let normalized_pre_slice = self.normalized_pre_slice;
        let byte_source = self.byte_source.clone();

        // Prefetch loop (spawns prefetches on the tokio scheduler).

        // Three concurrency limits bound the pipeline:
        // (a) rg_prefetch_kbytes_semaphore: bounds possibly compressed projected bytes
        //     in the pipeline. Primary memory bound, but does not account for decompression.
        // (b) rg_prefetch_semaphore: bounds row group count in the pipeline. Secondary
        //     bound, only binding for degenerate cases (many tiny row groups where
        //     (a) is not exhausted).
        // (c) prefetch channel depth: sized >= (b) so it is never the binding constraint.
        //     The channel is a handoff queue between the prefetch and decode tasks, not
        //     a concurrency gate.
        //
        // Note: in-flight concurrency is separately controlled inside the object store using
        // a combination of a bytes-based and count-based semaphore. These operate at the
        // network layer and are independent of the pipeline limits above.
        // The pipeline channel depth must be >= in-flight concurrency to avoid
        // stalling the prefetch loop before the semaphores are exhausted.

        let (prefetch_send, mut prefetch_recv) =
            tokio::sync::mpsc::channel(row_group_prefetch_size);

        let row_index = self.row_index.clone();

        let pipeline_budget = self.pipeline_budget.clone();

        let rg_prefetch_prev_all_spawned = Option::take(&mut self.rg_prefetch_prev_all_spawned);
        let rg_prefetch_current_all_spawned =
            Option::take(&mut self.rg_prefetch_current_all_spawned);

        let prefetch_task = AbortOnDropHandle(ASYNC.spawn(async move {
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
                projected_arrow_fields.clone(),
                row_index,
                verbose,
            )
            .await?;

            let mut row_group_data_fetcher = RowGroupDataFetcher {
                projection: projected_arrow_fields.clone(),
                is_full_projection,
                predicate,
                slice_range,
                memory_prefetch_func,
                metadata,
                byte_source,
                row_group_slice,
                row_group_mask,
                row_offset,
            };

            if let Some(rg_prefetch_prev_all_spawned) = rg_prefetch_prev_all_spawned {
                rg_prefetch_prev_all_spawned.wait().await;
            }

            while let Some(fetch_length) = row_group_data_fetcher.peek_next_bytes() {
                let fetch_length = usize::try_from(fetch_length)
                    .expect("ParquetReadImpl: fetch_length too large for usize: {fetch_length}");

                let permit = pipeline_budget.acquire(fetch_length).await;

                // Budget reserved, spawn request
                let Some(prefetch) = row_group_data_fetcher.next().await else {
                    // Mask skipped all remaining row groups between peek and next — release permits
                    drop(permit);
                    break;
                };

                if prefetch_send.send((prefetch?, permit)).await.is_err() {
                    break;
                }
            }

            drop(rg_prefetch_current_all_spawned);

            PolarsResult::Ok(())
        }));

        // Decode loop (spawns decodes on the computational executor).
        let (decode_send, mut decode_recv) = tokio::sync::mpsc::channel(self.config.num_pipelines);
        let decode_task = AbortOnDropHandle(ASYNC.spawn(async move {
            while let Some((prefetch_task, permits)) = prefetch_recv.recv().await {
                let row_group_data = prefetch_task.await.unwrap()?;
                let row_group_decoder = row_group_decoder.clone();
                let decode_fut = executor::spawn(TaskPriority::High, async move {
                    row_group_decoder.row_group_data_to_df(row_group_data).await
                });
                if decode_send.send((decode_fut, permits)).await.is_err() {
                    break;
                }
            }
            PolarsResult::Ok(())
        }));

        // Distributes morsels across pipelines. This does not perform any CPU or I/O bound work -
        // it is purely a dispatch loop. Run on the computational executor to reduce context switches.
        //
        // `last_morsel_pipelines` is precomputed by the multi-scan layer so the split budget
        // is shared across files in the scan.
        let last_morsel_pipelines = self.config.last_morsel_pipelines;
        let disable_morsel_split = self.disable_morsel_split;
        let distribute_task = executor::spawn(TaskPriority::High, async move {
            let mut morsel_seq = MorselSeq::default();
            // Note: We don't use this (it is handled by the bridge). But morsels require a source token.
            let source_token = SourceToken::new();

            // Decode first non-empty morsel.
            let mut next = None;
            loop {
                let Some((decode_fut, permits)) = decode_recv.recv().await else {
                    break;
                };
                let df = decode_fut.await?;
                if df.height() == 0 {
                    continue;
                }

                if disable_morsel_split {
                    if morsel_sender
                        .send_morsel(Morsel::new_unregistered(
                            df,
                            morsel_seq,
                            source_token.clone(),
                        ))
                        .await
                        .is_err()
                    {
                        return Ok(());
                    }
                    drop(permits);
                    morsel_seq = morsel_seq.successor();
                    continue;
                }

                next = Some((df, permits));
                break;
            }

            while let Some((df, permits)) = next.take() {
                // Try to decode the next non-empty morsel first, so we know
                // whether the df is the last morsel.

                // Important: Drop this before awaiting the next one, or could
                // deadlock if the permit limit is 1.
                drop(permits);

                loop {
                    let Some((decode_fut, permit)) = decode_recv.recv().await else {
                        break;
                    };
                    let next_df = decode_fut.await?;
                    if next_df.height() == 0 {
                        continue;
                    }
                    next = Some((next_df, permit));
                    break;
                }

                for df in split_to_morsels(
                    &df,
                    ideal_morsel_size,
                    next.is_none(),
                    last_morsel_pipelines,
                ) {
                    if morsel_sender
                        .send_morsel(Morsel::new_unregistered(
                            df,
                            morsel_seq,
                            source_token.clone(),
                        ))
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

        let join_task = ASYNC.spawn(async move {
            prefetch_task.await.unwrap()?;
            decode_task.await.unwrap()?;
            distribute_task.await?;
            Ok(())
        });

        (morsel_rx, AbortOnDropHandle(join_task))
    }

    /// Creates a `RowGroupDecoder` that turns `RowGroupData` into DataFrames.
    /// This must be called AFTER the following have been initialized:
    /// * `self.projected_arrow_fields`
    /// * `self.physical_predicate`
    pub(super) fn init_row_group_decoder(&mut self) -> RowGroupDecoder {
        let projected_arrow_fields = self.projected_arrow_fields.clone();
        let row_index = self.row_index.clone();
        let target_values_per_thread = self.config.target_values_per_thread;
        let predicate = self.predicate.clone();

        let filters_rows = predicate.as_ref().is_some_and(|p| p.filters_rows);
        let mut use_prefiltered =
            filters_rows && matches!(self.options.parallel, ParallelStrategy::Prefiltered);
        use_prefiltered |= filters_rows && matches!(self.options.parallel, ParallelStrategy::Auto);

        let predicate_field_indices: Arc<[usize]> =
            if use_prefiltered && let Some(predicate) = predicate.as_ref() {
                projected_arrow_fields
                    .iter()
                    .enumerate()
                    .filter_map(|(i, projected_field)| {
                        predicate
                            .live_columns
                            .contains(projected_field.output_name())
                            .then_some(i)
                    })
                    .collect()
            } else {
                Default::default()
            };

        let non_predicate_field_indices: Arc<[usize]> = if use_prefiltered {
            filtered_range(
                predicate_field_indices.as_ref(),
                projected_arrow_fields.len(),
            )
            .collect()
        } else {
            Default::default()
        };

        // The predicate columns, each with its decode filter when the decoder may evaluate
        // the predicate on the values as it decodes them.
        let staged = predicate.as_ref().and_then(|p| p.staged.as_ref());
        let predicate_columns: Arc<[PredicateColumn]> = match staged {
            Some(staged) if use_prefiltered => staged
                .column_predicates
                .iter()
                .map(|(name, p)| {
                    let dynamic = p
                        .dynamic
                        .iter()
                        .map(|d| DynamicConjunct::new(d.predicate.clone(), d.source.clone()))
                        .collect();
                    let Some(field_idx) = projected_arrow_fields
                        .iter()
                        .position(|f| f.output_name() == name)
                    else {
                        assert_eq!(Some(name), row_index.as_ref().map(|ri| &ri.name));
                        return PredicateColumn {
                            source: Source::RowIndex,
                            predicate: p.predicate.clone(),
                            decode_filter: None,
                            constant: None,
                            dynamic,
                        };
                    };
                    let projection = &projected_arrow_fields[field_idx];
                    let arrow_field = projection.arrow_field();
                    let constant = p.specialized.as_ref().and_then(|s| match s {
                        SpecializedColumnPredicate::Equal(sc) if !sc.is_null() => Some(sc.clone()),
                        _ => None,
                    });
                    let decode_filter = p
                        .predicate
                        .as_ref()
                        .filter(|_| filter_while_decoding(projection))
                        .map(|predicate| PredicateFilter {
                            predicate: Arc::new(ColumnPredicateExpr::new(
                                name.clone(),
                                DataType::from_arrow_field(arrow_field),
                                arrow_field.dtype.clone(),
                                predicate.clone(),
                                p.specialized.clone(),
                            )),
                            include_values: constant.is_none(),
                        });
                    PredicateColumn {
                        source: Source::Field(field_idx),
                        predicate: p.predicate.clone(),
                        decode_filter,
                        constant,
                        dynamic,
                    }
                })
                .collect(),
            _ => Default::default(),
        };
        let rest_predicate = match staged {
            Some(staged) if use_prefiltered => staged.rest.clone(),
            _ => predicate.as_ref().map(|p| p.predicate.clone()),
        };
        let rest_field_indices: Arc<[usize]> = predicate_field_indices
            .iter()
            .copied()
            .filter(|&i| {
                predicate_columns
                    .iter()
                    .all(|c| c.source != Source::Field(i))
            })
            .collect();
        // Until a row group is measured: the predicate columns with something to
        // evaluate, then the ones whose conjuncts all keep every row, with the rest.
        let (evaluated, unset): (Vec<usize>, Vec<usize>) =
            (0..predicate_columns.len()).partition(|&c| {
                let c = &predicate_columns[c];
                c.predicate.is_some() || c.dynamic.iter().any(|d| d.source.filters_rows())
            });
        let mut passes = vec![evaluated];
        if !unset.is_empty() {
            passes.push(unset);
        } else if !predicate_columns.is_empty() && !rest_field_indices.is_empty() {
            passes.push(Vec::new());
        }

        if use_prefiltered && self.verbose {
            eprintln!(
                "[ParquetFileReader]: Pre-filtered decode enabled ({} live [{} column predicates, {} rest], {} non-live)",
                predicate_field_indices.len(),
                predicate_columns.len(),
                rest_field_indices.len(),
                non_predicate_field_indices.len()
            )
        }

        RowGroupDecoder {
            num_pipelines: self.config.num_pipelines,
            projected_arrow_fields,
            row_index,
            predicate,
            use_prefiltered,
            predicate_field_indices,
            predicate_columns,
            rest_predicate,
            rest_field_indices,
            passes: Mutex::new(Arc::new(passes)),
            non_predicate_field_indices,
            target_values_per_thread,
        }
    }
}

/// Returns 0..len in a Vec, excluding indices in `exclude`.
/// `exclude` needs to be a sorted list of unique values.
fn filtered_range(exclude: &[usize], len: usize) -> impl Iterator<Item = usize> {
    if cfg!(debug_assertions) {
        assert!(exclude.windows(2).all(|x| x[1] > x[0]));
    }

    let mut j = 0;

    (0..len).filter(move |&i| {
        if j == exclude.len() || i != exclude[j] {
            true
        } else {
            j += 1;
            false
        }
    })
}

pub(crate) fn split_to_morsels(
    df: &DataFrame,
    ideal_morsel_size: usize,
    last_morsel: bool,
    last_morsel_pipelines: usize,
) -> impl Iterator<Item = DataFrame> + '_ {
    let mut n_morsels = if df.height() > 3 * ideal_morsel_size / 2 {
        // num_rows > (1.5 * ideal_morsel_size)
        (df.height() / ideal_morsel_size).max(2)
    } else {
        1
    };

    if last_morsel {
        n_morsels = n_morsels.max(last_morsel_pipelines);
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
        assert_eq!(
            filtered_range(&[1, 3], 7).collect::<Vec<_>>().as_slice(),
            &[0, 2, 4, 5, 6]
        );
        assert_eq!(
            filtered_range(&[1, 6], 7).collect::<Vec<_>>().as_slice(),
            &[0, 2, 3, 4, 5]
        );
    }
}
