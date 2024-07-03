use hashbrown::HashMap;
use polars_core::prelude::*;
use polars_core::utils::{accumulate_dataframes_vertical_unchecked, slice_offsets};

use crate::executors::sinks::group_by::ooc::GroupBySource;
use crate::executors::sinks::io::{block_thread_until_io_thread_done, IOThread};
use crate::operators::{DataChunk, FinalizedSink, Sink};

pub(super) fn default_slices<K, V, HB>(
    pre_agg_partitions: &[HashMap<K, V, HB>],
) -> Vec<Option<(usize, usize)>> {
    pre_agg_partitions
        .iter()
        .map(|agg_map| Some((0, agg_map.len())))
        .collect()
}

pub(super) fn compute_slices<K, V, HB>(
    pre_agg_partitions: &[HashMap<K, V, HB>],
    slice: Option<(i64, usize)>,
) -> Vec<Option<(usize, usize)>> {
    if let Some((offset, slice_len)) = slice {
        let total_len = pre_agg_partitions
            .iter()
            .map(|agg_map| agg_map.len())
            .sum::<usize>();

        if total_len <= slice_len {
            return default_slices(pre_agg_partitions);
        }

        let (mut offset, mut len) = slice_offsets(offset, slice_len, total_len);

        pre_agg_partitions
            .iter()
            .map(|agg_map| {
                if offset > agg_map.len() {
                    offset -= agg_map.len();
                    None
                } else {
                    let slice = Some((offset, std::cmp::min(len, agg_map.len())));
                    len = len.saturating_sub(agg_map.len() - offset);
                    offset = 0;
                    slice
                }
            })
            .collect::<Vec<_>>()
    } else {
        default_slices(pre_agg_partitions)
    }
}

pub(super) fn finalize_group_by(
    dfs: Vec<DataFrame>,
    output_schema: &Schema,
    slice: Option<(i64, usize)>,
    ooc_payload: Option<(IOThread, Box<dyn Sink>)>,
) -> PolarsResult<FinalizedSink> {
    let df = if dfs.is_empty() {
        DataFrame::empty_with_schema(output_schema)
    } else {
        let mut df = accumulate_dataframes_vertical_unchecked(dfs);
        // re init to check duplicates
        unsafe { DataFrame::new(std::mem::take(df.get_columns_mut())) }?
    };

    match ooc_payload {
        None => Ok(FinalizedSink::Finished(df)),
        Some((iot, sink)) => {
            // we wait until all chunks are spilled
            block_thread_until_io_thread_done(&iot);

            Ok(FinalizedSink::Source(Box::new(GroupBySource::new(
                iot, df, sink, slice,
            )?)))
        },
    }
}

pub(super) fn prepare_key(s: &Series, chunk: &DataChunk) -> Series {
    if s.len() == 1 && chunk.data.height() > 1 {
        s.new_from_index(0, chunk.data.height())
    } else {
        s.rechunk()
    }
}
