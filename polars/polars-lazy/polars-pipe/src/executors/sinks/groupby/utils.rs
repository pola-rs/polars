use hashbrown::HashMap;
use polars_core::prelude::*;
use polars_core::utils::{accumulate_dataframes_vertical_unchecked, slice_offsets};

use crate::executors::sinks::groupby::ooc::GroupBySource;
use crate::executors::sinks::groupby::ooc_state::OocState;
use crate::executors::sinks::io::block_thread_until_io_thread_done;
use crate::operators::{FinalizedSink, Sink};

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

pub(super) fn finalize_groupby(
    dfs: Vec<DataFrame>,
    output_schema: &Schema,
    ooc_state: &mut OocState,
    slice: Option<(i64, usize)>,
    split: Option<Box<dyn Sink>>,
) -> PolarsResult<FinalizedSink> {
    let df = if dfs.is_empty() {
        DataFrame::from(output_schema)
    } else {
        let mut df = accumulate_dataframes_vertical_unchecked(dfs);
        // re init to check duplicates
        unsafe { DataFrame::new(std::mem::take(df.get_columns_mut())) }?
    };

    if ooc_state.ooc {
        let mut iot = ooc_state.io_thread.lock().unwrap();
        // make sure that we reset the shared states
        // the OOC groupby will call split as well and it should
        // not send continue spilling to disk
        let iot = iot.take().unwrap();
        ooc_state.ooc = false;

        // we wait until all chunks are spilled
        block_thread_until_io_thread_done(&iot);

        Ok(FinalizedSink::Source(Box::new(GroupBySource::new(
            iot,
            df,
            split.unwrap(),
            slice,
        )?)))
    } else {
        Ok(FinalizedSink::Finished(df))
    }
}
