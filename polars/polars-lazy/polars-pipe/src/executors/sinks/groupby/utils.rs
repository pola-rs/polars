use hashbrown::HashMap;
use polars_core::utils::slice_offsets;

pub(super) fn compute_slices<K, V, HB>(
    pre_agg_partitions: &[HashMap<K, V, HB>],
    slice: Option<(i64, usize)>,
) -> Vec<Option<(usize, usize)>> {
    if let Some((offset, slice_len)) = slice {
        let total_len = pre_agg_partitions
            .iter()
            .map(|agg_map| agg_map.len())
            .sum::<usize>();
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
        pre_agg_partitions
            .iter()
            .map(|agg_map| Some((0, agg_map.len())))
            .collect()
    }
}
