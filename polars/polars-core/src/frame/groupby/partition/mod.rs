mod aggregations;
mod split;
use crate::frame::groupby::GroupedMap;
use crate::prelude::UInt32Chunked;
use crate::utils::{CustomIterTools, NoNull};
pub use aggregations::{AggState, PartitionAgg};

/// Get an index per group
pub fn group_maps_to_group_index(g_maps: &[GroupedMap<Option<u64>>]) -> UInt32Chunked {
    let len = g_maps.iter().map(|tbl| tbl.len()).sum();
    let ca: NoNull<UInt32Chunked> = g_maps
        .iter()
        .map(|tbl| tbl.iter().map(|(_, v)| v.0))
        .flatten()
        .trust_my_length(len)
        .collect();
    ca.into_inner()
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::df;
    use crate::prelude::*;
    use aggregations::PartitionAgg;
    use split::IntoGroupMap;

    #[test]
    fn test_partition_groups() -> Result<()> {
        // 1: 1, 2, 4
        // 2: 3, 6
        // 3: 5
        let df = df![
            "groups" => [1u32, 1, 2, 1, 3, 2],
            "vals" => [1i64, 2, 3, 4, 5, 6]
        ]?;

        let g_maps = df.column("groups")?.u32()?.group_maps();

        let ca = df.column("vals")?.i64()?;

        let mut iter = g_maps.iter().map(|g_map| ca.part_agg_sum(g_map));
        let mut first = iter.next().unwrap().unwrap();

        first.merge(iter.filter_map(|v| v).collect());
        let keys_idx = group_maps_to_group_index(&g_maps);
        let key_sort = keys_idx.argsort(false);
        let keys = df.column("groups")?.take(&keys_idx.sort(false));
        let values = first.finish(ca.dtype().clone()).take(&key_sort);

        assert_eq!(Vec::from(keys.u32()?), [Some(1), Some(2), Some(3)]);
        assert_eq!(Vec::from(values.i64()?), [Some(7), Some(9), Some(5)]);
        Ok(())
    }
}
