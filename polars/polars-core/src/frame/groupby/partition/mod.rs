mod aggregations;
mod split;
pub use aggregations::{AggState, PartitionAgg};

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
        let keys_idx = first.keys();
        let key_sort = keys_idx.argsort(false);
        let keys = df.column("groups")?.take(&keys_idx.sort(false));
        let values = first.finish(ca.dtype().clone()).take(&key_sort);

        assert_eq!(Vec::from(keys.u32()?), [Some(1), Some(2), Some(3)]);
        assert_eq!(Vec::from(values.i64()?), [Some(7), Some(9), Some(5)]);
        Ok(())
    }
}
