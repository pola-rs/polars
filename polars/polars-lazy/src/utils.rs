use std::path::PathBuf;

use polars_core::prelude::*;
use polars_plan::prelude::*;

/// Get a set of the data source paths in this LogicalPlan
pub(crate) fn agg_source_paths(
    root_lp: Node,
    paths: &mut PlHashSet<PathBuf>,
    lp_arena: &Arena<ALogicalPlan>,
) {
    lp_arena.iter(root_lp).for_each(|(_, lp)| {
        use ALogicalPlan::*;
        match lp {
            #[cfg(feature = "csv-file")]
            CsvScan { path, .. } => {
                paths.insert(path.clone());
            }
            #[cfg(feature = "parquet")]
            ParquetScan { path, .. } => {
                paths.insert(path.clone());
            }
            #[cfg(feature = "ipc")]
            IpcScan { path, .. } => {
                paths.insert(path.clone());
            }
            // always block parallel on anonymous sources
            // as we cannot know if they will lock or not.
            AnonymousScan { .. } => {
                paths.insert("anonymous".into());
            }
            _ => {}
        }
    })
}
