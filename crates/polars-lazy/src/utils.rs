use std::path::PathBuf;

use polars_core::prelude::*;
use polars_plan::prelude::*;

/// Get a set of the data source paths in this LogicalPlan
pub(crate) fn agg_source_paths(
    root_lp: Node,
    acc_paths: &mut PlHashSet<PathBuf>,
    lp_arena: &Arena<ALogicalPlan>,
) {
    lp_arena.iter(root_lp).for_each(|(_, lp)| {
        use ALogicalPlan::*;
        if let Scan { paths, .. } = lp {
            for path in paths.as_ref() {
                acc_paths.insert(path.clone());
            }
        }
    })
}
