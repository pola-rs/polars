use std::path::PathBuf;

pub(crate) use polars_plan::plans::ArenaLpIter;
use polars_plan::plans::IR;
use polars_utils::aliases::PlHashSet;
use polars_utils::arena::{Arena, Node};

/// Get a set of the data source paths in this LogicalPlan
pub(crate) fn agg_source_paths(
    root_lp: Node,
    acc_paths: &mut PlHashSet<PathBuf>,
    lp_arena: &Arena<IR>,
) {
    lp_arena.iter(root_lp).for_each(|(_, lp)| {
        use IR::*;
        if let Scan { paths, .. } = lp {
            for path in paths.as_ref() {
                acc_paths.insert(path.clone());
            }
        }
    })
}
