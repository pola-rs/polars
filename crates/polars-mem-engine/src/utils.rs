use std::path::Path;

pub(crate) use polars_plan::plans::ArenaLpIter;
use polars_plan::plans::{ScanSources, IR};
use polars_utils::aliases::PlHashSet;
use polars_utils::arena::{Arena, Node};

/// Get a set of the data source paths in this LogicalPlan
///
/// # Notes
///
/// - Scan sources with in-memory buffers are ignored.
pub(crate) fn agg_source_paths<'a>(
    root_lp: Node,
    acc_paths: &mut PlHashSet<&'a Path>,
    lp_arena: &'a Arena<IR>,
) {
    for (_, lp) in lp_arena.iter(root_lp) {
        if let IR::Scan { sources, .. } = lp {
            match sources {
                ScanSources::Files(paths) => acc_paths.extend(paths.iter().map(|p| p.as_path())),
                ScanSources::Buffers(_) => {
                    // Ignore
                },
            }
        }
    }
}
