use std::path::PathBuf;

use super::*;

pub(super) struct CountStar;

impl CountStar {
    pub(super) fn new() -> Self {
        Self
    }
}

impl OptimizationRule for CountStar {
    // Replace select count(*) from datasource with specialized map function
    fn optimize_plan(
        &mut self,
        lp_arena: &mut Arena<ALogicalPlan>,
        _expr_arena: &mut Arena<AExpr>,
        node: Node,
    ) -> Option<ALogicalPlan> {
        CountStar::is_simple_count_star_plan(node, lp_arena).map(|(paths, scan_type)| {
            let placeholder = ALogicalPlan::DataFrameScan {
                df: Arc::new(Default::default()),
                schema: Arc::new(Default::default()),
                output_schema: None,
                projection: None,
                selection: None,
            };
            let placeholder_node = lp_arena.add(placeholder);

            let alp = ALogicalPlan::MapFunction {
                input: placeholder_node,
                function: FunctionNode::Count { paths, scan_type },
            };
            lp_arena.replace(node, alp.clone());
            alp
        })
    }
}

impl CountStar {
    // Only optimize Projection -> Scan simple queries, no nested structures for now
    fn is_simple_count_star_plan(
        node: Node,
        lp_arena: &mut Arena<ALogicalPlan>,
    ) -> Option<(Arc<[PathBuf]>, FileScan)> {
        // Top node should be a projection
        let ALogicalPlan::Projection { input, .. } = lp_arena.get(node) else {
            return None;
        };

        // Leaf node should be a scan without predicates
        let ALogicalPlan::Scan {
            paths,
            scan_type,
            predicate: None,
            ..
        } = lp_arena.get(*input)
        else {
            return None;
        };

        // Do not support anonymous scans
        if matches!(scan_type, FileScan::Anonymous { .. }) {
            return None;
        }

        Some((paths.clone(), scan_type.clone()))
    }
}
