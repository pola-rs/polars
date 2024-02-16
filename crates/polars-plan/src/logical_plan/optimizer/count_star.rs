use std::path::PathBuf;

use super::*;

pub(super) struct CountStar;

impl CountStar {
    pub(super) fn new() -> Self {
        Self
    }
}

impl OptimizationRule for CountStar {
    fn optimize_plan(
        &mut self,
        lp_arena: &mut Arena<ALogicalPlan>,
        _expr_arena: &mut Arena<AExpr>,
        node: Node,
    ) -> Option<ALogicalPlan> {
        // Replace select count(*) from datasource with fast pass map function

        // Do not allow nested structures. Logical plan should be Project -> Scan
        let ALogicalPlan::Projection { input, .. } = lp_arena.get(node) else {
            return None;
        };

        #[allow(unused_variables)]
        let ALogicalPlan::Scan {
            paths, scan_type, ..
        } = lp_arena.get(*input)
        else {
            return None;
        };

        fn _add_map_function(
            lp_arena: &mut Arena<ALogicalPlan>,
            paths: Arc<[PathBuf]>,
            scan_type: FileScan,
        ) -> Option<ALogicalPlan> {
            // create a placeholder node as the map function needs a leaf node
            let placeholder = ALogicalPlan::DataFrameScan {
                df: Arc::new(Default::default()),
                schema: Arc::new(Default::default()),
                output_schema: None,
                projection: None,
                selection: None,
            };
            let node = lp_arena.add(placeholder);

            let alp = ALogicalPlan::MapFunction {
                input: node,
                function: FunctionNode::Count { paths, scan_type },
            };
            lp_arena.replace(node, alp.clone());
            Some(alp)
        }

        match scan_type {
            #[cfg(feature = "parquet")]
            sc @ FileScan::Parquet { .. } => _add_map_function(lp_arena, paths.clone(), sc.clone()),
            #[cfg(feature = "csv")]
            sc @ FileScan::Csv { .. } => _add_map_function(lp_arena, paths.clone(), sc.clone()),
            _ => None,
        }
    }
}
