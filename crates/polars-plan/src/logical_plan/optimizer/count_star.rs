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
            paths,
            scan_type,
            predicate: None,
            ..
        } = lp_arena.get(*input)
        else {
            return None;
        };

        fn _replace_count_with_specialized_function(
            lp_arena: &mut Arena<ALogicalPlan>,
            paths: Arc<[PathBuf]>,
            scan_type: FileScan,
            node: Node,
        ) -> Option<ALogicalPlan> {
            // create a placeholder node as the map function needs a leaf node
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
            Some(alp)
        }

        match scan_type {
            #[cfg(feature = "parquet")]
            sc @ FileScan::Parquet { .. } => {
                _replace_count_with_specialized_function(lp_arena, paths.clone(), sc.clone(), node)
            },
            #[cfg(feature = "csv")]
            sc @ FileScan::Csv { .. } => {
                _replace_count_with_specialized_function(lp_arena, paths.clone(), sc.clone(), node)
            },
            #[cfg(feature = "ipc")]
            sc @ FileScan::Ipc { .. } => {
                _replace_count_with_specialized_function(lp_arena, paths.clone(), sc.clone(), node)
            },
            _ => None,
        }
    }
}
