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

        // create a placeholder node as the map function needs a leaf node
        let placeholder = ALogicalPlan::DataFrameScan {
            df: Arc::new(Default::default()),
            schema: Arc::new(Default::default()),
            output_schema: None,
            projection: None,
            selection: None,
        };
        #[allow(unused_variables)]
        let placeholder_node = lp_arena.add(placeholder);

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

        #[cfg(all(feature = "parquet", feature = "csv"))]
        if matches!(scan_type, FileScan::Parquet { .. } | FileScan::Csv { .. }) {
            let alp = ALogicalPlan::MapFunction {
                input: placeholder_node,
                function: FunctionNode::Count {
                    paths: paths.clone(),
                    scan_type: scan_type.clone(),
                },
            };
            lp_arena.replace(node, alp.clone());
            return Some(alp);
        }
        None
    }
}
