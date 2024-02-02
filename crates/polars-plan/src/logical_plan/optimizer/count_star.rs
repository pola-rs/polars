use super::*;

pub(super) struct CountStar {
    nodes: Vec<Node>,
}

impl CountStar {
    pub(super) fn new(nodes: Vec<Node>) -> Self {
        Self { nodes }
    }
}

impl OptimizationRule for CountStar {
    fn optimize_plan(
        &mut self,
        lp_arena: &mut Arena<ALogicalPlan>,
        _expr_arena: &mut Arena<AExpr>,
        _node: Node,
    ) -> Option<ALogicalPlan> {
        // Replace count(*) with fast pass map function
        for node in &self.nodes {
            // create a useless node
            let temp_alp = ALogicalPlan::DataFrameScan {
                df: Arc::new(Default::default()),
                schema: Arc::new(Default::default()),
                output_schema: None,
                projection: None,
                selection: None,
            };
            let temp_node = lp_arena.add(temp_alp);

            let ALogicalPlan::Projection { input, .. } = lp_arena.get(*node) else {
                unreachable!();
            };

            let ALogicalPlan::Scan {
                paths, scan_type, ..
            } = lp_arena.get(*input)
            else {
                unreachable!();
            };
            #[cfg(feature = "parquet")]
            if matches!(scan_type, FileScan::Parquet { .. } | FileScan::Csv { .. }) {
                let alp = ALogicalPlan::MapFunction {
                    input: temp_node,
                    function: FunctionNode::Count {
                        paths: paths.clone(),
                        scan_type: scan_type.clone(),
                    },
                };
                lp_arena.replace(*node, alp)
            }
        }
        None
    }
}
