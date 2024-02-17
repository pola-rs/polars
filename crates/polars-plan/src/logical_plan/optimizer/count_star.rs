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

        let ALogicalPlan::Scan {
            paths,
            scan_type,
            predicate: None,
            ..
        } = lp_arena.get(*input)
        else {
            return None;
        };

        if matches!(scan_type, FileScan::Anonymous { .. }) {
            return None;
        }

        let paths = paths.clone();
        let scan_type = scan_type.clone();

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
}
