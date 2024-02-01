use super::*;

pub(super) struct CountStar {
    nodes: Vec<Node>,
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
            let ALogicalPlan::Projection { input, .. } = lp_arena.get(*node) else {
                unreachable!();
            };

            let ALogicalPlan::Scan {
                paths, scan_type, ..
            } = lp_arena.get(*input)
            else {
                unreachable!();
            };

            let alp = ALogicalPlan::MapFunction {
                input: Default::default(),
                function: FunctionNode::Count {
                    paths: paths.clone(),
                    scan_type: scan_type.clone(),
                },
            };
            lp_arena.replace(*node, alp)
        }
        None
    }
}
